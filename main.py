"""
GraphStrider: a data-agnostic, plan-and-retrieve KGQA pipeline (v3).

How a question is answered
--------------------------
1. planning (1 LLM call, RoG-style "Reasoning on Graphs"):
   The LLM never writes SPARQL and never walks the graph. Given the relation schema that was
   extracted automatically from the KB, it returns a small JSON *relation-path blueprint*:
       {"plans": [{"constraints": [{"entity": "Joel Zwick", "path": ["~directed_by", "in_language"]}]}]}
   `rel` walks subject -> object, `~rel` walks object -> subject. Several constraints in one plan are
   intersected; alternative plans are only tried if earlier ones return nothing.
2. entity_linking (0 LLM calls, BLINK-style bi-encoder):
   Every node label in the KB is embedded once (cached on disk) into a FAISS inner-product index.
   Each entity mention from the plan is embedded and linked by nearest-neighbour search
   (exact label matches are tried first). The top-k candidates are kept; the relation path picks the
   first candidate the path actually works for, so the plan also disambiguates the entity.
3. path_execution (deterministic): each constraint becomes one SPARQL property-path query
   (`VALUES ?topic { <e> } ?topic ^<p1>/<p2> ?answer`), run by pyoxigraph.
4. replanning (optional, 1 LLM call, only if every plan came back empty): the LLM gets feedback
   (where each path broke, which relations the linked entity really has) and proposes new plans.
5. nl_generation (optional, --with-nl / ad-hoc questions).

Nothing in this file is specific to MetaQA or to movies: namespaces, labels (rdfs:label, skos:prefLabel,
schema:name, ... or the IRI local name), relations and the few-shot examples are all discovered from the KB.

Web UI: `uv run streamlit run app.py` (see README.md and docs/IMPLEMENTATION.md).

Run folder:  outputs/run_<YYYYmmdd_HHMMSS>/
    run_config.json        config (key redacted), argv, versions, startup timings
    prompt_plan.txt        exact system prompt sent for planning
    prompt_answer.txt      exact system prompt sent for NL answers
    details.jsonl          one line per question: question, plans, linked entities, SPARQL, answers, metrics
    timings.csv            one row per question: per-stage seconds / tokens / cost
    queries.log            human-readable log of each question
    errors.log             full tracebacks
    summary.json / .txt    aggregated stats per stage + slowest questions + diagnosis

Env vars (in .env): KB_PATH, QA_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_QUESTIONS, LLM_MAX_TOKENS,
    LLM_PRICE_INPUT_PER_1M, LLM_PRICE_OUTPUT_PER_1M (USD, default 0 = local model),
    CACHE_DIR (default .cache), EMBED_BACKEND (fastembed | openai), EMBED_MODEL, EMBED_BASE_URL, EMBED_API_KEY,
    LINK_TOP_K (default 5), LINK_MIN_SCORE (default 0.75), MAX_PLANS (default 3), REPLAN_ROUNDS (default 1)
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import unquote

import faiss
import numpy as np
import pyoxigraph as ox
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CACHE_VERSION = 3
STAGES = ("planning", "entity_linking", "path_execution", "replanning", "nl_generation")
BRACKET_CHARS = re.compile(r"[\[\]]")

# Predicates whose objects are human-readable names for their subjects (used for linking, not for planning).
LABEL_PREDICATES = (
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://schema.org/name",
    "https://schema.org/name",
    "http://xmlns.com/foaf/0.1/name",
    "http://purl.org/dc/terms/title",
    "http://purl.org/dc/elements/1.1/title",
)
MAX_LITERAL_LABEL_CHARS = 64  # longer literals (descriptions, abstracts) are not linkable entities


@dataclass
class Config:
    kb_path: Path
    qa_path: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    max_questions: int | None = None
    cache_dir: Path = Path(".cache")
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0
    max_tokens: int | None = None
    embed_backend: str = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_base_url: str = ""
    embed_api_key: str = ""
    link_top_k: int = 5
    link_min_score: float = 0.75
    max_plans: int = 3
    max_hops: int = 4
    replan_rounds: int = 1


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw.isdigit() else default


def load_config() -> Config:
    load_dotenv()
    max_q_raw = os.getenv("MAX_QUESTIONS", "").strip()
    max_tok_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")
    return Config(
        kb_path=Path(os.getenv("KB_PATH", "kb.ttl")),
        qa_path=Path(os.getenv("QA_PATH", "MetaQA/1-hop/ntm/qa_dev.txt")),
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_model=os.getenv("LLM_MODEL", "local-model"),
        max_questions=int(max_q_raw) if max_q_raw.isdigit() else None,
        cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
        price_in_per_1m=_env_float("LLM_PRICE_INPUT_PER_1M", 0.0),
        price_out_per_1m=_env_float("LLM_PRICE_OUTPUT_PER_1M", 0.0),
        max_tokens=int(max_tok_raw) if max_tok_raw.isdigit() else None,
        embed_backend=os.getenv("EMBED_BACKEND", "fastembed").strip().lower(),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        embed_base_url=os.getenv("EMBED_BASE_URL", base_url),
        embed_api_key=os.getenv("EMBED_API_KEY", api_key),
        link_top_k=_env_int("LINK_TOP_K", 5),
        link_min_score=_env_float("LINK_MIN_SCORE", 0.75),
        max_plans=_env_int("MAX_PLANS", 3),
        max_hops=_env_int("MAX_HOPS", 4),
        replan_rounds=_env_int("REPLAN_ROUNDS", 1),
    )


def create_run_output_dir(base: str | Path = "outputs") -> Path:
    run_dir = Path(base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Tracing: time / tokens / cost per stage
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class LLMCall:
    text: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    estimated: bool


class Trace:
    """Collects per-stage timing and token usage for ONE question."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stages: dict[str, dict] = {}

    def _slot(self, name: str) -> dict:
        return self.stages.setdefault(name, {
            "seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0.0, "tokens_estimated": False})

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._slot(name)["seconds"] += time.perf_counter() - t0

    def add_llm(self, name: str, call: LLMCall) -> None:
        s = self._slot(name)
        s["seconds"] += call.seconds
        s["prompt_tokens"] += call.prompt_tokens
        s["completion_tokens"] += call.completion_tokens
        s["tokens_estimated"] |= call.estimated
        s["cost_usd"] += (call.prompt_tokens * self.cfg.price_in_per_1m
                          + call.completion_tokens * self.cfg.price_out_per_1m) / 1_000_000

    @property
    def total_seconds(self) -> float:
        return sum(s["seconds"] for s in self.stages.values())

    @property
    def total_cost(self) -> float:
        return sum(s["cost_usd"] for s in self.stages.values())

    def to_dict(self) -> dict:
        return {"total_seconds": round(self.total_seconds, 5), "total_cost_usd": round(self.total_cost, 8),
                "stages": {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in self.stages.items()}}


# --------------------------------------------------------------------------- #
# Store / LLM setup
# --------------------------------------------------------------------------- #

INVALID_PERCENT_ENCODING = re.compile(rb"%(?![0-9A-Fa-f]{2})")
RDF_FORMATS = {".ttl": ox.RdfFormat.TURTLE, ".nt": ox.RdfFormat.N_TRIPLES, ".nq": ox.RdfFormat.N_QUADS,
               ".trig": ox.RdfFormat.TRIG, ".rdf": ox.RdfFormat.RDF_XML, ".xml": ox.RdfFormat.RDF_XML,
               ".n3": ox.RdfFormat.N3}


def build_store(kb_path: Path) -> ox.Store:
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
    store = ox.Store()
    print(f"Loading knowledge base from {kb_path} ...")
    raw = INVALID_PERCENT_ENCODING.sub(b"%25", kb_path.read_bytes())
    fmt = RDF_FORMATS.get(kb_path.suffix.lower(), ox.RdfFormat.TURTLE)
    store.load(raw, format=fmt, base_iri=kb_path.resolve().as_uri())
    print("Knowledge base loaded.")
    return store


def build_llm_client(cfg: Config) -> OpenAI:
    return OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def chat(client: OpenAI, cfg: Config, messages: list[dict], temperature: float) -> LLMCall:
    kwargs = {"max_tokens": cfg.max_tokens} if cfg.max_tokens else {}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=cfg.llm_model, messages=messages,
                                          temperature=temperature, **kwargs)
    seconds = time.perf_counter() - t0
    text = THINK_BLOCK.sub("", resp.choices[0].message.content or "").strip()
    usage = resp.usage
    if usage and usage.total_tokens:
        return LLMCall(text, seconds, usage.prompt_tokens, usage.completion_tokens, False)
    prompt_est = estimate_tokens("".join(m["content"] for m in messages))
    return LLMCall(text, seconds, prompt_est, estimate_tokens(text), True)


# --------------------------------------------------------------------------- #
# KB introspection: relations, labels, linkable nodes (no dataset assumptions)
# --------------------------------------------------------------------------- #

def iri_local_name(iri: str) -> str:
    tail = re.split(r"[#/:]", iri.rstrip("/#"))[-1]
    return unquote(tail) or iri


def humanize(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)  # camelCase -> camel Case
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip().lower()


def term_key(term) -> str:
    """SPARQL/N-Triples serialization of a term; usable verbatim inside a query."""
    return str(term)


@dataclass
class Relation:
    name: str          # short name shown to the LLM (unique)
    iri: str
    count: int = 0
    literal_objects: int = 0
    datatypes: set[str] = field(default_factory=set)
    samples: list[tuple[str, str]] = field(default_factory=list)  # (subject key, object key)

    @property
    def object_kind(self) -> str:
        if self.literal_objects == 0:
            return "entity"
        if self.literal_objects == self.count:
            dts = ", ".join(sorted(iri_local_name(d) for d in self.datatypes)) or "string"
            return f"literal ({dts})"
        return "entity or literal"


@dataclass
class KBGraph:
    relations: dict[str, Relation]         # by short name
    by_iri: dict[str, Relation]
    labels: dict[str, str]                 # term key -> display label
    nodes: list[str]                       # linkable term keys
    signature: str

    def label(self, key: str) -> str:
        if key in self.labels:
            return self.labels[key]
        if key.startswith("<") and key.endswith(">"):
            return humanize(iri_local_name(key[1:-1]))
        return key


def introspect_kb(store: ox.Store, kb_path: Path, samples_per_relation: int = 25) -> KBGraph:
    stat = kb_path.stat()
    signature = f"v{CACHE_VERSION}:{kb_path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    rels: dict[str, Relation] = {}
    explicit_labels: dict[str, str] = {}
    literal_labels: dict[str, str] = {}
    nodes: dict[str, None] = {}
    label_preds = set(LABEL_PREDICATES)

    for quad in store.quads_for_pattern(None, None, None, None):
        s, p, o = quad.subject, quad.predicate, quad.object
        if not isinstance(s, ox.NamedNode):
            continue
        s_key = term_key(s)
        if p.value in label_preds:
            if isinstance(o, ox.Literal) and (o.language in (None, "", "en") or s_key not in explicit_labels):
                explicit_labels[s_key] = o.value
            nodes[s_key] = None
            continue
        rel = rels.get(p.value)
        if rel is None:
            rel = rels[p.value] = Relation(name="", iri=p.value)
        rel.count += 1
        nodes[s_key] = None
        if isinstance(o, ox.Literal):
            rel.literal_objects += 1
            rel.datatypes.add(o.datatype.value)
            if len(o.value) > MAX_LITERAL_LABEL_CHARS:
                continue
        elif not isinstance(o, ox.NamedNode):
            continue
        o_key = term_key(o)
        nodes[o_key] = None
        if isinstance(o, ox.Literal):
            literal_labels[o_key] = o.value
        if len(rel.samples) < samples_per_relation:
            rel.samples.append((s_key, o_key))

    # Unique, readable short names for relations: local name, disambiguated on collision.
    seen: dict[str, int] = {}
    for rel in sorted(rels.values(), key=lambda r: r.iri):
        base = re.sub(r"[^\w\-.]", "_", iri_local_name(rel.iri)) or "rel"
        seen[base] = seen.get(base, 0) + 1
        rel.name = base if seen[base] == 1 else f"{base}_{seen[base]}"

    labels = {**literal_labels, **explicit_labels}
    graph = KBGraph({r.name: r for r in rels.values()}, rels, labels, list(nodes), signature)
    for key in graph.nodes:
        labels.setdefault(key, graph.label(key))
    return graph


# --------------------------------------------------------------------------- #
# BLINK-style bi-encoder entity linking (FAISS nearest neighbour over label embeddings)
# --------------------------------------------------------------------------- #

class Embedder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.name = f"{cfg.embed_backend}:{cfg.embed_model}"
        if cfg.embed_backend == "fastembed":
            from fastembed import TextEmbedding
            self._model = TextEmbedding(cfg.embed_model)
        elif cfg.embed_backend == "openai":
            self._client = OpenAI(base_url=cfg.embed_base_url, api_key=cfg.embed_api_key)
        else:
            raise ValueError(f"Unknown EMBED_BACKEND '{cfg.embed_backend}' (use fastembed or openai)")

    def encode(self, texts: list[str], batch_size: int = 256, progress: bool = False) -> np.ndarray:
        if self.cfg.embed_backend == "fastembed":
            it = self._model.embed(texts, batch_size=batch_size)
            vecs = list(tqdm(it, total=len(texts), desc="Embedding entities", unit="ent") if progress else it)
        else:
            vecs = []
            batches = range(0, len(texts), batch_size)
            for i in (tqdm(batches, desc="Embedding entities", unit="batch") if progress else batches):
                resp = self._client.embeddings.create(model=self.cfg.embed_model, input=texts[i:i + batch_size])
                vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype="float32").reshape(len(texts), -1)
        faiss.normalize_L2(arr)
        return arr


@dataclass
class Candidate:
    key: str
    label: str
    score: float
    method: str  # "exact" | "dense"

    def to_dict(self) -> dict:
        return {"label": self.label, "term": self.key, "score": round(self.score, 4), "method": self.method}


class EntityLinker:
    """Bi-encoder linker: one vector per distinct label, exact inner-product (cosine) search with FAISS.

    Like BLINK, retrieval is two-stage: the bi-encoder returns a wide shortlist (RERANK_POOL x top-k) and a cheap
    second stage re-scores it. BLINK uses a cross-encoder there; we blend in character-level similarity instead,
    which costs microseconds and fixes nicknames/typos that name embeddings handle poorly ("chris nolan").
    """

    RERANK_POOL = 4

    def __init__(self, graph: KBGraph, embedder: Embedder, cfg: Config):
        t0 = time.perf_counter()
        self.graph, self.embedder, self.cfg = graph, embedder, cfg
        by_label: dict[str, list[str]] = {}
        for key in graph.nodes:
            by_label.setdefault(normalize_label(graph.labels[key]), []).append(key)
        by_label.pop("", None)
        self.label_texts = list(by_label)
        self.label_terms = [by_label[t] for t in self.label_texts]
        self.exact = {t: i for i, t in enumerate(self.label_texts)}
        vectors = self._load_or_embed()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.build_seconds = time.perf_counter() - t0

    def _load_or_embed(self) -> np.ndarray:
        digest = hashlib.sha1("\n".join([self.graph.signature, self.embedder.name, *self.label_texts])
                              .encode("utf-8")).hexdigest()[:16]
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.cache_dir / f"entity_vectors_{digest}.npy"
        if path.exists():
            vecs = np.load(path)
            if vecs.shape[0] == len(self.label_texts):
                print(f"Loaded cached entity embeddings ({vecs.shape[0]} labels) from {path}.")
                return vecs
        print(f"Embedding {len(self.label_texts)} entity labels with {self.embedder.name} (one-time, cached) ...")
        vecs = self.embedder.encode(self.label_texts, progress=True)
        np.save(path, vecs)
        return vecs

    def link_many(self, mentions: list[str]) -> dict[str, list[Candidate]]:
        mentions = list(dict.fromkeys(m for m in mentions if m.strip()))
        if not mentions:
            return {}
        k = max(1, self.cfg.link_top_k)
        normed = [normalize_label(m) for m in mentions]
        scores, ids = self.index.search(self.embedder.encode(normed), k * self.RERANK_POOL)
        out: dict[str, list[Candidate]] = {}
        for mention, norm_m, row_s, row_i in zip(mentions, normed, scores, ids):
            cands: list[Candidate] = []
            exact_id = self.exact.get(norm_m)
            if exact_id is not None:
                cands += [Candidate(key, self.graph.labels[key], 1.0, "exact") for key in self.label_terms[exact_id]]
            dense = []
            for score, idx in zip(row_s, row_i):
                if idx < 0 or idx == exact_id or score < self.cfg.link_min_score:
                    continue
                lexical = difflib.SequenceMatcher(None, norm_m, self.label_texts[idx]).ratio()
                dense.append((0.5 * float(score) + 0.5 * lexical, idx))
            for score, idx in sorted(dense, reverse=True)[:k]:
                cands += [Candidate(key, self.graph.labels[key], score, "dense") for key in self.label_terms[idx]]
            out[mention] = cands
        return out


# --------------------------------------------------------------------------- #
# RoG-style planner: one LLM call -> relation-path blueprint
# --------------------------------------------------------------------------- #

def _nice(label: str) -> bool:
    return 3 <= len(label) <= 40 and sum(ch.isalpha() for ch in label) >= 3


def _pick_sample(graph: KBGraph, rel: Relation) -> tuple[str, str] | None:
    for s, o in rel.samples:
        if _nice(graph.label(s)) and _nice(graph.label(o)):
            return s, o
    return rel.samples[0] if rel.samples else None


def build_schema_text(graph: KBGraph) -> str:
    lines = []
    for rel in sorted(graph.relations.values(), key=lambda r: -r.count):
        ex = [f'"{graph.label(s)}" -> "{graph.label(o)}"' for s, o in rel.samples[:25]
              if _nice(graph.label(s))][:2]
        lines.append(f"- {rel.name}: subject -> {rel.object_kind}; {rel.count} facts"
                     + (f"; e.g. {'; '.join(ex)}" if ex else ""))
    return "\n".join(lines) or "(no relations discovered)"


def _plan_json(*constraints: tuple[str, list[str]]) -> str:
    return json.dumps({"plans": [{"constraints": [{"entity": e, "path": p} for e, p in constraints]}]},
                      ensure_ascii=False)


def build_plan_examples(store: ox.Store, graph: KBGraph) -> list[str]:
    """Few-shot examples generated from real triples, with deliberately generic phrasing."""
    entity_rels = [r for r in sorted(graph.relations.values(), key=lambda r: -r.count) if r.samples]
    examples: list[str] = []
    if not entity_rels:
        return examples
    r1 = entity_rels[0]
    pick = _pick_sample(graph, r1)
    if pick:
        s, o = pick
        examples.append(f'Q: "what is the {humanize(r1.name)} of {graph.label(s)}"\n'
                        + _plan_json((graph.label(s), [r1.name])))
    r2 = next((r for r in entity_rels[1:] if r.object_kind == "entity"), r1)
    pick = _pick_sample(graph, r2)
    if pick:
        s, o = pick
        examples.append(f'Q: "which things have {humanize(r2.name)} {graph.label(o)}"\n'
                        + _plan_json((graph.label(o), [f"~{r2.name}"])))
        # 2-hop: from o back to its subjects, then along another relation of such a subject.
        try:
            rows = list(store.query(f"SELECT DISTINCT ?p WHERE {{ {s} ?p ?x }}"))
        except Exception:
            rows = []
        other = [graph.by_iri[t[0].value] for t in rows if t[0].value in graph.by_iri
                 and graph.by_iri[t[0].value].name != r2.name]
        if other:
            r3 = max(other, key=lambda r: r.count)
            examples.append(f'Q: "what is the {humanize(r3.name)} of the things whose {humanize(r2.name)} is '
                            f'{graph.label(o)}"\n' + _plan_json((graph.label(o), [f"~{r2.name}", r3.name])))
            o3 = next((t[0] for t in store.query(f"SELECT ?x WHERE {{ {s} <{r3.iri}> ?x }} LIMIT 1")), None)
            if o3 is not None:
                l3 = graph.label(term_key(o3))
                examples.append(f'Q: "which things have {humanize(r2.name)} {graph.label(o)} and '
                                f'{humanize(r3.name)} {l3}"\n'
                                + _plan_json((graph.label(o), [f"~{r2.name}"]), (l3, [f"~{r3.name}"])))
    return examples


def build_plan_system_prompt(store: ox.Store, graph: KBGraph, cfg: Config) -> str:
    examples = "\n\n".join(build_plan_examples(store, graph)) or "(no examples available)"
    return f"""You are a query planner for a knowledge graph. You do NOT answer the question and you do NOT write SPARQL.
You output a relation-path plan that a database engine will execute.

Relations (subject -> object):
{build_schema_text(graph)}

Output ONE JSON object and nothing else:
{{"plans": [{{"constraints": [{{"entity": "<topic entity>", "path": ["<rel>", "~<rel>", ...]}}]}}]}}

Rules:
1. "entity" is the named entity / value the question starts from, copied EXACTLY as written in the question.
2. "path" is the sequence of relations walked from that entity to the answer (1 to {cfg.max_hops} steps).
   "rel" walks subject -> object; "~rel" walks backwards object -> subject. Use ONLY the relation names above.
   Paraphrases count: map the question's wording to the closest relation by meaning, and check the direction
   against the examples in the relation list.
3. Several constraints in one plan are INTERSECTED (use this when the answer must satisfy several conditions).
4. Give 1 to {cfg.max_plans} alternative plans, most likely first. A later plan is only used if earlier ones find nothing.
5. Counting/formatting is done later: plan for the set of answer entities.

Examples:
{examples}"""


@dataclass
class Constraint:
    entity: str
    path: list[str]  # short names, "~" prefix = inverse


@dataclass
class Plan:
    constraints: list[Constraint]

    def to_dict(self) -> dict:
        return {"constraints": [{"entity": c.entity, "path": c.path} for c in self.constraints]}


def extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                elif ch != "\\":
                    esc = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def resolve_relation(graph: KBGraph, step: str) -> str | None:
    """Map an LLM-written step onto a known relation, keeping its direction. Returns '~name' / 'name'."""
    step = step.strip()
    inverse = step[:1] in ("~", "^", "-")
    name = step.lstrip("~^-").strip()
    if name.endswith(("_inverse", "_inv", "_reverse")) and name not in graph.relations:
        inverse, name = not inverse, re.sub(r"_(inverse|inv|reverse)$", "", name)
    if name.startswith("<") and name.endswith(">"):
        name = graph.by_iri[name[1:-1]].name if name[1:-1] in graph.by_iri else name
    if name not in graph.relations:
        if ":" in name and name.split(":", 1)[1] in graph.relations:
            name = name.split(":", 1)[1]
        else:
            lowered = {r.lower(): r for r in graph.relations}
            if name.lower() in lowered:
                name = lowered[name.lower()]
            else:
                close = difflib.get_close_matches(name, list(graph.relations), n=1, cutoff=0.8)
                if not close:
                    return None
                name = close[0]
    return ("~" if inverse else "") + name


def parse_plans(graph: KBGraph, text: str, cfg: Config) -> tuple[list[Plan], list[str]]:
    data, problems = extract_json(text), []
    if data is None:
        return [], ["output was not valid JSON"]
    raw_plans = data.get("plans") if isinstance(data, dict) else None
    if isinstance(data, dict) and raw_plans is None and "constraints" in data:
        raw_plans = [data]
    plans: list[Plan] = []
    for rp in raw_plans or []:
        constraints = []
        for rc in (rp.get("constraints", []) if isinstance(rp, dict) else []):
            if not isinstance(rc, dict) or not str(rc.get("entity", "")).strip():
                continue
            steps = rc.get("path") or []
            steps = [steps] if isinstance(steps, str) else steps
            resolved = [resolve_relation(graph, str(s)) for s in steps]
            bad = [s for s, r in zip(steps, resolved) if r is None]
            if bad:
                problems.append(f"unknown relation(s) {bad}")
                continue
            if 0 < len(resolved) <= cfg.max_hops:
                constraints.append(Constraint(str(rc["entity"]).strip(), resolved))
        if constraints:
            plans.append(Plan(constraints))
    if not plans and not problems:
        problems.append("no usable plan in output")
    return plans[:cfg.max_plans], problems


# --------------------------------------------------------------------------- #
# Deterministic path execution
# --------------------------------------------------------------------------- #

def path_expr(graph: KBGraph, path: list[str]) -> str:
    parts = []
    for step in path:
        rel = graph.relations[step.lstrip("~")]
        parts.append(("^" if step.startswith("~") else "") + f"<{rel.iri}>")
    return "/".join(parts)


def path_sparql(graph: KBGraph, topic_key: str, path: list[str]) -> str:
    # A multi-hop walk that returns to its starting point is never the intended answer (e.g. co-stars).
    no_self = " FILTER(?answer != ?topic)" if len(path) > 1 else ""
    return f"SELECT DISTINCT ?answer WHERE {{ VALUES ?topic {{ {topic_key} }} ?topic {path_expr(graph, path)} ?answer{no_self} }}"


def run_path(store: ox.Store, graph: KBGraph, topic_key: str, path: list[str]) -> list[str]:
    return [term_key(sol[0]) for sol in store.query(path_sparql(graph, topic_key, path)) if sol[0] is not None]


def break_point(store: ox.Store, graph: KBGraph, topic_key: str, path: list[str]) -> int:
    """Index of the first step after which the path yields nothing (len(path) if it never breaks)."""
    for i in range(1, len(path) + 1):
        q = f"ASK {{ VALUES ?topic {{ {topic_key} }} ?topic {path_expr(graph, path[:i])} ?x }}"
        if not store.query(q):
            return i - 1
    return len(path)


def entity_relations(store: ox.Store, graph: KBGraph, key: str) -> list[str]:
    q = f"SELECT DISTINCT ?p ?dir WHERE {{ {{ {key} ?p ?x BIND(0 AS ?dir) }} UNION {{ ?x ?p {key} BIND(1 AS ?dir) }} }}"
    out = []
    for sol in store.query(q):
        rel = graph.by_iri.get(sol[0].value)
        if rel:
            out.append(("~" if sol[1].value == "1" else "") + rel.name)
    return sorted(out)


@dataclass
class Attempt:
    plan: Plan
    answers: list[str]
    constraints: list[dict]
    sparql: list[str]


def execute_plan(ctx: "Context", plan: Plan, links: dict[str, list[Candidate]], trace: Trace) -> Attempt:
    """For every constraint, use the best-ranked candidate entity for which the path is non-empty."""
    answer_sets: list[list[str]] = []
    info, sparqls = [], []
    with trace.timer("path_execution"):
        for c in plan.constraints:
            cands = links.get(c.entity, [])
            entry = {"entity": c.entity, "path": c.path, "linked": None, "results": 0,
                     "candidates_tried": 0}
            found: list[str] = []
            for cand in cands:
                entry["candidates_tried"] += 1
                found = run_path(ctx.store, ctx.graph, cand.key, c.path)
                if found:
                    entry["linked"] = cand.to_dict()
                    sparqls.append(path_sparql(ctx.graph, cand.key, c.path))
                    break
            if not found and cands:
                top = cands[0]
                entry["linked"] = top.to_dict()
                entry["break_after_step"] = break_point(ctx.store, ctx.graph, top.key, c.path)
                entry["entity_relations"] = entity_relations(ctx.store, ctx.graph, top.key)
                sparqls.append(path_sparql(ctx.graph, top.key, c.path))
            entry["results"] = len(found)
            info.append(entry)
            answer_sets.append(found)
            if not found:
                break
    if len(answer_sets) == len(plan.constraints) and all(answer_sets):
        common = set(answer_sets[0]).intersection(*answer_sets[1:])
        answers = [a for a in answer_sets[0] if a in common]
    else:
        answers = []
    return Attempt(plan, answers, info, sparqls)


def replan_feedback(attempts: list[Attempt], problems: list[str]) -> str:
    lines = ["None of your plans returned any results. What happened:"]
    lines += [f"- {p}" for p in problems]
    for i, att in enumerate(attempts, 1):
        for c in att.constraints:
            if c["linked"] is None:
                lines.append(f"- plan {i}: entity \"{c['entity']}\" was not found in the graph; "
                             "copy the entity text exactly as it appears in the question.")
            elif not c["results"]:
                step = c.get("break_after_step", 0)
                where = (f"no results after step {step + 1} ({c['path'][step]})"
                         if step < len(c["path"]) else "no results")
                lines.append(f"- plan {i}: \"{c['entity']}\" -> \"{c['linked']['label']}\", path {c['path']}: "
                             f"{where}. Relations this entity actually has: {c.get('entity_relations')}")
    lines.append("Propose different plans (other relations or directions). Output only the JSON object.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# QA file parsing (brackets optional; if present they are only kept as gold entity for metrics)
# --------------------------------------------------------------------------- #

BRACKET_ENTITY = re.compile(r"\[(.*?)]")


@dataclass
class ParsedQuestion:
    question: str            # brackets stripped: what the LLM sees
    gold_entity: str | None  # bracketed entity text, if the dataset has it
    ground_truth: list[str]


def clean_question(q: str) -> str:
    return BRACKET_CHARS.sub("", q).strip()


def parse_qa_line(line: str) -> ParsedQuestion | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2 or not parts[0].strip():
        return None
    m = BRACKET_ENTITY.search(parts[0])
    return ParsedQuestion(clean_question(parts[0]), m.group(1) if m else None,
                          [a.strip() for a in parts[1].split("|") if a.strip()])


def read_qa_file(path: Path) -> Iterable[ParsedQuestion]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_qa_line(line)
            if parsed:
                yield parsed


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build_answer_system_prompt() -> str:
    return ('You are a strictly grounded answer-formatting assistant. Only use the "Raw results" given.\n'
            'If count is asked, use the exact "Result count". Do not invent facts.')


@dataclass
class Context:
    cfg: Config
    store: ox.Store
    client: OpenAI
    graph: KBGraph
    linker: EntityLinker
    plan_prompt: str
    answer_prompt: str


def process_question(ctx: Context, question: str, trace: Trace, with_nl: bool = False) -> dict:
    """Plan (1 LLM call) -> link -> execute; replan only if everything came back empty."""
    out: dict = {"llm_calls": 0, "replanned": False}
    question = clean_question(question)
    messages = [{"role": "system", "content": ctx.plan_prompt}, {"role": "user", "content": f'Q: "{question}"'}]
    attempts: list[Attempt] = []
    out["plan_rounds"] = []

    for round_no in range(ctx.cfg.replan_rounds + 1):
        stage = "planning" if round_no == 0 else "replanning"
        call = chat(ctx.client, ctx.cfg, messages, temperature=0.0)
        trace.add_llm(stage, call)
        out["llm_calls"] += 1
        plans, problems = parse_plans(ctx.graph, call.text, ctx.cfg)
        round_info = {"raw_llm_response": call.text, "plans": [p.to_dict() for p in plans], "problems": problems}
        out["plan_rounds"].append(round_info)

        with trace.timer("entity_linking"):
            links = ctx.linker.link_many([c.entity for p in plans for c in p.constraints])
        round_info["links"] = {m: [c.to_dict() for c in cands] for m, cands in links.items()}

        round_attempts = []
        for plan in plans:
            att = execute_plan(ctx, plan, links, trace)
            round_attempts.append(att)
            if att.answers:
                break
        attempts.extend(round_attempts)
        winner = next((a for a in round_attempts if a.answers), None)
        if winner or round_no == ctx.cfg.replan_rounds:
            break
        out["replanned"] = True
        messages += [{"role": "assistant", "content": call.text},
                     {"role": "user", "content": replan_feedback(round_attempts, problems)}]

    best = next((a for a in attempts if a.answers), attempts[-1] if attempts else None)
    out["attempts"] = [{"plan": a.plan.to_dict(), "constraints": a.constraints, "answers": len(a.answers)}
                       for a in attempts]
    out["plan_used"] = best.plan.to_dict() if best else None
    out["entities"] = best.constraints if best else []
    out["sparql"] = "\n".join(best.sparql) if best else ""
    out["predicted"] = [ctx.graph.label(k) for k in best.answers] if best else []

    if with_nl:
        results = out["predicted"]
        user = (f'Raw results: {", ".join(results) if results else "(no results found)"}\n'
                f'Result count: {len(results)}\nQuestion: "{question}"\nAnswer:')
        nl = chat(ctx.client, ctx.cfg, [{"role": "system", "content": ctx.answer_prompt},
                                        {"role": "user", "content": user}], temperature=0.2)
        trace.add_llm("nl_generation", nl)
        out["llm_calls"] += 1
        out["nl_answer"] = nl.text.strip()
    return out


# --------------------------------------------------------------------------- #
# Recording + summary
# --------------------------------------------------------------------------- #

def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


class RunRecorder:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.records: list[dict] = []
        self.jsonl = (run_dir / "details.jsonl").open("w", encoding="utf-8")
        self.qlog = (run_dir / "queries.log").open("w", encoding="utf-8")
        self.errlog = (run_dir / "errors.log").open("w", encoding="utf-8")
        self.csv_fh = (run_dir / "timings.csv").open("w", newline="", encoding="utf-8")
        cols = ["index", "status", "llm_calls", "total_s", "total_cost_usd"]
        for st in STAGES:
            cols += [f"{st}_s", f"{st}_prompt_tok", f"{st}_completion_tok"]
        self.csv = csv.writer(self.csv_fh)
        self.csv.writerow(cols)

    def record(self, detail: dict, trace: Trace) -> None:
        m = trace.to_dict()
        detail["metrics"] = m
        self.jsonl.write(json.dumps(detail, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        row = [detail.get("index"), detail.get("status"), detail.get("llm_calls", 0),
               m["total_seconds"], m["total_cost_usd"]]
        for st in STAGES:
            s = m["stages"].get(st, {})
            row += [s.get("seconds", 0), s.get("prompt_tokens", 0), s.get("completion_tokens", 0)]
        self.csv.writerow(row)
        self.csv_fh.flush()
        linked = [f"{e['entity']} -> {e['linked']['label'] if e.get('linked') else None} "
                  f"({e['linked']['method'] if e.get('linked') else 'unlinked'})"
                  for e in detail.get("entities") or []]
        self.qlog.write(f"[{detail.get('index')}] Q: {detail.get('question')}\n"
                        f"  status   : {detail.get('status')}  (llm calls: {detail.get('llm_calls', 0)}"
                        f"{', replanned' if detail.get('replanned') else ''})\n"
                        f"  plan     : {json.dumps(detail.get('plan_used'), ensure_ascii=False)}\n"
                        f"  linked   : {linked}\n"
                        f"  SPARQL   : {detail.get('sparql', '').replace(chr(10), ' || ')}\n"
                        f"  predicted: {str(detail.get('predicted'))[:300]}\n"
                        f"  truth    : {detail.get('ground_truth')}\n"
                        f"  time     : {m['total_seconds']:.3f}s  " +
                        " ".join(f"{k}={v['seconds']:.3f}s" for k, v in m["stages"].items()) + "\n\n")
        self.qlog.flush()
        self.records.append({"index": detail.get("index"), "question": detail.get("question"),
                             "status": detail.get("status"), "llm_calls": detail.get("llm_calls", 0),
                             "replanned": detail.get("replanned", False), "metrics": m})

    def error(self, question: str, exc: BaseException) -> None:
        self.errlog.write(f"Q: {question}\n{''.join(traceback.format_exception(exc))}\n{'-' * 60}\n")
        self.errlog.flush()

    def close(self) -> None:
        for fh in (self.jsonl, self.qlog, self.errlog, self.csv_fh):
            fh.close()


def summarize(records: list[dict], extra: dict) -> dict:
    n = len(records)
    totals = [r["metrics"]["total_seconds"] for r in records]
    stages = {}
    wall = sum(totals) or 1e-9
    for st in STAGES:
        rows = [r["metrics"]["stages"][st] for r in records if st in r["metrics"]["stages"]]
        if not rows:
            continue
        secs = [x["seconds"] for x in rows]
        stages[st] = {
            "calls": len(rows), "total_s": round(sum(secs), 3), "share_of_time": round(sum(secs) / wall, 4),
            "mean_s": round(sum(secs) / len(secs), 4), "p50_s": round(_pct(secs, 50), 4),
            "p95_s": round(_pct(secs, 95), 4), "max_s": round(max(secs), 4),
            "prompt_tokens": sum(x["prompt_tokens"] for x in rows),
            "completion_tokens": sum(x["completion_tokens"] for x in rows),
            "cost_usd": round(sum(x["cost_usd"] for x in rows), 6),
            "tokens_estimated": any(x["tokens_estimated"] for x in rows)}
        ct, s_total = stages[st]["completion_tokens"], sum(secs)
        if ct and s_total:
            stages[st]["completion_tok_per_s"] = round(ct / s_total, 2)

    slowest = []
    for r in sorted(records, key=lambda r: -r["metrics"]["total_seconds"])[:10]:
        st_map = r["metrics"]["stages"]
        dom = max(st_map, key=lambda k: st_map[k]["seconds"]) if st_map else None
        slowest.append({"index": r["index"], "question": r["question"], "total_s": r["metrics"]["total_seconds"],
                        "dominant_stage": dom, "status": r["status"]})

    summary = {
        "questions": n, "total_seconds": round(sum(totals), 3),
        "mean_seconds_per_question": round(sum(totals) / n, 4) if n else 0,
        "p50_s": round(_pct(totals, 50), 4), "p95_s": round(_pct(totals, 95), 4),
        "total_cost_usd": round(sum(r["metrics"]["total_cost_usd"] for r in records), 6),
        "mean_llm_calls_per_question": round(sum(r["llm_calls"] for r in records) / n, 3) if n else 0,
        "replanned_questions": sum(r["replanned"] for r in records),
        "stages": stages, "slowest_questions": slowest, **extra}
    summary["diagnosis"] = diagnose(summary)
    return summary


def diagnose(s: dict) -> list[str]:
    tips: list[str] = []
    st = s["stages"]
    if not st:
        return ["No completed stages to analyse."]
    top = max(st, key=lambda k: st[k]["total_s"])
    tips.append(f"Bottleneck: '{top}' = {st[top]['share_of_time']:.0%} of total time "
                f"(mean {st[top]['mean_s']}s, p95 {st[top]['p95_s']}s).")
    g = st.get("planning")
    if g:
        n = max(g["calls"], 1)
        avg_in, avg_out = g["prompt_tokens"] / n, g["completion_tokens"] / n
        if avg_in > 2000:
            tips.append(f"Planning prompt averages {avg_in:.0f} tokens/call: the relation list is large. "
                        "The system prompt is identical for every question, so enable prompt caching on the server.")
        if avg_out > 150:
            tips.append(f"Planning output averages {avg_out:.0f} tokens but a plan needs ~30-80. The model is likely "
                        "emitting reasoning. Check raw_llm_response in details.jsonl, set LLM_MAX_TOKENS, "
                        "or use a non-thinking model.")
        if g.get("completion_tok_per_s") and g["completion_tok_per_s"] < 15:
            tips.append(f"Generation speed is only {g['completion_tok_per_s']} tok/s: model/hardware bound "
                        "(try a smaller/quantized model or GPU offload).")
    if s.get("questions") and s.get("replanned_questions", 0) / s["questions"] > 0.2:
        tips.append(f"{s['replanned_questions']} questions needed a 2nd LLM call (replanning): first plans often "
                    "come back empty. Look at 'attempts' in details.jsonl for wrong relation directions.")
    e = st.get("entity_linking")
    if e and e["share_of_time"] > 0.10:
        tips.append("entity_linking is >10% of time: use a smaller EMBED_MODEL or the fastembed backend.")
    d = st.get("path_execution")
    if d and d["share_of_time"] > 0.10:
        tips.append("path_execution is >10% of time: plans fan out widely (inspect long paths / huge answer sets).")
    ent = s.get("entity_stats")
    if ent and ent.get("questions_with_entities"):
        q = ent["questions_with_entities"]
        if ent["dense"] / q > 0.10:
            tips.append(f"{ent['dense'] / q:.0%} of questions were linked by vector search rather than exact label "
                        "match: mentions differ from KB labels (check 'links' in details.jsonl).")
        if ent["not_linked"]:
            tips.append(f"{ent['not_linked']} question(s) had a mention with no candidate above LINK_MIN_SCORE="
                        "(lower it or raise LINK_TOP_K).")
    if any(v.get("tokens_estimated") for v in st.values()):
        tips.append("Token counts are ESTIMATED (server returned no usage). Cost figures are approximate.")
    return tips


def render_summary(s: dict) -> str:
    L = ["=" * 78, "RUN SUMMARY", "=" * 78,
         f"Questions: {s['questions']}  | total {s['total_seconds']}s | mean {s['mean_seconds_per_question']}s "
         f"| p50 {s['p50_s']}s | p95 {s['p95_s']}s | cost ${s['total_cost_usd']}",
         f"LLM calls/question: {s['mean_llm_calls_per_question']} | replanned: {s['replanned_questions']}"]
    if "accuracy" in s:
        a = s["accuracy"]
        L.append(f"Accuracy (any-hit): {a['any_hit']:.2%} | exact-set: {a['exact']:.2%} | "
                 f"errors: {a['errors']} | entity-link match vs gold: {a.get('entity_gold_match', 'n/a')}")
    L += ["", f"{'stage':<20}{'calls':>6}{'total_s':>9}{'share':>7}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}"
              f"{'in_tok':>9}{'out_tok':>9}{'cost$':>9}"]
    for k, v in s["stages"].items():
        L.append(f"{k:<20}{v['calls']:>6}{v['total_s']:>9}{v['share_of_time']:>7.0%}{v['mean_s']:>8}"
                 f"{v['p50_s']:>8}{v['p95_s']:>8}{v['max_s']:>8}{v['prompt_tokens']:>9}"
                 f"{v['completion_tokens']:>9}{v['cost_usd']:>9}")
    L += ["", "Startup timings (s): " + json.dumps(s.get("startup_timings", {})), "", "Slowest questions:"]
    L += [f"  #{x['index']} {x['total_s']}s [{x['dominant_stage']}] {x['status']} :: {x['question']}"
          for x in s["slowest_questions"]]
    L += ["", "Diagnosis:"] + [f"  - {t}" for t in s["diagnosis"]]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def norm(text: str) -> str:
    return normalize_label(text)


@dataclass
class EvalStats:
    total: int = 0
    correct: int = 0
    exact: int = 0
    errors: int = 0
    gold_checked: int = 0
    gold_match: int = 0
    aborted_early: bool = False

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _prompt_continue(question: str, status: str) -> bool:
    print(f"\nError processing question: {question}\nStatus: {status}")
    while True:
        try:
            ans = input("Continue evaluating? [Y/n]: ").strip().lower()
        except EOFError:
            return True
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def load_questions(cfg: Config) -> list[ParsedQuestion]:
    qs = list(read_qa_file(cfg.qa_path))
    return qs[:cfg.max_questions] if cfg.max_questions else qs


def evaluate(ctx: Context, rec: RunRecorder, with_nl: bool, prompt_on_error: bool,
             questions: list[ParsedQuestion] | None = None,
             on_progress: Callable[[int, int, dict, EvalStats], bool | None] | None = None) -> tuple[EvalStats, dict]:
    """Evaluate a QA set. `on_progress(done, total, detail, stats)` runs after every question;
    returning False stops the run early (used by the web UI's stop button)."""
    stats = EvalStats()
    ent = {"questions_with_entities": 0, "exact": 0, "dense": 0, "not_linked": 0}
    qs = questions if questions is not None else load_questions(ctx.cfg)
    error_seen = False

    for pq in (qs if on_progress else tqdm(qs, desc="Evaluating", unit="q")):
        stats.total += 1
        trace = Trace(ctx.cfg)
        detail = {"index": stats.total, "question": pq.question, "gold_entity": pq.gold_entity,
                  "ground_truth": pq.ground_truth}
        failed = False
        try:
            detail.update(process_question(ctx, pq.question, trace, with_nl))
            pred = {norm(a) for a in detail["predicted"]}
            truth = {norm(t) for t in pq.ground_truth}
            hit = bool(pred & truth)
            infos = detail["entities"]
            methods = {i["linked"]["method"] if i.get("linked") else "not_linked" for i in infos}
            if infos:
                ent["questions_with_entities"] += 1
                for key in ("exact", "dense", "not_linked"):
                    ent[key] += key in methods
            if pq.gold_entity:
                stats.gold_checked += 1
                gold_hit = norm(pq.gold_entity) in {norm(i["linked"]["label"]) for i in infos if i.get("linked")}
                detail["entity_matches_gold"] = gold_hit
                stats.gold_match += gold_hit
            stats.correct += hit
            stats.exact += (pred == truth)
            detail["status"] = "correct" if hit else ("entity_not_linked" if "not_linked" in methods
                                                      else "incorrect")
        except Exception as exc:
            failed = True
            stats.errors += 1
            detail["status"] = f"error: {exc}"
            rec.error(pq.question, exc)
        rec.record(detail, trace)
        if on_progress and on_progress(stats.total, len(qs), detail, stats) is False:
            stats.aborted_early = True
            break
        if failed and prompt_on_error and not error_seen:
            error_seen = True
            if not _prompt_continue(pq.question, detail["status"]):
                stats.aborted_early = True
                break
    return stats, ent


def accuracy_dict(stats: EvalStats) -> dict:
    return {"any_hit": stats.accuracy,
            "exact": stats.exact / stats.total if stats.total else 0.0,
            "errors": stats.errors,
            "entity_gold_match": (f"{stats.gold_match / stats.gold_checked:.2%}" if stats.gold_checked else "n/a"),
            "aborted_early": stats.aborted_early}


# --------------------------------------------------------------------------- #
# Setup shared by the CLI and the web UI (app.py)
# --------------------------------------------------------------------------- #

def build_context(cfg: Config, log: Callable[[str], None] = print) -> tuple[Context, dict[str, float]]:
    """Load the KB, introspect its schema, build the entity index and the planning prompt."""
    startup: dict[str, float] = {}
    t0 = time.perf_counter()
    store = build_store(cfg.kb_path)
    startup["kb_load"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    graph = introspect_kb(store, cfg.kb_path)
    startup["schema"] = round(time.perf_counter() - t0, 3)
    log(f"Schema: {len(graph.relations)} relations, {len(graph.nodes)} linkable nodes.")

    t0 = time.perf_counter()
    linker = EntityLinker(graph, Embedder(cfg), cfg)
    startup["entity_index_build"] = round(time.perf_counter() - t0, 3)
    log(f"Entity index: {linker.index.ntotal} label vectors ({linker.embedder.name}, FAISS flat IP).")

    t0 = time.perf_counter()
    plan_prompt = build_plan_system_prompt(store, graph, cfg)
    startup["plan_prompt"] = round(time.perf_counter() - t0, 3)
    ctx = Context(cfg, store, build_llm_client(cfg), graph, linker, plan_prompt, build_answer_system_prompt())
    return ctx, startup


def write_run_header(run_dir: Path, ctx: Context, startup: dict[str, float]) -> None:
    cfg, graph, linker = ctx.cfg, ctx.graph, ctx.linker
    (run_dir / "prompt_plan.txt").write_text(ctx.plan_prompt, encoding="utf-8")
    (run_dir / "prompt_answer.txt").write_text(ctx.answer_prompt, encoding="utf-8")
    cfg_dump = {k: str(v) for k, v in vars(cfg).items()}
    cfg_dump["llm_api_key"] = cfg_dump["embed_api_key"] = "***redacted***"
    (run_dir / "run_config.json").write_text(json.dumps({
        "started_at": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
        "config": cfg_dump, "python": platform.python_version(),
        "pyoxigraph": getattr(ox, "__version__", "unknown"), "faiss": getattr(faiss, "__version__", "unknown"),
        "embedder": linker.embedder.name,
        "plan_prompt_chars": len(ctx.plan_prompt),
        "plan_prompt_est_tokens": estimate_tokens(ctx.plan_prompt),
        "n_relations": len(graph.relations), "n_linkable_nodes": len(graph.nodes),
        "n_label_vectors": linker.index.ntotal, "startup_timings_s": startup}, indent=2), encoding="utf-8")


def finalize_run(rec: RunRecorder, extra: dict) -> dict | None:
    """Close the recorder and write summary.json / summary.txt. Returns the summary (None if nothing ran)."""
    rec.close()
    if not rec.records:
        return None
    summary = summarize(rec.records, extra)
    (rec.dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (rec.dir / "summary.txt").write_text(render_summary(summary), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def print_metrics(trace: Trace) -> None:
    print("\n--- Metrics ---")
    for name, s in trace.stages.items():
        tok = f" ({s['prompt_tokens']} in / {s['completion_tokens']} out tokens)" if s["prompt_tokens"] else ""
        print(f"{name:<18}: {s['seconds']:.4f}s{tok}")
    print(f"{'TOTAL':<18}: {trace.total_seconds:.4f}s | cost ${trace.total_cost:.6f}")


def ask_adhoc(ctx: Context, rec: RunRecorder, question: str) -> None:
    trace = Trace(ctx.cfg)
    detail = {"index": len(rec.records) + 1, "question": clean_question(question), "ground_truth": None}
    try:
        detail.update(process_question(ctx, question, trace, with_nl=True))
        detail["status"] = "answered"
        print(f"\nPlan: {json.dumps(detail['plan_used'], ensure_ascii=False)}")
        for e in detail["entities"]:
            linked = e.get("linked")
            print(f"Linked: {e['entity']!r} -> "
                  + (f"{linked['label']!r} ({linked['method']}, score {linked['score']})" if linked else "NOT FOUND"))
        if detail["replanned"]:
            print("(first plan returned nothing; replanned)")
        print(f"SPARQL:\n{detail['sparql']}")
        print(f"Answer: {detail['nl_answer']}")
    except Exception as exc:
        detail["status"] = f"error: {exc}"
        rec.error(question, exc)
        print(f"Error: {exc}")
    rec.record(detail, trace)
    print_metrics(trace)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphStrider")
    p.add_argument("-q", "--question", type=str, help="Ask ad-hoc question (no brackets needed)")
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL")
    p.add_argument("--with-nl", action="store_true", help="Also run NL answer stage during evaluation")
    p.add_argument("--yes", action="store_true", help="Never prompt to continue after an error")
    return p.parse_args(argv)


def main() -> int:
    args = parse_cli_args()
    cfg = load_config()
    run_dir = create_run_output_dir()
    print(f"Run folder: {run_dir}")
    try:
        ctx, startup = build_context(cfg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    write_run_header(run_dir, ctx, startup)

    rec = RunRecorder(run_dir)
    extra: dict = {"startup_timings": startup, "model": cfg.llm_model, "mode": "eval"}
    try:
        if args.question:
            extra["mode"] = "single"
            ask_adhoc(ctx, rec, args.question)
        elif args.interactive:
            extra["mode"] = "interactive"
            while True:
                try:
                    q = input("\nAsk> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in ("exit", "quit"):
                    break
                if q:
                    ask_adhoc(ctx, rec, q)
        else:
            stats, ent = evaluate(ctx, rec, args.with_nl, prompt_on_error=not args.yes)
            extra["entity_stats"] = ent
            extra["accuracy"] = accuracy_dict(stats)
            print(f"\nTotal: {stats.total} | Correct: {stats.correct} | Accuracy: {stats.accuracy:.2%}")
    finally:
        summary = finalize_run(rec, extra)
        if summary:
            print("\n" + render_summary(summary))
        print(f"\nAll artifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
