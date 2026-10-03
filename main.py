"""
GraphStrider: a Text-to-SPARQL retrieval pipeline (v2).

Changes vs v1
-------------
* No brackets needed. The LLM infers the entity itself and writes it as ex:entity_name.
* Safety net for the "LLM guessed the wrong name" problem: after generation, every ex:<entity>
  in the SPARQL is verified against the KB. Case problems are fixed and near-misses
  (ex:chris_nolan -> ex:christopher_nolan) are repaired by fuzzy matching. Unresolvable
  entities are flagged in the logs.
* Every stage (sparql_generation, entity_resolution, db_execution, nl_generation) is timed and
  its tokens/cost recorded per question; each run folder gets a summary with a bottleneck diagnosis.

Run folder:  outputs/run_<YYYYmmdd_HHMMSS>/
    run_config.json        config (key redacted), argv, versions, startup timings
    prompt_sparql.txt      exact system prompt sent for SPARQL generation
    prompt_answer.txt      exact system prompt sent for NL answers
    details.jsonl          one line per question: question, SPARQL, raw LLM output, entities, answers, metrics
    timings.csv            one row per question: per-stage seconds / tokens / cost (open in Excel/pandas)
    queries.log            human-readable log of each question
    errors.log             full tracebacks
    summary.json           aggregated stats per stage + slowest questions + diagnosis
    summary.txt            same, human readable

Env vars (in .env): KB_PATH, QA_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, MAX_QUESTIONS,
    SCHEMA_CACHE_PATH, LLM_PRICE_INPUT_PER_1M, LLM_PRICE_OUTPUT_PER_1M (USD, default 0 = local model),
    LLM_MAX_TOKENS (optional), FUZZY_CUTOFF (default 0.82)
"""

from __future__ import annotations

import argparse
import csv
import difflib
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
from typing import Iterable, Iterator

import pyoxigraph as ox
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:  # optional, much faster fuzzy matching
    from rapidfuzz import fuzz as _rf_fuzz, process as _rf_process
except ImportError:  # pragma: no cover
    _rf_fuzz = _rf_process = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EX_PREFIX = "http://example.org/"
SPARQL_PREFIX_HEADER = "PREFIX ex: <http://example.org/>\n"
BRACKET_CHARS = re.compile(r"[\[\]]")
EX_REF = re.compile(r"\bex:([^\s{}<>;,()]+)")
SCHEMA_VERSION = 2
STAGES = ("sparql_generation", "entity_resolution", "db_execution", "nl_generation")


@dataclass
class Config:
    kb_path: Path
    qa_path: Path
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    max_questions: int | None = None
    schema_cache_path: Path = Path("schema_cache.json")
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0
    max_tokens: int | None = None
    fuzzy_cutoff: float = 0.82


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def load_config() -> Config:
    load_dotenv()
    max_q_raw = os.getenv("MAX_QUESTIONS", "").strip()
    max_tok_raw = os.getenv("LLM_MAX_TOKENS", "").strip()
    return Config(
        kb_path=Path(os.getenv("KB_PATH", "kb.ttl")),
        qa_path=Path(os.getenv("QA_PATH", "MetaQA/1-hop/ntm/qa_dev.txt")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        llm_model=os.getenv("LLM_MODEL", "local-model"),
        max_questions=int(max_q_raw) if max_q_raw.isdigit() else None,
        schema_cache_path=Path(os.getenv("SCHEMA_CACHE_PATH", "schema_cache.json")),
        price_in_per_1m=_env_float("LLM_PRICE_INPUT_PER_1M", 0.0),
        price_out_per_1m=_env_float("LLM_PRICE_OUTPUT_PER_1M", 0.0),
        max_tokens=int(max_tok_raw) if max_tok_raw.isdigit() else None,
        fuzzy_cutoff=_env_float("FUZZY_CUTOFF", 0.82),
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


def build_store(kb_path: Path) -> ox.Store:
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
    store = ox.Store()
    print(f"Loading knowledge base from {kb_path} ...")
    raw = INVALID_PERCENT_ENCODING.sub(b"%25", kb_path.read_bytes())
    store.load(raw, format=ox.RdfFormat.TURTLE, base_iri=EX_PREFIX)
    print("Knowledge base loaded.")
    return store


def build_llm_client(cfg: Config) -> OpenAI:
    return OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)


def chat(client: OpenAI, cfg: Config, messages: list[dict], temperature: float) -> LLMCall:
    kwargs = {"max_tokens": cfg.max_tokens} if cfg.max_tokens else {}
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=cfg.llm_model, messages=messages,
                                          temperature=temperature, **kwargs)
    seconds = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage and usage.total_tokens:
        return LLMCall(text, seconds, usage.prompt_tokens, usage.completion_tokens, False)
    prompt_est = estimate_tokens("".join(m["content"] for m in messages))
    return LLMCall(text, seconds, prompt_est, estimate_tokens(text), True)


# --------------------------------------------------------------------------- #
# Entity helpers
# --------------------------------------------------------------------------- #

def entity_to_uri_local_name(entity: str) -> str:
    return entity.strip().lower().replace(" ", "_")


def entity_exists(store: ox.Store, local_name: str) -> bool:
    try:
        uri = ox.NamedNode(EX_PREFIX + local_name).value
        return bool(store.query("ASK { <%s> ?p ?o }" % uri)) or bool(store.query("ASK { ?s ?p <%s> }" % uri))
    except Exception:
        return False


class EntityIndex:
    """All entity local names in the KB, used to repair names the LLM got slightly wrong."""

    def __init__(self, store: ox.Store, predicate_set: set[str], cutoff: float):
        self.store, self.cutoff = store, cutoff
        t0 = time.perf_counter()
        q = "SELECT DISTINCT ?e WHERE { { ?e ?p ?o } UNION { ?s ?p ?e } FILTER(isIRI(?e)) }"
        names = []
        for sol in store.query(q):
            term = sol[0]
            if isinstance(term, ox.NamedNode) and term.value.startswith(EX_PREFIX):
                local = term.value[len(EX_PREFIX):]
                if local not in predicate_set:
                    names.append(local)
        self.names = names
        self.build_seconds = time.perf_counter() - t0

    def closest(self, name: str) -> tuple[str, float] | None:
        if not self.names:
            return None
        if _rf_process is not None:
            hit = _rf_process.extractOne(name, self.names, scorer=_rf_fuzz.ratio,
                                         score_cutoff=self.cutoff * 100)
            return (hit[0], hit[1] / 100) if hit else None
        first = difflib.get_close_matches(name, self.names, n=1, cutoff=self.cutoff)
        return (first[0], difflib.SequenceMatcher(None, name, first[0]).ratio()) if first else None


def resolve_entities(store: ox.Store, index: EntityIndex, predicate_set: set[str],
                     sparql: str) -> tuple[str, list[dict]]:
    """Verify every ex:<entity> in the SPARQL; fix case; fuzzy-repair near misses."""
    infos: list[dict] = []
    cache: dict[str, dict] = {}

    def repl(m: re.Match) -> str:
        raw, trail = m.group(1), ""
        while raw.endswith("."):
            raw, trail = raw[:-1], trail + "."
        if raw in predicate_set:
            return m.group(0)
        if raw not in cache:
            info = {"llm": raw}
            if entity_exists(store, raw):
                info.update(status="ok", resolved=raw)
            elif entity_exists(store, entity_to_uri_local_name(raw)):
                info.update(status="case_fixed", resolved=entity_to_uri_local_name(raw))
            else:
                near = index.closest(entity_to_uri_local_name(raw))
                if near:
                    info.update(status="repaired", resolved=near[0], score=round(near[1], 3))
                else:
                    info.update(status="not_found", resolved=raw)
            cache[raw] = info
            infos.append(info)
        return "ex:" + cache[raw]["resolved"] + trail

    return EX_REF.sub(repl, sparql), infos


# --------------------------------------------------------------------------- #
# QA file parsing (brackets optional; if present they are only kept as gold entity for metrics)
# --------------------------------------------------------------------------- #

BRACKET_ENTITY = re.compile(r"\[(.*?)]")


@dataclass
class ParsedQuestion:
    question: str            # brackets stripped: what the LLM sees
    gold_entity: str | None  # local name from brackets, if the dataset has them
    ground_truth: list[str]


def clean_question(q: str) -> str:
    return BRACKET_CHARS.sub("", q).strip()


def parse_qa_line(line: str) -> ParsedQuestion | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2 or not parts[0].strip():
        return None
    m = BRACKET_ENTITY.search(parts[0])
    return ParsedQuestion(clean_question(parts[0]),
                          entity_to_uri_local_name(m.group(1)) if m else None,
                          [a.strip() for a in parts[1].split("|") if a.strip()])


def read_qa_file(path: Path) -> Iterable[ParsedQuestion]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_qa_line(line)
            if parsed:
                yield parsed


# --------------------------------------------------------------------------- #
# Dynamic schema & few-shot examples (no brackets)
# --------------------------------------------------------------------------- #

def readable(local_name: str) -> str:
    return local_name.replace("_", " ")


PREDICATE_QUESTION_TEMPLATES: dict[str, tuple[str, str]] = {
    "directed_by": ("subject", "Who directed {entity}"),
    "written_by": ("subject", "Who wrote {entity}"),
    "starred_actors": ("object", "What movies did {entity} star in"),
    "has_genre": ("subject", "What genre is {entity}"),
    "has_tags": ("subject", "What are the tags for {entity}"),
    "in_language": ("subject", "What language is {entity} in"),
    "release_year": ("subject", "What year was {entity} released"),
    "has_imdb_rating": ("subject", "What is the IMDB rating of {entity}"),
    "has_imdb_votes": ("subject", "How many IMDB votes does {entity} have"),
}


def _fallback_template(predicate_local: str) -> tuple[str, str]:
    return "subject", f"What is the {predicate_local.replace('_', ' ')} of {{entity}}"


@dataclass
class SchemaExample:
    predicate: str
    direction: str
    entity_local: str
    answer_local: str
    question: str
    sparql: str


@dataclass
class KBSchema:
    predicates: list[str]
    examples: list[SchemaExample]


def discover_predicates(store: ox.Store) -> list[str]:
    preds = set()
    for sol in store.query("SELECT DISTINCT ?p WHERE { ?s ?p ?o }"):
        term = sol[0]
        if isinstance(term, ox.NamedNode) and term.value.startswith(EX_PREFIX):
            preds.add(term.value[len(EX_PREFIX):])
    return sorted(preds)


def sample_triple(store: ox.Store, predicate_local: str) -> tuple[str, str] | None:
    try:
        results = store.query(f"{SPARQL_PREFIX_HEADER}SELECT ?s ?o WHERE {{ ?s ex:{predicate_local} ?o }} LIMIT 1")
    except Exception:
        return None
    for sol in results:
        s_term, o_term = sol[0], sol[1]
        if not isinstance(s_term, ox.NamedNode) or not s_term.value.startswith(EX_PREFIX):
            continue
        if isinstance(o_term, ox.NamedNode) and o_term.value.startswith(EX_PREFIX):
            obj = o_term.value[len(EX_PREFIX):]
        else:
            obj = getattr(o_term, "value", str(o_term))
        return s_term.value[len(EX_PREFIX):], obj
    return None


def build_schema_examples(store: ox.Store, predicates: list[str]) -> list[SchemaExample]:
    examples = []
    for predicate in predicates:
        sample = sample_triple(store, predicate)
        if sample is None:
            continue
        subject_local, object_text = sample
        direction, template = PREDICATE_QUESTION_TEMPLATES.get(predicate, _fallback_template(predicate))
        if direction == "subject":
            entity_local, answer_local = subject_local, object_text
            sparql = f"{SPARQL_PREFIX_HEADER}SELECT ?answer WHERE {{ ex:{entity_local} ex:{predicate} ?answer }}"
        else:
            entity_local, answer_local = entity_to_uri_local_name(object_text), subject_local
            sparql = f"{SPARQL_PREFIX_HEADER}SELECT ?answer WHERE {{ ?answer ex:{predicate} ex:{entity_local} }}"
        examples.append(SchemaExample(predicate, direction, entity_local, answer_local,
                                      template.format(entity=readable(entity_local)), sparql))
    return examples


def get_or_build_schema(store: ox.Store, cfg: Config) -> KBSchema:
    stat = cfg.kb_path.stat()
    signature = f"v{SCHEMA_VERSION}:{stat.st_size}:{int(stat.st_mtime)}"
    if cfg.schema_cache_path.exists():
        try:
            cached = json.loads(cfg.schema_cache_path.read_text(encoding="utf-8"))
            if cached.get("kb_signature") == signature:
                print(f"Loaded cached schema ({len(cached['predicates'])} predicates).")
                return KBSchema(cached["predicates"], [SchemaExample(**e) for e in cached["examples"]])
        except Exception:
            pass
    print("Discovering KB schema...")
    local = discover_predicates(store)
    schema = KBSchema([f"ex:{p}" for p in local], build_schema_examples(store, local))
    cfg.schema_cache_path.write_text(json.dumps(
        {"kb_signature": signature, "predicates": schema.predicates,
         "examples": [vars(e) for e in schema.examples]}, indent=2, ensure_ascii=False), encoding="utf-8")
    return schema


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def build_system_prompt(schema: KBSchema) -> str:
    predicates_block = "\n".join(schema.predicates) or "(none discovered)"
    blocks = [f'Q: "{ex.question}"\n{ex.sparql}' for ex in schema.examples]
    genre = next((e for e in schema.examples if e.predicate == "has_genre"), None)
    if genre:
        g = readable(genre.answer_local)
        blocks.append(
            f'Q: "how many {g} genre movies in 2000"\n{SPARQL_PREFIX_HEADER}'
            f"SELECT ?answer WHERE {{ ?answer ex:has_genre ex:{genre.answer_local} . ?answer ex:release_year 2000 }}")
    examples_text = "\n\n".join(blocks) or "(no examples available)"
    return f"""You are a deterministic natural-language-to-SPARQL translator for a knowledge graph.
PREFIX ex: <http://example.org/>

Valid predicates:
{predicates_block}

Rules:
1. Find the named entity in the question (movie title, person, genre, language, year, tag).
2. Write it as ex:<local_name>, where local_name is the entity text EXACTLY as written in the question,
   lowercased, with spaces replaced by underscores (e.g. "Christopher Nolan" -> ex:christopher_nolan).
   Never abbreviate, shorten, translate or guess nicknames.
3. Use ONLY the predicates listed above. The answer variable must be ?answer.
4. Output exactly ONE raw SPARQL SELECT query. No markdown, no explanation.

Examples:
{examples_text}"""


def build_answer_system_prompt() -> str:
    return ('You are a strictly grounded answer-formatting assistant. Only use the "Raw results" given.\n'
            'If count is asked, use the exact "Result count". Do not invent facts.')


def strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:sparql|turtle)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"```\s*$", "", text).strip()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class Context:
    cfg: Config
    store: ox.Store
    client: OpenAI
    predicate_set: set[str]
    entity_index: EntityIndex
    sparql_prompt: str
    answer_prompt: str


def local_name_from_term(term) -> str:
    if isinstance(term, ox.NamedNode):
        v = term.value
        return v[len(EX_PREFIX):].replace("_", " ") if v.startswith(EX_PREFIX) else v
    return term.value if isinstance(term, ox.Literal) else str(term)


def execute_sparql(store: ox.Store, sparql: str) -> list[str]:
    results = store.query(sparql)
    variables = getattr(results, "variables", None)
    answers = []
    for sol in results:
        terms = (sol[v] for v in variables) if variables else iter(sol)
        answers.extend(local_name_from_term(t) for t in terms if t is not None)
    return answers


def process_question(ctx: Context, question: str, trace: Trace, with_nl: bool = False) -> dict:
    """Runs all stages. Fills `trace` progressively so partial timings survive exceptions."""
    out: dict = {}
    question = clean_question(question)

    call = chat(ctx.client, ctx.cfg,
                [{"role": "system", "content": ctx.sparql_prompt},
                 {"role": "user", "content": f'Q: "{question}"'}], temperature=0.0)
    trace.add_llm("sparql_generation", call)
    sparql = strip_markdown_fences(call.text)
    if "PREFIX ex:" not in sparql:
        sparql = SPARQL_PREFIX_HEADER + sparql
    out["raw_llm_response"], out["sparql_original"] = call.text, sparql

    with trace.timer("entity_resolution"):
        sparql, entities = resolve_entities(ctx.store, ctx.entity_index, ctx.predicate_set, sparql)
    out["sparql"], out["entities"] = sparql, entities

    with trace.timer("db_execution"):
        out["predicted"] = execute_sparql(ctx.store, sparql)

    if with_nl:
        results = out["predicted"]
        user = (f'Raw results: {", ".join(results) if results else "(no results found)"}\n'
                f'Result count: {len(results)}\nQuestion: "{question}"\nAnswer:')
        nl = chat(ctx.client, ctx.cfg, [{"role": "system", "content": ctx.answer_prompt},
                                        {"role": "user", "content": user}], temperature=0.2)
        trace.add_llm("nl_generation", nl)
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
        cols = ["index", "status", "total_s", "total_cost_usd"]
        for st in STAGES:
            cols += [f"{st}_s", f"{st}_prompt_tok", f"{st}_completion_tok"]
        self.csv = csv.writer(self.csv_fh)
        self.csv.writerow(cols)

    def record(self, detail: dict, trace: Trace) -> None:
        m = trace.to_dict()
        detail["metrics"] = m
        self.jsonl.write(json.dumps(detail, ensure_ascii=False) + "\n")
        self.jsonl.flush()
        row = [detail.get("index"), detail.get("status"), m["total_seconds"], m["total_cost_usd"]]
        for st in STAGES:
            s = m["stages"].get(st, {})
            row += [s.get("seconds", 0), s.get("prompt_tokens", 0), s.get("completion_tokens", 0)]
        self.csv.writerow(row)
        self.csv_fh.flush()
        self.qlog.write(f"[{detail.get('index')}] Q: {detail.get('question')}\n"
                        f"  status   : {detail.get('status')}\n"
                        f"  entities : {detail.get('entities')}\n"
                        f"  SPARQL   : {detail.get('sparql', '').replace(chr(10), ' ')}\n"
                        f"  predicted: {str(detail.get('predicted'))[:300]}\n"
                        f"  truth    : {detail.get('ground_truth')}\n"
                        f"  time     : {m['total_seconds']:.3f}s  " +
                        " ".join(f"{k}={v['seconds']:.3f}s" for k, v in m["stages"].items()) + "\n\n")
        self.qlog.flush()
        self.records.append({"index": detail.get("index"), "question": detail.get("question"),
                             "status": detail.get("status"), "metrics": m})

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
    g = st.get("sparql_generation")
    if g:
        n = max(g["calls"], 1)
        avg_in, avg_out = g["prompt_tokens"] / n, g["completion_tokens"] / n
        if avg_in > 2000:
            tips.append(f"SPARQL prompt averages {avg_in:.0f} tokens/call. Large prompt = slow prompt-eval. "
                        "Shrink the predicate list / few-shot examples (see prompt_sparql.txt), or enable prompt caching.")
        if avg_out > 120:
            tips.append(f"SPARQL output averages {avg_out:.0f} tokens but a query needs ~40-80. The model is likely "
                        "emitting reasoning/explanations. Check raw_llm_response in details.jsonl, set LLM_MAX_TOKENS, "
                        "or use a non-thinking model.")
        if g.get("completion_tok_per_s") and g["completion_tok_per_s"] < 15:
            tips.append(f"Generation speed is only {g['completion_tok_per_s']} tok/s: model/hardware bound "
                        "(try a smaller/quantized model or GPU offload).")
        if g["p95_s"] > 3 * max(g["p50_s"], 1e-6):
            tips.append("sparql_generation has high variance (p95 > 3x p50): see slowest_questions for outliers.")
    e = st.get("entity_resolution")
    if e and e["share_of_time"] > 0.10:
        tips.append("entity_resolution is >10% of time: fuzzy matching is slow. `pip install rapidfuzz`.")
    d = st.get("db_execution")
    if d and d["share_of_time"] > 0.10:
        tips.append("db_execution is >10% of time: inspect slow queries (unbounded patterns/missing predicates).")
    ent = s.get("entity_stats")
    if ent and ent.get("questions_with_entities"):
        q = ent["questions_with_entities"]
        fixed = (ent["repaired"] + ent["case_fixed"]) / q
        if fixed > 0.10:
            tips.append(f"{fixed:.0%} of questions needed entity repair: the LLM misnames entities. "
                        "Strengthen rule 2 in the prompt or add more name examples.")
        if ent["not_found"]:
            tips.append(f"{ent['not_found']} question(s) had an entity that could not be resolved "
                        "(status 'entity_not_found'): likely empty results.")
    if any(v.get("tokens_estimated") for v in st.values()):
        tips.append("Token counts are ESTIMATED (server returned no usage). Cost figures are approximate.")
    return tips


def render_summary(s: dict) -> str:
    L = ["=" * 78, "RUN SUMMARY", "=" * 78,
         f"Questions: {s['questions']}  | total {s['total_seconds']}s | mean {s['mean_seconds_per_question']}s "
         f"| p50 {s['p50_s']}s | p95 {s['p95_s']}s | cost ${s['total_cost_usd']}"]
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
    return text.strip().lower()


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


def evaluate(ctx: Context, rec: RunRecorder, with_nl: bool, prompt_on_error: bool) -> tuple[EvalStats, dict]:
    stats = EvalStats()
    ent = {"questions_with_entities": 0, "ok": 0, "case_fixed": 0, "repaired": 0, "not_found": 0}
    qs = list(read_qa_file(ctx.cfg.qa_path))
    if ctx.cfg.max_questions:
        qs = qs[:ctx.cfg.max_questions]
    error_seen = False

    for pq in tqdm(qs, desc="Evaluating", unit="q"):
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
            statuses = {i["status"] for i in infos}
            if infos:
                ent["questions_with_entities"] += 1
                for key in ("not_found", "repaired", "case_fixed"):
                    if key in statuses:
                        ent[key] += 1
                if statuses == {"ok"}:
                    ent["ok"] += 1
            if pq.gold_entity:
                stats.gold_checked += 1
                gold_hit = pq.gold_entity in {i["resolved"] for i in infos}
                detail["entity_matches_gold"] = gold_hit
                stats.gold_match += gold_hit
            stats.correct += hit
            stats.exact += (pred == truth)
            detail["status"] = "correct" if hit else ("entity_not_found" if "not_found" in statuses else "incorrect")
        except Exception as exc:
            failed = True
            stats.errors += 1
            detail["status"] = f"error: {exc}"
            rec.error(pq.question, exc)
        rec.record(detail, trace)
        if failed and prompt_on_error and not error_seen:
            error_seen = True
            if not _prompt_continue(pq.question, detail["status"]):
                stats.aborted_early = True
                break
    return stats, ent


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
        print(f"\nSPARQL:\n{detail['sparql']}")
        repaired = [e for e in detail["entities"] if e["status"] in ("repaired", "case_fixed")]
        if repaired:
            print(f"Entity repaired: {repaired}")
        if any(e["status"] == "not_found" for e in detail["entities"]):
            print("WARNING: an entity was not found in the KB; results are likely empty.")
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
    startup: dict[str, float] = {}

    t0 = time.perf_counter()
    try:
        store = build_store(cfg.kb_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    startup["kb_load"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    schema = get_or_build_schema(store, cfg)
    startup["schema"] = round(time.perf_counter() - t0, 3)

    predicate_set = {p[len("ex:"):] for p in schema.predicates}
    index = EntityIndex(store, predicate_set, cfg.fuzzy_cutoff)
    startup["entity_index_build"] = round(index.build_seconds, 3)
    print(f"Entity index: {len(index.names)} entities ({'rapidfuzz' if _rf_process else 'difflib'} matcher).")

    ctx = Context(cfg, store, build_llm_client(cfg), predicate_set, index,
                  build_system_prompt(schema), build_answer_system_prompt())

    (run_dir / "prompt_sparql.txt").write_text(ctx.sparql_prompt, encoding="utf-8")
    (run_dir / "prompt_answer.txt").write_text(ctx.answer_prompt, encoding="utf-8")
    cfg_dump = {k: str(v) for k, v in vars(cfg).items()}
    cfg_dump["llm_api_key"] = "***redacted***"
    (run_dir / "run_config.json").write_text(json.dumps({
        "started_at": datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
        "config": cfg_dump, "python": platform.python_version(),
        "pyoxigraph": getattr(ox, "__version__", "unknown"),
        "fuzzy_matcher": "rapidfuzz" if _rf_process else "difflib",
        "sparql_prompt_chars": len(ctx.sparql_prompt),
        "sparql_prompt_est_tokens": estimate_tokens(ctx.sparql_prompt),
        "n_predicates": len(predicate_set), "n_entities": len(index.names),
        "startup_timings_s": startup}, indent=2), encoding="utf-8")

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
            extra["accuracy"] = {
                "any_hit": stats.accuracy,
                "exact": stats.exact / stats.total if stats.total else 0.0,
                "errors": stats.errors,
                "entity_gold_match": (f"{stats.gold_match / stats.gold_checked:.2%}" if stats.gold_checked else "n/a"),
                "aborted_early": stats.aborted_early}
            print(f"\nTotal: {stats.total} | Correct: {stats.correct} | Accuracy: {stats.accuracy:.2%}")
    finally:
        rec.close()
        if rec.records:
            summary = summarize(rec.records, extra)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            text = render_summary(summary)
            (run_dir / "summary.txt").write_text(text, encoding="utf-8")
            print("\n" + text)
        print(f"\nAll artifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())