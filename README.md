# GraphStrider

Ask natural-language questions over any RDF knowledge graph. One cheap LLM call plans the answer,
and the database does the rest.

GraphStrider combines two ideas from the knowledge-graph QA literature:

- **RoG (Reasoning on Graphs)**: the LLM is a *planner*. It outputs a short relation-path blueprint
  (`Joel Zwick → ~directed_by → in_language`) instead of writing SPARQL or walking the graph step by step.
- **BLINK**: entity mentions are linked with a *bi-encoder*. Every entity label is embedded once into a
  FAISS index, so linking at question time is a millisecond nearest-neighbour search.

It is **data-agnostic**. Relations, labels, namespaces and even the prompt's few-shot examples are discovered
from the KB at startup, so nothing is hard-coded for a particular dataset. MetaQA (movies) is the bundled
benchmark.

```
question ──► LLM planner (1 call) ──► {"entity": "Joel Zwick", "path": ["~directed_by", "in_language"]}
                                              │
              FAISS bi-encoder linker ◄───────┘  "Joel Zwick" → <http://example.org/joel_zwick>
                                              │
              SPARQL property path   ◄────────┘  ?topic ^<directed_by>/<in_language> ?answer
                                              │
                                     answers: Greek
```

See **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** for the full design.

## Quick start

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies
cp .env.example .env         # then set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
```

Put your knowledge graph at `kb.ttl` (or set `KB_PATH`). Turtle, N-Triples, N-Quads, TriG, RDF/XML and N3 are
detected from the file extension. On the first start GraphStrider embeds every entity label (≈1 min for
40k labels on a laptop CPU). The vectors are cached in `.cache/`, so later starts take under a second.

### Web interface

```bash
uv run streamlit run app.py
```

| Tab | What it does |
|---|---|
| 💬 **Ask** | Ask a question and get the answer, the relation-path plan, the linked entities with scores, a diagram of the walked subgraph, the executed SPARQL, and the time/tokens/cost of every stage. |
| 🧪 **Evaluate** | Run a QA file with a live progress bar, running accuracy and the latest results. Writes a run folder exactly like the CLI. |
| 📊 **Runs & metrics** | Dashboard for any run in `outputs/`: KPIs, time per stage, outcome breakdown, per-question latency split by stage, diagnosis, and a question explorer that opens any row. A **Compare runs** view lines up accuracy and latency across runs. |
| 🕸️ **Knowledge graph** | The auto-discovered schema (relations, fact counts, examples), the generated planning prompt, and an entity-linking playground. |

The sidebar holds the LLM settings and the pipeline knobs (top-k candidates, minimum linking score,
alternative plans, max hops, replanning rounds, NL answer on/off).

### Command line

```bash
uv run main.py -q "what languages are spoken in films directed by Joel Zwick"   # one question
uv run main.py -i                                                               # interactive REPL
uv run main.py --yes                                                            # evaluate QA_PATH
QA_PATH=MetaQA/2-hop/ntm/qa_dev.txt MAX_QUESTIONS=100 uv run main.py --yes      # another QA file
```

`--with-nl` also generates natural-language answers during evaluation. `--yes` never stops to ask
after an error.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `KB_PATH` | `kb.ttl` | RDF file to load |
| `QA_PATH` | `MetaQA/1-hop/ntm/qa_dev.txt` | Evaluation file (`question<TAB>ans1\|ans2`; a `[bracketed]` entity is optional) |
| `MAX_QUESTIONS` | all | Limit the evaluation size |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | LM Studio on `localhost:1234` | Any OpenAI-compatible chat endpoint |
| `LLM_MAX_TOKENS` | server default | Cap on output tokens (a plan needs ~50) |
| `LLM_PRICE_INPUT_PER_1M`, `LLM_PRICE_OUTPUT_PER_1M` | `0` | USD prices, used for cost reporting |
| `EMBED_BACKEND` | `fastembed` | `fastembed` (local ONNX, CPU) or `openai` (any `/embeddings` endpoint) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for entity labels |
| `EMBED_BASE_URL`, `EMBED_API_KEY` | the LLM's | Only for `EMBED_BACKEND=openai` |
| `CACHE_DIR` | `.cache` | Where entity vectors are cached |
| `LINK_TOP_K` | `5` | Candidates kept per mention |
| `LINK_MIN_SCORE` | `0.75` | Minimum score for a vector-search candidate |
| `MAX_PLANS` | `3` | Alternative plans the LLM may propose |
| `MAX_HOPS` | `4` | Longest relation path allowed |
| `REPLAN_ROUNDS` | `1` | Extra LLM calls allowed when every plan returns nothing (`0` = strictly one call) |

## Outputs

Every CLI run, evaluation or web session writes `outputs/run_<YYYYmmdd_HHMMSS>/`:

| File | Content |
|---|---|
| `run_config.json` | Config (keys redacted), versions, KB stats, startup timings |
| `prompt_plan.txt`, `prompt_answer.txt` | The exact system prompts used |
| `details.jsonl` | One line per question: plans, LLM output, linking candidates, SPARQL, answers, metrics |
| `timings.csv` | Per-question seconds / tokens per stage |
| `queries.log` | Human-readable log |
| `errors.log` | Full tracebacks |
| `summary.json`, `summary.txt` | Aggregates per stage, slowest questions, accuracy, automatic diagnosis |

## Project layout

```
main.py                 pipeline + CLI (KB introspection, linker, planner, executor, metrics)
app.py                  Streamlit web interface
docs/IMPLEMENTATION.md  design and implementation notes
.env.example            configuration template
kb.ttl                  knowledge graph (not committed)
MetaQA/                 benchmark QA files (not committed)
outputs/                run folders (not committed)
.cache/                 entity-vector cache (not committed)
```
