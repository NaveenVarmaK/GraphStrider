"""
GraphStrider web UI (Streamlit).

    uv run streamlit run app.py

Tabs
----
Ask               ask a question; see the plan, linked entities, traversed subgraph, SPARQL, answer and per-stage cost
Evaluate          run a QA file with a live progress view; results land in outputs/run_<ts>/ like the CLI
Runs & metrics    dashboards for any run folder (CLI or UI) plus a cross-run comparison
Knowledge graph   the auto-discovered schema, the planning prompt and a standalone entity-linking playground

The heavy objects (KB store, FAISS entity index, planning prompt) are built once per process and shared.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import main as gs

st.set_page_config(page_title="GraphStrider", page_icon="🕸️", layout="wide")

OUTPUTS = Path("outputs")

# --------------------------------------------------------------------------- #
# Colors: stage identity (fixed categorical order) + reserved status colors
# --------------------------------------------------------------------------- #

# Old (v2) stage names keep their own slots so historic runs stay readable.
STAGE_ORDER = ["planning", "entity_linking", "path_execution", "replanning", "nl_generation",
               "sparql_generation", "entity_resolution", "db_execution"]
CATEGORICAL = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
               "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
STATUS_COLORS = {"correct": "#0ca30c", "answered": "#0ca30c", "incorrect": "#d03b3b",
                 "entity_not_linked": "#ec835a", "entity_not_found": "#ec835a", "error": "#fab219"}
STATUS_ICONS = {"correct": "✅", "answered": "✅", "incorrect": "❌", "entity_not_linked": "🔗",
                "entity_not_found": "🔗", "error": "⚠️"}
SURFACE = {"light": "#ffffff", "dark": "#0e1117"}
INK = {"light": "#31333f", "dark": "#fafafa"}


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def stage_scale(stages: list[str]) -> alt.Scale:
    pal = CATEGORICAL[theme()]
    ordered = [s for s in STAGE_ORDER if s in stages] + [s for s in stages if s not in STAGE_ORDER]
    return alt.Scale(domain=ordered, range=[pal[STAGE_ORDER.index(s) % 8] if s in STAGE_ORDER else pal[7]
                                            for s in ordered])


def status_key(status: str | None) -> str:
    status = status or "unknown"
    return "error" if status.startswith("error") else status


def stage_label(stage: str) -> str:
    return stage.replace("_", " ")


# --------------------------------------------------------------------------- #
# Shared pipeline objects
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading knowledge graph and building the entity index (first time only)…")
def load_pipeline(kb_path: str, embed_backend: str, embed_model: str):
    cfg = gs.load_config()
    cfg.kb_path, cfg.embed_backend, cfg.embed_model = Path(kb_path), embed_backend, embed_model
    ctx, startup = gs.build_context(cfg, log=lambda _msg: None)
    return ctx, startup


def sidebar_settings() -> tuple[gs.Context, dict, bool]:
    env = gs.load_config()
    st.sidebar.title("🕸️ GraphStrider")
    st.sidebar.caption("Plan-and-retrieve KGQA · BLINK linking · RoG planning")

    with st.sidebar.expander("Knowledge base & embeddings", expanded=False):
        kb_path = st.text_input("KB file", str(env.kb_path))
        embed_backend = st.selectbox("Embedding backend", ["fastembed", "openai"],
                                     index=0 if env.embed_backend == "fastembed" else 1)
        embed_model = st.text_input("Embedding model", env.embed_model)
    if not Path(kb_path).exists():
        st.error(f"KB file not found: `{kb_path}`. Set KB_PATH in .env or fix the path in the sidebar.")
        st.stop()
    ctx, startup = load_pipeline(kb_path, embed_backend, embed_model)

    with st.sidebar.expander("LLM", expanded=False):
        base_url = st.text_input("Base URL", ctx.cfg.llm_base_url)
        model = st.text_input("Model", ctx.cfg.llm_model)
        api_key = st.text_input("API key", ctx.cfg.llm_api_key, type="password")
        max_tokens = st.number_input("Max output tokens (0 = server default)", 0, 32768, ctx.cfg.max_tokens or 0, 64)

    st.sidebar.subheader("Pipeline knobs")
    top_k = st.sidebar.slider("Entity candidates (top-k)", 1, 20, ctx.cfg.link_top_k,
                              help="How many linked candidates each mention keeps. The plan's path picks among them.")
    min_score = st.sidebar.slider("Min. linking score", 0.0, 1.0, float(ctx.cfg.link_min_score), 0.01)
    max_plans = st.sidebar.slider("Alternative plans", 1, 5, ctx.cfg.max_plans)
    max_hops = st.sidebar.slider("Max hops per path", 1, 6, ctx.cfg.max_hops)
    replan = st.sidebar.slider("Replanning rounds", 0, 3, ctx.cfg.replan_rounds,
                               help="Extra LLM calls allowed only when every plan returns nothing.")
    with_nl = st.sidebar.toggle("Natural-language answer", value=True,
                                help="Adds one LLM call to phrase the answer. Off = graph answers only.")

    # The cached context is shared, so apply the knobs to it (local single-user tool).
    cfg = ctx.cfg
    if (base_url, api_key) != (cfg.llm_base_url, cfg.llm_api_key):
        cfg.llm_base_url, cfg.llm_api_key = base_url, api_key
        ctx.client = gs.build_llm_client(cfg)
    cfg.llm_model, cfg.max_tokens = model, (int(max_tokens) or None)
    cfg.link_top_k, cfg.link_min_score, cfg.replan_rounds = top_k, min_score, replan
    if (max_plans, max_hops) != (cfg.max_plans, cfg.max_hops):
        cfg.max_plans, cfg.max_hops = max_plans, max_hops
        ctx.plan_prompt = gs.build_plan_system_prompt(ctx.store, ctx.graph, cfg)

    st.sidebar.divider()
    st.sidebar.caption(f"KB: `{cfg.kb_path}` · {len(ctx.graph.relations)} relations · "
                       f"{ctx.linker.index.ntotal:,} label vectors\n\nLLM: `{cfg.llm_model}`")
    return ctx, startup, with_nl


# --------------------------------------------------------------------------- #
# Run folders
# --------------------------------------------------------------------------- #

def ui_recorder(ctx: gs.Context, startup: dict) -> gs.RunRecorder:
    """One run folder per browser session for questions asked in the Ask tab."""
    if "ui_rec" not in st.session_state:
        run_dir = gs.create_run_output_dir(OUTPUTS)
        gs.write_run_header(run_dir, ctx, startup)
        st.session_state.ui_rec = gs.RunRecorder(run_dir)
    return st.session_state.ui_rec


def write_summary(rec: gs.RunRecorder, extra: dict) -> dict:
    summary = gs.summarize(rec.records, extra)
    (rec.dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (rec.dir / "summary.txt").write_text(gs.render_summary(summary), encoding="utf-8")
    return summary


def list_runs() -> list[Path]:
    return sorted((p for p in OUTPUTS.glob("run_*") if (p / "summary.json").exists()), reverse=True)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_details(run_dir: Path) -> list[dict]:
    path = run_dir / "details.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def accuracy_of(summary: dict) -> dict:
    """Older runs stored accuracy as a bare float; newer ones as a dict."""
    acc = summary.get("accuracy")
    if isinstance(acc, (int, float)):
        return {"any_hit": float(acc)}
    return acc if isinstance(acc, dict) else {}


def run_label(run_dir: Path) -> str:
    s = load_json(run_dir / "summary.json")
    when = run_dir.name.removeprefix("run_")
    try:
        when = datetime.strptime(when, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    acc = accuracy_of(s).get("any_hit")
    acc_txt = f" · {acc:.0%} correct" if isinstance(acc, (int, float)) else ""
    return f"{when} · {s.get('mode', '?')} · {s.get('questions', 0)} q{acc_txt} · {s.get('model', '')}"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

def stage_bar(stage_seconds: dict[str, float], title: str, height: int = 180) -> alt.Chart:
    df = pd.DataFrame([{"stage": k, "seconds": v} for k, v in stage_seconds.items() if v is not None])
    df["label"] = [f"{v:.3f} s" if v < 1 else f"{v:.1f} s" for v in df["seconds"]]
    base = alt.Chart(df, title=title, height=height).encode(
        y=alt.Y("stage:N", sort=STAGE_ORDER, title=None, axis=alt.Axis(labelExpr="replace(datum.label, /_/g, ' ')")),
        x=alt.X("seconds:Q", title="seconds", scale=alt.Scale(nice=True, padding=40)),
        tooltip=[alt.Tooltip("stage:N"), alt.Tooltip("seconds:Q", format=".3f")])
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        color=alt.Color("stage:N", scale=stage_scale(list(df["stage"])), legend=None))
    text = base.mark_text(align="left", dx=4, color=INK[theme()], fontSize=11).encode(text="label:N")
    return bars + text


def status_bar(statuses: list[str], height: int = 180) -> alt.Chart:
    counts = pd.Series([status_key(s) for s in statuses]).value_counts().rename_axis("status").reset_index(name="questions")
    counts["label"] = [f"{STATUS_ICONS.get(s, '•')} {s.replace('_', ' ')}" for s in counts["status"]]
    domain = list(counts["label"])
    rng = [STATUS_COLORS.get(s, "#8b8b8b") for s in counts["status"]]
    return (alt.Chart(counts, title="Outcome per question", height=height)
            .mark_bar(cornerRadiusEnd=4, height={"band": 0.7})
            .encode(y=alt.Y("label:N", title=None, sort="-x"), x=alt.X("questions:Q", title="questions"),
                    color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
                    tooltip=["label:N", "questions:Q"]))


def latency_per_question(timings: pd.DataFrame, details: list[dict]) -> alt.Chart | None:
    stage_cols = [c for c in timings.columns if c.endswith("_s") and c != "total_s"]
    if timings.empty or not stage_cols:
        return None
    long = timings.melt(id_vars=["index"], value_vars=stage_cols, var_name="stage", value_name="seconds")
    long["stage"] = long["stage"].str.removesuffix("_s")
    long = long[long["seconds"] > 0]
    q_by_index = {d.get("index"): d.get("question", "") for d in details}
    long["question"] = long["index"].map(q_by_index).fillna("")
    stages = list(dict.fromkeys(long["stage"]))
    order = {s: i for i, s in enumerate(STAGE_ORDER)}
    long["order"] = long["stage"].map(lambda s: order.get(s, 99))
    return (alt.Chart(long, title="Time per question, split by stage", height=260)
            .mark_bar(stroke=SURFACE[theme()], strokeWidth=1)
            .encode(x=alt.X("index:O", title="question #", axis=alt.Axis(labelOverlap=True)),
                    y=alt.Y("sum(seconds):Q", title="seconds"),
                    color=alt.Color("stage:N", scale=stage_scale(stages), title="stage",
                                    legend=alt.Legend(orient="top", labelExpr="replace(datum.label, /_/g, ' ')")),
                    order=alt.Order("order:Q"),
                    tooltip=[alt.Tooltip("index:O", title="#"), alt.Tooltip("question:N"),
                             alt.Tooltip("stage:N"), alt.Tooltip("seconds:Q", format=".3f")]))


def single_series_bar(df: pd.DataFrame, x: str, y: str, title: str, fmt: str, x_title: str,
                      height: int = 220) -> alt.Chart:
    return (alt.Chart(df, title=title, height=height)
            .mark_bar(cornerRadiusEnd=4, color=CATEGORICAL[theme()][0], height={"band": 0.7})
            .encode(y=alt.Y(f"{y}:N", sort="-x", title=None), x=alt.X(f"{x}:Q", title=x_title),
                    tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)]))


# --------------------------------------------------------------------------- #
# Subgraph view for one answered question
# --------------------------------------------------------------------------- #

def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def path_subgraph_dot(ctx: gs.Context, constraints: list[dict], answers: set[str], per_layer: int = 6) -> str:
    """A sample of the walked subgraph: linked entity -> intermediate nodes -> answers."""
    pal, ink = CATEGORICAL[theme()], INK[theme()]
    nodes: dict[str, str] = {}   # key -> role
    edges: set[tuple[str, str, str]] = set()
    graph = ctx.graph
    for c in constraints:
        linked = c.get("linked")
        if not linked or not c.get("results"):
            continue
        topic = linked["term"]
        nodes[topic] = "topic"
        layers: list[list[tuple[str, str]]] = []
        frontier = [topic]
        for i, step in enumerate(c["path"]):
            values = " ".join(frontier)
            q = (f"SELECT ?a ?b WHERE {{ VALUES ?a {{ {values} }} ?a {gs.path_expr(graph, [step])} ?b }} "
                 f"LIMIT 2000")
            pairs = [(gs.term_key(s[0]), gs.term_key(s[1])) for s in ctx.store.query(q)]
            if i == len(c["path"]) - 1:
                pairs = [p for p in pairs if p[1] in answers]
            nxt = list(dict.fromkeys(b for _, b in pairs))[:per_layer]
            layers.append([p for p in pairs if p[1] in nxt])
            frontier = nxt
            if not frontier:
                break
        # keep only intermediate nodes that lead somewhere
        for i in range(len(layers) - 2, -1, -1):
            alive = {a for a, _ in layers[i + 1]}
            layers[i] = [p for p in layers[i] if p[1] in alive]
        for i, (layer, step) in enumerate(zip(layers, c["path"])):
            name = step.lstrip("~")
            label = f"← {name}" if step.startswith("~") else name
            for a, b in layer:
                nodes.setdefault(a, "mid")
                nodes[b] = "answer" if b in answers and i == len(layers) - 1 else nodes.get(b, "mid")
                edges.add((a, b, label))
    if not edges:
        return ""
    fill = {"topic": pal[0], "mid": "#8b8b8b", "answer": pal[2]}
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.6;',
             f'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 fontcolor="white" penwidth=0];',
             f'edge [fontname="Helvetica" fontsize=9 color="{ink}" fontcolor="{ink}" arrowsize=0.6];']
    ids = {k: f"n{i}" for i, k in enumerate(nodes)}
    for key, role in nodes.items():
        lines.append(f'{ids[key]} [label="{_dot_escape(graph.label(key)[:40])}" fillcolor="{fill[role]}"];')
    for a, b, label in sorted(edges):
        lines.append(f'{ids[a]} -> {ids[b]} [label="{_dot_escape(label)}"];')
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_ask(ctx: gs.Context, startup: dict, with_nl: bool) -> None:
    st.subheader("Ask the knowledge graph")
    with st.form("ask", border=False):
        cols = st.columns([6, 1])
        question = cols[0].text_input("Question", placeholder="e.g. what languages are spoken in films directed by Joel Zwick",
                                      label_visibility="collapsed")
        submitted = cols[1].form_submit_button("Ask", type="primary", width="stretch")

    if submitted and question.strip():
        rec = ui_recorder(ctx, startup)
        trace = gs.Trace(ctx.cfg)
        detail = {"index": len(rec.records) + 1, "question": gs.clean_question(question), "ground_truth": None}
        with st.spinner("Planning → linking → executing…"):
            try:
                detail.update(gs.process_question(ctx, question, trace, with_nl=with_nl))
                detail["status"] = "answered"
            except Exception as exc:  # surfaced in the UI and in errors.log
                detail["status"] = f"error: {exc}"
                rec.error(question, exc)
        rec.record(detail, trace)
        write_summary(rec, {"startup_timings": startup, "model": ctx.cfg.llm_model, "mode": "web"})
        st.session_state.last = detail
        st.session_state.setdefault("history", []).insert(0, detail)

    detail = st.session_state.get("last")
    if not detail:
        st.info("Ask a question to see the plan, the linked entities, the traversed subgraph and the cost of each stage.")
        return
    render_question_detail(ctx, detail, live=True)

    history = st.session_state.get("history", [])
    if len(history) > 1:
        st.divider()
        st.markdown("##### This session")
        st.dataframe(pd.DataFrame([{
            "#": d["index"], "question": d["question"], "status": d.get("status"),
            "answers": len(d.get("predicted") or []), "LLM calls": d.get("llm_calls", 0),
            "seconds": d["metrics"]["total_seconds"], "cost $": d["metrics"]["total_cost_usd"]} for d in history]),
            hide_index=True, width="stretch")
        st.caption(f"Logged to `{st.session_state.ui_rec.dir}`")


def render_question_detail(ctx: gs.Context | None, d: dict, live: bool = False) -> None:
    status = status_key(d.get("status"))
    m = d.get("metrics", {"total_seconds": 0, "total_cost_usd": 0, "stages": {}})
    if status == "error":
        st.error(d.get("status"))
        return
    predicted = d.get("predicted") or []
    if d.get("nl_answer"):
        st.success(d["nl_answer"])
    elif predicted:
        st.success(", ".join(predicted[:25]) + (" …" if len(predicted) > 25 else ""))
    else:
        st.warning("The graph returned no answer for any plan.")

    tok_in = sum(s.get("prompt_tokens", 0) for s in m["stages"].values())
    tok_out = sum(s.get("completion_tokens", 0) for s in m["stages"].values())
    k = st.columns(5)
    k[0].metric("Total time", f"{m['total_seconds']:.2f} s")
    k[1].metric("LLM calls", d.get("llm_calls", "–"), help="1 = plan only. +1 for replanning, +1 for the NL answer.")
    k[2].metric("Answers", len(predicted))
    k[3].metric("Tokens in / out", f"{tok_in:,} / {tok_out:,}")
    k[4].metric("Cost", f"${m['total_cost_usd']:.6f}")
    if d.get("ground_truth"):
        st.caption(f"{STATUS_ICONS.get(status, '')} **{status.replace('_', ' ')}** · ground truth: "
                   + ", ".join(d["ground_truth"]))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Relation-path plan")
        plan = d.get("plan_used")
        if plan:
            for c in plan.get("constraints", []):
                chain = " → ".join([f"**{c['entity']}**", *[f"`{p}`" for p in c["path"]], "**?answer**"])
                st.markdown(chain)
            if len(plan.get("constraints", [])) > 1:
                st.caption("Several constraints → their answers are intersected.")
        if d.get("replanned"):
            st.caption("🔁 The first plans returned nothing; the LLM replanned with feedback.")
        if ctx is not None and d.get("entities"):
            dot = path_subgraph_dot(ctx, d["entities"], set(_answer_keys(ctx, d)))
            if dot:
                st.graphviz_chart(dot, width="stretch")
                st.caption("Sample of the walked subgraph · blue = linked entity · gray = intermediate · "
                           "green = answer · `← rel` = relation walked backwards")
    with right:
        stage_s = {k: v["seconds"] for k, v in m["stages"].items()}
        if stage_s:
            st.altair_chart(stage_bar(stage_s, "Time per stage"), width="stretch")
        st.markdown("##### Linked entities")
        rows = []
        for e in d.get("entities") or []:
            linked = e.get("linked")
            if isinstance(linked, dict):
                rows.append({"mention": e.get("entity"), "linked to": linked.get("label"), "method": linked.get("method"),
                             "score": linked.get("score"), "path results": e.get("results"),
                             "candidates tried": e.get("candidates_tried")})
            else:  # v2 run format
                rows.append({"mention": e.get("llm"), "linked to": e.get("resolved"), "method": e.get("status"),
                             "score": e.get("score")})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if predicted:
        with st.expander(f"All {len(predicted)} answers"):
            st.dataframe(pd.DataFrame({"answer": predicted}), hide_index=True, width="stretch")
    if d.get("sparql"):
        with st.expander("Executed SPARQL"):
            st.code(d["sparql"], language="sparql")
    rounds = d.get("plan_rounds") or []
    if rounds:
        with st.expander("LLM rounds: raw output, parsed plans, candidates"):
            for i, r in enumerate(rounds, 1):
                st.markdown(f"**Round {i}** {'(replanning)' if i > 1 else '(planning)'}")
                st.code(r.get("raw_llm_response", ""), language="json")
                if r.get("problems"):
                    st.caption("Problems: " + "; ".join(r["problems"]))
                for mention, cands in (r.get("links") or {}).items():
                    st.markdown(f"Candidates for *{mention}*")
                    st.dataframe(pd.DataFrame(cands), hide_index=True, width="stretch")
    if d.get("attempts"):
        with st.expander("Plan attempts"):
            st.json(d["attempts"], expanded=False)


def _answer_keys(ctx: gs.Context, d: dict) -> list[str]:
    """Recover answer term keys for the subgraph view by re-running the winning constraint paths."""
    keys: set[str] | None = None
    for c in d.get("entities") or []:
        linked = c.get("linked")
        if not isinstance(linked, dict) or not c.get("results"):
            return []
        found = set(gs.run_path(ctx.store, ctx.graph, linked["term"], c["path"]))
        keys = found if keys is None else keys & found
    return sorted(keys or [])


def tab_evaluate(ctx: gs.Context, startup: dict) -> None:
    st.subheader("Evaluate on a QA file")
    st.caption("QA format: one question per line, `question<TAB>answer1|answer2`. "
               "A `[bracketed]` entity is optional; it is only used to score entity linking.")
    qa_files = sorted({str(p) for p in Path(".").glob("**/qa_*.txt") if ".venv" not in p.parts})
    default = str(ctx.cfg.qa_path)
    if default not in qa_files:
        qa_files.insert(0, default)
    c = st.columns([4, 1, 1])
    qa_path = c[0].selectbox("QA file", qa_files, index=qa_files.index(default))
    n = c[1].number_input("Questions", 1, 100_000, ctx.cfg.max_questions or 20)
    c[2].write("")
    start = c[2].button("Run evaluation", type="primary", width="stretch")
    with_nl = st.toggle("Also generate natural-language answers", value=False,
                        help="Scoring only uses the graph answers. The extra LLM call per question is usually the "
                             "slowest stage, so it is off by default here.")
    if not start:
        if "last_eval_dir" in st.session_state:
            st.divider()
            render_run_dashboard(Path(st.session_state.last_eval_dir), ctx, "eval")
        return

    cfg = replace(ctx.cfg, qa_path=Path(qa_path), max_questions=int(n))
    questions = gs.load_questions(cfg)
    run_dir = gs.create_run_output_dir(OUTPUTS)
    gs.write_run_header(run_dir, ctx, startup)
    rec = gs.RunRecorder(run_dir)
    extra = {"startup_timings": startup, "model": ctx.cfg.llm_model, "mode": "eval", "qa_path": qa_path}

    bar = st.progress(0.0, text="Starting…")
    kpis = st.columns(4)
    slots = [k.empty() for k in kpis]
    table = st.empty()
    recent: list[dict] = []

    def on_progress(done: int, total: int, detail: dict, stats: gs.EvalStats) -> None:
        bar.progress(done / total, text=f"{done}/{total} · {detail['question'][:80]}")
        secs = [r["metrics"]["total_seconds"] for r in rec.records]
        slots[0].metric("Done", f"{done}/{total}")
        slots[1].metric("Correct so far", f"{stats.accuracy:.1%}")
        slots[2].metric("Mean time", f"{sum(secs) / len(secs):.2f} s")
        slots[3].metric("Errors", stats.errors)
        recent.insert(0, {"#": detail["index"], "status": f"{STATUS_ICONS.get(status_key(detail.get('status')), '')} "
                                                           f"{status_key(detail.get('status'))}",
                          "question": detail["question"], "predicted": ", ".join((detail.get("predicted") or [])[:5]),
                          "truth": ", ".join(detail.get("ground_truth") or []),
                          "s": detail["metrics"]["total_seconds"]})
        table.dataframe(pd.DataFrame(recent[:15]), hide_index=True, width="stretch")

    stats = gs.EvalStats()
    try:
        stats, ent = gs.evaluate(replace(ctx, cfg=cfg), rec, with_nl,
                                 prompt_on_error=False, questions=questions, on_progress=on_progress)
        extra["entity_stats"] = ent
        extra["accuracy"] = gs.accuracy_dict(stats)
    finally:  # also runs when the user presses Streamlit's Stop: partial runs are still summarized
        gs.finalize_run(rec, extra)
    bar.progress(1.0, text=f"Finished · {stats.correct}/{stats.total} correct · saved to {run_dir}")
    st.session_state.last_eval_dir = str(run_dir)
    st.divider()
    render_run_dashboard(run_dir, ctx, "eval")


def render_run_dashboard(run_dir: Path, ctx: gs.Context | None, scope: str) -> None:
    """`scope` keeps widget keys unique when the same run is shown in two tabs."""
    s = load_json(run_dir / "summary.json")
    if not s:
        st.warning("This run has no summary yet.")
        return
    details = load_details(run_dir)
    acc = accuracy_of(s)
    k = st.columns(7)
    k[0].metric("Questions", s.get("questions", 0))
    k[1].metric("Correct (any hit)", f"{acc['any_hit']:.1%}" if "any_hit" in acc else "–")
    k[2].metric("Exact answer set", f"{acc['exact']:.1%}" if "exact" in acc else "–")
    k[3].metric("Mean time / q", f"{s.get('mean_seconds_per_question', 0):.2f} s")
    k[4].metric("p95 time", f"{s.get('p95_s', 0):.2f} s")
    k[5].metric("LLM calls / q", s.get("mean_llm_calls_per_question", "–"))
    k[6].metric("Cost", f"${s.get('total_cost_usd', 0):.4f}")
    if acc.get("entity_gold_match") not in (None, "n/a"):
        st.caption(f"Entity linking matched the gold entity in **{acc['entity_gold_match']}** of questions · "
                   f"replanned: {s.get('replanned_questions', 0)} · model `{s.get('model', '?')}`")

    c1, c2 = st.columns(2)
    stages = s.get("stages", {})
    if stages:
        c1.altair_chart(stage_bar({k: v["total_s"] for k, v in stages.items()}, "Total time per stage"),
                        width="stretch")
    if details:
        c2.altair_chart(status_bar([d.get("status") for d in details]), width="stretch")

    timings_path = run_dir / "timings.csv"
    if timings_path.exists():
        timings = pd.read_csv(timings_path)
        chart = latency_per_question(timings, details)
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    if stages:
        with st.expander("Stage statistics (tokens, percentiles, throughput)"):
            st.dataframe(pd.DataFrame(stages).T.rename_axis("stage"), width="stretch")
    if s.get("diagnosis"):
        with st.expander("Diagnosis", expanded=True):
            for tip in s["diagnosis"]:
                st.markdown(f"- {tip}")

    if details:
        st.markdown("##### Questions")
        df = pd.DataFrame([{
            "#": d.get("index"), "status": status_key(d.get("status")), "question": d.get("question"),
            "answers": len(d.get("predicted") or []), "LLM calls": d.get("llm_calls"),
            "seconds": d.get("metrics", {}).get("total_seconds")} for d in details])
        options = sorted(df["status"].unique())
        chosen = st.multiselect("Filter by outcome", options, default=options, key=f"{scope}_flt_{run_dir.name}")
        view = df[df["status"].isin(chosen)]
        event = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key=f"{scope}_tbl_{run_dir.name}")
        rows = event.selection.rows if event and event.selection else []
        if rows:
            idx = view.iloc[rows[0]]["#"]
            d = next(x for x in details if x.get("index") == idx)
            st.markdown(f"###### #{idx} · {d.get('question')}")
            render_question_detail(ctx, d)
        else:
            st.caption("Select a row to inspect its plan, linking and SPARQL.")

    dl = st.columns(3)
    for col, name in zip(dl, ["summary.json", "details.jsonl", "timings.csv"]):
        path = run_dir / name
        if path.exists():
            col.download_button(f"Download {name}", path.read_bytes(), file_name=f"{run_dir.name}_{name}",
                                key=f"{scope}_dl_{run_dir.name}_{name}")


def tab_runs(ctx: gs.Context) -> None:
    runs = list_runs()
    if not runs:
        st.info("No runs yet. Use the Evaluate tab or `uv run main.py`.")
        return
    view, compare = st.tabs(["Single run", "Compare runs"])
    with view:
        chosen = st.selectbox("Run", runs, format_func=run_label)
        st.caption(f"`{chosen}`")
        render_run_dashboard(chosen, ctx, "runs")
    with compare:
        evals = [r for r in runs if "any_hit" in accuracy_of(load_json(r / "summary.json"))]
        if not evals:
            st.info("Comparison needs evaluation runs (they carry accuracy).")
            return
        picked = st.multiselect("Evaluation runs", evals, default=evals[:min(6, len(evals))], format_func=run_label)
        if not picked:
            return
        rows = []
        for r in picked:
            s = load_json(r / "summary.json")
            acc = accuracy_of(s)
            rows.append({"run": run_label(r), "accuracy": acc["any_hit"], "exact": acc.get("exact"), "mean_s": s.get("mean_seconds_per_question", 0),
                         "llm_calls": s.get("mean_llm_calls_per_question"), "questions": s.get("questions"),
                         "cost_usd": s.get("total_cost_usd", 0), "folder": r.name})
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)  # two measures on different scales -> two charts, never a dual axis
        c1.altair_chart(single_series_bar(df, "accuracy", "run", "Correct (any hit)", ".1%", "share of questions"),
                        width="stretch")
        c2.altair_chart(single_series_bar(df, "mean_s", "run", "Mean seconds per question", ".2f", "seconds"),
                        width="stretch")
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"accuracy": st.column_config.NumberColumn(format="percent"),
                                    "exact": st.column_config.NumberColumn(format="percent")})


def tab_kg(ctx: gs.Context, startup: dict) -> None:
    graph = ctx.graph
    k = st.columns(4)
    k[0].metric("Relations", len(graph.relations))
    k[1].metric("Facts", f"{sum(r.count for r in graph.relations.values()):,}")
    k[2].metric("Linkable nodes", f"{len(graph.nodes):,}")
    k[3].metric("Label vectors", f"{ctx.linker.index.ntotal:,}")
    st.caption("Startup: " + " · ".join(f"{stage_label(k)} {v:.2f}s" for k, v in startup.items())
               + f" · embedder `{ctx.linker.embedder.name}`")

    rel_df = pd.DataFrame([{"relation": r.name, "facts": r.count, "object": r.object_kind, "IRI": r.iri,
                            "example": next((f"{graph.label(s)} → {graph.label(o)}" for s, o in r.samples), "")}
                           for r in sorted(graph.relations.values(), key=lambda r: -r.count)])
    c1, c2 = st.columns([2, 3])
    c1.altair_chart(single_series_bar(rel_df, "facts", "relation", "Facts per relation", ",", "facts",
                                      height=max(160, 28 * len(rel_df))), width="stretch")
    c2.dataframe(rel_df, hide_index=True, width="stretch")

    st.markdown("##### Entity-linking playground")
    st.caption("The BLINK-style linker on its own: exact label match first, then FAISS nearest neighbours "
               "re-scored with character similarity. No LLM involved.")
    mention = st.text_input("Mention", placeholder="e.g. chris nolan", key="kg_mention")
    if mention.strip():
        cands = ctx.linker.link_many([mention]).get(mention, [])
        if not cands:
            st.warning("No candidate above the minimum linking score.")
        else:
            df = pd.DataFrame([c.to_dict() for c in cands])
            df["relations"] = [", ".join(gs.entity_relations(ctx.store, graph, c.key)) for c in cands]
            st.dataframe(df, hide_index=True, width="stretch",
                         column_config={"score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
            st.caption("`relations` = what the entity is connected by (`~rel` = it is the object). "
                       "The planner's path decides which candidate is used.")

    with st.expander("Planning system prompt (generated from this KB)"):
        st.code(ctx.plan_prompt, language="text")
        st.caption(f"~{gs.estimate_tokens(ctx.plan_prompt)} tokens · identical for every question, so server-side "
                   "prompt caching applies.")


# --------------------------------------------------------------------------- #

def run() -> None:
    ctx, startup, with_nl = sidebar_settings()
    ask, evaluate, runs, kg = st.tabs(["💬 Ask", "🧪 Evaluate", "📊 Runs & metrics", "🕸️ Knowledge graph"])
    with ask:
        tab_ask(ctx, startup, with_nl)
    with evaluate:
        tab_evaluate(ctx, startup)
    with runs:
        tab_runs(ctx)
    with kg:
        tab_kg(ctx, startup)


run()
