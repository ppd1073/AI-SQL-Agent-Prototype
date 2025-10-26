"""
ai_sql_analyst.py
A compact, production-like NL→SQL analyst helper with:
- .env-based config (NO hardcoded secrets)
- Schema extraction (writes artifacts/schema_card.{md,json})
- Token-efficient schema excerpt for prompting
- Deterministic NL→SQL fallback (zero LLM tokens)
- Resilient LLM caller (only when enabled) + caching
- SELECT-only safety guard
- Query execution (SQLAlchemy)
- Validation checks
- Auto-visualization (bar/line) + quick insight sentence
- Optional CSV/PNG exports
"""

from __future__ import annotations

import os, re, json, hashlib, datetime as dt
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# -----------------------------
# 0) Setup & folders
# -----------------------------
load_dotenv()

# Turn LLM usage off if there's no key or if USE_LLM=0 in .env
USE_LLM = os.getenv("USE_LLM", "1") == "1" and bool(os.getenv("OPENAI_API_KEY"))
if not USE_LLM:
    print("ℹ️ LLM disabled (no API key or USE_LLM=0). Using deterministic SQL templates.")

Path("artifacts/cache").mkdir(parents=True, exist_ok=True)
Path("artifacts/kpi_out").mkdir(parents=True, exist_ok=True)
Path("artifacts/llm_cache").mkdir(parents=True, exist_ok=True)
Path("artifacts/plots").mkdir(parents=True, exist_ok=True)

# Use a lighter default model while testing
PRIMARY_MODEL  = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", PRIMARY_MODEL)
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "4000"))
ROWS_PREVIEW = int(os.getenv("ROWS_PREVIEW", "50"))

# -----------------------------
# 1) DB engine
# -----------------------------
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text

STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "4000"))

def make_engine():
    """
    Prefer DATABASE_URL if valid; otherwise build from components.
    Validates by running SELECT 1. Also uses URL.create to auto-quote passwords.
    """
    env_url = os.getenv("DATABASE_URL")

    # Component pieces (used for the safe fallback)
    pg_user = os.getenv("PGUSER", "postgres")
    pg_pw   = os.getenv("PGPASSWORD", "****")   # raw; URL.create will quote if needed
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = int(os.getenv("PGPORT", "5432"))
    pg_db   = os.getenv("PGDATABASE", "CapstoneProject1")

    # Try DATABASE_URL first (if present)
    if env_url:
        try:
            eng = create_engine(
                env_url,
                connect_args={"options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"},
                pool_pre_ping=True,
            )
            # Validate the connection quickly
            with eng.connect() as con:
                con.execute(text("SELECT 1"))
            return eng
        except Exception as e:
            print(f"⚠️ DATABASE_URL failed ({e}); falling back to component URL.")

    # Safe fallback using URL.create (handles passwords with '@' etc.)
    url = URL.create(
        "postgresql+psycopg2",
        username=pg_user,
        password=pg_pw,   # do NOT percent-encode here; URL.create handles it
        host=pg_host,
        port=pg_port,
        database=pg_db,
    )
    return create_engine(
        url,
        connect_args={"options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"},
        pool_pre_ping=True,
    )

engine = make_engine()

# -----------------------------
# 2) Schema card (writes once; cheap)
# -----------------------------
SCHEMA_SQL = """
SELECT c.table_name, c.column_name, c.data_type
FROM information_schema.columns c
WHERE c.table_schema = 'public'
ORDER BY c.table_name, c.ordinal_position;
"""

def write_schema_cards() -> int:
    card: dict[str, list[dict]] = {}
    with engine.connect() as con:
        for t, col, dtp in con.execute(text(SCHEMA_SQL)).fetchall():
            card.setdefault(t, []).append({"column": col, "type": dtp})

    md_lines = []
    for t, cols in card.items():
        md_lines.append(f"### {t}")
        for x in cols:
            md_lines.append(f"- {x['column']} : {x['type']}")
        md_lines.append("")

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    Path("artifacts/schema_card.md").write_text("\n".join(md_lines), encoding="utf-8")
    Path("artifacts/schema_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(card)

if not Path("artifacts/schema_card.json").exists():
    n_tables = write_schema_cards()
    print(f"Schema cards written ({n_tables} tables) ✅")
else:
    print("Schema cards exist ✅")

# -----------------------------
# 3) Schema excerpt (token saver)
# -----------------------------
def build_schema_excerpt(max_chars=1600) -> str:
    try:
        sc = json.loads(Path("artifacts/schema_card.json").read_text(encoding="utf-8"))
    except Exception:
        try:
            return Path("artifacts/schema_card.md").read_text(encoding="utf-8")[:max_chars]
        except Exception:
            return ""

    # keep only commonly used tables (edit as your domain grows)
    keep = {
        "w_invoiceline_f", "w_sales_class_d", "w_sales_agent_d",
        "w_financial_summary_sales_f", "w_time_d",
        "w_job_f", "w_job_shipment_f", "w_sub_job_f",
        "w_financial_summary_cost_f", "w_machine_type_d"
    }
    lines = []
    for t, cols in sc.items():
        if t not in keep:
            continue
        lines.append(f"### {t}")
        for c in cols:
            lines.append(f"- {c['column']} : {c['type']}")
        lines.append("")
    return "\n".join(lines)[:max_chars]

SCHEMA = build_schema_excerpt()
print(f"Schema excerpt length: {len(SCHEMA)}")

# -----------------------------
# 4) Resilient LLM caller + caches
# -----------------------------
# Build clients ONLY if LLM is enabled.
from langchain_openai import ChatOpenAI

def _make_llm(model_name: str):
    return ChatOpenAI(model=model_name, temperature=0)

if USE_LLM:
    _llm_primary  = _make_llm(PRIMARY_MODEL)
    _llm_fallback = _make_llm(FALLBACK_MODEL)
else:
    _llm_primary = _llm_fallback = None

def llm_invoke(messages):
    """Primary model; on 429/backoff, try fallback once."""
    if not USE_LLM:
        raise RuntimeError("LLM disabled")
    try:
        return _llm_primary.invoke(messages)
    except Exception as e1:
        msg = str(e1).lower()
        if "rate limit" in msg or "429" in msg or "tpm" in msg or "insufficient_quota" in msg:
            import time; time.sleep(2.5)
            try:
                return _llm_fallback.invoke(messages)
            except Exception as e2:
                raise RuntimeError(f"Both models failed. Primary: {e1}\nFallback: {e2}")
        raise

# Small JSON caches to avoid repeated LLM calls
LLM_SQL_CACHE = Path("artifacts/llm_cache/nl2sql.json")
LLM_SUM_CACHE = Path("artifacts/llm_cache/summary.json")

def _load_json(p: Path) -> dict:
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _save_json(p: Path, obj: dict):
    tmp = Path(str(p)+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)

_sql_cache = _load_json(LLM_SQL_CACHE)
_sum_cache = _load_json(LLM_SUM_CACHE)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

# -----------------------------
# 5) NL→SQL prompt + helpers
# -----------------------------
def ensure_select(sql: str):
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Refusing non-SELECT SQL.")
    for bad in [" update "," delete "," insert "," drop "," alter "," truncate "," create "]:
        if bad in s:
            raise ValueError("Potentially destructive SQL detected.")

def strip_fences(s: str) -> str:
    m = re.search(r"```sql(.*?)```", s, re.S|re.I)
    if m: return m.group(1).strip()
    m = re.search(r"```(.*?)```", s, re.S|re.I)
    return m.group(1).strip() if m else s.strip()

FEW_SHOTS = """
RULES:
- If the user asks for "invoiced amount", "invoice totals", "total invoiced", "billing", or "revenue from invoices",
  YOU MUST use table w_invoiceline_f (column invoice_amount) joined to w_sales_class_d and/or w_sales_agent_d.
  DO NOT use w_financial_summary_sales_f for invoiced totals.
- If the user asks for "forecast vs actual", "variance", or time-based summaries,
  use w_financial_summary_sales_f joined to w_time_d and appropriate dimensions.

Example Q: Which sales class has the highest total invoiced amount?
Example SQL:
SELECT sc.sales_class_desc, SUM(i.invoice_amount) AS total_invoiced_amount
FROM w_invoiceline_f i
JOIN w_sales_class_d sc ON i.sales_class_id = sc.sales_class_id
GROUP BY sc.sales_class_desc
ORDER BY total_invoiced_amount DESC
LIMIT 1;

Example Q: Show invoice totals and counts by sales class.
Example SQL:
SELECT sc.sales_class_desc,
       SUM(i.invoice_amount) AS total_invoiced,
       COUNT(DISTINCT i.invoice_id) AS num_invoices
FROM w_invoiceline_f i
JOIN w_sales_class_d sc ON i.sales_class_id = sc.sales_class_id
GROUP BY sc.sales_class_desc
ORDER BY total_invoiced DESC;

Example Q: Monthly forecast vs actual and variance.
Example SQL:
SELECT t.time_year, t.time_quarter, t.time_month,
       SUM(fs.actual_amount)  AS total_actual,
       SUM(fs.forcast_amount) AS total_forecast,
       (SUM(fs.actual_amount) - SUM(fs.forcast_amount)) AS variance
FROM w_financial_summary_sales_f fs
JOIN w_time_d t ON fs.report_end_date_id = t.time_id
GROUP BY t.time_year, t.time_quarter, t.time_month
ORDER BY t.time_year, t.time_quarter, t.time_month;
"""

SYS = """You are a SQL assistant for PostgreSQL.
Return ONE safe SELECT statement only (no prose, no comments).
Follow the RULES exactly. If question contains 'invoice' or 'invoiced',
use w_invoiceline_f.invoice_amount joined to w_sales_class_d or w_sales_agent_d.
For forecast/actual variance, use w_financial_summary_sales_f with w_time_d.
"""

# Optional: simple metric+time hinting
def parse_time_window(question: str) -> Tuple[Optional[dt.date], Optional[dt.date], str]:
    q = question.lower()
    today = dt.date.today()
    if "last year" in q:
        y = today.year - 1
        return dt.date(y,1,1), dt.date(y,12,31), f"{y}"
    if "this year" in q or "current year" in q:
        y = today.year
        return dt.date(y,1,1), dt.date(y,12,31), f"{y}"
    m = re.search(r"(in|for)\s+(\d{4})", q)
    if m:
        y = int(m.group(2))
        return dt.date(y,1,1), dt.date(y,12,31), f"{y}"
    m = re.search(r"last\s+(\d{1,3})\s+days?", q)
    if m:
        n = int(m.group(1))
        return today - dt.timedelta(days=n), today, f"last {n} days"
    return None, None, ""

def build_hint(question: str) -> str:
    q = question.lower()
    hint = ""
    if any(k in q for k in ["invoice","invoiced","billing"]):
        hint = "Use w_invoiceline_f.invoice_amount joined to w_sales_class_d or w_sales_agent_d. Do NOT use w_financial_summary_sales_f."
        time_key = "invoice_sent_date"
    elif any(k in q for k in ["forecast","variance","actual"]):
        hint = "Use w_financial_summary_sales_f joined to w_time_d."
        time_key = "report_end_date_id"
    else:
        time_key = None

    start, end, label = parse_time_window(question)
    if time_key and start and end:
        tbl = "w_invoiceline_f" if "invoic" in q or "billing" in q else "w_financial_summary_sales_f"
        hint += f" Join w_time_d and filter {label}: {tbl}.{time_key} -> w_time_d.time_id between {start:%Y-%m-%d} and {end:%Y-%m-%d}."
    return hint.strip()

# --- Deterministic (no-LLM) SQL helpers ---
def _int_from_text(q: str, default: int = 5) -> int:
    m = re.search(r"\btop\s+(\d{1,3})\b", q, re.I)
    if m:
        return int(m.group(1))
    return default

def _time_filter_clause(fact_alias: str, time_key: str, question: str) -> str:
    """
    Builds a JOIN to w_time_d and BETWEEN filter when the fact table stores a *time_id*.
    Returns a SQL snippet starting with a newline (or empty string if no time filter).
    """
    start, end, _ = parse_time_window(question)
    if not start or not end:
        return ""
    return f"""
JOIN w_time_d t ON {fact_alias}.{time_key} = t.time_id
WHERE t.time_date BETWEEN DATE '{start:%Y-%m-%d}' AND DATE '{end:%Y-%m-%d}'"""

def _time_where_between_on_joined_t(question: str) -> str:
    """
    ONLY emits a WHERE clause, intended for queries that already JOIN w_time_d t...
    """
    start, end, _ = parse_time_window(question)
    if not start or not end:
        return ""
    return f"\nWHERE t.time_date BETWEEN DATE '{start:%Y-%m-%d}' AND DATE '{end:%Y-%m-%d}'"

def deterministic_sql(question: str) -> str:
    """
    Heuristic, SELECT-only SQL templates for common intents.
    This runs with zero LLM tokens.
    """
    q = question.lower().strip()

    # 1) Totals and counts by sales class
    if ("invoice" in q or "invoiced" in q) and ("count" in q or "counts" in q):
        time_join = _time_filter_clause("i", "invoice_sent_date", question)
        return f"""
SELECT sc.sales_class_desc,
       SUM(i.invoice_amount)                    AS total_invoiced,
       COUNT(DISTINCT i.invoice_id)             AS num_invoices
FROM w_invoiceline_f i
JOIN w_sales_class_d sc ON i.sales_class_id = sc.sales_class_id
{time_join if time_join else ""}
GROUP BY sc.sales_class_desc
ORDER BY total_invoiced DESC;""".strip()

    # 2) Which sales class has the highest total invoiced amount?
    if ("which" in q or "highest" in q) and "sales class" in q and ("invoice" in q or "invoiced" in q):
        time_join = _time_filter_clause("i", "invoice_sent_date", question)
        return f"""
SELECT sc.sales_class_desc,
       SUM(i.invoice_amount) AS total_invoiced_amount
FROM w_invoiceline_f i
JOIN w_sales_class_d sc ON i.sales_class_id = sc.sales_class_id
{time_join if time_join else ""}
GROUP BY sc.sales_class_desc
ORDER BY total_invoiced_amount DESC
LIMIT 1;""".strip()

    # 3) Top N sales agents by invoiced amount
    if ("top" in q and "agent" in q) or ("sales agent" in q and ("invoice" in q or "invoiced" in q)):
        topn = _int_from_text(q, default=5)
        time_join = _time_filter_clause("i", "invoice_sent_date", question)
        return f"""
SELECT sa.sales_agent_name,
       SUM(i.invoice_amount) AS total_invoiced
FROM w_invoiceline_f i
JOIN w_sales_agent_d sa ON i.sales_agent_id = sa.sales_agent_id
{time_join if time_join else ""}
GROUP BY sa.sales_agent_name
ORDER BY total_invoiced DESC
LIMIT {topn};""".strip()

    # 4) Monthly forecast vs actual (with variance)
    if "forecast" in q and "actual" in q:
        where_between = _time_where_between_on_joined_t(question)
        return f"""
SELECT t.time_year, t.time_quarter, t.time_month,
       SUM(fs.actual_amount)  AS total_actual,
       SUM(fs.forcast_amount) AS total_forecast,
       (SUM(fs.actual_amount) - SUM(fs.forcast_amount)) AS variance
FROM w_financial_summary_sales_f fs
JOIN w_time_d t ON fs.report_end_date_id = t.time_id
{where_between}
GROUP BY t.time_year, t.time_quarter, t.time_month
ORDER BY t.time_year, t.time_quarter, t.time_month;""".strip()

    # 5) Largest pending shipments
    if "pending" in q or ("unshipped" in q) or ("partial" in q and "ship" in q):
        return """
SELECT j.job_id,
       js.requested_quantity,
       js.actual_quantity,
       (js.requested_quantity - js.actual_quantity) AS qty_pending
FROM w_job_shipment_f js
JOIN w_sub_job_f sj ON js.sub_job_id = sj.sub_job_id
JOIN w_job_f j      ON sj.job_id = j.job_id
WHERE js.actual_quantity < js.requested_quantity
ORDER BY qty_pending DESC
LIMIT 20;""".strip()

    # Fallback template (safe but generic): totals by sales class
    if "invoice" in q or "invoiced" in q:
        return """
SELECT sc.sales_class_desc,
       SUM(i.invoice_amount) AS total_invoiced
FROM w_invoiceline_f i
JOIN w_sales_class_d sc ON i.sales_class_id = sc.sales_class_id
GROUP BY sc.sales_class_desc
ORDER BY total_invoiced DESC;""".strip()

    # If nothing matched, raise a clear message so you can extend templates
    raise ValueError(
        "No deterministic template matched this question. "
        "Extend deterministic_sql() with a new case for your intent."
    )

def nl_to_sql_direct(question: str) -> str:
    """
    Try LLM (if enabled), else fall back to deterministic SQL templates.
    Never crashes on quota/key errors; always returns a SELECT.
    """
    if not USE_LLM:
        sql = deterministic_sql(question)
        ensure_select(sql)
        return sql

    # Build the hint as before (kept for consistency when LLM is on)
    hint = build_hint(question)
    prompt = f"""{FEW_SHOTS}

Database schema (excerpt):
{SCHEMA}

User question: {question}
Hint: {hint}
Return only a single SQL SELECT statement (no prose)."""

    key = _hash(PRIMARY_MODEL + "|" + prompt)
    cached = _sql_cache.get(key)
    if cached:
        sql = strip_fences(cached); ensure_select(sql); return sql

    try:
        resp = llm_invoke([{"role":"system","content":SYS},
                           {"role":"user","content":prompt}])
        sql = strip_fences(resp.content); ensure_select(sql)
        _sql_cache[key] = sql; _save_json(LLM_SQL_CACHE, _sql_cache)
        return sql
    except Exception as e:
        # On any LLM failure (quota/rate/no key), use deterministic SQL
        print(f"⚠️ LLM unavailable ({e}); using deterministic SQL.")
        sql = deterministic_sql(question)
        ensure_select(sql)
        return sql

# -----------------------------
# 6) Run SQL + validation
# -----------------------------
def run_sql_query(sql: str) -> pd.DataFrame:
    with engine.connect() as con:
        df = pd.read_sql(text(sql), con)
    return df if len(df) <= ROWS_PREVIEW else df.head(ROWS_PREVIEW)

def validate_df(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    max_rows_warn: int = 200000
) -> List[str]:
    warns: List[str] = []
    if df is None or df.empty:
        return ["No rows returned."]
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing: warns.append(f"Missing expected columns: {missing}")
    if value_cols:
        for c in value_cols:
            if c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    warns.append(f"Column '{c}' is not numeric (dtype={df[c].dtype}).")
                null_ratio = df[c].isna().mean()
                if null_ratio > 0.05:
                    warns.append(f"Column '{c}' has {null_ratio:.1%} missing values.")
                if df[c].abs().sum() == 0:
                    warns.append(f"Sum of '{c}' is zero; check filters/joins.")
            else:
                warns.append(f"Value column '{c}' not found in result.")
    if len(df) > max_rows_warn:
        warns.append(f"Large result: {len(df):,} rows. Consider narrowing the query.")
    return warns

# -----------------------------
# 7) Auto-viz + quick insight
# -----------------------------
def _topn_with_others(df, x_col, y_col, n=10):
    d = df[[x_col, y_col]].copy().groupby(x_col, as_index=False)[y_col].sum()
    d = d.sort_values(y_col, ascending=False)
    if len(d) <= n: return d
    head = d.head(n)
    tail_sum = d.iloc[n:][y_col].sum()
    return pd.concat([head, pd.DataFrame({x_col:["Others"], y_col:[tail_sum]})], ignore_index=True)

def auto_viz(df: pd.DataFrame, title: Optional[str] = None, topn=10):
    if df is None or df.empty:
        print("No data to visualize."); return

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    # Try to coerce YYYYMMDD style integers if present
    for c in df.columns:
        if c.lower() in {"date","time","time_id","invoice_sent_date","report_end_date_id"} and c not in dt_cols:
            try:
                col = df[c].astype(str).str.slice(0,8)
                df["_tmp_dt"] = pd.to_datetime(col, format="%Y%m%d", errors="coerce")
                if df["_tmp_dt"].notna().mean() > 0.8:
                    dt_cols.append("_tmp_dt"); break
                df.drop(columns=["_tmp_dt"], errors="ignore", inplace=True)
            except Exception:
                pass

    # line chart if (dt + numeric)
    if num_cols and dt_cols:
        y, t = num_cols[0], dt_cols[0]
        d = df[[t, y]].dropna().groupby(t, as_index=False)[y].sum().sort_values(t)
        plt.figure(figsize=(10,4))
        plt.plot(d[t], d[y])
        plt.xticks(rotation=45, ha="right"); plt.xlabel("Time"); plt.ylabel(y.replace("_"," ").title())
        if title: plt.title(title)
        plt.tight_layout(); plt.show(); return

    # bar chart if (categorical + numeric)
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if num_cols and obj_cols:
        y, x = num_cols[0], obj_cols[0]
        d = _topn_with_others(df, x, y, n=topn)
        plt.figure(figsize=(10,5))
        plt.bar(d[x].astype(str), d[y].values)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(x.replace("_"," ").title()); plt.ylabel(y.replace("_"," ").title())
        if title: plt.title(title)
        plt.tight_layout(); plt.show(); return

    print("(No obvious chart type) – showing head():")
    print(df.head(10).to_string(index=False))

def quick_insights(df: pd.DataFrame, cat_col: str, val_col: str, top=3) -> str:
    try:
        d = df[[cat_col, val_col]].copy().groupby(cat_col, as_index=False)[val_col].sum().sort_values(val_col, ascending=False)
        if d.empty: return "(no data)"
        total = d[val_col].sum(); head = d.head(top)
        parts = [f"{head.iloc[i][cat_col]} ({head.iloc[i][val_col]:,.0f})" for i in range(min(top, len(head)))]
        return f"Top {min(top, len(head))}: " + ", ".join(parts) + f". Total = {total:,.0f}. {head.iloc[0][cat_col]} = {head.iloc[0][val_col]/total:.1%} of total."
    except Exception:
        return "(insight unavailable)"

def save_bar_png(df, x_col, y_col, path, title=None, top=10):
    d = df[[x_col, y_col]].copy().groupby(x_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False).head(top)
    plt.figure(figsize=(10,5)); plt.bar(d[x_col].astype(str), d[y_col].values)
    plt.xticks(rotation=45, ha="right")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

# -----------------------------
# 8) Summary (cached)
# -----------------------------
def summarize(question: str, sql: str, df: pd.DataFrame) -> str:
    if not USE_LLM:
        return "(summary disabled: LLM off)"

    key = _hash(question + "|" + sql)
    cached = _sum_cache.get(key)
    if cached: return cached

    sample = df.to_dict(orient="records")[:10]
    prompt = f"""Summarize in ≤4 sentences for a business audience.
Question: {question}
SQL: {sql}
First rows JSON: {json.dumps(sample, ensure_ascii=False)}"""
    try:
        resp = llm_invoke([{"role":"user","content":prompt}])
        txt = resp.content
    except Exception as e:
        txt = f"(summary unavailable: {e})"

    _sum_cache[key] = txt; _save_json(LLM_SUM_CACHE, _sum_cache)
    return txt

# -----------------------------
# 9) One-call pipeline (visual-first)
# -----------------------------
def ask_v2(
    question: str,
    required_cols: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    preview: int = 10,
    summarize_answer: bool = False,  # default off (saves tokens)
    auto_plot: bool = True,
    export_png: bool = False,
    export_csv: bool = False,
    topn_for_plot: int = 10,
):
    sql = nl_to_sql_direct(question)
    print("SQL:\n", sql)

    df = run_sql_query(sql)
    print(f"\nRows: {len(df)}")
    print(df.head(preview).to_string(index=False))

    warns = validate_df(df, required_cols=required_cols, value_cols=value_cols)
    if warns:
        print("\n⚠️ Validation warnings:")
        for w in warns: print(" -", w)
    else:
        print("\n✅ Validation checks passed.")

    if summarize_answer:
        print("\nSummary:")
        print(summarize(question, sql, df))

    if auto_plot:
        # If possible, compute a quick insight (first object + first numeric)
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if obj_cols and num_cols:
            print("\nInsight:", quick_insights(df, obj_cols[0], num_cols[0]))
        auto_viz(df, title=question, topn=topn_for_plot)

    if export_csv:
        name = re.sub(r"[^a-zA-Z0-9]+", "_", question).strip("_")[:60]
        csv_path = Path("artifacts") / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    if export_png and auto_plot:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if obj_cols and num_cols:
            name = re.sub(r"[^a-zA-Z0-9]+", "_", question).strip("_")[:60]
            png_path = Path("artifacts/plots") / f"{name}.png"
            save_bar_png(df, obj_cols[0], num_cols[0], str(png_path), title=question, top=topn_for_plot)
            print(f"Saved PNG: {png_path}")

    return df, sql


# -----------------------------
# 10) Examples (run as a script)
# -----------------------------
if __name__ == "__main__":
    print("\n=== Demo ===")
    # Example 1
    ask_v2(
        "Show invoice totals and counts by sales class",
        required_cols=["sales_class_desc","total_invoiced","num_invoices"],
        value_cols=["total_invoiced","num_invoices"],
        summarize_answer=False,
        export_png=True, export_csv=True,
        topn_for_plot=10,
    )

    # Example 2
    ask_v2(
        "Which sales class has the highest total invoiced amount?",
        required_cols=["sales_class_desc","total_invoiced_amount"],
        value_cols=["total_invoiced_amount"],
        summarize_answer=False,
        export_png=False,
    )

    # Example 3
    ask_v2(
        "Monthly forecast vs actual and variance in 2014",
        required_cols=["time_year","time_quarter","time_month","total_actual","total_forecast","variance"],
        value_cols=["total_actual","total_forecast","variance"],
        summarize_answer=False,
        export_png=True,
    )
