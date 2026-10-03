import os

import pandas as pd
import plotly.express as px
import streamlit as st

import agent
import config

st.set_page_config(page_title="Data Lakehouse Agent", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_secret(name, default=""):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:
        return default


def password_gate():
    """Optional: set APP_PASSWORD in secrets to stop strangers burning your API quota."""
    pw = get_secret("APP_PASSWORD")
    if not pw or st.session_state.get("authed"):
        return
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == pw:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


@st.cache_resource(show_spinner="Starting the data engine (first load can take a few seconds)...")
def get_agent():
    # One DuckDB connection + LLM client shared by all sessions.
    return agent.build_agent()


def choose_chart(df: pd.DataFrame, plan):
    """Return (kind, x, y, color). Honors the LLM's suggestion when it is valid,
    otherwise falls back to a heuristic."""
    if df.shape[0] == 0:
        return "table", None, None, None
    num_cols = df.select_dtypes("number").columns.tolist()
    if not num_cols:
        return "table", None, None, None
    if df.shape[0] == 1:
        return "metric", None, None, None

    hint = getattr(plan, "chart_type", "auto")
    hx, hy = getattr(plan, "x_column", ""), getattr(plan, "y_column", "")
    x = hx if hx in df.columns else None
    y = hy if hy in df.columns and hy in num_cols else None

    if x is None:
        non_num = [c for c in df.columns if c not in num_cols]
        x = non_num[0] if non_num else df.columns[0]
    if y is None:
        y = next((c for c in num_cols if c != x), num_cols[0])

    color = next((c for c in df.columns if c not in (x, y) and c not in num_cols), None)
    is_time = pd.api.types.is_datetime64_any_dtype(df[x])

    if hint == "table":
        return "table", None, None, None
    if is_time or hint == "line":
        return "line", x, y, color
    if hint == "pie" and df.shape[0] <= 8 and color is None:
        return "pie", x, y, None
    return "bar", x, y, color


def draw_chart(df, plan, key):
    kind, x, y, color = choose_chart(df, plan)
    if kind == "table":
        st.info("This result is best read as a table.")
        return
    if kind == "metric":
        nums = df.select_dtypes("number").columns[:4]
        cols = st.columns(len(nums))
        for c, name in zip(cols, nums):
            val = df[name].iloc[0]
            c.metric(name.replace("_", " ").title(), f"{val:,.2f}" if isinstance(val, float) else f"{val:,}")
        return

    plot_df = df.copy()
    # Numeric IDs like payment_type=1,2,3 are categories, not a continuous axis.
    if pd.api.types.is_integer_dtype(plot_df[x]) and plot_df[x].nunique() <= 30:
        plot_df[x] = plot_df[x].astype(str)

    common = dict(template="plotly_dark", title=f"{y} by {x}")
    if kind == "line":
        fig = px.line(plot_df, x=x, y=y, color=color, markers=True, **common)
    elif kind == "pie":
        fig = px.pie(plot_df, names=x, values=y, template="plotly_dark", title=common["title"])
    else:
        fig = px.bar(plot_df, x=x, y=y, color=color or x, barmode="group", **common)
    st.plotly_chart(fig, width="stretch", key=key)


def render_entry(entry, idx):
    if entry["error"]:
        st.error(entry["error"])
        if entry.get("sql"):
            with st.expander("View last attempted SQL"):
                st.code(entry["sql"], language="sql")
        return

    if entry["summary"]:
        st.markdown(f"**{entry['summary']}**")

    df = entry["df"]
    if df.empty:
        st.warning("The query ran but returned no rows.")
    else:
        left, right = st.columns(2)
        with left:
            st.caption(f"Data ({len(df):,} rows)")
            st.dataframe(df, width="stretch")
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                               file_name="result.csv", mime="text/csv", key=f"dl_{idx}")
        with right:
            st.caption("Visualization")
            draw_chart(df, entry["plan"], key=f"chart_{idx}")
        if entry["truncated"]:
            st.warning(f"Result capped at {config.ROW_LIMIT:,} rows. Ask a narrower question for the full picture.")

    with st.expander("View generated SQL"):
        st.code(entry["sql"], language="sql")
        if entry["attempts"] > 1:
            st.caption(f"Self-corrected: succeeded on attempt {entry['attempts']}.")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("📊 Conversational Data Lakehouse")
st.caption("Ask questions about NYC yellow-taxi trips in plain English. The AI writes the SQL, "
           "DuckDB runs it on Parquet files, and you get a chart and a summary.")

password_gate()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    ag = get_agent()
except Exception as e:
    st.error(f"Could not start the data engine: {e}")
    st.stop()

# Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    labels = {"local": "Full local dataset", "sample": "Sample dataset", "remote": "Remote Parquet (HTTPS)"}
    st.write(f"**Data source:** {labels.get(ag.source_kind, ag.source_kind)}")
    st.write(f"**Model:** `{config.MODEL_NAME}`")
    st.write(f"**Security lockdown:** {'on ✅' if ag.locked else 'validation only ⚠️'}")
    if ag.source_kind == "sample":
        info = agent.load_sample_info()
        if info:
            st.warning(f"Sample of {info['sample_rows']:,} of {info['total_rows']:,} trips "
                       f"({info['sample_rows'] / info['total_rows']:.1%}). Counts and totals are "
                       "understated; averages and percentages are representative.")
        else:
            st.warning("Running on a sample: counts and totals are understated.")

    st.header("Try asking")
    examples = [
        "Revenue by payment type and each one's share",
        "Top 10 pickup neighborhoods by trips",
        "Monthly revenue trend",
        "Average tip percentage for credit card trips by borough",
        "What share of trips go to JFK Airport?",
    ]
    for q in examples:
        if st.button(q, key=f"ex_{q}", width="stretch"):
            st.session_state.pending = q
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# Conversation history ----------------------------------------------------------
for i, m in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.write(m["question"])
    with st.chat_message("assistant"):
        render_entry(m, i)

# New question ----------------------------------------------------------------
question = st.chat_input("Ask a business question (e.g. 'now break that down by month')")
question = question or st.session_state.pop("pending", None)

if question:
    if len(st.session_state.messages) >= config.MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit reached for this session. Clear the conversation to continue.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("🧠 Writing SQL and querying the data..."):
            try:
                history = [{"question": m["question"], "sql": m["sql"]}
                           for m in st.session_state.messages if not m["error"]]
                res = ag.ask(question, history)
            except Exception as e:  # LLM/API/network problems (not SQL problems)
                st.error(f"The AI service had a problem (not your question): {e}")
                st.stop()

    st.session_state.messages.append(dict(
        question=question, sql=res.sql, df=res.df if res.df is not None else pd.DataFrame(),
        truncated=res.truncated, attempts=res.attempts, plan=res.plan,
        summary=res.summary, error=res.error))
    st.rerun()