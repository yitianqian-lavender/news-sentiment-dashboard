"""
News Sentiment Trend Dashboard
by Yitian Qian

This dashboard is a tool to analyze the sentiment of news articles.
It uses the Lexicon and BERT models to analyze the sentiment of the news articles.
It also uses the Guardian API to fetch the news articles.

"""

from datetime import date as _date
import time, requests, urllib.parse as ul
import numpy as np, pandas as pd, plotly.express as px, streamlit as st
import nltk
import os
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Unified color map
COLOR_MAP = {
    "POSITIVE": "#2ca02c", "Positive": "#2ca02c",
    "NEGATIVE": "#d62728", "Negative": "#d62728",
    "Neutral": "#1f77b4"
}

def today() -> _date: return _date.today()

try:
    secret_key = st.secrets.get("GUARDIAN_KEY")
except Exception:
    secret_key = None

GUARDIAN_KEY = os.getenv("GUARDIAN_KEY", "")

# Guardian API fetch with progress bar
def fetch_guardian(api_key: str, query: str, from_date=None, pages=3) -> pd.DataFrame:
    rows = []
    base = "https://content.guardianapis.com/search"
    progress = st.progress(0)
    for p in range(1, pages + 1):
        params = {
            "api-key": api_key, "q": query, "page": p,
            "page-size": 200, "show-fields": "headline"
        }
        if from_date: params["from-date"] = from_date
        data = requests.get(f"{base}?{ul.urlencode(params)}", timeout=15).json()
        if data.get("response", {}).get("status") != "ok":
            st.warning(f"Guardian error: {data}"); break
        for itm in data["response"]["results"]:
            rows.append((itm["webTitle"], itm["webPublicationDate"][:10]))
        progress.progress(p / pages)
        if p >= data["response"]["pages"]: break
        time.sleep(0.3)
    progress.empty()
    return pd.DataFrame(rows, columns=["title", "date"])

# Lexicon sentiment
@st.cache_data(show_spinner="Scoring with VADER…")
def add_lex(df):
    def vader_label(text):
        score = sia.polarity_scores(text)['compound']
        return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
    out = df.copy()
    out["lex_label"] = out["title"].astype(str).apply(vader_label)
    return out

# BERT binary sentiment (SafeTensors)
@st.cache_resource
def get_bert():
    from transformers import pipeline
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1)

bert_clf = get_bert()

@st.cache_data(show_spinner="Predicting with BERT…")
def add_bert(df):
    out = df.copy()
    preds = bert_clf(out["title"].tolist(), batch_size=32, truncation=True)
    out["bert_label"] = [p["label"].upper() for p in preds]
    return out

# Daily count grouping
def daily_counts(data, col):
    tbl = data.groupby(["date", col]).size().unstack(fill_value=0)
    rng = pd.date_range(tbl.index.min(), tbl.index.max(), freq="D").date
    return tbl.reindex(rng, fill_value=0)

# Line chart with smoothing toggle
def line_chart(tbl, y_cols, title, smooth=True):
    tidy = tbl.reset_index().rename(columns={"index": "date"}) if "date" not in tbl.columns else tbl.copy()
    tidy = tidy.sort_values("date").loc[:, ~tidy.columns.duplicated()]
    cols = [c for c in y_cols if c in tidy.columns]
    if smooth: tidy[cols] = tidy[cols].rolling(7, min_periods=1).mean()
    fig = px.line(tidy, x="date", y=cols,
                  color_discrete_map=COLOR_MAP,
                  labels={"value": "Count", "date": "Date"},
                  title=title)
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

# Streamlit UI
st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")
st.title("News Sentiment Trend Dashboard")

with st.sidebar:
    st.header("Upload CSV")
    csv_file = st.file_uploader("CSV with title & date columns", type="csv")
    st.header("Fetch from Guardian API")
    gd_key = st.text_input("Guardian key", value=GUARDIAN_KEY, type="password")
    gd_query = st.text_input("Keyword(s)", value="AI")
    gd_from = st.text_input("From date YYYY-MM-DD (optional)")
    gd_pages = st.slider("Pages (200 rows each)", 1, 20, 3)
    gd_fetch = st.button("Fetch from Guardian")
    st.header("Local filter & engine")
    kw_filter = st.text_input("Keyword filter (local)")
    engine = st.radio("Sentiment engine", ["Lexicon", "BERT", "Compare"])
    st.header("Chart options")
    smooth_toggle = st.checkbox("7-day smoothing", value=True)
    pie_mode = st.selectbox("Pie chart labels", ["percent", "value", "percent+value"])

# Load data
raw_df = pd.DataFrame(columns=["title", "date"])
if csv_file: raw_df = pd.read_csv(csv_file)
if gd_fetch and gd_key and gd_query.strip():
    with st.spinner("Fetching headlines…"):
        fetched = fetch_guardian(gd_key.strip(), gd_query.strip(), gd_from.strip() or None, gd_pages)
    st.toast(f"Fetched {len(fetched)} rows from Guardian")
    raw_df = pd.concat([raw_df, fetched], ignore_index=True)
    st.session_state.pop("date_slider", None)
if raw_df.empty: st.warning("Upload a CSV or fetch headlines first."); st.stop()

df = raw_df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["date"])
if kw_filter: df = df[df["title"].str.contains(kw_filter, case=False, na=False)]
if df.empty: st.warning("No rows match your filter."); st.stop()

min_d, max_d = df["date"].min(), df["date"].max()
d_start, d_end = st.slider("Date range", min_value=min_d, max_value=max_d,
                           value=(min_d, max_d), format="MMM DD YY", key="date_slider")
df = df[(df["date"] >= d_start) & (df["date"] <= d_end)]

df = add_lex(df)
if engine in ("BERT", "Compare"): df = add_bert(df)

# KPIs
st.metric("Date Range", f"{df['date'].min()} → {df['date'].max()}")
avg_daily = len(df) / ((df['date'].max() - df['date'].min()).days + 1)
st.metric("Avg daily headlines", f"{avg_daily:.1f}")
today_rows = df[df["date"] == _date.today()]
st.metric("Today - Positive", int(today_rows["lex_label"].eq("Positive").sum()))
st.metric("Today - Negative", int(today_rows["lex_label"].eq("Negative").sum()))
active_col = "lex_label" if engine == "Lexicon" else ("bert_label" if engine == "BERT" else None)
if engine != "Compare":
    total_rows = len(df)
    pos_rows = int(df[active_col].str.contains("POS", case=False).sum())
    st.metric("Total headlines", total_rows)
    st.metric("Positive ratio", f"{pos_rows / total_rows:.0%}")

# Visualizations
if engine == "Lexicon":
    g = daily_counts(df, "lex_label")
    line_chart(g, ["Positive", "Negative", "Neutral"], "Daily sentiment (Lexicon)", smooth_toggle)
elif engine == "BERT":
    g = daily_counts(df, "bert_label")
    line_chart(g, ["POSITIVE", "NEGATIVE"], "Daily sentiment (BERT)", smooth_toggle)
else:
    lex_tbl = daily_counts(df, "lex_label")
    bert_tbl = daily_counts(df, "bert_label")
    lex_pos = lex_tbl.get("Positive", pd.Series(0, index=lex_tbl.index))
    bert_pos = bert_tbl.get("POSITIVE", pd.Series(0, index=bert_tbl.index))
    merged = pd.DataFrame({"date": lex_pos.index, "Lex_Pos": lex_pos.values, "BERT_Pos": bert_pos.values})
    line_chart(merged, ["Lex_Pos", "BERT_Pos"], "Positive trend — Lexicon vs BERT", smooth_toggle)

    col1, col2 = st.columns(2, gap="small")
    pie_lex = df["lex_label"].value_counts().rename_axis("label").reset_index(name="count")
    fig1 = px.pie(pie_lex, names="label", values="count", title="Lexicon polarity",
                  color="label", color_discrete_map=COLOR_MAP, hole=0.3)
    fig1.update_traces(textinfo=pie_mode)
    col1.plotly_chart(fig1, use_container_width=True)
    col1.metric("Lexicon Pos %", f"{df['lex_label'].eq('Positive').mean():.0%}")

    pie_bert = df["bert_label"].value_counts().rename_axis("label").reset_index(name="count")
    fig2 = px.pie(pie_bert, names="label", values="count", title="BERT polarity",
                  color="label", color_discrete_map=COLOR_MAP, hole=0.3)
    fig2.update_traces(textinfo=pie_mode)
    col2.plotly_chart(fig2, use_container_width=True)
    col2.metric("BERT Pos %", f"{df['bert_label'].eq('POSITIVE').mean():.0%}")

# Download
@st.cache_data
def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")
st.download_button("Download current CSV", to_csv_bytes(df),
                   "headlines_sentiment.csv", "text/csv")
with st.expander(f"Raw data (showing {min(len(df),1000)} of {len(df)} rows)"):
    st.dataframe(df.head(1000))
