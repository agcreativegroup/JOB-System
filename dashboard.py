# ========================================
#  JOB MARKET DASHBOARD + RECOMMENDER
# ========================================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # quiet TensorFlow CPU note

print(os.listdir("models"))
import io
import ast
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

# --- DL / NLP ---
import tensorflow as tf
from tensorflow.keras.models import load_model
try:
    # prefer TF-keras if present (Transformers compat)
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
except Exception:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Job Market Dashboard", layout="wide")
st.title("Job Market Dynamics Dashboard + Job Recommender")

# -----------------------------
# Helpers (cache-aware)
# -----------------------------
STOPWORDS = {
    "and","or","to","a","the","in","of","for","on","with","at","by","an","is","are",
    "from","as","it","this","that","be","we","i","you","they","our","us","your","needed","looking","want","-",
    ":", "&", "amp", ".", ",", "(", ")", "[", "]", "{", "}", "/", "\\"
}

def parse_keywords(cell):
    """Safely parse the keywords column into a clean list."""
    if isinstance(cell, list):
        raw = cell
    else:
        try:
            raw = ast.literal_eval(str(cell))
        except Exception:
            raw = []
    cleaned = []
    for w in raw:
        w = str(w).strip().lower()
        if not w:
            continue
        if w in {"-", ":", ".", ",", "&", "amp"}:
            continue
        if w in STOPWORDS:
            continue
        cleaned.append(w)
    return cleaned

@st.cache_data(show_spinner=False)
def load_csv(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like)
    return df

@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns: ['title','link','published_date','is_hourly','hourly_low','hourly_high','budget','country','job_title_clean','keywords','year_month']
    if 'year_month_str' not in df.columns:
        # build from published_date or year_month
        if 'published_date' in df.columns:
            tm = pd.to_datetime(df['published_date'], errors='coerce')
            df['year_month_str'] = tm.dt.to_period('M').astype(str)
        elif 'year_month' in df.columns:
            df['year_month_str'] = pd.to_datetime(df['year_month']).dt.to_period('M').astype(str)
        else:
            # fallback: everything as one month
            df['year_month_str'] = '1970-01'
    else:
        df['year_month_str'] = pd.to_datetime(df['year_month_str'], errors='coerce')
        df['year_month_str'] = df['year_month_str'].dt.to_period('M').astype(str)

    # Remote flag from title
    df['remote'] = df.get('job_title_clean', df.get('title','')).astype(str).str.contains(
        r"\b(remote|work from home|telecommute|wfh)\b", case=False, na=False
    )

    # Numeric hourly
    for c in ['hourly_low','hourly_high','budget']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Average hourly if present
    if {'hourly_low','hourly_high'}.issubset(df.columns):
        df['avg_hourly'] = (df['hourly_low'].fillna(0) + df['hourly_high'].fillna(0)) / 2.0

    # Clean keywords to list
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(parse_keywords)
    else:
        df['keywords'] = [[] for _ in range(len(df))]

    # Drop dupes by link if available
    if 'link' in df.columns:
        df = df.drop_duplicates(subset=['link'])
    else:
        df = df.drop_duplicates()

    return df

@st.cache_resource(show_spinner=False)
def load_lstm():
    try:
        return load_model("models/job_lstm_model.h5")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_cnn():
    try:
        return load_model("models/job_cnn_model.h5")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_bert():
    # prefer local fine-tuned folder if it exists
    local_dir = "models/job_bert_model"
    try:
        if os.path.isdir(local_dir):
            tok = BertTokenizer.from_pretrained(local_dir)
            mdl = TFBertForSequenceClassification.from_pretrained(local_dir)
            return (tok, mdl)
        return None
    except Exception:
        return None

# -----------------------------
# Sidebar: Load & Real-time fetch
# -----------------------------
st.sidebar.header(" Data Source")

uploaded = st.sidebar.file_uploader("Upload cleaned CSV (e.g., eda_jobs.csv)", type=["csv"])
realtime_url = st.sidebar.text_input("Optional: CSV/JSON URL for real-time jobs")
fetch_btn = st.sidebar.button("Fetch & Merge Real-time Data")

if uploaded:
    base_df = load_csv(uploaded)
else:
    st.sidebar.info("Upload a CSV to start. (You can also add a URL to fetch more data.)")
    st.stop()

data = preprocess(base_df)

# --- Real-time fetch / merge (optional) ---
if fetch_btn and realtime_url.strip():
    try:
        r = requests.get(realtime_url.strip(), timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type","").lower()

        # Support CSV or JSON
        if "json" in content_type or realtime_url.strip().lower().endswith(".json"):
            new_raw = pd.json_normalize(r.json())
        else:
            new_raw = pd.read_csv(io.StringIO(r.text))

        new_df = preprocess(new_raw)
        # Concatenate & refresh cache
        data = pd.concat([data, new_df], ignore_index=True)
        data = preprocess(data)  # re-run to normalize post-merge
        st.sidebar.success(f"Merged real-time rows: {len(new_df)}")
    except Exception as e:
        st.sidebar.error(f"Fetch failed: {e}")

# Save merged/cached dataset (optional)
if st.sidebar.button(" Save merged dataset as eda_jobs_merged.csv"):
    out_name = "eda_jobs_merged.csv"
    data.to_csv(out_name, index=False)
    st.sidebar.success(f"Saved {out_name}")

# -----------------------------
# Tabs Layout
# -----------------------------
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview",
    " Remote vs Onsite",
    " Job Forecast",
    " Keyword Forecast",
    " Country Forecast",
    " Recommendations"
])

# -----------------------------
# 0. Overview (quick stats + salary map)
# -----------------------------
with tab0:
    st.subheader(" Overview & Salary Map")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Jobs", f"{len(data):,}")
    col_b.metric("Countries", data['country'].nunique() if 'country' in data.columns else 0)
    col_c.metric("Months Covered", data['year_month_str'].nunique())
    remote_share = (100.0 * data['remote'].mean()) if 'remote' in data.columns else 0.0
    col_d.metric("Remote Share", f"{remote_share:.1f}%")

    if 'avg_hourly' in data.columns and 'country' in data.columns:
        country_salary = (
            data.dropna(subset=['avg_hourly','country'])
                .groupby('country', as_index=False)['avg_hourly'].mean()
        )
        fig_map = px.choropleth(
            country_salary, locations="country", locationmode="country names",
            color="avg_hourly", hover_name="country",
            title="Average Hourly Rate by Country", color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No hourly fields to build salary map (needs hourly_low & hourly_high).")

# -----------------------------
# 1. Remote vs Onsite Trend
# -----------------------------
with tab1:
    st.subheader(" Remote vs Onsite Jobs Over Time")
    remote_trend = (data.groupby(['year_month_str','remote'])
                        .size().reset_index(name='count'))
    fig_remote = px.line(
        remote_trend, x="year_month_str", y="count", color="remote",
        title="Remote vs Onsite Job Trend"
    )
    st.plotly_chart(fig_remote, use_container_width=True)

# -----------------------------
# 2. Job Market Forecast (Prophet)
# -----------------------------
with tab2:
    st.subheader("Job Market Forecast (Next 6 Months)")
    job_trend = data.groupby('year_month_str').size().reset_index(name='count')
    job_trend = job_trend.rename(columns={'year_month_str':'ds','count':'y'})
    job_trend['ds'] = pd.to_datetime(job_trend['ds'])

    if len(job_trend) >= 3:
        model = Prophet()
        model.fit(job_trend[['ds','y']])
        future = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future)
        fig_forecast = px.line(forecast, x="ds", y="yhat",
                               title="Predicted Job Market Trend (Next 6 Months)")
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Need at least 3 monthly points to forecast.")

# -----------------------------
# 3. Keyword Forecasting
# -----------------------------
with tab3:
    st.subheader(" Keyword-Level Forecasting")
    # explode keywords over months
    trend_keywords = []
    for _, row in data[['year_month_str','keywords']].iterrows():
        for kw in row['keywords']:
            trend_keywords.append((row['year_month_str'], kw))
    if trend_keywords:
        trend_df = pd.DataFrame(trend_keywords, columns=['year_month_str','keyword'])
        keyword_trend = (trend_df
                         .groupby(['year_month_str','keyword'])
                         .size().reset_index(name='count'))

        top_keywords = trend_df['keyword'].value_counts().head(8).index
        selected_kw = st.selectbox("Select a Keyword:", top_keywords)

        kw_data = keyword_trend[keyword_trend['keyword'] == selected_kw].copy()
        kw_data = kw_data.rename(columns={'year_month_str':'ds','count':'y'})
        kw_data['ds'] = pd.to_datetime(kw_data['ds'])

        fig_kw_hist = px.line(keyword_trend[keyword_trend['keyword'].isin(top_keywords)],
                              x='year_month_str', y='count', color='keyword',
                              title="Top Keyword Frequencies Over Time")
        st.plotly_chart(fig_kw_hist, use_container_width=True)

        if len(kw_data) >= 3:
            model_kw = Prophet()
            model_kw.fit(kw_data[['ds','y']])
            future_kw = model_kw.make_future_dataframe(periods=6, freq='M')
            forecast_kw = model_kw.predict(future_kw)
            fig_kw = px.line(forecast_kw, x="ds", y="yhat",
                             title=f"Forecast for '{selected_kw}' (Next 6 Months)")
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Need ≥3 points for this keyword to forecast.")
    else:
        st.info("No keywords available in dataset.")

# -----------------------------
# 4. Country-Level Keyword Forecast
# -----------------------------
with tab4:
    st.subheader(" Country-Level Keyword Forecast")
    if 'country' not in data.columns:
        st.info("No 'country' column found.")
    else:
        trend_list = []
        for _, row in data[['year_month_str','country','keywords']].iterrows():
            for kw in row['keywords']:
                trend_list.append((row['year_month_str'], row['country'], kw))
        if trend_list:
            tdf = pd.DataFrame(trend_list, columns=['year_month_str','country','keyword'])
            agg = tdf.groupby(['year_month_str','country','keyword']).size().reset_index(name='count')
            top_kw_country = tdf['keyword'].value_counts().head(10).index
            selected_kw2 = st.selectbox("Keyword", top_kw_country)
            sub = agg[agg['keyword'] == selected_kw2]
            if sub.empty:
                st.info("No data for this keyword.")
            else:
                chosen_country = st.selectbox("Country", sorted(sub['country'].unique()))
                cdata = sub[sub['country'] == chosen_country].copy()
                cdata = cdata.rename(columns={'year_month_str':'ds','count':'y'})
                cdata['ds'] = pd.to_datetime(cdata['ds'])
                if len(cdata) >= 3:
                    model_c = Prophet()
                    model_c.fit(cdata[['ds','y']])
                    future_c = model_c.make_future_dataframe(periods=6, freq='M')
                    forecast_c = model_c.predict(future_c)
                    fig_c = px.line(forecast_c, x="ds", y="yhat",
                                    title=f"Forecast for '{selected_kw2}' in {chosen_country}")
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.info("Need ≥3 points for this country-keyword pair to forecast.")
        else:
            st.info("No country/keyword pairs available.")

# -----------------------------
# 5. Job Recommendation System
# -----------------------------
with tab5:
    st.subheader(" Job Recommendation System")

    import pickle
    # Load only Keras Tokenizer + LabelEncoder
    try:
        with open("models/keras_tokenizer.pkl", "rb") as f:
            keras_tokenizer = pickle.load(f)
        with open("models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        st.error(f"Tokenizer/LabelEncoder not found in models/: {e}")
        st.stop()

    user_input = st.text_input("Enter a job title or description:")

    colL, colC, colB = st.columns(3)
    lstm_model = load_lstm()
    cnn_model  = load_cnn()
    bert_pack  = load_bert()

    if st.button("Get Recommendations"):
        if not user_input.strip():
            st.warning("Please enter a job description.")
        else:
            # ---------------------
            # LSTM + CNN (Keras Tokenizer)
            # ---------------------
            try:
                seq = keras_tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(seq, maxlen=50, padding="post", truncating="post")
            except Exception as e:
                st.error(f"Keras Tokenizer error: {e}")
                padded = None

            # LSTM Prediction
            with colL:
                if lstm_model is not None and padded is not None:
                    try:
                        pred_lstm = np.argmax(lstm_model.predict(padded), axis=1)
                        st.success(" LSTM: " + label_encoder.inverse_transform(pred_lstm)[0])
                    except Exception as e:
                        st.warning(f"LSTM error: {e}")
                else:
                    st.warning(" LSTM model not found or tokenizer failed.")

            # CNN Prediction
            with colC:
                if cnn_model is not None and padded is not None:
                    try:
                        pred_cnn = np.argmax(cnn_model.predict(padded), axis=1)
                        st.success(" CNN: " + label_encoder.inverse_transform(pred_cnn)[0])
                    except Exception as e:
                        st.warning(f"CNN error: {e}")
                else:
                    st.warning(" CNN model not found or tokenizer failed.")

           # ---------------------
# BERT (HuggingFace Tokenizer)
# ---------------------
with colB:
    if bert_pack is not None:
        try:
            tokenizer_bert, model_bert = bert_pack
            st.write("✅ BERT model loaded successfully.")  # Debug message

            enc = tokenizer_bert(
                [user_input],
                truncation=True,
                padding=True,
                max_length=32,
                return_tensors="tf"
            )
            # Explicitly pass inputs
            outputs = model_bert(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                training=False
            )
            logits = outputs.logits
            pred_id = int(tf.argmax(logits, axis=1).numpy()[0])

            st.success(" BERT: " + label_encoder.inverse_transform([pred_id])[0])
        except Exception as e:
            st.error(f" BERT prediction error: {e}")
    else:
        st.error(" BERT model not found (expected folder: /job_bert_model/)")


    st.caption(" Tip: Place your trained models in ./models (job_lstm_model.h5, job_cnn_model.h5, job_bert_model/, keras_tokenizer.pkl, label_encoder.pkl).")
