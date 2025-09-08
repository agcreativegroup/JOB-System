import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['cnn_job_classifier'] = load_model("cnn_job_classifier.h5", compile=False)
    except Exception:
        models['cnn_job_classifier'] = None

    try:
        models['cnn_salary_predictor'] = load_model("cnn_salary_predictor.h5", compile=False)
    except Exception:
        models['cnn_salary_predictor'] = None

    try:
        with open("cnn_tokenizer.pkl", "rb") as f:
            models['cnn_tokenizer'] = pickle.load(f)
    except Exception:
        models['cnn_tokenizer'] = None

    try:
        with open("category_scaler.pkl", "rb") as f:
            models['category_scaler'] = pickle.load(f)
    except Exception:
        models['category_scaler'] = None

    try:
        with open("job_category_encoder.pkl", "rb") as f:
            models['job_category_encoder'] = pickle.load(f)
    except Exception:
        models['job_category_encoder'] = None

    return models

models = load_models()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📊 Job Market Analysis")
page = st.sidebar.radio(
    "Select Task:",
    [
        "🏠 Home Dashboard",
        "📊 Task 1: Title-Salary Correlation",
        "🚀 Task 2: Emerging Job Categories",
        "📈 Task 3: High-Demand Roles (LSTM - placeholder)",  # ← keep label consistent
        "🌍 Task 4: Hourly Rates by Country",
        "🎯 Task 5: Job Recommendation Engine + Salary Prediction",
        "📋 Task 6: Market Dynamics Dashboard",
        "🏡 Task 7: Remote Work Trends",
        "🔮 Task 8: CNN Demand Prediction"
    ]
)

uploaded = st.sidebar.file_uploader("Upload Job Data (CSV)", type=["csv"])
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

# -------------------------
# Home Dashboard
# -------------------------
if page == "🏠 Home Dashboard":
    st.title("🏠 Job Market Dashboard")
    if df is not None and not df.empty:
        st.metric("Total Jobs", f"{len(df):,}")
        if "country" in df.columns:
            st.metric("Countries", df['country'].nunique())
        if "budget" in df.columns and pd.api.types.is_numeric_dtype(df['budget']):
            st.metric("Avg Salary", f"${df['budget'].mean():,.0f}")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())
    else:
        st.info("📂 Upload a dataset in the sidebar.")

# -------------------------
# Task 1: Title-Salary Correlation
# -------------------------
elif page == "📊 Task 1: Title-Salary Correlation":
    st.title("📊 Job Title-Salary Correlation")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    elif {"title", "budget"}.issubset(df.columns):
        # Ensure numeric budget
        df = df.copy()
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
        df['title_len'] = df['title'].astype(str).str.len()
        fig = px.scatter(df.dropna(subset=["budget"]), x="title_len", y="budget",
                         title="Salary vs Job Title Length")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset must contain 'title' and 'budget' columns.")

# -------------------------
# Task 2: Emerging Job Categories
# -------------------------
elif page == "🚀 Task 2: Emerging Job Categories":
    st.title("🚀 Emerging Job Categories")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    elif "job_category" in df.columns:
        cat_counts = df['job_category'].value_counts().head(10)
        st.subheader("Top Job Categories")
        st.bar_chart(cat_counts)
    else:
        st.warning("Dataset must contain 'job_category'.")

# -------------------------
# Task 3: High-Demand Roles (LSTM placeholder)
# -------------------------
elif page == "📈 Task 3: High-Demand Roles (LSTM - placeholder)":
    st.title("📈 High-Demand Roles Prediction (LSTM)")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    else:
        # Detect a date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)

        if not date_col:
            st.error("❌ No date-like column found in dataset. Add 'published_date' or similar.")
        else:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            daily = df.dropna(subset=[date_col]).groupby(df[date_col].dt.date).size()

            st.write(f"📊 Found {len(daily)} daily data points from {daily.index.min()} → {daily.index.max()}")

            if len(daily) == 0:
                st.error("❌ No valid daily job counts. Check your dataset.")
            else:
                hist_df = pd.DataFrame({"date": pd.to_datetime(daily.index), "jobs": daily.values})

                # Try loading model + scaler
                try:
                    lstm_model = load_model("lstm_job_volume.h5", compile=False)
                    with open("job_volume_scaler.pkl", "rb") as f:
                        scaler = pickle.load(f)
                except Exception as e:
                    st.warning(f"⚠️ Could not load LSTM model or scaler: {e}")
                    lstm_model, scaler = None, None

                forecast_df = None
                if lstm_model is not None and scaler is not None and len(hist_df) >= 30:
                    try:
                        daily_vals = hist_df["jobs"].values.reshape(-1, 1)
                        scaled = scaler.transform(daily_vals)
                        X_seq = np.expand_dims(scaled[-30:], axis=0)

                        preds = lstm_model.predict(X_seq)
                        preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

                        last_date = hist_df["date"].iloc[-1]
                        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(preds))

                        forecast_df = pd.DataFrame({"date": future_dates, "jobs": preds})
                        st.success("✅ Forecast generated with LSTM model.")
                    except Exception as e:
                        st.error(f"❌ Error during LSTM prediction: {e}")

                # Plot
                fig = px.line(hist_df, x="date", y="jobs", title="Job Market Volume (History & Forecast)")
                if forecast_df is not None:
                    fig.add_scatter(x=forecast_df["date"], y=forecast_df["jobs"],
                                    mode="lines+markers", name="Forecast")
                else:
                    st.warning("⚠️ Showing history only (no forecast).")
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Task 4: Hourly Rates by Country
# -------------------------
elif page == "🌍 Task 4: Hourly Rates by Country":
    st.title("🌍 Hourly Rates by Country")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    elif {'country', 'budget'}.issubset(df.columns):
        tmp = df.copy()
        tmp['budget'] = pd.to_numeric(tmp['budget'], errors='coerce')
        avg_rates = tmp.groupby("country")['budget'].mean().reset_index().dropna()
        fig = px.bar(avg_rates, x="country", y="budget", title="Average Pay by Country")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset must contain 'country' and 'budget'.")

# -------------------------
# Task 5: Job Recommendation + Salary Prediction
# -------------------------
elif page == "🎯 Task 5: Job Recommendation Engine + Salary Prediction":
    st.title("🎯 Job Recommendation & Salary Prediction")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    else:
        user_input = st.text_area("Enter your skills or job preferences:")
        if st.button("Recommend Jobs"):
            desc_col = df['description'] if 'description' in df.columns else pd.Series([''] * len(df))
            text_field = df['title'].fillna('') + " " + desc_col.fillna('')
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(text_field)
            user_vec = tfidf.transform([user_input])
            sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
            top_idx = sim.argsort()[-5:][::-1]
            cols = [c for c in ['title', 'country', 'job_category', 'budget'] if c in df.columns]
            result = df.iloc[top_idx][cols].copy()
            st.subheader("🔎 Recommended Jobs")
            st.dataframe(result)

            # Salary prediction using CNN
            if models['cnn_salary_predictor'] is not None and models['cnn_tokenizer'] is not None:
                seq = models['cnn_tokenizer'].texts_to_sequences([user_input])
                seq_pad = pad_sequences(seq, maxlen=50)
                salary_pred = models['cnn_salary_predictor'].predict(seq_pad)
                st.success(f"💰 Predicted Salary: ${salary_pred[0][0]:,.0f}")
            else:
                st.info("⚠️ Salary predictor not available.")

# -------------------------
# Task 6: Market Dynamics
# -------------------------
elif page == "📋 Task 6: Market Dynamics Dashboard":
    st.title("📋 Job Market Dynamics")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    elif "published_date" in df.columns:
        tmp = df.copy()
        tmp['published_date'] = pd.to_datetime(tmp['published_date'], errors='coerce')
        monthly = tmp.dropna(subset=['published_date']).groupby(tmp['published_date'].dt.to_period("M")).size()
        monthly = monthly.rename("jobs").to_frame()
        # Convert PeriodIndex -> Timestamp (month start) for plotting
        monthly.index = monthly.index.to_timestamp()
        st.line_chart(monthly)
    else:
        st.warning("Dataset must contain 'published_date'.")

# -------------------------
# Task 7: Remote Work Trends
# -------------------------
elif page == "🏡 Task 7: Remote Work Trends":
    st.title("🏡 Remote Work Trends")
    if df is None:
        st.info("📂 Upload a dataset in the sidebar.")
    elif "description" in df.columns:
        remote_jobs = df['description'].astype(str).str.contains(r"\b(remote|wfh)\b", case=False, regex=True).mean() * 100
        st.metric("Remote Job %", f"{remote_jobs:.2f}%")
    else:
        st.warning("Dataset must contain 'description'.")

# -------------------------
# Task 8: CNN Demand Prediction
# -------------------------
elif page == "🔮 Task 8: CNN Demand Prediction":
    st.title("🔮 CNN Job Demand Prediction")
    if models['cnn_job_classifier'] is not None and models['cnn_tokenizer'] is not None:
        user_input = st.text_input("Enter a job title to predict demand:")
        if st.button("Predict Demand"):
            seq = models['cnn_tokenizer'].texts_to_sequences([user_input])
            seq_pad = pad_sequences(seq, maxlen=50)
            pred = models['cnn_job_classifier'].predict(seq_pad)
            class_idx = int(np.argmax(pred))
            categories = ["Low Demand", "Medium Demand", "High Demand"]
            st.success(f"Predicted Demand: {categories[class_idx]} ({pred[0][class_idx]*100:.1f}% confidence)")
    else:
        st.info("⚠️ CNN classifier not available. Upload models or check paths.")
