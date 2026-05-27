import streamlit as st
import joblib
import pandas as pd
import numpy as np

 
model = joblib.load("final_ads_revenue_model.pkl")
scaler = joblib.load("scaler.pkl")
 
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="🎬",
    layout="centered"
)

 
st.title("🎬 YouTube Ad Revenue Predictor")
st.write("Enter your video details to predict ad revenue!")
st.divider()

 
st.subheader("📊 Video Details")

watch_time = st.slider(
    "⏱️ Total Watch Time (minutes)",
    min_value=14659,
    max_value=61557,
    value=37544
)

views = st.slider(
    "👀 Views",
    min_value=9521,
    max_value=10468,
    value=9999
)

likes = st.slider(
    "👍 Likes",
    min_value=195,
    max_value=2061,
    value=1099
)

video_length = st.slider(
    "🎥 Video Length (minutes)",
    min_value=2,
    max_value=30,
    value=16
)

st.divider()

# ✅ Auto Calculate
like_rate = likes / (views + 1)
log_views = np.log1p(views)
log_likes = np.log1p(likes)

# ✅ Show Calculated Values
st.subheader("🔢 Auto Calculated")
col1, col2, col3 = st.columns(3)
col1.metric("Like Rate", f"{like_rate:.4f}")
col2.metric("Log Views", f"{log_views:.4f}")
col3.metric("Log Likes", f"{log_likes:.4f}")

st.divider()

# ✅ Dollar to Rupees Conversion
USD_TO_INR = 84.0

if st.button("🚀 Predict Revenue", use_container_width=True):

    new_data = pd.DataFrame({
        'watch_time_minutes': [watch_time],
        'log_likes': [log_likes],
        'like_rate': [like_rate],
        'video_length_minutes': [video_length],
        'log_views': [log_views]
    })

    # ✅ Scale
    new_data_scaled = scaler.transform(new_data)

    # ✅ Predict
    prediction = np.expm1(model.predict(new_data_scaled))
    revenue_usd = prediction[0]
    revenue_inr = revenue_usd * USD_TO_INR

    st.divider()
    st.subheader("💰 Prediction Result")

    # ✅ INR Results
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Expected",
                f"₹{revenue_inr * 0.9:,.2f}")
    col2.metric("Predicted Revenue",
                f"₹{revenue_inr:,.2f}",
                delta="Best Estimate")
    col3.metric("Max Expected",
                f"₹{revenue_inr * 1.1:,.2f}")

    st.divider()

    # ✅ USD Show
    st.info(f"💵 In USD: ${revenue_usd:.2f}")

    # ✅ Performance Rating
    if revenue_inr >= 25000:
        st.success("🔥 Excellent! High Revenue Video!")
    elif revenue_inr >= 16000:
        st.info("✅ Good! Average Revenue Video!")
    else:
        st.warning("📈 Low Revenue — Try increasing watch time!")

    # ✅ Tips
    st.subheader("💡 Tips to Improve Revenue")
    if watch_time < 30000:
        st.write("⏱️ Increase watch time — #1 revenue driver!")
    if like_rate < 0.10:
        st.write("👍 Improve like rate!")
    if video_length < 10:
        st.write("🎥 Make longer videos!")

    st.divider()
    st.caption("🤖 Powered by XGBoost | R² Score: 94.58%")