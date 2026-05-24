import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ✅ Page Config
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="🎬",
    layout="centered"
)

# ✅ Load Model
model = joblib.load("final_ads_revenue_model.pkl")

# ✅ Title
st.title("🎬 YouTube Ad Revenue Predictor")
st.write("Enter your video details to predict ad revenue!")
st.divider()

# ✅ Input Fields (Realistic Range!)
st.subheader("📊 Video Details")

watch_time = st.slider(
    "⏱️ Watch Time (minutes)",
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

# ✅ Predict Button
if st.button("🚀 Predict Revenue", use_container_width=True):

    new_data = pd.DataFrame({
        'watch_time_minutes': [watch_time],
        'log_likes': [log_likes],
        'like_rate': [like_rate],
        'video_length_minutes': [video_length],
        'log_views': [log_views]
    })

    prediction = np.expm1(model.predict(new_data))
    revenue = prediction[0]

    st.divider()
    st.subheader("💰 Prediction Result")

    # Revenue Range Display
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Expected", f"${revenue * 0.9:.2f}")
    col2.metric("Predicted Revenue", f"${revenue:.2f}", delta="Best Estimate")
    col3.metric("Max Expected", f"${revenue * 1.1:.2f}")

    st.divider()

    # ✅ Performance Rating
    if revenue >= 300:
        st.success("🔥 Excellent! High Revenue Video!")
    elif revenue >= 200:
        st.info("✅ Good! Average Revenue Video!")
    else:
        st.warning("📈 Low Revenue — Try increasing watch time!")

    # ✅ Tips
    st.subheader("💡 Tips to Improve Revenue")
    if watch_time < 30000:
        st.write("⏱️ Increase watch time — it's the #1 revenue driver!")
    if like_rate < 0.10:
        st.write("👍 Improve like rate — engage your audience better!")
    if video_length < 10:
        st.write("🎥 Make longer videos — more ad placements!")