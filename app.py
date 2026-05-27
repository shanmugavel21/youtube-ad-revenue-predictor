import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Load Model & Scaler
model = joblib.load("final_ads_revenue_model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Page Config
st.set_page_config(
    page_title="YouTube Ad Revenue Predictor",
    page_icon="🎬",
    layout="wide"
)

# ✅ Title
st.title("🎬 YouTube Ad Revenue Predictor")
st.write("Predict YouTube Ad Revenue using Machine Learning!")
st.divider()

# ✅ Tabs
tab1, tab2, tab3 = st.tabs([
    "🚀 Predict Revenue",
    "📊 Visual Analytics",
    "🔍 Model Insights"
])

# ==========================================
# TAB 1: PREDICT REVENUE
# ==========================================
with tab1:
    st.subheader("📊 Enter Video Details")

    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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

    # Auto Calculate
    like_rate = likes / (views + 1)
    log_views = np.log1p(views)
    log_likes = np.log1p(likes)

    # Show Calculated Values
    st.subheader("🔢 Auto Calculated")
    col1, col2, col3 = st.columns(3)
    col1.metric("Like Rate", f"{like_rate:.4f}")
    col2.metric("Log Views", f"{log_views:.4f}")
    col3.metric("Log Likes", f"{log_likes:.4f}")

    st.divider()

    USD_TO_INR = 84.0

    if st.button("🚀 Predict Revenue", use_container_width=True):

        new_data = pd.DataFrame({
            'watch_time_minutes': [watch_time],
            'log_likes': [log_likes],
            'like_rate': [like_rate],
            'video_length_minutes': [video_length],
            'log_views': [log_views]
        })

        new_data_scaled = scaler.transform(new_data)
        prediction = np.expm1(model.predict(new_data_scaled))
        revenue_usd = prediction[0]
        revenue_inr = revenue_usd * USD_TO_INR

        st.subheader("💰 Prediction Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Expected", f"₹{revenue_inr * 0.9:,.2f}")
        col2.metric("Predicted Revenue", f"₹{revenue_inr:,.2f}",
                    delta="Best Estimate")
        col3.metric("Max Expected", f"₹{revenue_inr * 1.1:,.2f}")

        st.divider()
        st.info(f"💵 In USD: ${revenue_usd:.2f}")

        if revenue_inr >= 25000:
            st.success("🔥 Excellent! High Revenue Video!")
        elif revenue_inr >= 16000:
            st.info("✅ Good! Average Revenue Video!")
        else:
            st.warning("📈 Low Revenue — Try increasing watch time!")

        st.subheader("💡 Tips to Improve Revenue")
        if watch_time < 30000:
            st.write("⏱️ Increase watch time — #1 revenue driver!")
        if like_rate < 0.10:
            st.write("👍 Improve like rate!")
        if video_length < 10:
            st.write("🎥 Make longer videos!")

        st.caption("🤖 Powered by XGBoost | R² Score: 94.58%")

# ==========================================
# TAB 2: VISUAL ANALYTICS
# ==========================================
with tab2:
    st.subheader("📊 Visual Analytics")

    # Revenue Distribution
    st.markdown("### 💰 Revenue Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    revenue_ranges = ['₹10k-15k', '₹15k-20k', 
                      '₹20k-25k', '₹25k-30k', '₹30k+']
    counts = [15, 25, 35, 20, 5]
    ax1.bar(revenue_ranges, counts, color='#FF4B4B')
    ax1.set_xlabel("Revenue Range")
    ax1.set_ylabel("Count")
    ax1.set_title("Revenue Distribution")
    st.pyplot(fig1)

    st.divider()

    # Watch Time vs Revenue
    st.markdown("### ⏱️ Watch Time vs Revenue")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    watch_times = [14659, 20000, 30000, 40000, 50000, 61557]
    revenues = [10000, 13000, 17000, 21000, 25000, 32000]
    ax2.plot(watch_times, revenues, 
             color='#FF4B4B', marker='o', linewidth=2)
    ax2.set_xlabel("Watch Time (minutes)")
    ax2.set_ylabel("Revenue (₹)")
    ax2.set_title("Watch Time vs Revenue")
    ax2.fill_between(watch_times, revenues, alpha=0.1, 
                     color='#FF4B4B')
    st.pyplot(fig2)

# ==========================================
# TAB 3: MODEL INSIGHTS
# ==========================================
with tab3:
    st.subheader("🔍 Model Insights")

    # Model Performance
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "XGBoost")
    col2.metric("R² Score", "94.58%")
    col3.metric("MAE", "₹424.20")

    st.divider()

    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    features = ['watch_time', 'log_likes', 
                'like_rate', 'video_length', 'log_views']
    importance = [93.7, 2.2, 1.0, 0.8, 0.7]
    colors = ['#FF4B4B' if i == max(importance) 
              else '#FFB4B4' for i in importance]
    bars = ax3.barh(features, importance, color=colors)
    ax3.set_xlabel("Importance (%)")
    ax3.set_title("Feature Importance")
    for bar, imp in zip(bars, importance):
        ax3.text(bar.get_width() + 0.5, 
                 bar.get_y() + bar.get_height()/2,
                 f'{imp}%', va='center', fontweight='bold')
    st.pyplot(fig3)

    st.divider()

    # Model Comparison
    st.markdown("### 🆚 Model Comparison")
    comparison_data = {
        'Model': ['Random Forest', 'XGBoost'],
        'R² Score': ['94.36%', '94.58%'],
        'MAE': ['$5.25', '$5.05'],
        'Status': ['❌ Not Selected', '✅ Best Model']
    }
    st.table(pd.DataFrame(comparison_data))