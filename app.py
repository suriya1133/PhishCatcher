import streamlit as st
from ml import predict_url as ml_predict
from test import predict_url as dl_predict
from evaluate import evaluate_all
import io
import sys
import os
import warnings

warnings.filterwarnings('ignore')

def capture_prediction(func, url):
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        func(url)
    finally:
        sys.stdout = sys_stdout
    return buffer.getvalue()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🔐 Phishing URL Detector")

# ❌ Exit Button (clean way to stop the app)
if st.button("❌ Exit App"):
    st.warning("Shutting down the app... 👋")
    st.stop()  # Stops execution, safe method
    # If you want to forcefully kill the Python process (use only if necessary):
    # os._exit(0)

# --- URL Input & Model Selection ---
url_input = st.text_input("🔗 Enter a URL to analyze:", placeholder="https://example.com")
model_option = st.selectbox("🧠 Select Detection Model:", ["Machine Learning", "Deep Learning", "Both"])

# --- Show Model Accuracies ---
with st.expander("📊 View Model Performance (Evaluated Live from Test Data)"):
    with st.spinner("Evaluating models on test data..."):
        try:
            ml_results, dl_results = evaluate_all()
            st.subheader("🔍 Machine Learning Models")
            for name, metrics in ml_results.items():
                st.markdown(f"**{name}**")
                st.write({k: f"{v*100:.2f}%" for k, v in metrics.items()})

            st.subheader("🤖 Deep Learning Models")
            for name, metrics in dl_results.items():
                st.markdown(f"**{name}**")
                st.write({k: f"{v*100:.2f}%" for k, v in metrics.items()})
        except Exception as e:
            st.error(f"❌ Failed to evaluate models: {e}")

# --- URL Prediction Execution ---
if st.button("🚨 Analyze URL"):
    if not url_input:
        st.warning("Please enter a URL to analyze.")
    else:
        with st.spinner("Analyzing..."):
            if model_option == "Machine Learning":
                result = capture_prediction(ml_predict, url_input)
                st.code(result)
            elif model_option == "Deep Learning":
                result = capture_prediction(dl_predict, url_input)
                st.code(result)
            else:
                st.subheader("🔍 Machine Learning Result")
                ml_result = capture_prediction(ml_predict, url_input)
                st.code(ml_result)

                st.subheader("🤖 Deep Learning Result")
                dl_result = capture_prediction(dl_predict, url_input)
                st.code(dl_result)

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Stay safe online! 🛡️ | You Can't See Me After This")
