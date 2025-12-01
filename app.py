import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, pipeline

# --- Page Configuration ---
st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")
st.title("🧠 Intelligent Customer Feedback Analysis System")

# --- Load TensorFlow model + tokenizer ---
model_path = r"sentiment_model"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path, from_pt=False)
    summarizer = pipeline("summarization", model="t5-small")
    return tokenizer, model, summarizer

# Load models
with st.spinner("Loading models..."):
    tokenizer, model, summarizer = load_model()

st.success("✅ Models loaded successfully!")

# --- Prediction function ---
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    
    # Get confidence score
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    confidence = probabilities[predicted_class]
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    
    return {
        "label": label_map[predicted_class],
        "score": float(confidence)
    }

# --- Summarization function ---
def summarize_text(text):
    try:
        summary = summarizer(str(text), max_length=40, min_length=10, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# --- Input Section ---
st.subheader("📝 Enter Customer Feedback")

feedback_text = st.text_area(
    "Type or paste customer feedback below:",
    height=150,
    placeholder="Example: This product is amazing! I've been using it for a month now and it has exceeded all my expectations. The quality is outstanding and the customer service was very helpful when I had questions."
)

# --- Analyze Button ---
if st.button("🔍 Analyze Feedback", type="primary"):
    if feedback_text.strip():
        with st.spinner("Analyzing feedback..."):
            # Get sentiment
            sentiment_result = predict_sentiment(feedback_text)
            
            # Get summary
            summary = summarize_text(feedback_text)
        
        # --- Display Results ---
        st.success("✅ Analysis Completed!")
        
        # Create three columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Original Text:")
            st.info(feedback_text)
        
        with col2:
            st.markdown("### 📊 Sentiment Analysis:")
            
            # Color-coded sentiment display
            if sentiment_result['label'] == "POSITIVE":
                st.markdown(f"**Label:** :green[{sentiment_result['label']}]")
                st.markdown(f"**Confidence:** :green[{sentiment_result['score']:.4f} ({sentiment_result['score']*100:.2f}%)]")
            else:
                st.markdown(f"**Label:** :red[{sentiment_result['label']}]")
                st.markdown(f"**Confidence:** :red[{sentiment_result['score']:.4f} ({sentiment_result['score']*100:.2f}%)]")
        
        # Summary section (full width)
        st.markdown("### ✨ Summary:")
        st.success(summary)
        
        # --- Visual Sentiment Indicator ---
        st.markdown("### 📈 Confidence Visualization")
        st.progress(sentiment_result['score'])
        
    else:
        st.warning("⚠️ Please enter some feedback text to analyze.")


