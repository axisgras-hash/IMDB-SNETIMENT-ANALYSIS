# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model
# import streamlit as st
# import time

# # Let call word Index
# word_index = imdb.get_word_index()
# reverse_word_index = {value:key for key,value in word_index.items()}
# max_len = 500
# model = load_model('simple_rnn_imbd.keras')

# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])


# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word,2) + 3 for word in words]  # numerical array
#     padded_review = sequence.pad_sequences([encoded_review],maxlen=max_len)
#     return padded_review

# st.title('IMDB Movies Review Sentiment Analysis')
# st.write('Enter a movie review to classifiy it as Positive or Negative.')

# user_input = st.text_area('Movie Review: ')

# if st.button('Classify🔍'):
#     processed_input = preprocess_text(user_input)

#     prediction = model.predict(processed_input)[0][0]
#     sentiment = 'Positive' if prediction > 0.5 else 'Negative'

#     with st.spinner('Wait for it...'):
#         time.sleep(3)

#     st.success(f'Sentiment: {sentiment}')
#     st.info(f'Prediction Score: {prediction}')

# else:
#     st.warning('Please enter a movie review!!')
    
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Load Data & Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("simple_rnn_imbd.keras")

model = load_trained_model()

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
max_len = 500


# -------------------------------
# Helper Functions
# -------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review


def predict_sentiment(text):
    processed_input = preprocess_text(text)
    prediction = model.predict(processed_input)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    return sentiment, prediction


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ App Settings")

st.sidebar.markdown("### About Model")
st.sidebar.info(
    """
    - Model: Simple RNN  
    - Dataset: IMDB Movie Reviews  
    - Max Review Length: 500 words  
    """
)

threshold = st.sidebar.slider(
    "Sentiment Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.write(
    """
    - Write full sentences  
    - Avoid very short reviews  
    - Try both positive & negative samples  
    """
)

# -------------------------------
# Main UI
# -------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown(
    "Enter a movie review below and let the AI predict whether it's **Positive** or **Negative**."
)

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "✍️ Movie Review",
        height=200,
        placeholder="Example: This movie was absolutely fantastic with brilliant acting..."
    )

with col2:
    st.markdown("### Quick Examples")
    if st.button("Positive Example"):
        user_input = "The movie was amazing, I loved the performances and storyline!"
    if st.button("Negative Example"):
        user_input = "This was the worst movie I have ever seen. Completely disappointing."

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review first!")
    else:
        with st.spinner("Analyzing review..."):
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            sentiment, prediction = predict_sentiment(user_input)

        st.markdown("## 🧠 Prediction Result")

        if prediction > threshold:
            st.success(f"### ✅ Sentiment: {sentiment}")
        else:
            st.error(f"### ❌ Sentiment: {sentiment}")

        st.metric("Confidence Score", f"{prediction:.4f}")

# -------------------------------
# Footer Section
# -------------------------------
st.markdown("---")
st.markdown(
    """
    ### 📌 About This Project
    
    This web application uses a **Recurrent Neural Network (RNN)** trained on the 
    IMDB movie review dataset to classify sentiments.
    
    **Tech Stack Used:**
    - TensorFlow / Keras  
    - Streamlit  
    - NumPy  
    
    Built for experimentation, learning, and deployment practice.
    
    ---
    👨‍💻 Developed with Streamlit | Deep Learning Sentiment Analysis Demo
    """
)

