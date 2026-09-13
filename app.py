import streamlit as st
import speech_recognition as sr
import tempfile
import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Twitter Voice Sentiment",
    page_icon="🐦",
    layout="centered"
)

# =====================================================
# SETUP & NLTK DOWNLOADS
# =====================================================
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
    <style>
    /* Force text and labels to be white */
    label, p, div[role="radiogroup"] label p {
        color: white !important;
    }
    .stTextArea textarea {
        color: black !important;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.35), transparent 30%),
            radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.30), transparent 30%),
            linear-gradient(135deg, #0f172a, #1e1b4b, #111827);
        color: white;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 30px;
    }
    .result-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .positive {
        color: #4ade80;
        font-size: 30px;
        font-weight: 800;
    }
    .negative {
        color: #fb7185;
        font-size: 30px;
        font-weight: 800;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 40px;
        font-size: 14px;
    }
    .stTextArea textarea, .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-size: 16px !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #6c757d !important;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.markdown('<div class="main-title">🐦 Twitter Voice Sentiment</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Convert speech to text and detect Hate Speech & Sentiment using your Custom ML Model 🤖</div>', unsafe_allow_html=True)

# =====================================================
# LOAD CUSTOM MODEL & CLEANING FUNCTION
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("twitter_sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'twitter_sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in your repository.")
        st.stop()

with st.spinner("🤖 Loading Custom Twitter Model..."):
    model, vectorizer = load_artifacts()

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"&[a-z]+;", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(clean_tokens)

# =====================================================
# SENTIMENT DISPLAY FUNCTION
# =====================================================
def show_sentiment(text):
    cleaned_text = clean_tweet(text)
    vectorized_input = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(vectorized_input)[0]
    
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = probabilities[prediction] * 100

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Sentiment Result")

    if prediction == 1:
        st.markdown('<div class="negative">🚨 HATE SPEECH / NEGATIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="positive">✅ NON-HATE / POSITIVE</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Confidence Score")
    st.progress(min(int(confidence), 100))
    st.write(f"**{confidence:.2f}%**")
    
    with st.expander("🔍 See Model Interpretation (Cleaned Text)"):
        st.write(f"*{cleaned_text if cleaned_text else '[No recognizable text/words remaining]'}*")
        
    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# SPEECH-TO-TEXT FUNCTION
# =====================================================
def convert_speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text, None
    except sr.UnknownValueError:
        return None, "❌ Could not understand the audio. Try speaking clearer."
    except sr.RequestError:
        return None, "❌ Google Speech Recognition service is unavailable. Check your internet."
    except Exception as e:
        return None, f"❌ Error processing audio: {str(e)}"

# =====================================================
# OPTION CARD
# =====================================================
st.subheader("🎯 Choose Input Method")
option = st.radio(
    "Select an option:", 
    ["🎤 Talk", "📁 Upload Audio", "⌨️ Type Text"], 
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

# =====================================================
# INPUT LOGIC
# =====================================================
if option == "🎤 Talk":
    st.subheader("🎤 Record Your Voice")
    st.write("Click the microphone button below and speak naturally.")

    audio_value = st.audio_input("🎙️ Click here to record your voice")

    if audio_value is not None:
        st.success("✅ Recording completed!")
        audio_bytes = audio_value.getvalue()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            audio_path = temp_file.name

        with st.spinner("🎧 Converting speech to text..."):
            text, error = convert_speech_to_text(audio_path)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if error:
            st.error(error)
        else:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### 📝 Transcribed Text")
            st.write(f"**{text}**")
            st.markdown("</div>", unsafe_allow_html=True)
            show_sentiment(text)

elif option == "📁 Upload Audio":
    st.subheader("📁 Upload Your Audio")
    st.write("Upload a WAV audio file and analyze its sentiment.")

    uploaded_file = st.file_uploader("🎵 Choose an audio file", type=["wav"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🔍 Analyze Audio"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                audio_path = temp_file.name

            with st.spinner("🎧 Converting speech to text..."):
                text, error = convert_speech_to_text(audio_path)

            if os.path.exists(audio_path):
                os.remove(audio_path)

            if error:
                st.error(error)
            else:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Transcribed Text")
                st.write(f"**{text}**")
                st.markdown("</div>", unsafe_allow_html=True)
                show_sentiment(text)

else:
    st.subheader("⌨️ Type or Paste Text")
    st.write("Paste a tweet or sentence to analyze its sentiment instantly.")
    
    user_input = st.text_area("Tweet text:", placeholder="Type here...", label_visibility="collapsed")
    
    if st.button("🔍 Analyze Text"):
        if user_input.strip():
            show_sentiment(user_input)
        else:
            st.warning("Please enter some text to analyze.")

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    '<div class="footer">'
    '🤖 Powered by Custom TF-IDF Model • Speech Recognition • Streamlit'
    '</div>',
    unsafe_allow_html=True
)