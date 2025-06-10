import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
import os
import tempfile
import speech_recognition as sr
import torch
from groq import Groq
from TTS.api import TTS
from pydub import AudioSegment

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Voice Q&A App", layout="centered")
st.title("🎤 Voice Q&A with Groq + TTS")

uploaded_audio = st.file_uploader("Upload your question (WAV or M4A format)", type=["wav", "m4a"], key="uploader")
user_text = None

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as f:
        f.write(uploaded_audio.read())
        audio_path = f.name

    if uploaded_audio.name.endswith(".m4a"):
        wav_path = audio_path.replace(".m4a", ".wav")
        sound = AudioSegment.from_file(audio_path)
        sound.export(wav_path, format="wav")
        audio_path = wav_path

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            user_text = recognizer.recognize_whisper(audio_data, model="base")
            st.success(f"Recognized Text: {user_text}")
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

if user_text:
    st.header("Response from Groq AI")
    with st.spinner("Generating response..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_text}],
                model="llama-3-70b-8192"
            )
            answer = chat_completion.choices[0].message.content
            st.success("Answer:")
            st.write(answer)

            st.header("Listen to the Answer")

            @st.cache_resource
            def load_tts():
                return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

            tts = load_tts()
            tts_file_path = tempfile.mktemp(suffix=".wav")
            tts.tts_to_file(text=answer, file_path=tts_file_path)

            with open(tts_file_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/wav")
        except Exception as e:
            st.error(f"Groq API Error: {e}")
