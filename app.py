import streamlit as st
import os
import whisper
import requests
import torchaudio
import soundfile as sf
import io
import tempfile
from transformers import pipeline

st.set_page_config(page_title="Voice Q&A App", page_icon="🔊")
st.title("🔊 Voice-based Question Answering App")

st.markdown("""
Upload a voice recording (.wav or .m4a), and this app will:
1. Transcribe your question using Whisper
2. Generate an answer using the Groq API (LLaMA 3)
3. Convert the response back to speech
""")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

uploaded_file = st.file_uploader("Upload your voice question (.wav or .m4a)", type=["wav", "m4a"])

if uploaded_file is not None:
    if uploaded_file.name.endswith((".wav", ".m4a")):
        with st.spinner("Transcribing with Whisper..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            model = load_whisper_model()
            result = model.transcribe(tmp_file_path)
            question = result["text"]

        st.markdown("**Transcribed Question:**")
        st.success(question)

        with st.spinner("Generating answer via Groq..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            answer = response.json()["choices"][0]["message"]["content"]
            st.markdown("**Answer:**")
            st.info(answer)

        with st.spinner("Converting answer to speech..."):
            tts = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
            tts_output = tts(answer)

            audio_array = tts_output["waveform"].numpy()
            sample_rate = tts_output["sampling_rate"]

            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array.T, sample_rate, format="WAV")
            st.audio(audio_buffer.getvalue(), format="audio/wav")
    else:
        st.warning("Please upload a .wav or .m4a file.")
