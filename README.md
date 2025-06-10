---
title: TTS
emoji: 🐨
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
license: mit
---

# 🐨 Voice Q&A with Groq + TTS

This Streamlit app lets you:

- Upload a voice file (`.wav` or `.m4a`)
- Automatically transcribe it using Whisper via SpeechRecognition
- Generate a response using Groq’s LLaMA 3 (70B)
- Convert the response to speech using a TTS model (`Tacotron2-DDC`)