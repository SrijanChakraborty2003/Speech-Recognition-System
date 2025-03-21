import streamlit as st
import whisper
import tempfile
import os

# Load Whisper model with caching
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("Speech-to-Text Transcription")

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name  # Store temp file path

    st.write("Transcribing...")

    with st.spinner("Processing audio... Please wait."):
        try:
            result = model.transcribe(temp_audio_path)
            transcription_text = result["text"]
        finally:
            os.remove(temp_audio_path)  # Always remove the temp file

    st.success("Transcription Completed!")
    st.write("**Transcription:**")
    st.write(transcription_text)
