import streamlit as st
import whisper
import tempfile
@st.cache_resource
def load_model():
    return whisper.load_model("base")
model = load_model()
st.title("Speech-to-Text Transcription")
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name
    st.write("Transcribing...")
    result = model.transcribe(temp_audio_path)
    st.success("Transcription Completed!")
    st.write("**Transcription:**")
    st.write(result["text"])