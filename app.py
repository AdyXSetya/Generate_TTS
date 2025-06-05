import streamlit as st
import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types

st.set_page_config(page_title="Gemini TTS", layout="centered")
st.title("🔊 Gemini Text to Speech Generator")

# Load API Key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found. Please set GEMINI_API_KEY in secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# Input Form
text_input = st.text_area("Enter your text (supports SSML):", value="""<speak>ngerasa gaya kamu gitu-gitu aja ; <break time="150ms"/> tas rantai ini nge-boost look kamu jadi classy dalam sekejap</speak>
<speak><emphasis level="strong">warna</emphasis> banyak <prosody rate="90%" pitch="+2%">pilih yang paling kamu</prosody></speak>""")

voice_name = st.selectbox("Choose voice:", [
    "Zephyr", "Echo", "Fable", "Onyx", "Nova", "Shimmer"
])

if st.button("Generate Audio"):
    with st.spinner("Generating audio..."):

        model = "gemini-2.5-flash-preview-tts"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text_input),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        )

        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                inline_data = part.inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                if file_extension is None:
                    file_extension = ".wav"
                    data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)

                # Show download button
                b64_audio = base64.b64encode(data_buffer).decode()
                href = f'<a href="data:audio/{file_extension};base64,{b64_audio}" download="output{file_extension}">Download Audio File</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.audio(data_buffer, format=f"audio/{file_extension.replace('.', '')}")
            else:
                st.warning(chunk.text)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except Exception:
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except Exception:
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}
