import streamlit as st
import base64
import mimetypes
import struct
from google import genai
from google.genai import types
from urllib.parse import unquote

# --- FUNGSI UTAMA HARUS DIDEFINISIKAN DI ATAS ---
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

# --- SAMPAI SINI ---

# Load API Key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API Key not found. Please set GEMINI_API_KEY in secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# Get query parameters from URL
query_params = st.experimental_get_query_params()
text_input = query_params.get("text", [""])[0]  # Default to empty string if not provided
voice_name = query_params.get("voice", ["Zephyr"])[0]  # Default to "Zephyr" if not provided

# Decode text input (in case it's URL-encoded)
text_input = unquote(text_input)

# Check if both required parameters are present
if text_input and voice_name:
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

            b64_audio = base64.b64encode(data_buffer).decode()
            url_download = f"data:audio/{file_extension[1:]};base64,{b64_audio}"

            result = {
                "audio": b64_audio,
                "mime_type": inline_data.mime_type,
                "file_extension": file_extension,
                "url_download": url_download,
            }

            st.json(result)
            break
else:
    st.json({
        "error": "Missing required parameters",
        "required": ["text", "voice"],
        "example_url": "?text=Hello%20World&voice=Zephyr"
    })
