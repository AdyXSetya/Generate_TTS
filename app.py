import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.server.server import Server
import json
from google import genai
from google.genai import types
import base64
import mimetypes
import struct
import os

# === Fungsi utilitas parsing MIME dan WAV ===
def parse_audio_mime_type(mime_type: str):
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

# === Endpoint handler ===
@st.server_request_handler("/generate")
def handle_generate(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get("text")
            voice = data.get("voice", "Zephyr")

            if not text:
                return {"error": "Missing 'text' field"}, 400

            # Inisialisasi client Gemini
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                return {"error": "API Key not set"}, 500

            client = genai.Client(api_key=api_key)

            model = "gemini-2.5-flash-preview-tts"
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
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

                    audio_b64 = base64.b64encode(data_buffer).decode("utf-8")
                    return {
                        "audio_base64": audio_b64,
                        "file_extension": file_extension,
                        "mime_type": inline_data.mime_type,
                    }

            return {"error": "Failed to generate audio"}, 500

        except Exception as e:
            return {"error": str(e)}, 500

    return {"error": "Method not allowed"}, 405

# === UI Streamlit (opsional) ===
st.title("🔊 TTS API on Streamlit")
st.markdown("Endpoint: `https://your-app-url.streamlit.app/generate`") 
st.markdown("Kirim POST request dari Postman/JavaScript ke endpoint tersebut.")
