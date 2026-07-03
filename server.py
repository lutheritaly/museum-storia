import os
import json
import io
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama
from kokoro import KPipeline
import soundfile as sf

app = FastAPI(redirect_slashes=False)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website (like Netlify) to talk to Luther
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, GET, everything
    allow_headers=["*"],  # Allows all custom browser headers
)

# Enable CORS so the phone browser at museo.scattiearte.it can securely talk to Luther
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://storia.scattiearte.it"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the local offline artifact registry we built earlier
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "artifacts.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    artifact_db = json.load(f)["artifacts"]

# Initialize the Kokoro TTS pipeline locally in RAM
# 'a' stands for American English; swap to 'i' if using Italian voices natively later
tts_pipeline = KPipeline(lang_code='a')

class TourInteraction(BaseModel):
    beacon_id: str
    user_input: str

def get_artifact_by_beacon(beacon_id: str):
    for artifact in artifact_db:
        if artifact["beacon_id"] == beacon_id:
            return artifact
    return None

async def audio_stream_generator(system_prompt: str, user_message: str):
    """
    Executes a true async streaming pipeline:
    Ollama (Llama 3.2) Streams Text -> Buffer -> Kokoro Voices Sentence -> Streams Audio Bytes
    """
    try:
        # Request a streaming response from the local Ollama instance
        response_stream = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            stream=True
        )

        sentence_buffer = ""
        
        for chunk in response_stream:
            text_fragment = chunk['message']['content']
            sentence_buffer += text_fragment
            
            # As soon as a full sentence structure completes, voice it immediately
            if any(punctuation in text_fragment for punctuation in ['.', '!', '?']):
                clean_sentence = sentence_buffer.strip()
                if clean_sentence:
                    # Kokoro synthesizes the raw audio array
                    generator = tts_pipeline(clean_sentence, voice='af_bella', speed=1.0)
                    for _, _, audio in generator:
                        # Convert the raw floating-point audio array directly to standard WAV bytes
                        byte_io = io.BytesIO()
                        sf.write(byte_io, audio, 24000, format='WAV')
                        yield byte_io.getvalue()
                        await asyncio.sleep(0.001) # Yield control back to the event loop
                
                sentence_buffer = "" # Clear the buffer for the next sentence

    except Exception as e:
        print(f"Streaming pipeline failure: {e}")
        yield b""

from fastapi.responses import FileResponse
import os

from fastapi.responses import StreamingResponse
import re

from fastapi.responses import StreamingResponse
import re
import numpy as np

from fastapi.responses import StreamingResponse
import re
import numpy as np
import io
import soundfile as sf

from fastapi.responses import StreamingResponse
import re
import numpy as np

@app.post("/api/interact")
@app.post("/api/interact/")
async def interact(interaction: TourInteraction):
    artifact = get_artifact_by_beacon(interaction.beacon_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Beacon assignment missing from registry.")

    async def audio_stream_generator():
        try:
            response_stream = ollama.chat(
                model='llama3.2:3b',
                messages=[
                    {'role': 'system', 'content': artifact["system_prompt"]},
                    {'role': 'user', 'content': interaction.user_input}
                ],
                stream=True
            )

            sentence_buffer = ""
            for chunk in response_stream:
                text_token = chunk['message']['content']
                sentence_buffer += text_token
                print(text_token, end="", flush=True)

                if any(punc in text_token for punc in [".", "!", "?", "\n"]):
                    sentences = re.split(r'(?<=[.!?\n])\s+', sentence_buffer)
                    for raw_sentence in sentences[:-1]:
                        clean_sentence = raw_sentence.strip()
                        if len(clean_sentence) > 2:
                            generator = tts_pipeline(clean_sentence, voice='af_bella', speed=1.0)
                            for _, _, audio in generator:
                                if hasattr(audio, "numpy"):
                                    audio_array = audio.numpy()
                                elif hasattr(audio, "cpu"):
                                    audio_array = audio.cpu().numpy()
                                else:
                                    audio_array = np.array(audio)
                                
                                # Convert the float32 array to raw 16-bit signed PCM bytes
                                pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
                                yield pcm_bytes
                    sentence_buffer = sentences[-1]

            if sentence_buffer.strip():
                generator = tts_pipeline(sentence_buffer.strip(), voice='af_bella', speed=1.0)
                for _, _, audio in generator:
                    if hasattr(audio, "numpy"):
                        audio_array = audio.numpy()
                    elif hasattr(audio, "cpu"):
                        audio_array = audio.cpu().numpy()
                    else:
                        audio_array = np.array(audio)
                        
                    pcm_bytes = (audio_array * 32767).astype(np.int16).tobytes()
                    yield pcm_bytes

        except Exception as e:
            print(f"\n[Pipeline Failure]: {e}")
            yield b""

    # Return raw binary data stream
    return StreamingResponse(audio_stream_generator(), media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    # Fire up the ASGI server locally on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)