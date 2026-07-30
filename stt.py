import json
import os
from torch import device
import whisper

model = whisper.load_model("turbo", device = "cuda")

for file in os.listdir("audios"):       
    result = model.transcribe(f"audios/{file}")
        
    with open(f"transcripts/{file.split('.')[0]}.txt", "w") as f:
        f.write(result)

    chunks = []
    for segment in result["segments"]:
            chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    with open(f"jsons/{file.split('.')[0]}.json", "w") as f:
            json.dump(chunks_with_metadata,f)

    