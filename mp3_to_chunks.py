import json
import os
from torch import device
import whisper

os.makedirs("transcripts", exist_ok=True)
os.makedirs("jsons", exist_ok=True)

model = whisper.load_model("small", device = "cuda")

for file in os.listdir("audios"):       
    result = model.transcribe(f"audios/{file}",fp16 = False, language = "en")
        
    with open(f"transcripts/{file.split('.')[0]}.txt", "w") as f:
        f.write(result["text"])

    chunks = []
    for segment in result["segments"]:
            chunks.append({"chunk_id": segment["id"], "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
    chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

    with open(f"jsons/{file.split('.')[0]}.json", "w") as f:
            json.dump(chunks_with_metadata,f)
