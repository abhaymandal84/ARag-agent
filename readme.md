# RAG AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about your own video content (lectures, courses, tutorials). It transcribes your videos, embeds the transcripts, and at query time retrieves the most relevant passages and feeds them to an LLM, so answers are grounded in your material and can point back to the exact video and timestamp.

## How It Works

```
videos/  ->  mp3  ->  transcript JSON  ->  embeddings (joblib)  ->  retrieval + LLM answer
        ffmpeg     Whisper            preprocess_json          prompt generation
```

| Stage | Script | Tool | Output |
|-------|--------|------|--------|
| 1. Collect | (manual) | - | Video files in `videos/` |
| 2. Video to audio | `video_to_mp3` | ffmpeg | One `.mp3` per video |
| 3. Audio to text | `mp3_to_json` | Whisper | One `.json` per audio file (timestamped segments) |
| 4. Text to vectors | `preprocess_json` | Embedding model, pandas, joblib | A DataFrame with embeddings saved as a `.joblib` file |
| 5. Ask questions | Query script | Embedding model + LLM | Answer grounded in your videos |

## Prerequisites

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on your `PATH`
- [OpenAI Whisper](https://github.com/openai/whisper) (or a compatible implementation)
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `requests` (or the SDK of your chosen LLM provider)
- An embedding model and an LLM (local or hosted)
- Enough disk space: audio and transcripts are small, but keep the original videos accessible

Verify ffmpeg is installed:

```bash
ffmpeg -version
```

## Project Structure

```
.
├── videos/              # Step 1: your raw video files
├── audios/              # Step 2 output: extracted mp3 files
├── jsons/               # Step 3 output: Whisper transcripts
├── video_to_mp3         # Step 2 script
├── mp3_to_json          # Step 3 script
├── preprocess_json      # Step 4 script
├── embeddings.joblib    # Step 4 output: DataFrame with embeddings
└── README.md
```

Folder names may differ in your setup. Adjust the paths inside the scripts if so.

## Usage

### Step 1: Collect your videos

Move all your video files into the `videos/` folder.

- Use clear, consistent file names. The names or numbering are what the assistant will cite when it points to a source.
- Supported formats are anything ffmpeg can read (mp4, mkv, mov, webm, etc.).

### Step 2: Convert videos to mp3

Run `video_to_mp3`. It uses ffmpeg to extract the audio track from every video in `videos/` and saves it as an mp3.

```bash
python video_to_mp3
```

Check that every video has a matching mp3 before moving on.

### Step 3: Convert mp3 to JSON

Run `mp3_to_json`. It uses Whisper to transcribe each mp3 and writes a JSON file containing the transcript split into timestamped segments (start time, end time, text).

```bash
python mp3_to_json
```

Notes:
- Transcription is the slowest step. A GPU speeds it up considerably.
- Larger Whisper models are more accurate but slower. Pick based on audio quality and language.
- If your videos are not in English, set the language explicitly so Whisper does not mis-detect it.

### Step 4: Convert JSON to vectors

Run `preprocess_json`. It reads all the JSON transcripts, converts each text chunk into an embedding vector, stores everything (video name, timestamps, text, embedding) in a pandas DataFrame, and saves it as a joblib pickle.

```bash
python preprocess_json
```

This only needs to be re-run when you add or change videos.

### Step 5: Prompt generation and querying the LLM

1. Load the joblib file into memory (this is your vector store).
2. Embed the user's question using the **same embedding model** used in Step 4.
3. Compute cosine similarity between the question embedding and every stored chunk.
4. Select the top-k most similar chunks.
5. Build a prompt containing the question and those chunks (with video name and timestamps).
6. Send the prompt to the LLM and return its answer.

Example prompt structure:

```
You are a teaching assistant. Answer the question using ONLY the transcript
excerpts below. Each excerpt includes the video name and timestamps. In your
answer, tell the user which video and timestamp to watch for more detail.
If the excerpts do not contain the answer, say so.

Excerpts:
{retrieved_chunks}

Question: {user_question}
```

## Configuration

Decide and keep consistent:

- **Embedding model:** must be identical in Steps 4 and 5, or similarity scores are meaningless.
- **LLM:** any chat-capable model, hosted or local.
- **Top-k:** how many chunks to retrieve (commonly 3 to 5). Higher gives more context but more noise and cost.
- **Chunk size:** how many transcript segments are merged into one chunk before embedding. Smaller chunks give precise timestamps, larger chunks give more context.

## Tips for Better Results

- Use clean audio. Transcription errors propagate to every later step.
- Merge very short Whisper segments into larger chunks so each embedding carries enough meaning.
- Instruct the LLM to answer only from the provided excerpts, to reduce hallucination.
- Always return video name and timestamps with the answer so users can verify it.

## Troubleshooting

| Problem | Likely cause / fix |
|---------|--------------------|
| `ffmpeg: command not found` | ffmpeg is not installed or not on `PATH` |
| Whisper is very slow | Use a smaller model or run on a GPU |
| Irrelevant answers | Check that the same embedding model is used in Steps 4 and 5; try adjusting top-k or chunk size |
| Answers ignore your videos | Tighten the prompt so the LLM answers only from the excerpts |
| New videos not found | Re-run Steps 2 to 4 after adding videos |
| Garbled transcripts | Set the correct language in Whisper, or use a larger model |

## Limitations

- Answer quality depends on transcript accuracy and retrieval quality.
- The assistant only knows what is in your videos.
- Content that is only shown visually (slides, code on screen) is not captured, since only the audio is transcribed.
