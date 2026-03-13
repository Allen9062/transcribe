# Transcribe

Batch transcription with `faster-whisper`.

## Project Structure
- `srcs/transcribe.py`: batch transcription script
- `input/`: drop source audio or video files here
- `output/`: generated `.txt` transcripts
- `models/`: optional local model cache for Windows or offline usage

## Requirements
- Python 3.11
- `pip install faster-whisper`
- `ffmpeg` available on `PATH`
- NVIDIA GPU + CUDA runtime if you want to run with `--device cuda`

## Supported Input Formats
- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.ogg`
- `.mp4`
- `.webm`

## Script Flow
`srcs/transcribe.py` does the following:

1. Resolve the project root from the script location.
2. Read files from `input/`.
3. Filter files by supported extension.
4. Load one Whisper model for the whole batch.
5. Transcribe each file with:
   - `language="zh"`
   - `beam_size=5`
   - `vad_filter=True`
   - `condition_on_previous_text=False`
6. Write one `.txt` file per input into `output/`.
7. Print model load time, file duration, transcription time, language, and text stats.

## Notes About Current Behavior
- The script expects `input/` to already exist. If the folder is missing, it exits early.
- Output filenames use the input stem, for example `sample.m4a -> output/sample.txt`.
- `--compute-type` defaults to:
  - `int8` on `cpu`
  - `float16` on `cuda`
- On CPU, the script automatically uses all available CPU threads.
- If you pass a local path to `--model`, `WhisperModel` will load from that folder instead of downloading from Hugging Face.

## FFmpeg Setup on Windows
1. Download a static build: https://www.gyan.dev/ffmpeg/builds/
2. Extract it and add the `bin` folder to `PATH`
3. Verify:

```powershell
ffmpeg -version
```

## Usage

### CPU Example
1. Put audio files into `input/`
2. Run:

```powershell
python srcs\transcribe.py --model base --device cpu
```

### GPU Example
Use a CUDA-capable GPU:

```powershell
python srcs\transcribe.py --model large-v3 --device cuda
```

## Using a Local Model Folder
This is useful on Windows if Hugging Face model download fails because symlink creation is blocked.

Download the model into `models/` first:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Systran/faster-whisper-large-v3', local_dir='models/faster-whisper-large-v3')"
```

Then run the script with the local model path:

```powershell
python srcs\transcribe.py --model models/faster-whisper-large-v3 --device cuda
```

## Available Model Names
- `tiny`
- `base`
- `small`
- `medium`
- `large-v3`
- `distil-large-v3`

You can also pass a local model directory path to `--model`.
