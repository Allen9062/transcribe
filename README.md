# Transcribe

Simple batch transcription with `faster-whisper`.

## Structure
- srcs/transcribe.py
- input/ (drop audio files here)
- output/ (transcripts will be written here)

## Requirements
- Python 3.11
- pip install faster-whisper
- ffmpeg available on PATH

## FFmpeg (Windows)
1) Download a static build: https://www.gyan.dev/ffmpeg/builds/
2) Extract and add the `bin` folder to PATH
3) Verify:
   - ffmpeg -version

## Usage
1) Put audio files into input/
2) Run:
   - python srcs\\transcribe.py --model base --device cpu
3) Output text files will appear in output/

## Models
- tiny / base / small / medium / large-v3 / distil-large-v3
