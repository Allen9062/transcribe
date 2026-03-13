from pathlib import Path
import argparse
import os
import time


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".webm"}


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _default_compute_type(device: str) -> str:
    return "int8" if device == "cpu" else "float16"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="base",
        help="tiny/base/small/medium/large-v3/distil-large-v3",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Defaults to int8 on cpu and float16 on cuda.",
    )
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    root = Path(__file__).resolve().parents[1]
    input_dir = root / "input"
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Input folder not found: {input_dir}")
        return

    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"No audio files found in: {input_dir}")
        return

    compute_type = args.compute_type or _default_compute_type(args.device)
    model_kwargs = {
        "device": args.device,
        "compute_type": compute_type,
    }
    if args.device == "cpu":
        model_kwargs["cpu_threads"] = os.cpu_count() or 1

    print(f"Found {len(audio_files)} file(s) in: {input_dir}")
    t0 = time.perf_counter()
    model = WhisperModel(args.model, **model_kwargs)
    t1 = time.perf_counter()
    print(
        f"Model loaded: {args.model} on {args.device} "
        f"({compute_type}, {t1 - t0:.2f}s)"
    )

    for audio_path in audio_files:
        file_start = time.perf_counter()
        size_bytes = audio_path.stat().st_size
        print(f"Transcribing: {audio_path.name} ({_format_size(size_bytes)})")

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        out_path = output_dir / f"{audio_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")

        file_end = time.perf_counter()
        duration_sec = float(getattr(info, "duration", 0.0) or 0.0)
        language = getattr(info, "language", "unknown")
        language_prob = getattr(info, "language_probability", None)
        char_count = len(text)
        word_count = len(text.split()) if text else 0

        print(
            "Duration: {} | Transcribe: {:.2f}s | Total: {:.2f}s".format(
                _format_duration(duration_sec),
                file_end - file_start,
                file_end - file_start,
            )
        )
        if language_prob is None:
            print(f"Language: {language}")
        else:
            print(f"Language: {language} ({float(language_prob):.2%})")
        print(f"Text stats: {char_count} chars, {word_count} words")
        print(f"Wrote: {out_path}")

    t2 = time.perf_counter()
    print(f"Total time: {t2 - t0:.2f}s")


if __name__ == "__main__":
    main()
