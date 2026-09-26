# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "kokoro>=0.9.4,<0.10",
#     "soundfile>=0.12",
#     "httpx[socks]",
#     "nbformat>=5.10,<6",
#     "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
# ]
# ///
"""Render every subtitle cue as natural English narration with the local Kokoro-82M model."""

import argparse
import json
from importlib.metadata import version

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from build_assets import AUDIO, DATA, audio_name, build_cues

MODEL = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000


def spoken_words(result, offset, fallback):
    """Group Kokoro's per-token timestamps into words, keeping trailing punctuation and spacing."""
    words, pending, start = [], "", None
    for token in result.tokens:
        if start is None and token.start_ts is not None:
            start = float(token.start_ts) + offset
        pending += token.text + token.whitespace
        if token.whitespace or token is result.tokens[-1]:
            words.append([round(start if start is not None else fallback, 3), pending])
            fallback = words[-1][0]
            pending, start = "", None
    if pending:
        words.append([round(fallback, 3), pending])
    return words


def reusable(voice, speed):
    """Kokoro output is not byte-identical between runs, so keep clips whose text, voice and speed are unchanged."""
    manifest = AUDIO / "narration-manifest.json"
    if not manifest.exists():
        return {}
    previous = json.loads(manifest.read_text())
    if previous.get("voice") != voice or previous.get("speed") != speed or "words" not in previous["clips"][0]:
        return {}
    return {clip["text"]: clip for clip in previous["clips"]}


def render(cues, voice, speed):
    pipeline = None
    kept = reusable(voice, speed)
    AUDIO.mkdir(parents=True, exist_ok=True)
    clips = []
    for number, cue in enumerate(cues, 1):
        unchanged = kept.get(cue["text"])
        if unchanged and unchanged["file"] == audio_name(number) and (AUDIO / audio_name(number)).exists():
            clips.append(unchanged)
            print(f"{number:02d} {unchanged['seconds']:5.2f}s  kept     {cue['text'][:55]}")
            continue
        if pipeline is None:
            pipeline = KPipeline(lang_code="a", repo_id=MODEL)
        chunks, words, offset = [], [], 0.0
        for result in pipeline(cue["text"], voice=voice, speed=speed):
            words += spoken_words(result, offset, words[-1][0] if words else 0.0)
            chunks.append(result.audio.numpy())
            offset += len(chunks[-1]) / SAMPLE_RATE
        audio = np.concatenate(chunks)
        spoken = "".join(word[1] for word in words).strip()
        if spoken != cue["text"]:
            raise ValueError(f"Cue {number} word timings do not reproduce the subtitle:\n{spoken!r}\n{cue['text']!r}")
        sf.write(AUDIO / audio_name(number), audio, SAMPLE_RATE, format="MP3")
        seconds = round(len(audio) / SAMPLE_RATE, 2)
        clips.append({"file": audio_name(number), "text": cue["text"], "seconds": seconds,
                      "slot_seconds": cue["end"] - cue["start"], "words": words})
        print(f"{number:02d} {seconds:5.2f}s {len(words):3d} words  {cue['text'][:55]}")
    for stale in set(AUDIO.glob("episode-02-cue-*.mp3")) - {AUDIO / clip["file"] for clip in clips}:
        stale.unlink()
    manifest = {"model": MODEL, "kokoro_version": version("kokoro"), "voice": voice, "speed": speed, "clips": clips}
    (AUDIO / "narration-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default="af_heart", help="Kokoro American English voice, e.g. af_heart, af_bella, am_michael")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    cues = build_cues(json.loads((DATA / "results.json").read_text()))
    clips = render(cues, args.voice, args.speed)
    overruns = [(i, c["seconds"]) for i, c in enumerate(clips, 1) if c["seconds"] > c["slot_seconds"]]
    total = sum(c["seconds"] for c in clips)
    print(f"Wrote {len(clips)} clips, {total:.1f}s of narration. Cues longer than their slot (the lab holds the clock): {overruns}")


if __name__ == "__main__":
    main()
