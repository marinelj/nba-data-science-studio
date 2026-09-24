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


def render(cues, voice, speed):
    pipeline = KPipeline(lang_code="a", repo_id=MODEL)
    AUDIO.mkdir(parents=True, exist_ok=True)
    for old in AUDIO.glob("episode-02-cue-*.mp3"):
        old.unlink()
    clips = []
    for number, cue in enumerate(cues, 1):
        audio = np.concatenate([result.audio.numpy() for result in pipeline(cue["text"], voice=voice, speed=speed)])
        sf.write(AUDIO / audio_name(number), audio, SAMPLE_RATE, format="MP3")
        seconds = round(len(audio) / SAMPLE_RATE, 2)
        clips.append({"file": audio_name(number), "text": cue["text"], "seconds": seconds,
                      "slot_seconds": cue["end"] - cue["start"]})
        print(f"{number:02d} {seconds:5.2f}s  {cue['text'][:70]}")
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
