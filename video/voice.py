"""Turkish voiceover for the demo video, one clip per scene -> public/vo/NN.mp3 + public/vo/timing.json.

Uses ElevenLabs when ELEVENLABS_API_KEY is set in the environment (voice id from ELEVENLABS_VOICE_ID,
model eleven_multilingual_v2); otherwise the free Microsoft Edge neural voice tr-TR-AhmetNeural.
Only the script text below is sent to the speech service.
"""
import asyncio
import json
import os
import subprocess
from pathlib import Path

LINES = [
    ("Biz Ankara'dan Asteria Roket takımıyız. Mars harita görevini seçtik.", "We are team Asteria Roket from Ankara. We chose the Martian Map challenge."),
    ("Projemiz Mars Pathfinder: Mars'ta yürüyecek ilk astronot için bir rota haritası.", "Our project, Mars Pathfinder: a route map for the first astronaut to walk on Mars."),
    ("NASA verisiyle dağları, kraterleri ve tozu görüyor; bilim duraklarını ve dönüş süresini planlıyoruz.", "With NASA data we see mountains, craters and dust, and plan science stops and the time to get back."),
    ("Gerçek Curiosity görüntülerinde kayaları işaretleyip, rotayı aralarından geçiriyoruz.", "On real Curiosity images we mark the rocks and lead the route between them."),
    ("Hedefimiz, Mars'taki ilk adımların daha bilinçli atılması.", "Our goal: better-informed first steps on Mars."),
]
OUT = Path(__file__).parent / "public/vo"
FFPROBE_FF = None


def trim(path):
    """Cut leading/trailing silence (speech untouched)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe(); tmp = path.with_suffix(".tmp.mp3")
    filt = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse"
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(path), "-af", filt, "-b:a", "192k", str(tmp)], check=True)
    tmp.replace(path)


def duration(path):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True).stderr
    t = out.split("Duration: ")[1].split(",")[0]
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


async def edge(text, path):
    import edge_tts
    await edge_tts.Communicate(text, "tr-TR-AhmetNeural", rate="+4%").save(str(path))


def eleven(text, path, previous="", following=""):
    import urllib.request
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "X2nuXhfEmBYjbVTFw3li")
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}", method="POST",
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json", "Accept": "audio/mpeg"},
                                 data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2", "previous_text": previous, "next_text": following,
                                                  "voice_settings": {"stability": .55, "similarity_boost": .8, "style": .15, "use_speaker_boost": True}}).encode())
    path.write_bytes(urllib.request.urlopen(req, timeout=120).read())


def load_dotenv():
    """Read ELEVENLABS_* from video/.env or the project .env (both git-ignored) if not already set."""
    for path in (Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"):
        if path.is_file():
            for line in path.read_text("utf-8").splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip().startswith("ELEVENLABS_") and not os.environ.get(key.strip()):
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")


def split_take(take):
    """Split one web-generated take (all lines, paragraph pauses) at its longest silences into NN.mp3. No tempo change."""
    import re
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    log = subprocess.run([ff, "-i", str(take), "-af", "silencedetect=noise=-38dB:d=0.25", "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    gaps = sorted(zip(starts, ends), key=lambda g: g[0] - g[1])[:len(LINES) - 1]
    cuts = [0.0] + sorted((a + b) / 2 for a, b in gaps) + [duration(take)]
    if len(cuts) != len(LINES) + 1:
        raise SystemExit(f"found {len(cuts) - 1} segments, expected {len(LINES)}; put a paragraph break between lines")
    for i in range(len(LINES)):
        path = OUT / f"{i:02d}.mp3"
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(take), "-ss", f"{cuts[i]:.3f}", "-to", f"{cuts[i + 1]:.3f}", "-b:a", "192k", str(path)], check=True)
        trim(path)


def main():
    import sys
    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    take = OUT / "doruk.mp3"
    if take.is_file() and "--edge" not in sys.argv:
        provider = "elevenlabs Doruk (web take, split at pauses)"
        split_take(take)
    elif "--edge" in sys.argv or not os.environ.get("ELEVENLABS_API_KEY"):
        provider = "edge-tts tr-TR-AhmetNeural"
    else:
        provider = "elevenlabs"
    timing = []
    for i, (tr, en) in enumerate(LINES):
        path = OUT / f"{i:02d}.mp3"
        if provider.startswith("elevenlabs Doruk"):
            pass
        elif provider == "elevenlabs":
            eleven(tr, path, " ".join(x[0] for x in LINES[:i]), " ".join(x[0] for x in LINES[i + 1:]))
        else:
            asyncio.run(edge(tr, path))
        trim(path)
        timing.append({"file": f"vo/{i:02d}.mp3", "seconds": round(duration(path), 2), "tr": tr, "en": en})
        print(i, timing[-1]["seconds"], tr)
    (OUT / "timing.json").write_text(json.dumps({"provider": provider, "lines": timing}, ensure_ascii=False, indent=1), "utf-8")
    total = sum(t["seconds"] for t in timing)
    print("total", round(total, 2))
    if total > 26:
        raise SystemExit("voice longer than 26 s: shorten the text rather than speeding it up")


if __name__ == "__main__":
    main()
