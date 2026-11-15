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
    ("Mars'ta bir astronot, nereye güvenle yürür?", "Where can an astronaut safely walk on Mars?"),
    ("Yörünge haritası iki yüz metreyi görür; tehlike, yirmi beş santimlik bir kaya.", "Orbital maps see 200 m. The hazard is a 25 cm rock."),
    ("Pathfinder; bilim duraklarını, dönüşü, tozu ve jeolojiyi, NASA verisiyle tek rotada planlar.", "Pathfinder plans science stops, the return, dust and geology in one route, from NASA open data."),
    ("Gerçek Curiosity karelerinde model kayayı işaretler, stereo doğrular, rota dolaşır.", "On real Curiosity frames, the model flags rocks, stereo confirms them, the route goes around."),
    ("Eski model kaya bulamıyordu; yenisi, görmediği elli dokuz kayanın otuz birini buluyor.", "The old model found none. The new one finds 31 of 59 rocks in scenes it never saw."),
    ("Açık NASA verisiyle, Mars'ta daha güvenli adımlar.", "Built on open NASA data. For safer steps on Mars."),
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
    await edge_tts.Communicate(text, "tr-TR-AhmetNeural", rate="+16%", pitch="-2Hz").save(str(path))


def eleven(text, path):
    import urllib.request
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}", method="POST",
                                 headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json", "Accept": "audio/mpeg"},
                                 data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2",
                                                  "voice_settings": {"stability": .45, "similarity_boost": .8, "style": .35}}).encode())
    path.write_bytes(urllib.request.urlopen(req, timeout=120).read())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provider = "elevenlabs" if os.environ.get("ELEVENLABS_API_KEY") else "edge-tts tr-TR-AhmetNeural"
    timing = []
    for i, (tr, en) in enumerate(LINES):
        path = OUT / f"{i:02d}.mp3"
        if provider == "elevenlabs":
            eleven(tr, path)
        else:
            asyncio.run(edge(tr, path))
        trim(path)
        timing.append({"file": f"vo/{i:02d}.mp3", "seconds": round(duration(path), 2), "tr": tr, "en": en})
        print(i, timing[-1]["seconds"], tr)
    (OUT / "timing.json").write_text(json.dumps({"provider": provider, "lines": timing}, ensure_ascii=False, indent=1), "utf-8")
    print("total", round(sum(t["seconds"] for t in timing), 2))


if __name__ == "__main__":
    main()
