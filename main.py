import os, json, random, asyncio, csv, re, datetime
import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (AudioFileClip, VideoFileClip, ImageClip, ColorClip,
                            CompositeVideoClip, concatenate_videoclips, vfx)
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import wave
from moviepy.editor import CompositeAudioClip
import moviepy.audio.fx.all as afx


def make_music(dur, path="bg.wav"):
    sr = 44100
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    chords = [[220.00, 261.63, 329.63], [174.61, 220.00, 261.63],
              [130.81, 164.81, 196.00], [196.00, 246.94, 293.66]]
    out = np.zeros_like(t)
    seg = 6.0
    for k in range(int(dur // seg) + 1):
        s, e = int(k * seg * sr), min(int((k + 1) * seg * sr), len(t))
        if s >= e:
            break
        tt = t[s:e]
        env = np.clip(np.minimum((tt - tt[0]) / 1.5, (tt[-1] - tt) / 1.5), 0, 1)
        for f in chords[k % 4]:
            out[s:e] += env * (np.sin(2*np.pi*f*tt) + 0.3*np.sin(2*np.pi*2*f*tt))
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.5 * 32767).astype(np.int16)
    st = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(st.tobytes())
    return path


def add_music(voice, dur):
    m = AudioFileClip(make_music(dur + 1)).subclip(0, dur)
    m = m.fx(afx.volumex, 0.18).fx(afx.audio_fadein, 1).fx(afx.audio_fadeout, 2)
    return CompositeAudioClip([voice, m])


W, H = 1080, 1920
BRAND = "MIND SKETCH"
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
# Pixabay key (the secret may still be named PEXELS_API_KEY in your repo; the code calls Pixabay)
STOCK_KEY = os.environ.get("PIXABAY_API_KEY") or os.environ["PEXELS_API_KEY"]
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"] if m]
# A different voice from the Tom Facts channel so the two channels sound different.
VOICE = os.environ.get("VOICE", "en-US-AriaNeural")
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"
USED_CLIPS = "used_clips.txt"
LOG = "log.csv"
SEP = " || "
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# ONE niche only: everyday psychology. Only well-established, widely taught concepts
# (no diagnoses, no mental illnesses, no treatment advice, no disputed findings).
CONCEPTS = [
    "procrastination", "confirmation bias", "anchoring effect", "sunk cost fallacy",
    "spotlight effect", "mere exposure effect", "halo effect", "bystander effect",
    "peak-end rule", "hedonic adaptation", "availability heuristic", "loss aversion",
    "framing effect", "primacy and recency effect", "IKEA effect", "fundamental attribution error",
    "curse of knowledge", "planning fallacy", "self-serving bias", "social proof",
    "reciprocity", "foot-in-the-door technique", "habit loops (cue, routine, reward)",
    "why unfinished tasks stay stuck in your mind", "first impressions",
    "overthinking everyday decisions", "choice overload in shopping",
    "the false consensus effect", "the contrast effect", "the endowment effect",
    "cognitive dissonance", "the illusion of control", "optimism bias",
    "the Baader-Meinhof (frequency) illusion", "hindsight bias", "groupthink",
    "body language misconceptions", "why we procrastinate on easy tasks",
    "why we forget why we walked into a room", "why we compare ourselves to others",
    "why we remember embarrassing moments", "why we say yes when we want to say no",
    "why habits are hard to break", "why we love routines",
]
ANGLES = [
    "a relatable everyday moment where it happens to you without you noticing",
    "why the brain does this (the simple reason)",
    "one easy trick to notice it or use it in daily life",
    "a surprising thing most people get wrong about it",
]
HOOKS = [
    "start by describing a very specific relatable moment ('You open your phone to check one thing...')",
    "start by telling the viewer something they do without knowing it ('You do this every single day...')",
    "start with a bold, slightly provocative statement about the viewer's own brain",
    "start with a question the viewer instantly says 'yes' to in their head",
]


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


def pick_concept():
    used = {l.split(SEP)[1].lower() for l in read_lines(HISTORY, 400) if SEP in l}
    fresh = [c for c in CONCEPTS if c.lower() not in used]
    return random.choice(fresh or CONCEPTS)


def gemini_json(prompt, temperature=0.9):
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    last_err = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY}, json=body, timeout=120)
            if r.status_code == 200:
                parts = r.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                out = json.loads(text)
                print("Model used:", model)
                return out
            last_err = f"{model}: {r.status_code} {r.text[:200]}"
        except Exception as e:
            last_err = f"{model}: {type(e).__name__}: {e}"
        print("Model failed ->", last_err)
    raise RuntimeError("All Gemini models failed. Last: " + str(last_err))


def fact_check(meta):
    prompt = f"""You are a strict, skeptical psychology editor who checks short video scripts for accuracy before publishing.
Title: {meta['title']}
Script: {meta['script']}
Do these checks:
1) NAME: is the named concept a real, standard concept that psychology textbooks teach under that name?
2) MECHANISM: write the standard textbook explanation of why it happens, then judge whether the script's explanation matches it.
   Mark mechanism_ok=false if the script gives a simplified or invented cause (for example 'your brain is lazy', 'your brain is wired to...', 'hardwired',
   evolution or brain-region claims that are not firmly established) or a pop-psychology metaphor presented as the scientific reason.
3) CLAIMS: mark accurate=false if the main claim is false, overstated, disputed, or failed to replicate (power posing, ego depletion, learning styles,
   left-brain/right-brain, 10% of the brain, Mozart effect, decision fatigue as a proven law, 'studies prove', exact percentages).
4) SAFETY: mark safe=false if it diagnoses, mentions mental illnesses, gives therapy, treatment or medical advice.
Hedged, general wording ('often', 'tends to', 'most people') is fine. Be strict: when unsure, fail the check.
"confidence" = how sure you are (1-10) that the script is accurate and safe to publish. Use 8 or more only when you are certain.
Return ONLY JSON: {{"concept_ok": true or false, "mechanism_ok": true or false, "accurate": true or false, "safe": true or false, "confidence": 1-10, "issue": "short reason"}}"""
    return gemini_json(prompt, temperature=0.0)


def passes_check(check):
    return (check.get("accurate") is True and check.get("safe") is True
            and check.get("mechanism_ok") is True and check.get("concept_ok") is True
            and check.get("confidence", 0) >= 8)


BANNED = ["depress", "anxiety", "adhd", "trauma", "disorder", "therapy", "therapist", "diagnos", "cure",
          "treatment", "medication", "suicid", "self-harm", "ptsd", "bipolar", "schizo", "autis", "ocd",
          "mental illness", "mental health", "lazy", "wired to", "hardwired", "hard-wired", "studies prove",
          "science proves", "proven"]


def hook_of(meta):
    first = meta["scenes"][0]["line"].strip()
    return re.split(r"(?<=[.!?])\s", first)[0]


def recent_hooks(n=60):
    out = []
    for l in read_lines(HISTORY, n):
        parts = l.split(SEP)
        if len(parts) > 2:
            out.append(parts[2])
    return out


def _tokens(t):
    return set(re.findall(r"[a-z']+", t.lower()))


def score_idea(meta):
    """Simple rule-based quality score (0-8). It checks craft rules only; it does not predict views."""
    pts, hook = 0, hook_of(meta)
    hw = len(hook.split())
    pts += hw <= 9                                            # short hook
    pts += bool(re.search(r"\b(you|your|you're)\b", hook.lower()) or hook.endswith("?"))  # speaks to the viewer
    pts += 62 <= len(meta["script"].split()) <= 92            # tight length
    kws = [s["keyword"].lower() for s in meta["scenes"]]
    pts += len(set(kws)) == len(kws)                          # varied visuals
    pts += all(2 <= len(k.split()) <= 5 for k in kws)         # concrete, filmable keywords
    abstract = ("psycholog", "bias", "brain", "mind", "concept")
    pts += not any(a in k for k in kws for a in abstract)     # no abstract footage searches
    pts += meta["scenes"][-1]["line"].rstrip().endswith("?")  # comment-driving ending
    old = [_tokens(h) for h in recent_hooks()]
    new = _tokens(hook)
    pts += not any(new and len(new & o) / len(new | o) > 0.5 for o in old)  # distinct from recent hooks
    return pts


MIN_SCORE = 7


def parse_scenes(meta):
    """Validate the scenes returned by Gemini and build the full script. Returns None if unusable."""
    scenes = meta.get("scenes")
    if not isinstance(scenes, list) or not (4 <= len(scenes) <= 8):
        return None
    clean = []
    for s in scenes:
        if not isinstance(s, dict):
            return None
        line = str(s.get("line", "")).strip()
        kw = str(s.get("keyword", "")).strip()
        if not line or not kw:
            return None
        clean.append({"line": line, "keyword": kw})
    script = " ".join(s["line"] for s in clean)
    wc = len(script.split())
    if not (62 <= wc <= 100):
        return None
    low = script.lower() + " " + str(meta.get("title", "")).lower()
    if any(w in low for w in BANNED):
        print("Rejected by banned-word filter")
        return None
    first_sentence = clean[0]["line"].replace("?", ".").replace("!", ".").split(".")[0]
    if len(first_sentence.split()) > 12:
        print("Rejected: hook too long")
        return None
    if not clean[-1]["line"].rstrip().endswith("?"):
        print("Rejected: last line is not a question")
        return None
    meta["title"] = str(meta.get("title", "")).strip().strip('"').strip()[:70]
    if not meta["title"]:
        return None
    tags = meta.get("tags")
    meta["tags"] = [str(t)[:30] for t in tags][:10] if isinstance(tags, list) and tags else ["psychology facts", "psychology"]
    meta["scenes"] = clean
    meta["script"] = script
    return meta


def make_script():
    concept = pick_concept()
    angle = random.choice(ANGLES)
    hook = random.choice(HOOKS)
    titles = [l.split(SEP)[0] for l in read_lines(HISTORY, 150)]
    hooks = recent_hooks()
    prompt = f"""You write viral-quality YouTube Shorts for the channel Mind Sketch, which explains everyday psychology in simple, warm, human words.
Concept: {concept}. Angle: {angle}. Hook style: {hook}.
Voice: a smart, calm friend talking to ONE person ('you'). Contractions, short sentences, a little humor or wonder. Never a textbook, never a robot.
Structure (it must keep people watching to the end):
1) HOOK, max 9 words, creates curiosity or recognition in the first 2 seconds. Never start with 'Did you know'.
2) Name the concept simply, then explain WHY it happens in plain words.
3) One concrete everyday example the viewer instantly recognizes.
4) One tiny practical takeaway.
5) LAST line: a question about THIS concept that people want to answer in the comments. It should echo the opening idea so the video loops naturally when it replays.
Do NOT repeat these titles: {titles}. Do NOT open like these recent hooks: {hooks}.
STRICT RULES: only well-established, widely taught psychology. Use general language ('often', 'tends to', 'most people'), never 'studies prove'.
Explain WHY it happens using the standard textbook explanation in plain words. Do NOT invent causes and do NOT use metaphors as the cause
(never 'your brain is lazy', 'wired to', 'hardwired', 'tricks you'). No evolution or brain-region claims unless they are firmly established.
No exact percentages or statistics. No diagnoses, no mental illnesses (no depression, anxiety disorders, ADHD, trauma, etc.), no therapy or treatment advice, no medical claims, no disputed findings.
Split the narration into 5 or 6 SCENES. Each scene has the exact words the narrator says (8-20 words) and a stock-footage search phrase showing what the viewer should SEE during those words.
Visual keywords: 2-5 words, a concrete filmable scene (a person, place or object doing something), e.g. 'woman scrolling phone bed', 'crowd crossing street', 'man staring window rain', 'coffee cup desk morning'. Vary the shots. No abstract words (psychology, bias, brain, mind). No text, logos or famous people.
Total narration 70-90 words (about 30 seconds). Cut every sentence that does not move the story forward.
Return ONLY JSON with this shape:
{{"title": "curiosity-driven title, max 70 chars, like 'Why You Procrastinate Even When You Know Better'",
"scenes": [{{"line": "narration words", "keyword": "visual search phrase"}}],
"tags": ["6 to 8 tags: always include psychology facts, psychology, human behavior, mind, did you know, plus the specific concept name"]}}"""
    for attempt in range(8):
        meta = gemini_json(prompt)
        meta = parse_scenes(meta)
        if not meta:
            print("Bad structure, regenerating")
            continue
        sc = score_idea(meta)
        print("Idea score:", sc, "/ 8 | hook:", hook_of(meta))
        if sc < MIN_SCORE:
            print("Rejected: idea score too low, regenerating")
            continue
        check = fact_check(meta)
        print("Fact-check:", check)
        if passes_check(check):
            meta["concept"] = concept
            return meta
        print("Rejected, regenerating:", meta.get("title"))
    raise RuntimeError("No script passed the structure and fact checks")


async def _tts(text, path):
    try:
        await edge_tts.Communicate(text, VOICE, rate="+4%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="en").save(path)


def _get_with_retry(url, params, tries=3):
    import time
    for k in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(3 * (k + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def get_clip(query, i, used):
    r = _get_with_retry("https://pixabay.com/api/videos/",
                        {"key": STOCK_KEY, "q": query, "per_page": 40, "safesearch": "true"})
    hits = [h for h in r.json().get("hits", []) if str(h.get("id")) not in used]
    random.shuffle(hits)
    words = [w for w in query.lower().split() if len(w) > 2]

    def score(h):
        tags = (h.get("tags") or "").lower()
        return sum(1 for w in words if w in tags)

    hits.sort(key=lambda h: -score(h))  # clips whose tags match the scene come first
    for h in hits:
        url = None
        for size in ("large", "medium", "small"):
            d = h["videos"].get(size) or {}
            if d.get("url") and (d.get("width") or 0) <= 1920:
                url = d["url"]
                break
        if not url:
            continue
        path = f"clip{i}.mp4"
        with requests.get(url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(path, "wb") as out:
                for chunk in dl.iter_content(1 << 20):
                    out.write(chunk)
        used.add(str(h["id"]))
        with open(USED_CLIPS, "a", encoding="utf-8") as f:
            f.write(str(h["id"]) + "\n")
        return path
    return None


def safe_clip(query, i, used):
    try:
        return get_clip(query, i, used)
    except Exception as e:
        print("Clip search failed for", query, "->", e)
        return None


def fit(clip, dur):
    clip = clip.without_audio()
    s = max(W / clip.w, H / clip.h) * 1.01
    clip = clip.resize(s)
    clip = clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=W, height=H)
    if clip.duration < dur:
        clip = clip.fx(vfx.loop, duration=dur)
    return clip.subclip(0, dur)


def scene_durations(scenes, total):
    """Split the total duration between scenes in proportion to the words narrated in each."""
    words = [max(len(s["line"].split()), 1) for s in scenes]
    s = sum(words)
    return [total * w / s for w in words]


def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise RuntimeError("No font found")


def _caption_image(text):
    font = _font(76)
    img = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines, line = [], ""
    for w in text.split():
        test = (line + " " + w).strip()
        if line and d.textlength(test, font=font) > W - 200:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    lh = 94
    y = (420 - lh * len(lines)) // 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((W - tw) / 2, y), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=7, stroke_fill=(0, 0, 0, 255))
        y += lh
    return np.array(img)


def _brand_image():
    font = _font(44)
    img = Image.new("RGBA", (W, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(BRAND, font=font)
    d.text(((W - tw) / 2, 20), BRAND, font=font, fill=(255, 159, 28, 235),
           stroke_width=3, stroke_fill=(0, 0, 0, 200))
    return np.array(img)


def _cta_image(text="Follow for daily psychology"):
    font = _font(52)
    img = Image.new("RGBA", (W, 110), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=font)
    d.text(((W - tw) / 2, 20), text, font=font, fill=(255, 255, 255, 255),
           stroke_width=5, stroke_fill=(0, 0, 0, 255))
    return np.array(img)


def add_subs(video, text, dur):
    """English captions (same language as the voice), 3-4 words at a time, plus a small brand tag."""
    try:
        words = text.split()
        parts = [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]
        total = sum(len(p) for p in parts)
        t = 0.0
        layers = [video]
        # soft dark layer so the white captions stay readable on any footage
        layers.append(ColorClip((W, H), color=(0, 0, 0)).set_opacity(0.22).set_duration(video.duration))
        layers.append(ImageClip(_brand_image()).set_duration(video.duration).set_position(("center", 150)))
        if video.duration > 6:  # small, non-pushy follow reminder in the last seconds
            layers.append(ImageClip(_cta_image()).set_start(video.duration - 2.2)
                          .set_duration(2.2).set_position(("center", 1500)))
        for p in parts:
            d = dur * len(p) / total
            c = (ImageClip(_caption_image(p))
                 .set_start(t).set_duration(d)
                 .set_position(("center", 1080)))
            layers.append(c)
            t += d
        out = CompositeVideoClip(layers).set_duration(video.duration)
        return out.set_audio(video.audio)
    except Exception as e:
        print("Subtitles skipped:", e)
        return video


def build_video(meta):
    asyncio.run(_tts(meta["script"], "voice.mp3"))
    audio = AudioFileClip("voice.mp3")
    total = audio.duration + 0.6
    used = set(read_lines(USED_CLIPS, 2000))
    scenes = meta["scenes"]
    durs = scene_durations(scenes, total)
    clips, last_path = [], None
    for i, (sc, d) in enumerate(zip(scenes, durs)):
        p = safe_clip(sc["keyword"], i, used) or safe_clip("person thinking", i, used) or last_path
        if not p:
            continue
        last_path = p
        clips.append(fit(VideoFileClip(p), max(d, 1.5)))
    if not clips:
        raise RuntimeError("No stock clips found")
    video = concatenate_videoclips(clips).set_audio(add_music(audio, total))
    video = add_subs(video, meta["script"], audio.duration)
    video.write_videofile("out.mp4", fps=30, codec="libx264", audio_codec="aac",
                          audio_bitrate="192k", preset="veryfast", threads=2, logger=None,
                          ffmpeg_params=["-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    meta["audio_sec"] = round(audio.duration, 1)
    return "out.mp4"


def get_creds():
    cid = os.environ["YT_CLIENT_ID"].strip()
    sec = os.environ["YT_CLIENT_SECRET"].strip()
    rt = os.environ["YT_REFRESH_TOKEN"].strip()
    creds = Credentials(
        None,
        refresh_token=rt,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=sec,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())  # fail fast, before rendering the video
    print("YouTube authorization OK")
    return creds


def upload(path, meta, creds):
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": meta["title"][:95],
            "description": (meta["script"]
                            + "\n\nWhat should we explain next? Tell us in the comments!"
                            + "\nSimple psychology facts every day. Subscribe to Mind Sketch for more!"
                            + "\n\nEducational content only, not medical or psychological advice."
                            + "\n\n#Shorts #Psychology #PsychologyFacts #HumanBehavior #DidYouKnow"),
            "tags": meta["tags"],
            "categoryId": "27",
        },
        "status": {"privacyStatus": PRIVACY, "selfDeclaredMadeForKids": False},
    }
    req = yt.videos().insert(
        part="snippet,status", body=body,
        media_body=MediaFileUpload(path, chunksize=-1, resumable=True),
    )
    import time
    from googleapiclient.errors import HttpError
    resp, tries = None, 0
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            tries += 1
            if e.resp.status in (500, 502, 503, 504) and tries < 4:
                time.sleep(5 * tries)
                continue
            raise
    print("Uploaded:", resp["id"])


if __name__ == "__main__":
    creds = get_creds()
    meta = make_script()
    print("Title:", meta["title"], "| Concept:", meta["concept"])
    video_path = build_video(meta)
    upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(meta["title"] + SEP + meta["concept"] + SEP + hook_of(meta) + "\n")
    try:  # one row per video; fill the last 4 columns by hand from YouTube Studio
        new = not os.path.exists(LOG)
        with open(LOG, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "title", "concept", "hook", "words", "audio_sec", "scenes", "idea_score",
                            "views", "viewed_vs_swiped_pct", "avg_view_pct", "subs_gained"])
            w.writerow([datetime.date.today().isoformat(), meta["title"], meta["concept"], hook_of(meta),
                        len(meta["script"].split()), meta.get("audio_sec", ""), len(meta["scenes"]),
                        score_idea(meta), "", "", "", ""])
    except Exception as e:
        print("Log skipped:", e)
