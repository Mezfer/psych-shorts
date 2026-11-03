import os, json, random, asyncio, csv, re, datetime, wave
import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (AudioFileClip, VideoFileClip, ImageClip, ColorClip,
                            CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, vfx)
import moviepy.audio.fx.all as afx
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Arabic text needs letter joining + right-to-left ordering before PIL can draw it.
try:
    import arabic_reshaper
    _reshaper = arabic_reshaper.ArabicReshaper()
except Exception:
    _reshaper = None
try:
    from bidi.algorithm import get_display
except Exception:
    try:
        from bidi import get_display
    except Exception:
        get_display = None


try:
    from PIL import features as _features
    HAS_RAQM = bool(_features.check("raqm")) and bool(_features.check("fribidi"))
except Exception:
    HAS_RAQM = False
print("Arabic text engine:", "raqm" if HAS_RAQM else "reshaper+bidi")


def _layout():
    L = getattr(ImageFont, "Layout", None)
    if HAS_RAQM:
        return L.RAQM if L else ImageFont.LAYOUT_RAQM
    return L.BASIC if L else ImageFont.LAYOUT_BASIC


def ar(text):
    """Text to draw. With raqm the library shapes and orders Arabic itself, so we must NOT do it twice."""
    if HAS_RAQM:
        return text
    if _reshaper:
        text = _reshaper.reshape(text)
    return get_display(text) if get_display else text


def _tw(d, text, font):
    if HAS_RAQM:
        return d.textlength(text, font=font, direction="rtl", language="ar")
    return d.textlength(ar(text), font=font)


def _draw(d, xy, text, font, **kw):
    if HAS_RAQM:
        d.text(xy, text, font=font, direction="rtl", language="ar", **kw)
    else:
        d.text(xy, ar(text), font=font, **kw)


W, H = 1080, 1920
BRAND = os.environ.get("BRAND", "تحدّي التخمين")
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
# Pixabay key (the secret may still be named PEXELS_API_KEY in your repo; the code calls Pixabay)
STOCK_KEY = os.environ.get("PIXABAY_API_KEY") or os.environ["PEXELS_API_KEY"]
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"] if m]
VOICE = os.environ.get("VOICE", "ar-SA-HamedNeural")  # other good options: ar-EG-SalmaNeural, ar-EG-ShakirNeural
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"
USED_CLIPS = "used_clips.txt"
LOG = "log.csv"
SEP = " || "
FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Guess-challenge Shorts: one surprising TRUE fact per video, stable and easy to verify.
TOPICS = [
    "animals and their abilities", "the human body (general facts only, no medical advice)",
    "space and planets", "world geography and famous landmarks", "food and drinks around the world",
    "famous inventions and who made them", "oceans and sea creatures", "weather and natural phenomena",
    "everyday objects and how they work", "sports facts that never change", "big numbers and sizes of famous things",
    "birds and insects", "mountains, deserts and rivers", "how things are made",
]
BANNED = {"اكتئاب", "قلق", "علاج", "تشخيص", "مرض", "سياسة", "الحرب", "الحروب", "الدين", "قمار", "مخدرات",
          "انتحار", "جنس", "إرهاب", "اضمن", "مضمون", "ترامب", "اسرائيل", "إسرائيل"}
ABSTRACT = ("concept", "idea", "fact", "trivia", "quiz", "question", "answer", "challenge", "knowledge")
COUNTDOWN_LINE = "خمسة... أربعة... ثلاثة... اثنان... واحد..."


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


def pick_topic():
    used = [l.split(SEP)[1] for l in read_lines(HISTORY, 40) if SEP in l]
    fresh = [t for t in TOPICS if t not in used[-6:]]
    return random.choice(fresh or TOPICS)


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
    prompt = f"""You are a strict fact-checker for short Arabic trivia videos. Check the facts, in any language.
Title: {meta['title']}
Question: {meta['hook']}
Clue: {meta['setup']}
Answer and explanation: {meta['reveal']}
Extra fact: {meta['extra']}
Checks:
1) ANSWER: is the answer to the question correct and unambiguous (only one reasonable answer)?
2) STABLE: is every claim a well-established fact that does not change over time and is not disputed? (No prices, no 'latest' records, no rankings that change.)
3) DETAILS: are all numbers and the extra fact correct? Mark false if any number is wrong, rounded misleadingly, or unsure.
4) SAFE: no politics, religion, medical advice, violence, gambling or sensitive claims.
Be strict: when unsure, fail the check. "confidence" = how sure you are (1-10) that everything is correct. Use 8 or more only when certain.
Return ONLY JSON: {{"answer_correct": true or false, "stable": true or false, "details_ok": true or false, "safe": true or false, "confidence": 1-10, "issue": "short reason"}}"""
    return gemini_json(prompt, temperature=0.0)


def passes_check(c):
    return (c.get("answer_correct") is True and c.get("stable") is True and c.get("details_ok") is True
            and c.get("safe") is True and c.get("confidence", 0) >= 8)


def _tokens(t):
    return set(re.findall(r"[؀-ۿ]+|[a-z0-9']+", t.lower()))


def words(t):
    return len(t.split())


def recent(idx, n=60):
    out = []
    for l in read_lines(HISTORY, n):
        p = l.split(SEP)
        if len(p) > idx:
            out.append(p[idx])
    return out


def parse_meta(meta):
    """Validate Gemini's JSON and build the scene list. Returns None when unusable."""
    try:
        parts = {}
        for k in ("hook", "setup", "reveal", "extra", "cta"):
            d = meta[k]
            line, kw = str(d["line"]).strip(), str(d["keyword"]).strip()
            if not line or not re.fullmatch(r"[A-Za-z ]{3,40}", kw):
                print("Bad scene:", k)
                return None
            parts[k] = {"line": line, "keyword": kw.lower()}
    except Exception:
        return None
    title = str(meta.get("title", "")).strip().strip('"').strip()[:60]
    answer = str(meta.get("answer", "")).strip()
    if not title or not answer:
        return None
    narration = " ".join(parts[k]["line"] for k in ("hook", "setup", "reveal", "extra", "cta"))
    wc = words(narration)
    if not (38 <= wc <= 85):
        print("Rejected: length", wc)
        return None
    if _tokens(narration + " " + title) & BANNED:
        print("Rejected by banned-word filter")
        return None
    if words(parts["hook"]["line"]) > 12:
        print("Rejected: hook too long")
        return None
    if not parts["cta"]["line"].rstrip().endswith(("؟", "?")):
        print("Rejected: last line is not a question")
        return None
    if not parts["hook"]["line"].rstrip().endswith(("؟", "?")):
        print("Rejected: hook is not a question")
        return None
    scenes = [
        {"kind": "hook", **parts["hook"]},
        {"kind": "setup", **parts["setup"]},
        {"kind": "countdown", "line": COUNTDOWN_LINE, "keyword": "clock ticking close up"},
        {"kind": "reveal", **parts["reveal"]},
        {"kind": "extra", **parts["extra"]},
        {"kind": "cta", **parts["cta"]},
    ]
    tags = meta.get("tags")
    meta["tags"] = ([str(t)[:30] for t in tags][:10] if isinstance(tags, list) and tags
                    else ["تحدي", "خمن", "معلومات", "shorts"])
    meta.update({"title": title, "answer": answer, "scenes": scenes, "script": narration,
                 "hook_line": parts["hook"]["line"]})
    return meta


MIN_SCORE = 7


def score_idea(meta):
    """Rule-based craft score (0-8). It checks the writing rules only; it does not predict views."""
    pts = 0
    hook = meta["hook_line"]
    pts += words(hook) <= 10                                              # short hook
    pts += hook.rstrip().endswith(("؟", "?"))                             # a question to guess
    pts += 40 <= words(meta["script"]) <= 75                             # tight length
    kws = [s["keyword"] for s in meta["scenes"]]
    pts += len(set(kws)) == len(kws)                                      # varied visuals
    pts += all(2 <= len(k.split()) <= 5 for k in kws)                    # concrete footage searches
    pts += not any(a in k for k in kws for a in ABSTRACT)                # no abstract searches
    pts += words(meta["scenes"][-1]["line"]) <= 12                       # short comment call
    new = _tokens(hook)
    old = [_tokens(h) for h in recent(2)]
    pts += not any(new and len(new & o) / len(new | o) > 0.5 for o in old)  # distinct from recent hooks
    return pts


def make_script():
    topic = pick_topic()
    titles = [l.split(SEP)[0] for l in read_lines(HISTORY, 150)]
    hooks = recent(2)
    answers = recent(3)
    prompt = f"""You write viral YouTube Shorts in ARABIC for a "guess the answer" challenge channel with MrBeast-style energy: a big claim, curiosity, a countdown, then the answer.
Write all spoken lines in simple Arabic that every Arab viewer understands (Modern Standard Arabic mixed with easy everyday words). No heavy classical words. Short sentences. Energetic, friendly voice, talking to ONE viewer.
Category: {topic}.
Pick ONE surprising but TRUE, well-established fact with a short, clear, single answer (a number, an animal, a place, an object). It must be stable over time and easy to verify. No prices, no changing records or rankings, no politics, religion, medicine, violence or gambling.
Do NOT reuse these titles: {titles}. Do NOT open like these recent hooks: {hooks}. Do NOT reuse these answers: {answers}.
Return the fields below. Each "line" is the exact words the narrator says. Each "keyword" is an ENGLISH stock-footage search phrase (2-5 words, a concrete filmable scene such as 'cheetah running savanna', 'astronaut floating space', 'huge waterfall aerial'). Never abstract words. Vary the shots.
- hook: the challenge question, 5-10 words, ends with '؟', creates instant curiosity (e.g. 'هل تستطيع تخمين أسرع حيوان على الأرض؟').
- setup: 1-2 short sentences with a clue or the stakes, 10-20 words. Do NOT give the answer.
- reveal: the answer plus the one-line reason, 12-25 words.
- extra: one amazing related TRUE fact, 10-20 words.
- cta: one short question for the comments, 5-10 words, ends with '؟' (for example asking what they guessed).
- title: Arabic, max 55 characters, a challenge feel, honest (no lies).
- answer: the answer in 1-4 Arabic words.
- tags: 6 to 8 tags, include 'تحدي', 'خمن', 'معلومات', 'shorts' and the specific subject.
Total narration (hook+setup+reveal+extra+cta) must be 45-70 words.
Return ONLY JSON:
{{"title": "", "answer": "", "hook": {{"line": "", "keyword": ""}}, "setup": {{"line": "", "keyword": ""}},
"reveal": {{"line": "", "keyword": ""}}, "extra": {{"line": "", "keyword": ""}}, "cta": {{"line": "", "keyword": ""}}, "tags": []}}"""
    for attempt in range(8):
        meta = parse_meta(gemini_json(prompt))
        if not meta:
            print("Bad structure, regenerating")
            continue
        sc = score_idea(meta)
        print("Idea score:", sc, "/ 8 | hook:", meta["hook_line"])
        if sc < MIN_SCORE:
            print("Rejected: idea score too low, regenerating")
            continue
        check = fact_check(meta)
        print("Fact-check:", check)
        if passes_check(check):
            meta["concept"] = topic
            return meta
        print("Rejected, regenerating:", meta.get("title"))
    raise RuntimeError("No script passed the structure and fact checks")


# ---------------------------------------------------------------- audio
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
    m = m.fx(afx.volumex, 0.15).fx(afx.audio_fadein, 1).fx(afx.audio_fadeout, 2)
    return CompositeAudioClip([voice, m])


async def _tts_one(text, path):
    try:
        await edge_tts.Communicate(text, VOICE, rate="+4%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="ar").save(path)


async def _tts_all(scenes):
    paths = []
    for i, sc in enumerate(scenes):
        p = f"voice{i}.mp3"
        await _tts_one(sc["line"], p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- footage
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
    ws = [w for w in query.lower().split() if len(w) > 2]

    def score(h):
        tags = (h.get("tags") or "").lower()
        return sum(1 for w in ws if w in tags)

    hits.sort(key=lambda h: -score(h))
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


# ---------------------------------------------------------------- text overlays
def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size, layout_engine=_layout())
    raise RuntimeError("No font found")


def _caption_image(text):
    font = _font(78)
    img = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines, line = [], ""
    for w in text.split():
        test = (line + " " + w).strip()
        if line and _tw(d, test, font) > W - 200:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    lh = 100
    y = (420 - lh * len(lines)) // 2
    for ln in lines:
        tw = _tw(d, ln, font)
        _draw(d, ((W - tw) / 2, y), ln, font, fill=(255, 255, 255, 255),
              stroke_width=8, stroke_fill=(0, 0, 0, 255))
        y += lh
    return np.array(img)


def _text_image(text, size, height, color, stroke=4):
    font = _font(size)
    img = Image.new("RGBA", (W, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tw = _tw(d, text, font)
    _draw(d, ((W - tw) / 2, (height - size) / 2 - 8), text, font, fill=color,
          stroke_width=stroke, stroke_fill=(0, 0, 0, 230))
    return np.array(img)


def _digit_image(n):
    font = _font(480)
    img = Image.new("RGBA", (W, 620), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = str(n)
    tw = d.textlength(s, font=font, direction="ltr") if HAS_RAQM else d.textlength(s, font=font)
    d.text(((W - tw) / 2, 20), s, font=font, fill=(255, 159, 28, 255),
           stroke_width=14, stroke_fill=(0, 0, 0, 255))
    return np.array(img)


def _brand_image():
    return _text_image(BRAND, 46, 95, (255, 159, 28, 235), stroke=3)


def _cta_image():
    return _text_image("تابعنا لتحدٍّ جديد كل يوم", 54, 115, (255, 255, 255, 255), stroke=5)


def build_overlays(video, timeline):
    """Dark layer, brand tag, captions (3 words at a time, synced per scene), countdown digits, end card."""
    try:
        total = video.duration
        layers = [video,
                  ColorClip((W, H), color=(0, 0, 0)).set_opacity(0.30).set_duration(total),
                  ImageClip(_brand_image()).set_duration(total).set_position(("center", 150))]
        for start, adur, sc in timeline:
            if sc["kind"] == "countdown":
                step = max(adur, 4.0) / 5
                for k, n in enumerate([5, 4, 3, 2, 1]):
                    layers.append(ImageClip(_digit_image(n)).set_start(start + k * step).set_duration(step)
                                  .set_position(("center", 560)))
                continue
            ws = sc["line"].split()
            parts = [" ".join(ws[i:i + 3]) for i in range(0, len(ws), 3)]
            tot = sum(len(p) for p in parts) or 1
            t = start
            for p in parts:
                d = adur * len(p) / tot
                layers.append(ImageClip(_caption_image(p)).set_start(t).set_duration(d)
                              .set_position(("center", 1080)))
                t += d
        if total > 6:
            layers.append(ImageClip(_cta_image()).set_start(total - 2.2).set_duration(2.2)
                          .set_position(("center", 250)))
        return CompositeVideoClip(layers).set_duration(total).set_audio(video.audio)
    except Exception as e:
        print("Overlays skipped:", e)
        return video


# ---------------------------------------------------------------- build
def build_video(meta):
    scenes = meta["scenes"]
    paths = asyncio.run(_tts_all(scenes))
    used = set(read_lines(USED_CLIPS, 2000))
    voices, clips, timeline, t, last = [], [], [], 0.0, None
    for i, (sc, vp) in enumerate(zip(scenes, paths)):
        a = AudioFileClip(vp)
        dur = a.duration + 0.3
        if sc["kind"] == "countdown":
            dur = max(dur, 4.5)  # leave real suspense time
        voices.append(a.set_start(t))
        timeline.append((t, a.duration, sc))
        p = safe_clip(sc["keyword"], i, used) or safe_clip("clock ticking", i, used) or last
        if not p:
            raise RuntimeError("No stock clips found")
        last = p
        clips.append(fit(VideoFileClip(p), dur))
        t += dur
    total = t
    voice = CompositeAudioClip(voices).set_duration(total)
    video = concatenate_videoclips(clips).set_audio(add_music(voice, total))
    video = build_overlays(video, timeline)
    video.write_videofile("out.mp4", fps=30, codec="libx264", audio_codec="aac",
                          audio_bitrate="192k", preset="veryfast", threads=2, logger=None,
                          ffmpeg_params=["-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    meta["audio_sec"] = round(total, 1)
    return "out.mp4"


# ---------------------------------------------------------------- YouTube
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
                            + "\n\nكم كانت إجابتك؟ اكتبها في التعليقات!"
                            + "\nتحدٍّ جديد كل يوم. اشترك في القناة حتى لا يفوتك."
                            + "\n\n#Shorts #تحدي #خمن #معلومات #معلومات_عامة"),
            "tags": meta["tags"],
            "categoryId": "24",
            "defaultLanguage": "ar",
            "defaultAudioLanguage": "ar",
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
    print("Title:", meta["title"], "| Topic:", meta["concept"])
    video_path = build_video(meta)
    upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(meta["title"] + SEP + meta["concept"] + SEP + meta["hook_line"] + SEP + meta["answer"] + "\n")
    try:  # one row per video; fill the last 4 columns by hand from YouTube Studio
        new = not os.path.exists(LOG)
        with open(LOG, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "title", "topic", "hook", "words", "total_sec", "idea_score",
                            "views", "viewed_vs_swiped_pct", "avg_view_pct", "subs_gained"])
            w.writerow([datetime.date.today().isoformat(), meta["title"], meta["concept"], meta["hook_line"],
                        len(meta["script"].split()), meta.get("audio_sec", ""), score_idea(meta),
                        "", "", "", ""])
    except Exception as e:
        print("Log skipped:", e)
