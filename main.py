import os, json, random, asyncio, csv, re, datetime, wave
import numpy as np
import requests
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (AudioFileClip, ImageClip, CompositeVideoClip, CompositeAudioClip,
                            concatenate_videoclips)
import moviepy.audio.fx.all as afx
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# "Tu préfères ?" — French 15-second choice Shorts. Gemini writes the dilemma, Pixabay gives one photo per option,
# edge-tts reads it, everything else (music, ticks, layout) is generated here. No faces, no copyrighted material.

W, H = 1080, 1920
TEST = os.environ.get("TEST_RENDER") == "1"          # local layout test: no Gemini / Pixabay / YouTube
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
STOCK_KEY = os.environ.get("PIXABAY_API_KEY") or os.environ.get("PEXELS_API_KEY", "")  # the secret may still be named PEXELS_API_KEY
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
          "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash"] if m]
VOICE = os.environ.get("VOICE", "fr-FR-DeniseNeural")   # other good options: fr-FR-HenriNeural, fr-FR-VivienneMultilingualNeural
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history_fr.txt"
USED_IMGS = "used_imgs.txt"
LOG = "log.csv"
SEP = " || "
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
COL_A = (233, 69, 96)     # red
COL_B = (30, 136, 229)    # blue

CATEGORIES = {
    "food": "food and drinks everybody knows (pizza, chocolate, fries, ice cream, breakfast...)",
    "superpower": "funny harmless superpowers and magic abilities",
    "money_life": "everyday life and lifestyle choices (house, car, holidays, free time, sleep)",
    "animals": "animals as pets or animals in absurd situations (nothing cruel)",
    "tech": "phones, games, internet, social networks, robots",
    "travel": "travel and places (sea, mountain, city, countryside, famous countries)",
    "school_work": "school, work and daily habits that everyone knows",
    "absurd": "absurd, silly and funny situations that make people laugh",
    "time": "time, age, past and future choices",
    "senses": "everyday sensations: weather, seasons, smells, sounds, tastes",
}
BANNED = {"sexe", "sexuel", "porno", "mort", "mourir", "tuer", "suicide", "drogue", "alcool", "cigarette", "guerre",
          "politique", "religion", "dieu", "islam", "juif", "chrétien", "cancer", "maladie", "viol", "terroriste",
          "racisme", "nazi", "arme", "sang", "pari", "casino"}


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


def recent(idx, n=80):
    out = []
    for l in read_lines(HISTORY, n):
        p = l.split(SEP)
        if len(p) > idx:
            out.append(p[idx])
    return out


def next_episode():
    return 1 + len(read_lines(HISTORY, 100000))


def _tokens(t):
    return set(re.findall(r"[a-zàâäçéèêëîïôöùûüÿœ0-9']+", t.lower()))


def words(t):
    return len(t.split())


def too_similar(a, b):
    """True when this dilemma resembles one of the recent ones (same two options in any order, or mostly the same words)."""
    new = _tokens(a) | _tokens(b)
    for line in read_lines(HISTORY, 150):
        p = line.split(SEP)
        if len(p) < 4:
            continue
        old = _tokens(p[2]) | _tokens(p[3])
        if new and old and len(new & old) / len(new | old) > 0.5:
            return True
    return False


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


# ---------------------------------------------------------------- script
def pick_category():
    last = recent(1, 10)[-3:]
    pool = [k for k in CATEGORIES if k not in last] or list(CATEGORIES)
    return random.choice(pool)


def clean_option(s):
    s = str(s).strip().strip('"«»').strip()
    s = re.sub(r"^(tu préfères|tu preferes)\s*", "", s, flags=re.I)
    s = s.rstrip(" ?!.,;…")
    return s[:1].lower() + s[1:] if s else s


def parse_meta(meta):
    try:
        a, b = clean_option(meta["a"]), clean_option(meta["b"])
        ka, kb = str(meta["keyword_a"]).strip().lower(), str(meta["keyword_b"]).strip().lower()
        title = str(meta["title"]).strip().strip('"')
        cta = str(meta["cta"]).strip()
    except Exception:
        return None
    if not (a and b and title and cta):
        return None
    if not (2 <= words(a) <= 9 and 2 <= words(b) <= 9):
        print("Rejected: option length")
        return None
    if a.lower() == b.lower() or _tokens(a) == _tokens(b):
        return None
    for k in (ka, kb):
        if not re.fullmatch(r"[a-z ]{3,30}", k):
            print("Rejected: bad image keyword", k)
            return None
    if (_tokens(a + " " + b + " " + title) & BANNED):
        print("Rejected by banned-word filter")
        return None
    if ka == kb:
        return None
    if words(cta) > 12:
        cta = "A ou B ? Dis-le en commentaire !"
    if not cta.rstrip().endswith(("?", "!")):
        cta = cta.rstrip(" .") + " !"
    tags = meta.get("tags")
    tags = [str(t).strip().lstrip("#")[:30] for t in tags if str(t).strip()][:10] if isinstance(tags, list) else []
    meta.update({
        "a": a, "b": b, "keyword_a": ka, "keyword_b": kb, "title": title[:90], "cta": cta,
        "tags": tags or ["tu préfères", "quiz", "choix", "shorts"],
        "pinned_comment": str(meta.get("pinned_comment", "")).strip()[:200],
    })
    return meta


def judge(meta):
    """French proofreader + balance check. Returns corrected options or None."""
    prompt = f"""You are a strict native French proofreader and editor for a "Tu préfères ?" YouTube Shorts channel.
The video says: "Tu préfères {meta['a']}, ou {meta['b']} ?"
Check:
1) GRAMMAR: both options are correct French and each one completes "Tu préfères ... ?" naturally (usually an infinitive phrase). Fix any spelling, accent or grammar mistake.
2) BALANCED: neither option is obviously better, so people really argue in the comments.
3) SAFE: nothing about sex, violence, death, illness, drugs, alcohol, politics, religion, gambling, or any real person or brand scandal.
4) CLEAR: understandable in 3 seconds, and each option can be shown with a simple photo.
Return ONLY JSON: {{"ok": true or false, "a": "corrected option A, lowercase first letter, no final punctuation", "b": "corrected option B, same rules", "issue": "short reason"}}"""
    try:
        out = gemini_json(prompt, temperature=0.0)
    except Exception as e:
        print("Judge failed:", e)
        return None
    if out.get("ok") is not True:
        print("Judge rejected:", out.get("issue"))
        return None
    a, b = clean_option(out.get("a") or meta["a"]), clean_option(out.get("b") or meta["b"])
    return a, b


def make_script():
    cat = pick_category()
    old = [f"{x} / {y}" for x, y in zip(recent(2, 40), recent(3, 40))]
    prompt = f"""You write viral YouTube Shorts "Tu préfères ?" (Would you rather?) in FRENCH for a general audience of teenagers and adults.
Everything spoken or written is in correct, natural, everyday French.
CATEGORY for this video: {CATEGORIES[cat]}
Write ONE dilemma with two options A and B. Rules:
- The video says: "Tu préfères {{a}}, ou {{b}} ?" so a and b must each complete that sentence naturally: an infinitive phrase such as "vivre sans téléphone pendant un an" or "avoir toujours trop chaud". 3 to 8 words each, lowercase first letter, no final punctuation.
- Both options are about equally attractive or equally annoying, so that viewers split 50/50 and argue in the comments. Make it surprising, funny or tempting. No yes/no questions.
- Each option must be easy to show with ONE simple, concrete, filmable photo. keyword_a and keyword_b are ENGLISH stock-photo search phrases of 1 to 3 words (a concrete object, animal, food or place such as 'pizza', 'snowy mountain', 'cute cat', 'smartphone'). Never abstract words, never the same keyword twice.
- Totally safe: nothing about sex, violence, death, illness, drugs, alcohol, politics, religion, gambling, or real people.
- Do NOT repeat or resemble these recent dilemmas: {old[-40:]}
Fields:
- title: French YouTube title, max 80 characters, curiosity, starts with 'Tu préfères' and contains both options or a teaser, may end with a single emoji.
- cta: French call to action, 5-9 words, asks to answer in the comments, for example 'A ou B ? Dis-le en commentaire !'.
- pinned_comment: a friendly French comment the channel can pin (max 18 words) that asks people to vote.
- tags: 6 to 8 tags in French, include 'tu préfères', 'quiz', 'choix', 'shorts'.
Return ONLY JSON:
{{"title": "", "a": "", "b": "", "keyword_a": "", "keyword_b": "", "cta": "", "pinned_comment": "", "tags": []}}"""
    for attempt in range(10):
        meta = parse_meta(gemini_json(prompt))
        if not meta:
            continue
        if too_similar(meta["a"], meta["b"]):
            print("Rejected: too similar to a past video")
            continue
        fixed = judge(meta)
        if not fixed:
            continue
        meta["a"], meta["b"] = fixed
        if not (2 <= words(meta["a"]) <= 9 and 2 <= words(meta["b"]) <= 9):
            continue
        meta["category"] = cat
        meta["episode"] = next_episode()
        meta["hook_line"] = "Tu préfères"
        print(f"Attempt {attempt + 1}: OK  {meta['a']}  |  {meta['b']}")
        return meta
    raise RuntimeError("No dilemma passed the checks.")


# ---------------------------------------------------------------- audio
def make_music(dur, path="bg.wav"):
    sr = 44100
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    chords = [[261.63, 329.63, 392.00], [220.00, 261.63, 329.63],
              [174.61, 220.00, 261.63], [196.00, 246.94, 293.66]]
    out = np.zeros_like(t)
    seg = 3.0
    for k in range(int(dur // seg) + 1):
        s, e = int(k * seg * sr), min(int((k + 1) * seg * sr), len(t))
        if s >= e:
            break
        tt = t[s:e]
        env = np.clip(np.minimum((tt - tt[0]) / 0.6, (tt[-1] - tt) / 0.6), 0, 1)
        for f in chords[k % 4]:
            out[s:e] += env * (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt))
        # soft beat: a short low thump twice per second
        for b in np.arange(0, e - s, int(sr * 0.5)):
            n = min(int(sr * 0.12), e - s - b)
            if n > 0:
                x = np.arange(n) / sr
                out[s + b:s + b + n] += 1.2 * np.sin(2 * np.pi * 70 * x) * np.exp(-30 * x)
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.5 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
    return path


def add_music(voice, dur):
    m = AudioFileClip(make_music(dur + 1)).subclip(0, dur)
    m = m.fx(afx.volumex, 0.16).fx(afx.audio_fadein, 0.5).fx(afx.audio_fadeout, 1.5)
    return CompositeAudioClip([voice, m])


def make_sfx(total, ticks, pops, ding_at, path="sfx.wav"):
    """Own generated effects: a pop when each option appears, a tick per countdown number, a bell at the end card."""
    sr = 44100
    out = np.zeros(int(sr * (total + 1)))

    def burst(t0, freq, dur, amp, decay):
        i = int(t0 * sr)
        if i >= len(out):
            return
        n = min(int(dur * sr), len(out) - i)
        tt = np.arange(n) / sr
        out[i:i + n] += amp * np.sin(2 * np.pi * freq * tt) * np.exp(-decay * tt)

    for t0 in pops:
        burst(t0, 600, 0.12, 0.7, 30)
    for k, t0 in enumerate(ticks):
        burst(t0, 1500 if k == len(ticks) - 1 else 1000, 0.14, 0.8, 28)
    if ding_at is not None:
        for f in (880, 1320):
            burst(ding_at, f, 0.9, 0.4, 5)
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.6 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
    return path


def _fake_voice(text, path):
    """Test mode only: silence of a speaking-like length."""
    sr = 22050
    n = int(sr * max(0.8, len(text) * 0.065))
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())


async def _tts_one(text, path):
    if TEST:
        _fake_voice(text, path)
        return
    try:
        await edge_tts.Communicate(text, VOICE, rate="+6%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="fr").save(path)


async def _tts_all(lines):
    paths = []
    for i, line in enumerate(lines):
        p = f"voice{i}.mp3"
        await _tts_one(line, p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- images
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


def _fallback_photo(path, col):
    """Plain gradient card if no photo can be found, so the video still renders."""
    img = Image.new("RGB", (1280, 853))
    px = ImageDraw.Draw(img)
    for y in range(853):
        k = y / 853
        px.line([(0, y), (1280, y)], fill=tuple(int(c * (0.5 + 0.5 * k)) for c in col))
    img.save(path, quality=92)
    return path


def get_image(query, i, used, col):
    path = f"img{i}.jpg"
    if TEST:
        return _fallback_photo(path, col)
    for q in (query, query.split()[-1] if " " in query else None):
        if not q:
            continue
        try:
            r = _get_with_retry("https://pixabay.com/api/",
                                {"key": STOCK_KEY, "q": q, "image_type": "photo", "orientation": "horizontal",
                                 "per_page": 30, "safesearch": "true", "min_width": 1000})
            hits = [h for h in r.json().get("hits", []) if str(h.get("id")) not in used]
            random.shuffle(hits)
            for h in hits[:6]:
                url = h.get("largeImageURL") or h.get("webformatURL")
                if not url:
                    continue
                with requests.get(url, stream=True, timeout=60) as dl:
                    dl.raise_for_status()
                    with open(path, "wb") as out:
                        for chunk in dl.iter_content(1 << 20):
                            out.write(chunk)
                Image.open(path).convert("RGB").save(path, quality=92)  # fail here if the file is not a real image
                used.add(str(h["id"]))
                with open(USED_IMGS, "a", encoding="utf-8") as f:
                    f.write(str(h["id"]) + "\n")
                return path
        except Exception as e:
            print("Image search failed for", q, "->", e)
    print("No photo for", query, "- using a color card")
    return _fallback_photo(path, col)


# ---------------------------------------------------------------- drawing
def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise RuntimeError("No font found")


def _wrap(d, text, size, maxw, maxlines, maxh=10**6):
    """Largest font (<= size) so that the text fits in maxlines lines of width maxw."""
    while size >= 28:
        font = _font(size)
        lines, line = [], ""
        for w in text.split():
            test = (line + " " + w).strip()
            if line and d.textlength(test, font=font) > maxw:
                lines.append(line)
                line = w
            else:
                line = test
        if line:
            lines.append(line)
        if (len(lines) <= maxlines and len(lines) * int(size * 1.18) <= maxh
                and all(d.textlength(l, font=font) <= maxw for l in lines)):
            return font, lines, size
        size -= 4
    font = _font(28)
    return font, text.split(), 28


def _text_block(text, width, height, size, color, maxlines=3, stroke=6, bg=None):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg:
        d.rounded_rectangle([0, 0, width - 1, height - 1], radius=36, fill=bg)
    font, lines, size = _wrap(d, text, size, width - 90, maxlines, height - 24)
    lh = int(size * 1.18)
    y = (height - lh * len(lines)) // 2
    for ln in lines:
        tw = d.textlength(ln, font=font)
        d.text(((width - tw) / 2, y), ln, font=font, fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += lh
    return np.array(img)


PH = 680                    # panel height
PA_Y, PB_Y = 270, 970       # panel tops
PW = 960                    # panel width
PX = (W - PW) // 2


def _panel_image(photo, col):
    """Photo cropped to the panel, with a colored frame."""
    im = Image.open(photo).convert("RGB")
    s = max((PW + 60) / im.width, (PH + 60) / im.height)   # a little bigger than the panel so a slow zoom has room
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return np.array(im)


def _frame_image(col, letter):
    img = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, PW - 1, PH - 1], radius=40, outline=col + (255,), width=14)
    d.ellipse([22, 22, 142, 142], fill=col + (255,), outline=(255, 255, 255, 255), width=6)
    f = _font(88)
    tw = d.textlength(letter, font=f)
    d.text((82 - tw / 2, 82 - 52), letter, font=f, fill=(255, 255, 255, 255))
    return np.array(img)


def _rounded_mask(w, h, r=40):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return np.array(m).astype(float) / 255.0


def _bg_image():
    ys = np.linspace(0, 1, H)[:, None]
    top, bot = np.array([18, 10, 46]), np.array([46, 20, 92])
    arr = (top[None, None, :] * (1 - ys[:, :, None]) + bot[None, None, :] * ys[:, :, None])
    arr = np.repeat(arr, W, axis=1).astype(np.uint8)
    Image.fromarray(arr).save("bg_fr.png")
    return "bg_fr.png"


def _digit_image(n):
    f = _font(520)
    img = Image.new("RGBA", (W, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = str(n)
    tw = d.textlength(s, font=f)
    d.text(((W - tw) / 2, 10), s, font=f, fill=(255, 214, 10, 255), stroke_width=16, stroke_fill=(0, 0, 0, 255))
    return np.array(img)


# ---------------------------------------------------------------- build
def panel_clip(photo, col, dur, zoom):
    arr = _panel_image(photo, col)
    base = ImageClip(arr)
    if zoom:
        base = base.resize(lambda t: 1.0 + 0.035 * t)
    inner = CompositeVideoClip([base.set_position("center")], size=(PW, PH)).set_duration(dur)
    mask = ImageClip(_rounded_mask(PW, PH), ismask=True).set_duration(dur)
    return inner.set_mask(mask)


def build_video(meta, zoom=True):
    a, b = meta["a"], meta["b"]
    lines = ["Tu préfères...", a + "...", "ou " + b + " ?", meta["cta"]]
    paths = asyncio.run(_tts_all(lines))
    auds = [AudioFileClip(p) for p in paths]
    used = set(read_lines(USED_IMGS, 4000))
    photo_a = get_image(meta["keyword_a"], 0, used, COL_A)
    photo_b = get_image(meta["keyword_b"], 1, used, COL_B)

    # timeline
    t_hook = 0.0
    d_hook = auds[0].duration + 0.15
    t_a = t_hook + d_hook
    d_a = auds[1].duration + 0.35
    t_b = t_a + d_a
    d_b = auds[2].duration + 0.9
    t_cd = t_b + d_b
    d_cd = 3.0
    t_cta = t_cd + d_cd
    d_cta = auds[3].duration + 1.0
    total = t_cta + d_cta

    voices = [auds[0].set_start(t_hook), auds[1].set_start(t_a + 0.1),
              auds[2].set_start(t_b + 0.1), auds[3].set_start(t_cta + 0.2)]
    voice = CompositeAudioClip(voices).set_duration(total)
    mix = add_music(voice, total)
    try:
        sfx = AudioFileClip(make_sfx(total, [t_cd + k for k in range(3)], [t_a, t_b], t_cta)).subclip(0, total)
        mix = CompositeAudioClip([mix, sfx.fx(afx.volumex, 0.5)]).set_duration(total)
    except Exception as e:
        print("Sound effects skipped:", e)

    layers = [ImageClip(_bg_image()).set_duration(total)]
    # header: big in the center during the hook, small at the top afterwards
    big = _text_block("TU PRÉFÈRES ?", W, 330, 130, (255, 255, 255, 255), maxlines=2, stroke=9)
    layers.append(ImageClip(big).set_start(0).set_duration(d_hook).set_position(("center", 790)))
    head = _text_block("TU PRÉFÈRES ?", W, 130, 90, (255, 255, 255, 255), maxlines=1, stroke=7)
    layers.append(ImageClip(head).set_start(d_hook).set_duration(total - d_hook).set_position(("center", 95)))
    ep = _text_block(f"Choix n°{meta['episode']}", 500, 70, 40, (255, 214, 10, 255), maxlines=1, stroke=4)
    layers.append(ImageClip(ep).set_start(d_hook).set_duration(total - d_hook).set_position(("center", 205)))

    for (ph, col, letter, y, t0, txt) in ((photo_a, COL_A, "A", PA_Y, t_a, a), (photo_b, COL_B, "B", PB_Y, t_b, b)):
        dur = total - t0
        layers.append(panel_clip(ph, col, dur, zoom).set_start(t0).set_position((PX, y)).crossfadein(0.2))
        layers.append(ImageClip(_frame_image(col, letter)).set_start(t0).set_duration(dur).set_position((PX, y)))
        label = _text_block(txt[:1].upper() + txt[1:], PW - 80, 190, 62, (255, 255, 255, 255), maxlines=3, stroke=6,
                            bg=(0, 0, 0, 165))
        layers.append(ImageClip(label).set_start(t0 + 0.1).set_duration(dur - 0.1)
                      .set_position((PX + 40, y + PH - 215)))

    # countdown over the panels
    for k, n in enumerate([3, 2, 1]):
        layers.append(ImageClip(_digit_image(n)).set_start(t_cd + k).set_duration(1.0).set_position(("center", 650)))
    # end card pill on the seam between the two panels
    pill = _text_block(meta["cta"], 940, 190, 58, (255, 255, 255, 255), maxlines=2, stroke=5, bg=(20, 10, 46, 235))
    layers.append(ImageClip(pill).set_start(t_cta).set_duration(d_cta).set_position(("center", 865)))

    video = CompositeVideoClip(layers, size=(W, H)).set_duration(total).set_audio(mix)
    video.write_videofile("out.mp4", fps=30, codec="libx264", audio_codec="aac",
                          audio_bitrate="192k", preset="veryfast", threads=2, logger=None,
                          ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
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
    desc = (f"Tu préfères {meta['a']}, ou {meta['b']} ?\n\n"
            "Réponds A ou B en commentaire, et dis-nous pourquoi !\n"
            "Un nouveau choix impossible chaque jour : abonne-toi pour ne rien rater.\n\n"
            "#Shorts #TuPréfères #Quiz #Choix")
    body = {
        "snippet": {
            "title": meta["title"][:95],
            "description": desc,
            "tags": meta["tags"],
            "categoryId": "24",
            "defaultLanguage": "fr",
            "defaultAudioLanguage": "fr",
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
    return resp["id"]


LOG_HEADER = ["date", "title", "category", "a", "b", "video_id", "views", "viewed_vs_swiped_pct", "avg_view_pct",
              "likes", "comments", "subs_gained", "pinned_comment"]


def write_log(meta, vid):
    """One row per video. Fill views / viewed_vs_swiped_pct / avg_view_pct / likes / comments / subs_gained by hand
    from YouTube Studio (edit log.csv on GitHub)."""
    try:
        if os.path.exists(LOG):
            with open(LOG, encoding="utf-8", newline="") as f:
                first = next(csv.reader(f), [])
            if first != LOG_HEADER:
                os.replace(LOG, "log_old.csv")
        new = not os.path.exists(LOG)
        with open(LOG, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_HEADER)
            w.writerow([datetime.date.today().isoformat(), meta["title"], meta["category"], meta["a"], meta["b"],
                        vid, "", "", "", "", "", "", meta.get("pinned_comment", "")])
    except Exception as e:
        print("Log skipped:", e)


def write_summary(meta, vid):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"### {meta['title']}\nhttps://youtube.com/shorts/{vid}\n\n"
                    f"- A: {meta['a']}\n- B: {meta['b']}\n- category: {meta['category']}\n"
                    f"- comment to pin by hand: {meta.get('pinned_comment', '')}\n")
    except Exception as e:
        print("Summary skipped:", e)


def render(meta):
    try:
        return build_video(meta, zoom=True)
    except Exception as e:
        print("Render with zoom failed, retrying without zoom:", type(e).__name__, e)
        return build_video(meta, zoom=False)


if __name__ == "__main__":
    if TEST:
        meta = {"title": "Tu préfères vivre sans téléphone ou sans pizza ? 🍕", "a": "vivre un an sans téléphone",
                "b": "ne plus jamais manger de pizza", "keyword_a": "smartphone", "keyword_b": "pizza",
                "cta": "A ou B ? Dis-le en commentaire !", "tags": ["tu préfères"], "category": "test",
                "episode": 1, "pinned_comment": ""}
        print("Test render ->", render(meta))
        raise SystemExit
    creds = get_creds()
    meta = make_script()
    print("Title:", meta["title"], "| Category:", meta["category"])
    video_path = render(meta)
    vid = upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(SEP.join([meta["title"], meta["category"], meta["a"], meta["b"]]) + "\n")
    write_log(meta, vid)
    write_summary(meta, vid)
