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

# Each video uses ONE challenge format. Formats rotate (never the same one twice in a row) and, once you have
# enough results in log.csv, formats that hold viewers better are picked a bit more often.
SUBJECTS = [
    "animals and their abilities", "the human body (general facts only, no medical advice)",
    "space and planets", "food and drinks around the world", "famous inventions and who made them",
    "oceans and sea creatures", "weather and natural phenomena", "everyday objects and how they work",
    "sports facts that never change", "big numbers and sizes of famous things", "birds and insects",
    "mountains, deserts and rivers", "how things are made", "space exploration firsts",
]
FORMATS = {
    "guess_fact": "GUESS THE ANSWER about ONE surprising fact (subject area: {sub}). The answer is short and single "
                  "(a number, an animal, a place or an object). Example hook: 'هل تستطيع تخمين أسرع حيوان على الأرض؟'",
    "guess_country": "GUESS THE COUNTRY from 2-3 clues (a landmark, a food, geography or the flag colors). The clues must "
                     "point to exactly ONE country and the answer is that country. Example hook: 'خمّن الدولة من هذه الألغاز الثلاثة!'",
    "math_riddle": "A very short MATH or LOGIC riddle that a person can solve in 5 seconds in their head (simple arithmetic "
                   "with a small trick). The hook plus the setup must contain the COMPLETE puzzle. It must have exactly "
                   "ONE numeric answer: put it in 'answer_number'. Example hook: 'هل تحل هذه المسألة في 5 ثوانٍ؟'",
    "true_false": "TRUE OR FALSE: one surprising claim that is clearly TRUE, or a common myth that is clearly FALSE, "
                  "according to well-established knowledge. The viewer decides. The answer is 'صحيح' or 'خطأ'. "
                  "Example hook: 'صحيح أم خطأ؟ الموز ينمو على أشجار!'",
    "what_next": "WHAT HAPPENS NEXT: a simple, safe everyday or physics situation with ONE certain, well-known outcome. "
                 "Never something dangerous to imitate. Example hook: 'ماذا سيحدث عندما ...؟'",
    "only_first": "THE ONLY / THE FIRST / THE BIGGEST: a challenge about something that is the only one, the first or the "
                  "record holder in a well-established, never-changing way (history, space exploration, sports, "
                  "inventions, nature). The question must contain the word 'الوحيد' or 'الأول' or 'الأكبر', and the reveal "
                  "must include the year or place if it exists. Example: 'ما هي الرياضة الوحيدة التي لُعبت على القمر؟' "
                  "(answer: الغولف, Apollo 14, 1971).",
    "unusual_fact": "UNUSUAL MYSTERY: a question about a TRUE fact from nature or history that few people know, "
                    "with a short answer. Example hook: 'هل تعرف سبب هذا الغموض؟'",
}
CTA_STYLES = {
    "comment_guess": "ask what they guessed, for example 'ما هو تخمينك؟ اكتبه في التعليقات؟'",
    "tag_friend": "challenge them to send it to a friend, for example 'هل سيعرف صديقك الإجابة؟ أرسله له الآن؟'",
    "subscribe_next": "invite them to come back for the next challenge, for example 'هل أنت مستعد لتحدي الغد؟ اشترك الآن؟'",
    "score_comment": "ask whether they got it right, for example 'هل أجبت بشكل صحيح؟ اكتب نعم أو لا؟'",
    "replay": "invite them to watch again and check, for example 'هل لاحظت كل شيء؟ شاهد مرة أخرى؟'",
}
FORMAT_RULES = {
    "true_false": "The claim must be clearly true or clearly false by well-established knowledge, and the declared answer must match.",
    "guess_country": "The clues must identify exactly ONE country, with no other country possible.",
    "math_riddle": "Solve the puzzle yourself step by step; the numeric answer must be exactly right.",
}
BANNED = {"اكتئاب", "قلق", "علاج", "تشخيص", "مرض", "سياسة", "الحرب", "الحروب", "الدين", "قمار", "مخدرات",
          "انتحار", "جنس", "إرهاب", "اضمن", "مضمون", "ترامب", "اسرائيل", "إسرائيل"}
ABSTRACT = ("concept", "idea", "fact", "trivia", "quiz", "question", "answer", "challenge", "knowledge")
COUNTDOWN_LINE = "خمسة... أربعة... ثلاثة... اثنان... واحد..."


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


LOG_HEADER = ["date", "title", "format", "subject", "hook", "cta_style", "words", "total_sec", "idea_score",
              "video_id", "views", "viewed_vs_swiped_pct", "avg_view_pct", "likes", "comments", "subs_gained",
              "title_alts", "pinned_comment"]


def _num(x):
    try:
        return float(str(x).replace("%", "").replace(",", ".").strip())
    except Exception:
        return None


def metric_rows():
    """Rows of log.csv that you filled in by hand. Needs at least 8 rows before it is used at all."""
    try:
        with open(LOG, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return [], None
    for col in ("viewed_vs_swiped_pct", "views"):
        have = [r for r in rows if _num(r.get(col)) is not None]
        if len(have) >= 8:
            return have, col
    return [], None


def format_weights():
    w = {k: 1.0 for k in FORMATS}
    w["only_first"] = 1.5  # starting prior: a "the only / the first" question got the most views so far; learned results replace it
    have, col = metric_rows()
    if not have:
        return w
    mean = sum(_num(r[col]) for r in have) / len(have)
    if mean <= 0:
        return w
    for k in FORMATS:
        vals = [_num(r[col]) for r in have if r.get("format") == k]
        if len(vals) >= 2:  # a format needs 2+ measured videos before its weight changes; weights stay in 0.5-2.0
            w[k] = min(2.0, max(0.5, (sum(vals) / len(vals)) / mean))
    return w


def pick_format():
    last = recent(1, 10)[-2:]
    w = format_weights()
    pool = [k for k in FORMATS if k not in last] or list(FORMATS)
    return random.choices(pool, weights=[w[k] for k in pool])[0]


def pick_cta_style():
    last = recent(4, 10)[-2:]
    pool = [k for k in CTA_STYLES if k not in last] or list(CTA_STYLES)
    # "come back tomorrow" gets double weight: almost all viewers are new, so returning viewers are the gap to close
    return random.choices(pool, weights=[2.0 if k == "subscribe_next" else 1.0 for k in pool])[0]


def next_episode():
    """Episode number of this challenge series (counts the Arabic challenge videos already in history.txt)."""
    return 1 + sum(1 for l in read_lines(HISTORY, 100000) if len(l.split(SEP)) >= 4)


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


def _line(meta, kind):
    return next(s["line"] for s in meta["scenes"] if s["kind"] == kind)


def fact_check(meta):
    prompt = f"""You are a strict fact-checker for short Arabic trivia videos. Check the facts, in any language.
Format: {meta.get('fmt', '')}
Title: {meta['title']}
Question: {_line(meta, 'hook')}
Clue: {_line(meta, 'setup')}
Answer and explanation: {_line(meta, 'reveal')}
Declared answer: {meta['answer']}
Extra fact: {_line(meta, 'extra')}
Checks:
1) ANSWER: is the answer to the question correct and unambiguous (only one reasonable answer)? {FORMAT_RULES.get(meta.get('fmt'), '')}
2) STABLE: is every claim a well-established fact that does not change over time and is not disputed? (No prices, no 'latest' records, no rankings that change.)
3) DETAILS: are all numbers and the extra fact correct? Mark false if any number is wrong, rounded misleadingly, or unsure.
4) SAFE: no politics, religion, medical advice, violence, gambling or sensitive claims.
Be strict: when unsure, fail the check. "confidence" = how sure you are (1-10) that everything is correct. Use 8 or more only when certain.
Return ONLY JSON: {{"answer_correct": true or false, "stable": true or false, "details_ok": true or false, "safe": true or false, "confidence": 1-10, "issue": "short reason"}}"""
    return gemini_json(prompt, temperature=0.0)


def verify_math(meta):
    """Second, independent solve of the riddle (the model does not see our answer). Numbers must match exactly."""
    try:
        want = float(meta.get("answer_number"))
        out = gemini_json("Solve this puzzle carefully, step by step in your head. Question: "
                          + _line(meta, "hook") + " " + _line(meta, "setup")
                          + ' Return ONLY JSON: {"answer_number": number}', temperature=0.0)
        return abs(float(out["answer_number"]) - want) < 1e-9
    except Exception:
        return False


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


def best_hook(cands):
    """Pick the strongest hook among the options by simple rules (short, a number or time limit, not like recent hooks)."""
    old = [_tokens(h) for h in recent(2)]
    best = None
    for c in cands:
        c = str(c).strip()
        if not c or not c.endswith(("؟", "?")) or words(c) > 10:
            continue
        t = _tokens(c)
        sc = (2 if re.search(r"\d|ثوان|ثانية|خمس", c) else 0) + (1 if words(c) <= 8 else 0)
        if any(t and len(t & o) / len(t | o) > 0.5 for o in old):
            sc -= 3
        sc += random.random() * 0.5
        if best is None or sc > best[0]:
            best = (sc, c)
    return best[1] if best else None


def best_title(cands):
    cs = [str(c).strip().strip('"').strip() for c in cands if str(c).strip()]
    cs = [c for c in cs if 8 <= len(c) <= 60]
    if not cs:
        return ""
    return max(cs, key=lambda c: (("؟" in c or "?" in c) + bool(re.search(r"\d|ثوان", c)) - len(c) / 200))


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
    opts = meta.get("hook_options")
    opts = opts if isinstance(opts, list) else []
    parts["hook"]["line"] = best_hook([parts["hook"]["line"]] + opts) or parts["hook"]["line"]
    alts = meta.get("title_alts")
    alts = [str(a) for a in alts] if isinstance(alts, list) else []
    title = best_title([meta.get("title", "")] + alts)
    answer = str(meta.get("answer", "")).strip()
    if not title or not answer:
        return None
    narration = " ".join(parts[k]["line"] for k in ("hook", "setup", "reveal", "extra", "cta"))
    wc = words(narration)
    if not (34 <= wc <= 80):
        print("Rejected: length", wc)
        return None
    if _tokens(narration + " " + title) & BANNED:
        print("Rejected by banned-word filter")
        return None
    if words(parts["hook"]["line"]) > 12:
        print("Rejected: hook too long")
        return None
    cta_line = parts["cta"]["line"].strip()
    if not cta_line.endswith(("؟", "?")):
        # the model often ends the call to action with "!" or "." : fix it instead of rejecting
        cta_line = cta_line.rstrip(" .!،,…\"'«»") + "؟"
        parts["cta"]["line"] = cta_line
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
                 "hook_line": parts["hook"]["line"],
                 "title_alts": [a.strip() for a in alts if a.strip() and a.strip() != title][:2],
                 "pinned_comment": str(meta.get("pinned_comment", "")).strip()[:200]})
    return meta


MIN_SCORE = 7


def score_idea(meta):
    """Rule-based craft score (0-8). It checks the writing rules only; it does not predict views."""
    pts = 0
    hook = meta["hook_line"]
    pts += words(hook) <= 10                                              # short hook
    pts += hook.rstrip().endswith(("؟", "?"))                             # a question to guess
    pts += 34 <= words(meta["script"]) <= 75                             # tight length
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
    fmt = pick_format()
    sub = random.choice(SUBJECTS)
    cta = pick_cta_style()
    print("Format:", fmt, "| CTA style:", cta)
    titles = [l.split(SEP)[0] for l in read_lines(HISTORY, 150)]
    hooks = recent(2)
    answers = recent(3)
    prompt = f"""You write viral YouTube Shorts in ARABIC for a quiz and challenge channel with MrBeast-style energy: a big claim, curiosity, a 5-second countdown, then the answer.
Write all spoken lines in simple Arabic that every Arab viewer understands (Modern Standard Arabic mixed with easy everyday words). Short sentences. Energetic, friendly voice talking to ONE viewer.
FORMAT FOR THIS VIDEO: {FORMATS[fmt].format(sub=sub)}
All facts must be TRUE, well established, stable over time and easy to verify. Never invent anything. No prices, no changing records or rankings, no politics, religion, medicine, violence or gambling. If you are not sure of a fact, pick a different one.
Do NOT reuse these titles: {titles}. Do NOT open like these recent hooks: {hooks}. Do NOT reuse these answers: {answers}.
Each "line" is the exact words the narrator says. Each "keyword" is an ENGLISH stock-footage search phrase (2-5 words, a concrete filmable scene such as 'cheetah running savanna', 'astronaut floating space', 'huge waterfall aerial', 'chalkboard math equations'). Never abstract words. Vary the shots.
Fields:
- hook: the challenge question, 5-10 words, ends with '؟', creates instant curiosity.
- hook_options: 3 different ways to ask the SAME challenge (4-9 words each, each ends with '؟'); one may use a number or 'في 5 ثوانٍ'.
- setup: 1-2 short sentences with the clue or the full puzzle, 10-20 words. Do NOT give the answer.
- reveal: the answer plus the one-line reason, 12-25 words.
- extra: one amazing related TRUE fact, 10-20 words.
- cta: 5-10 words, ends with '؟'. Style: {CTA_STYLES[cta]}.
- title: Arabic, max 55 characters, a challenge feel that starts with 'تحدي' when it fits, honest (no lies), for example 'تحدي الرياضة الوحيدة التي لُعبت على القمر!'; title_alts: 2 other title options with different wording.
- pinned_comment: a friendly comment the channel can pin (max 20 words) that invites replies.
- answer: the answer in 1-4 Arabic words. answer_number: the numeric answer for math riddles, otherwise null.
- tags: 6 to 8 tags, include 'تحدي', 'خمن', 'معلومات', 'shorts' and the specific subject.
Total narration (hook+setup+reveal+extra+cta) must be {'45-58' if fmt == 'only_first' else '34-48'} words.
Return ONLY JSON:
{{"title": "", "title_alts": ["", ""], "answer": "", "answer_number": null, "pinned_comment": "", "hook_options": ["", "", ""],
"hook": {{"line": "", "keyword": ""}}, "setup": {{"line": "", "keyword": ""}}, "reveal": {{"line": "", "keyword": ""}},
"extra": {{"line": "", "keyword": ""}}, "cta": {{"line": "", "keyword": ""}}, "tags": []}}"""
    for attempt in range(10):
        meta = parse_meta(gemini_json(prompt))
        if not meta:
            print("Bad structure, regenerating")
            continue
        meta["fmt"], meta["cta_style"], meta["subject"] = fmt, cta, (sub if fmt == "guess_fact" else "")
        sc = score_idea(meta)
        print("Idea score:", sc, "/ 8 | hook:", meta["hook_line"])
        if sc < MIN_SCORE:
            print("Rejected: idea score too low, regenerating")
            continue
        if fmt == "math_riddle" and not verify_math(meta):
            print("Rejected: independent math check disagrees")
            continue
        check = fact_check(meta)
        print("Fact-check:", check)
        if passes_check(check):
            meta["concept"] = fmt
            meta["episode"] = next_episode()
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


def make_sfx(total, ticks, ding_at, path="sfx.wav"):
    """Own generated sound effects (no copyright issues): a tick per countdown number and a bell on the answer."""
    sr = 44100
    out = np.zeros(int(sr * (total + 1)))

    def burst(t0, freq, dur, amp, decay):
        i = int(t0 * sr)
        if i >= len(out):
            return
        n = min(int(dur * sr), len(out) - i)
        tt = np.arange(n) / sr
        out[i:i + n] += amp * np.sin(2 * np.pi * freq * tt) * np.exp(-decay * tt)

    for k, t0 in enumerate(ticks):
        burst(t0, 1500 if k == len(ticks) - 1 else 1000, 0.14, 0.8, 28)
    if ding_at is not None:
        for f in (880, 1320, 1760):
            burst(ding_at, f, 1.1, 0.45, 4)
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.6 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
    return path


def countdown_step(adur):
    return max(adur, 4.0) / 5


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


def _episode_image(n):
    return _text_image(f"تحدي رقم {n}", 42, 95, (255, 255, 255, 235), stroke=4)


def _cta_image():
    return _text_image("تابعنا لتحدٍّ جديد كل يوم", 54, 115, (255, 255, 255, 255), stroke=5)


def build_overlays(video, timeline, episode=None):
    """Dark layer, brand tag, captions (3 words at a time, synced per scene), countdown digits, end card."""
    try:
        total = video.duration
        layers = [video,
                  ColorClip((W, H), color=(0, 0, 0)).set_opacity(0.30).set_duration(total),
                  ImageClip(_brand_image()).set_duration(total).set_position(("center", 150))]
        if episode:
            layers.append(ImageClip(_episode_image(episode)).set_duration(total).set_position(("center", 245)))
        for start, adur, sc in timeline:
            if sc["kind"] == "countdown":
                step = countdown_step(adur)
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
                          .set_position(("center", 350)))
        return CompositeVideoClip(layers).set_duration(total).set_audio(video.audio)
    except Exception as e:
        print("Overlays skipped:", e)
        return video


# ---------------------------------------------------------------- build
def _brand_bg():
    """Own background (navy gradient + faint question marks) for the countdown and end scenes."""
    img = Image.new("RGB", (W, H))
    px = ImageDraw.Draw(img)
    for y in range(H):
        k = y / H
        px.line([(0, y), (W, y)], fill=(int(12 + 20 * k), int(20 + 30 * k), int(52 + 40 * k)))
    d = ImageDraw.Draw(img, "RGBA")
    try:
        f = _font(520)
    except Exception:
        f = ImageFont.load_default()
    d.text((W // 2, H // 2), "?", font=f, fill=(255, 190, 40, 38), anchor="mm")
    img.save("brand_bg.png")
    return "brand_bg.png"


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
        if sc["kind"] in ("countdown", "cta"):
            clips.append(ImageClip(_brand_bg()).set_duration(dur))
            t += dur
            continue
        p = safe_clip(sc["keyword"], i, used) or last
        if not p:
            raise RuntimeError("No stock clips found")
        last = p
        clips.append(fit(VideoFileClip(p), dur))
        t += dur
    total = t
    voice = CompositeAudioClip(voices).set_duration(total)
    mix = add_music(voice, total)
    try:
        ticks, ding = [], None
        for st, adur, sc in timeline:
            if sc["kind"] == "countdown":
                ticks = [st + k * countdown_step(adur) for k in range(5)]
            elif sc["kind"] == "reveal":
                ding = st
        sfx = AudioFileClip(make_sfx(total, ticks, ding)).subclip(0, total).fx(afx.volumex, 0.5)
        mix = CompositeAudioClip([mix, sfx]).set_duration(total)
    except Exception as e:
        print("Sound effects skipped:", e)
    video = concatenate_videoclips(clips).set_audio(mix)
    video = build_overlays(video, timeline, meta.get("episode"))
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
            "title": (meta["title"] + (f" | تحدي رقم {meta['episode']}" if meta.get("episode") else ""))[:95],
            "description": ((f"تحدي رقم {meta['episode']}\n\n" if meta.get("episode") else "") + meta["script"]
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
    return resp["id"]


def write_log(meta, vid):
    """One row per video. Fill views / viewed_vs_swiped_pct / avg_view_pct / likes / comments / subs_gained by hand
    from YouTube Studio (edit log.csv on GitHub). After 8+ filled rows the formats are weighted by results."""
    try:
        if os.path.exists(LOG):
            with open(LOG, encoding="utf-8", newline="") as f:
                first = next(csv.reader(f), [])
            if first != LOG_HEADER:
                os.replace(LOG, "log_old.csv")  # old layout kept, new file starts clean
        new = not os.path.exists(LOG)
        with open(LOG, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_HEADER)
            w.writerow([datetime.date.today().isoformat(), meta["title"], meta["fmt"], meta.get("subject", ""),
                        meta["hook_line"], meta["cta_style"], len(meta["script"].split()), meta.get("audio_sec", ""),
                        score_idea(meta), vid, "", "", "", "", "", "",
                        " | ".join(meta.get("title_alts", [])), meta.get("pinned_comment", "")])
    except Exception as e:
        print("Log skipped:", e)


def write_summary(meta, vid):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"### {meta['title']}\nhttps://youtube.com/shorts/{vid}\n\n"
                    f"- format: {meta['fmt']} | CTA style: {meta['cta_style']} | idea score: {score_idea(meta)}/8\n"
                    f"- other titles: {' | '.join(meta.get('title_alts', []))}\n"
                    f"- comment to pin by hand: {meta.get('pinned_comment', '')}\n")
    except Exception as e:
        print("Summary skipped:", e)


if __name__ == "__main__":
    creds = get_creds()
    meta = make_script()
    print("Title:", meta["title"], "| Format:", meta["concept"])
    video_path = build_video(meta)
    vid = upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(SEP.join([meta["title"], meta["concept"], meta["hook_line"], meta["answer"], meta["cta_style"]]) + "\n")
    write_log(meta, vid)
    write_summary(meta, vid)
