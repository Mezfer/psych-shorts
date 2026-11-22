import os, json, random, asyncio, csv, re, datetime, wave, time
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

# تحدّي التخمين — "game" Shorts: 3 questions (easy / medium / hard), 8 seconds each, a draining timer bar, three
# answer buttons and a green check on the right answer. Questions come from a fixed country table (flags from
# flagcdn.com), so every answer is correct by construction: no AI fact-checking needed.

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
TEST = os.environ.get("TEST_RENDER") == "1"
BRAND = os.environ.get("BRAND", "تحدّي التخمين")
VOICE = os.environ.get("VOICE", "ar-SA-HamedNeural")  # other good options: ar-EG-SalmaNeural, ar-EG-ShakirNeural
PRIVACY = os.environ.get("PRIVACY", "public")
HISTORY = "history.txt"
LOG = "log.csv"
SEP = " || "
THINK = 8.0  # seconds to answer
FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# code, Arabic name, Arabic capital (None = disputed / several capitals: no capital question), continent, tier (1 easy .. 3 hard)
COUNTRIES = [
    ("eg", "مصر", "القاهرة", "africa", 1), ("sa", "السعودية", "الرياض", "asia", 1),
    ("ma", "المغرب", "الرباط", "africa", 1), ("dz", "الجزائر", "الجزائر العاصمة", "africa", 1),
    ("tn", "تونس", "تونس", "africa", 1), ("ae", "الإمارات", "أبوظبي", "asia", 1),
    ("qa", "قطر", "الدوحة", "asia", 1), ("kw", "الكويت", "مدينة الكويت", "asia", 1),
    ("iq", "العراق", "بغداد", "asia", 1), ("sy", "سوريا", "دمشق", "asia", 1),
    ("lb", "لبنان", "بيروت", "asia", 1), ("jo", "الأردن", "عمّان", "asia", 1),
    ("ly", "ليبيا", "طرابلس", "africa", 1), ("sd", "السودان", "الخرطوم", "africa", 1),
    ("ye", "اليمن", "صنعاء", "asia", 1), ("om", "عُمان", "مسقط", "asia", 1),
    ("bh", "البحرين", "المنامة", "asia", 1), ("tr", "تركيا", "أنقرة", "asia", 1),
    ("fr", "فرنسا", "باريس", "europe", 1), ("de", "ألمانيا", "برلين", "europe", 1),
    ("it", "إيطاليا", "روما", "europe", 1), ("es", "إسبانيا", "مدريد", "europe", 1),
    ("gb", "بريطانيا", "لندن", "europe", 1), ("ru", "روسيا", "موسكو", "europe", 1),
    ("br", "البرازيل", "برازيليا", "samerica", 1), ("ar", "الأرجنتين", "بوينس آيرس", "samerica", 1),
    ("us", "الولايات المتحدة", "واشنطن", "namerica", 1), ("ca", "كندا", "أوتاوا", "namerica", 1),
    ("mx", "المكسيك", "مكسيكو سيتي", "namerica", 1), ("jp", "اليابان", "طوكيو", "asia", 1),
    ("cn", "الصين", "بكين", "asia", 1), ("in", "الهند", "نيودلهي", "asia", 1),
    ("au", "أستراليا", "كانبيرا", "oceania", 1), ("pt", "البرتغال", "لشبونة", "europe", 1),
    ("kr", "كوريا الجنوبية", "سيول", "asia", 1), ("gr", "اليونان", "أثينا", "europe", 1),
    # tier 2
    ("mr", "موريتانيا", "نواكشوط", "africa", 2), ("so", "الصومال", "مقديشو", "africa", 2),
    ("nl", "هولندا", "أمستردام", "europe", 2), ("be", "بلجيكا", "بروكسل", "europe", 2),
    ("ch", "سويسرا", "برن", "europe", 2), ("se", "السويد", "ستوكهولم", "europe", 2),
    ("no", "النرويج", "أوسلو", "europe", 2), ("pl", "بولندا", "وارسو", "europe", 2),
    ("ua", "أوكرانيا", "كييف", "europe", 2), ("at", "النمسا", "فيينا", "europe", 2),
    ("dk", "الدنمارك", "كوبنهاغن", "europe", 2), ("fi", "فنلندا", "هلسنكي", "europe", 2),
    ("ie", "أيرلندا", "دبلن", "europe", 2), ("cz", "التشيك", "براغ", "europe", 2),
    ("hu", "المجر", "بودابست", "europe", 2), ("ro", "رومانيا", "بوخارست", "europe", 2),
    ("hr", "كرواتيا", "زغرب", "europe", 2), ("rs", "صربيا", "بلغراد", "europe", 2),
    ("co", "كولومبيا", "بوغوتا", "samerica", 2), ("cl", "تشيلي", "سانتياغو", "samerica", 2),
    ("pe", "بيرو", "ليما", "samerica", 2), ("ve", "فنزويلا", "كاراكاس", "samerica", 2),
    ("cu", "كوبا", "هافانا", "namerica", 2), ("ng", "نيجيريا", "أبوجا", "africa", 2),
    ("gh", "غانا", "أكرا", "africa", 2), ("ke", "كينيا", "نيروبي", "africa", 2),
    ("et", "إثيوبيا", "أديس أبابا", "africa", 2), ("sn", "السنغال", "داكار", "africa", 2),
    ("za", "جنوب أفريقيا", None, "africa", 2), ("th", "تايلاند", "بانكوك", "asia", 2),
    ("vn", "فيتنام", "هانوي", "asia", 2), ("my", "ماليزيا", "كوالالمبور", "asia", 2),
    ("sg", "سنغافورة", "سنغافورة", "asia", 2), ("ph", "الفلبين", "مانيلا", "asia", 2),
    ("id", "إندونيسيا", "جاكرتا", "asia", 2), ("pk", "باكستان", "إسلام آباد", "asia", 2),
    ("ir", "إيران", "طهران", "asia", 2), ("af", "أفغانستان", "كابول", "asia", 2),
    ("bd", "بنغلاديش", "دكا", "asia", 2), ("nz", "نيوزيلندا", "ويلينغتون", "oceania", 2),
    ("is", "آيسلندا", "ريكيافيك", "europe", 2), ("kz", "كازاخستان", "أستانا", "asia", 2),
    # tier 3
    ("dj", "جيبوتي", "جيبوتي", "africa", 3), ("km", "جزر القمر", "موروني", "africa", 3),
    ("bt", "بوتان", "تيمفو", "asia", 3), ("np", "نيبال", "كاتماندو", "asia", 3),
    ("la", "لاوس", "فيينتيان", "asia", 3), ("kh", "كمبوديا", "بنوم بنه", "asia", 3),
    ("mm", "ميانمار", "نايبيداو", "asia", 3), ("mg", "مدغشقر", "أنتاناناريفو", "africa", 3),
    ("mz", "موزمبيق", "مابوتو", "africa", 3), ("ao", "أنغولا", "لواندا", "africa", 3),
    ("zm", "زامبيا", "لوساكا", "africa", 3), ("zw", "زيمبابوي", "هراري", "africa", 3),
    ("ug", "أوغندا", "كمبالا", "africa", 3), ("tz", "تنزانيا", "دودوما", "africa", 3),
    ("cm", "الكاميرون", "ياوندي", "africa", 3), ("ml", "مالي", "باماكو", "africa", 3),
    ("ne", "النيجر", "نيامي", "africa", 3), ("td", "تشاد", "نجامينا", "africa", 3),
    ("er", "إريتريا", "أسمرة", "africa", 3), ("si", "سلوفينيا", "ليوبليانا", "europe", 3),
    ("sk", "سلوفاكيا", "براتيسلافا", "europe", 3), ("bg", "بلغاريا", "صوفيا", "europe", 3),
    ("ee", "إستونيا", "تالين", "europe", 3), ("lv", "لاتفيا", "ريغا", "europe", 3),
    ("lt", "ليتوانيا", "فيلنيوس", "europe", 3), ("mt", "مالطا", "فاليتا", "europe", 3),
    ("lu", "لوكسمبورغ", "لوكسمبورغ", "europe", 3), ("ge", "جورجيا", "تبليسي", "asia", 3),
    ("am", "أرمينيا", "يريفان", "asia", 3), ("az", "أذربيجان", "باكو", "asia", 3),
    ("py", "باراغواي", "أسونسيون", "samerica", 3), ("uy", "الأوروغواي", "مونتيفيديو", "samerica", 3),
    ("ec", "الإكوادور", "كيتو", "samerica", 3), ("cr", "كوستاريكا", "سان خوسيه", "namerica", 3),
    ("pa", "بنما", "بنما", "namerica", 3), ("jm", "جامايكا", "كينغستون", "namerica", 3),
    ("fj", "فيجي", "سوفا", "oceania", 3), ("pg", "بابوا غينيا الجديدة", "بورت مورسبي", "oceania", 3),
]
CBY = {c[0]: c for c in COUNTRIES}
# flags that look almost the same must never appear together as answer and wrong option
LOOKALIKES = [{"td", "ro"}, {"nl", "lu"}, {"sk", "si"}, {"ie", "ci"}, {"id", "pl"}, {"ye", "sy"}, {"eg", "sy"},
              {"eg", "ye"}, {"eg", "iq"}, {"sy", "iq"}, {"ye", "iq"}, {"ml", "sn"}, {"kw", "ps"}]

LEVELS = [(1, "سهل", (46, 204, 113)), (2, "متوسط", (255, 159, 28)), (3, "صعب", (231, 76, 60))]
LETTERS = ["أ", "ب", "ج"]
Q_TEXT = {
    "flag": "أي دولة يمثلها هذا العلم؟",
    "capital": "ما هي عاصمة {name}؟",
    "which_flag": "أي علم يخص دولة {name}؟",
}
INTROS = [
    "ثلاثة أسئلة، سهل ومتوسط وصعب. لديك ثماني ثوانٍ لكل سؤال. هل تستطيع إكمالها؟",
    "هل تصل إلى المستوى الثالث؟ ثلاثة أسئلة وثماني ثوانٍ لكل سؤال. هيا نبدأ!",
    "تحدي اليوم ثلاثة مستويات. سهل، ثم متوسط، ثم صعب. جاهز؟",
]
OUTROS = [
    ("كم أجبت بشكل صحيح من ثلاثة؟ اكتب نتيجتك في التعليقات!", "score_comment"),
    ("هل وصلت إلى المستوى الصعب؟ اكتب نتيجتك وأرسل التحدي لصديقك!", "tag_friend"),
    ("تحدٍّ جديد كل يوم. اشترك حتى لا يفوتك التحدي القادم، واكتب نتيجتك!", "subscribe_next"),
]
TITLES = [
    "تحدّي {kind}: هل تصل إلى المستوى الصعب؟",
    "ثلاثة أسئلة: سهل ومتوسط وصعب! هل تنجح في {kind}؟",
    "اختبر معلوماتك في {kind} ⏱️ ثماني ثوانٍ لكل سؤال",
    "هل تعرف كل هذه {kind}؟ تحدٍّ من ثلاثة مستويات",
]


def read_lines(path, n):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()][-n:]


def recent(idx, n=60):
    out = []
    for l in read_lines(HISTORY, n):
        p = l.split(SEP)
        if len(p) > idx:
            out.append(p[idx])
    return out


def next_episode():
    return 1 + sum(1 for l in read_lines(HISTORY, 100000) if len(l.split(SEP)) >= 4)


# ---------------------------------------------------------------- questions
def _lookalike(a, b):
    return any({a, b} <= s for s in LOOKALIKES)


def _recent_names():
    names = set()
    for sig in " ".join(recent(3, 60)).split("؛"):
        if ":" in sig:
            names.add(sig.split(":", 1)[1].strip())
    return names


def make_question(level, tier, used_codes, recent_names):
    pool = [c for c in COUNTRIES if c[4] == tier and c[0] not in used_codes and c[1] not in recent_names]
    if len(pool) < 3:
        pool = [c for c in COUNTRIES if c[4] == tier and c[0] not in used_codes]
    ans = random.choice(pool)
    kinds = ["flag", "capital", "which_flag"]
    if ans[2] is None:
        kinds.remove("capital")
    if level == 1:
        kinds = [k for k in kinds if k != "which_flag"] or ["flag"]
    kind = random.choice(kinds)
    same = [c for c in COUNTRIES if c[0] != ans[0] and c[3] == ans[3] and not _lookalike(ans[0], c[0])
            and (kind != "capital" or c[2] is not None)]
    if len(same) < 2:
        same = [c for c in COUNTRIES if c[0] != ans[0] and not _lookalike(ans[0], c[0])
                and (kind != "capital" or c[2] is not None)]
    if level == 1:                       # easy: wrong options come from the well-known countries
        easy = [c for c in same if c[4] == 1]
        same = easy if len(easy) >= 2 else same
    wrong = []
    for c in random.sample(same, len(same)):
        if all(not _lookalike(c[0], w[0]) for w in wrong):
            wrong.append(c)
        if len(wrong) == 2:
            break
    if len(wrong) < 2:
        return None
    opts = [ans] + wrong
    random.shuffle(opts)
    correct = opts.index(ans)
    q = {"kind": kind, "level": level, "answer": ans, "options": opts, "correct": correct,
         "text": Q_TEXT[kind].format(name=ans[1]),
         "signature": f"{kind}:{ans[1]}"}
    if kind == "flag":
        q["option_texts"] = [o[1] for o in opts]
        q["reveal"] = f"الإجابة الصحيحة: {ans[1]}"
    elif kind == "capital":
        q["option_texts"] = [o[2] for o in opts]
        q["reveal"] = f"عاصمة {ans[1]} هي {ans[2]}"
    else:
        q["option_texts"] = None
        q["reveal"] = f"هذا هو علم {ans[1]}"
    q["codes"] = [o[0] for o in opts] if kind == "which_flag" else [ans[0]]
    return q


def _fake_flag(path, code):
    rnd = random.Random(code)
    img = Image.new("RGB", (640, 427), tuple(rnd.randint(40, 230) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for k in range(3):
        d.rectangle([0, k * 142, 640, k * 142 + 70], fill=tuple(rnd.randint(40, 230) for _ in range(3)))
    img.save(path)
    return path


def get_flag(code):
    path = f"flag_{code}.png"
    if os.path.exists(path):
        return path
    if TEST:
        return _fake_flag(path, code)
    for k in range(3):
        try:
            r = requests.get(f"https://flagcdn.com/w640/{code}.png", timeout=30)
            if r.status_code == 200 and len(r.content) > 300:
                with open(path, "wb") as f:
                    f.write(r.content)
                Image.open(path).convert("RGBA")   # must be a real image
                return path
        except Exception as e:
            print("Flag download failed", code, e)
        time.sleep(2 * (k + 1))
    return None


def make_script():
    recent_names = _recent_names()
    for attempt in range(12):
        used, qs, ok = set(), [], True
        for level, _, _ in LEVELS:
            q = make_question(level, level, used, recent_names)
            if not q:
                ok = False
                break
            used.update(o[0] for o in q["options"])
            qs.append(q)
        if not ok:
            continue
        need = {c for q in qs for c in q["codes"]}
        if all(get_flag(c) for c in need):
            break
    else:
        raise RuntimeError("Could not build a question set (flags unavailable?)")
    kinds = {q["kind"] for q in qs}
    kind_name = ("العواصم" if kinds == {"capital"} else "الأعلام" if "capital" not in kinds else "الأعلام والعواصم")
    intro = random.choice(INTROS)
    last_cta = recent(4, 10)[-2:]
    outros = [o for o in OUTROS if o[1] not in last_cta] or OUTROS
    outro, cta_style = random.choice(outros)
    old_titles = [l.split(SEP)[0] for l in read_lines(HISTORY, 100)]
    titles = [t.format(kind=kind_name) for t in TITLES]
    titles = [t for t in titles if t not in old_titles] or titles
    title = random.choice(titles)
    episode = next_episode()
    narration = " ".join([intro] + [q["text"] for q in qs] + [outro])
    return {"title": title, "episode": episode, "questions": qs, "intro": intro, "outro": outro,
            "cta_style": cta_style, "hook_line": qs[0]["text"], "kind_name": kind_name,
            "answers": "؛".join(q["signature"] for q in qs), "script": narration,
            "tags": ["تحدي", "خمن", "أعلام", "عواصم", "خمن الدولة", "ألغاز", "معلومات عامة", "shorts"],
            "pinned_comment": "اكتب نتيجتك من 3 في التعليقات، ومن وصل إلى المستوى الصعب؟"}


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
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.5 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
    return path


def add_music(voice, dur):
    m = AudioFileClip(make_music(dur + 1)).subclip(0, dur)
    m = m.fx(afx.volumex, 0.14).fx(afx.audio_fadein, 0.5).fx(afx.audio_fadeout, 1.5)
    return CompositeAudioClip([voice, m])


def make_sfx(total, ticks, dings, pops, path="sfx.wav"):
    """Own generated effects: a pop when a question appears, a tick every second, a bell on each right answer."""
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
        burst(t0, 520, 0.18, 0.7, 20)
        burst(t0 + 0.08, 780, 0.18, 0.6, 20)
    for t0, high in ticks:
        burst(t0, 1500 if high else 1000, 0.12, 0.7, 30)
    for t0 in dings:
        for f in (880, 1320, 1760):
            burst(t0, f, 1.0, 0.4, 4)
    out /= max(np.abs(out).max(), 1e-9)
    pcm = (out * 0.6 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.repeat(pcm[:, None], 2, axis=1).tobytes())
    return path


def _fake_voice(text, path):
    sr = 22050
    n = int(sr * max(0.8, len(text.split()) * 0.42))
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())


async def _tts_one(text, path):
    if TEST:
        _fake_voice(text, path)
        return
    try:
        await edge_tts.Communicate(text, VOICE, rate="+4%").save(path)
    except Exception as e:
        print("edge-tts failed, using gTTS:", e)
        from gtts import gTTS
        gTTS(text, lang="ar").save(path)


async def _tts_all(lines):
    paths = []
    for i, line in enumerate(lines):
        p = f"voice{i}.mp3"
        await _tts_one(line, p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- drawing
def _font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size, layout_engine=_layout())
    raise RuntimeError("No font found")


def _fit_lines(d, text, size, maxw, maxlines, minsize=30):
    while size >= minsize:
        font = _font(size)
        lines, line = [], ""
        for w in text.split():
            test = (line + " " + w).strip()
            if line and _tw(d, test, font) > maxw:
                lines.append(line)
                line = w
            else:
                line = test
        if line:
            lines.append(line)
        if len(lines) <= maxlines and all(_tw(d, l, font) <= maxw for l in lines):
            return font, lines, size
        size -= 4
    return _font(minsize), text.split(), minsize


def _center_text(d, cx, cy, text, size, maxw, maxlines, fill, stroke=0, stroke_fill=(0, 0, 0)):
    font, lines, size = _fit_lines(d, text, size, maxw, maxlines)
    lh = int(size * 1.25)
    y = cy - lh * len(lines) / 2
    for ln in lines:
        tw = _tw(d, ln, font)
        _draw(d, (cx - tw / 2, y), ln, font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        y += lh


def _bg():
    ys = np.linspace(0, 1, H)[:, None, None]
    a, b = np.array([14, 18, 52]), np.array([44, 24, 92])
    arr = (a * (1 - ys) + b * ys)
    arr = np.broadcast_to(arr, (H, W, 3)).astype(np.uint8)
    return Image.fromarray(arr).convert("RGBA")


def _paste_contain(img, flag_path, box, border=6, radius=22):
    x0, y0, x1, y1 = box
    fl = Image.open(flag_path).convert("RGBA")
    s = min((x1 - x0 - 2 * border) / fl.width, (y1 - y0 - 2 * border) / fl.height)
    fl = fl.resize((max(1, int(fl.width * s)), max(1, int(fl.height * s))), Image.LANCZOS)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    fx, fy = cx - fl.width // 2, cy - fl.height // 2
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([fx - border, fy - border, fx + fl.width + border, fy + fl.height + border],
                        radius=radius, fill=(255, 255, 255, 255))
    img.alpha_composite(fl, (fx, fy))


# layout (kept away from the Shorts interface: right edge icons and the bottom 270 px)
BX, BW = 70, 830
OPT_Y = {"text": [1060, 1215, 1370], "flags": [640, 905, 1170]}
OPT_H = {"text": 130, "flags": 225}
TIMER_Y = 1545


def _header(img, episode):
    d = ImageDraw.Draw(img)
    _center_text(d, W // 2, 105, BRAND, 50, 900, 1, (255, 190, 40, 255), stroke=3)
    _center_text(d, W // 2, 178, f"تحدي رقم {episode}", 40, 700, 1, (255, 255, 255, 235), stroke=3)


def _level_chip(img, level, idx, total=3):
    name, col = next((n, c) for l, n, c in LEVELS if l == level)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([W // 2 - 300, 245, W // 2 + 300, 345], radius=50, fill=col + (255,))
    _center_text(d, W // 2, 295, f"المستوى {level} · {name}", 54, 540, 1, (255, 255, 255, 255), stroke=3)
    _center_text(d, W // 2, 385, f"السؤال {idx} من {total}", 36, 500, 1, (200, 210, 255, 255))


def question_image(q, idx, episode):
    img = _bg()
    _header(img, episode)
    _level_chip(img, q["level"], idx)
    d = ImageDraw.Draw(img)
    mode = "flags" if q["kind"] == "which_flag" else "text"
    _center_text(d, W // 2, 520 if mode == "text" else 500, q["text"], 66, 940, 2, (255, 255, 255, 255), stroke=5)
    if mode == "text":
        _paste_contain(img, get_flag(q["answer"][0]), (W // 2 - 340, 600, W // 2 + 340, 1010))
    for i in range(3):
        y, h = OPT_Y[mode][i], OPT_H[mode]
        d.rounded_rectangle([BX, y, BX + BW, y + h], radius=34, fill=(255, 255, 255, 255))
        r = h // 2 - 18
        cx, cy = BX + BW - r - 22, y + h // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(60, 52, 140, 255))
        _center_text(d, cx, cy, LETTERS[i], int(r * 1.15), r * 2, 1, (255, 255, 255, 255))
        if mode == "text":
            _center_text(d, BX + (BW - 2 * r - 44) // 2, cy, q["option_texts"][i], 58, BW - 2 * r - 120, 1, (30, 30, 60, 255))
        else:
            _paste_contain(img, get_flag(q["codes"][i]), (BX + 30, y + 14, BX + BW - 2 * r - 60, y + h - 14), border=5, radius=16)
    return np.array(img.convert("RGB"))


def reveal_overlay(q):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    mode = "flags" if q["kind"] == "which_flag" else "text"
    for i in range(3):
        y, h = OPT_Y[mode][i], OPT_H[mode]
        if i == q["correct"]:
            d.rounded_rectangle([BX - 8, y - 8, BX + BW + 8, y + h + 8], radius=40, fill=(46, 204, 113, 120),
                                outline=(46, 204, 113, 255), width=12)
            cx, cy = BX + 85, y + h // 2          # check mark, drawn with lines (no font needed)
            d.ellipse([cx - 48, cy - 48, cx + 48, cy + 48], fill=(46, 204, 113, 255), outline=(255, 255, 255, 255), width=6)
            d.line([(cx - 24, cy + 2), (cx - 6, cy + 22), (cx + 28, cy - 20)], fill=(255, 255, 255, 255), width=14, joint="curve")
        else:
            d.rounded_rectangle([BX, y, BX + BW, y + h], radius=34, fill=(10, 10, 30, 175))
    return np.array(img)


def timer_strip(remaining, total=THINK):
    img = Image.new("RGBA", (W, 110), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    frac = max(0.0, min(1.0, remaining / total))
    col = (46, 204, 113) if frac > 0.5 else (255, 159, 28) if frac > 0.25 else (231, 76, 60)
    x0, x1, y0, y1 = BX + 130, BX + BW, 32, 80
    d.rounded_rectangle([x0, y0, x1, y1], radius=24, fill=(255, 255, 255, 60))
    if frac > 0.02:
        d.rounded_rectangle([x1 - int((x1 - x0) * frac), y0, x1, y1], radius=24, fill=col + (255,))
    num = str(max(1, int(np.ceil(remaining - 1e-6))))
    f = _font(78)
    d.ellipse([BX, 6, BX + 112, 104], fill=(255, 255, 255, 255))
    tw = d.textlength(num, font=f)
    d.text((BX + 56 - tw / 2, 55 - 42), num, font=f, fill=(30, 30, 60, 255))
    return np.array(img)


def intro_image(episode, kind_name):
    img = _bg()
    _header(img, episode)
    d = ImageDraw.Draw(img)
    _center_text(d, W // 2, 640, f"تحدّي {kind_name}", 118, 940, 2, (255, 255, 255, 255), stroke=7)
    _center_text(d, W // 2, 900, "ثماني ثوانٍ لكل سؤال", 52, 900, 1, (255, 214, 10, 255), stroke=4)
    for i, (lvl, name, col) in enumerate(LEVELS):
        y = 1040 + i * 150
        d.rounded_rectangle([W // 2 - 330, y, W // 2 + 330, y + 115], radius=58, fill=col + (255,))
        _center_text(d, W // 2, y + 58, f"المستوى {lvl} · {name}", 58, 600, 1, (255, 255, 255, 255), stroke=3)
    return np.array(img.convert("RGB"))


def outro_image(episode):
    img = _bg()
    _header(img, episode)
    d = ImageDraw.Draw(img)
    _center_text(d, W // 2, 700, "كم نتيجتك؟", 130, 940, 1, (255, 255, 255, 255), stroke=7)
    _center_text(d, W // 2, 900, "من 3", 150, 600, 1, (255, 214, 10, 255), stroke=8)
    _center_text(d, W // 2, 1130, "اكتبها في التعليقات", 66, 900, 1, (255, 255, 255, 255), stroke=5)
    _center_text(d, W // 2, 1290, "تحدٍّ جديد كل يوم · اشترك الآن", 52, 940, 1, (255, 190, 40, 255), stroke=4)
    return np.array(img.convert("RGB"))


# ---------------------------------------------------------------- build
def build_video(meta):
    qs = meta["questions"]
    lines = [meta["intro"]]
    for q in qs:
        lines.append(q["text"] + "... " + "، ".join(q["option_texts"] or ["الأول", "الثاني", "الثالث"]))
        lines.append(q["reveal"])
    lines.append(meta["outro"])
    paths = asyncio.run(_tts_all(lines))
    auds = [AudioFileClip(p) for p in paths]

    voices, clips, ticks, dings, pops = [], [], [], [], []
    t = 0.0
    # intro
    d_intro = auds[0].duration + 0.5
    clips.append(ImageClip(intro_image(meta["episode"], meta["kind_name"])).set_duration(d_intro))
    voices.append(auds[0].set_start(0.15))
    t += d_intro
    for i, q in enumerate(qs):
        qa, ra = auds[1 + 2 * i], auds[2 + 2 * i]
        think = max(THINK, float(np.ceil((qa.duration + 2.5) * 2) / 2))
        d_rev = ra.duration + 1.2
        base = ImageClip(question_image(q, i + 1, meta["episode"])).set_duration(think + d_rev)
        layers = [base]
        steps = int(think * 2)
        for k in range(steps):
            rem = think - k * 0.5
            layers.append(ImageClip(timer_strip(rem, think)).set_start(k * 0.5).set_duration(0.5)
                          .set_position((0, TIMER_Y)))
        layers.append(ImageClip(reveal_overlay(q)).set_start(think).set_duration(d_rev))
        layers.append(ImageClip(timer_strip(0.0, think)).set_start(think).set_duration(d_rev).set_position((0, TIMER_Y)))
        clips.append(CompositeVideoClip(layers, size=(W, H)).set_duration(think + d_rev))
        voices.append(qa.set_start(t + 0.25))
        voices.append(ra.set_start(t + think + 0.1))
        pops.append(t)
        for k in range(int(think)):
            left = think - k
            ticks.append((t + k, left <= 3))
        dings.append(t + think)
        t += think + d_rev
    # outro
    d_out = auds[-1].duration + 1.0
    clips.append(ImageClip(outro_image(meta["episode"])).set_duration(d_out))
    voices.append(auds[-1].set_start(t + 0.2))
    t += d_out
    total = t

    voice = CompositeAudioClip(voices).set_duration(total)
    mix = add_music(voice, total)
    try:
        sfx = AudioFileClip(make_sfx(total, ticks, dings, pops)).subclip(0, total).fx(afx.volumex, 0.5)
        mix = CompositeAudioClip([mix, sfx]).set_duration(total)
    except Exception as e:
        print("Sound effects skipped:", e)
    video = concatenate_videoclips(clips).set_audio(mix)
    video.write_videofile("out.mp4", fps=24, codec="libx264", audio_codec="aac",
                          audio_bitrate="192k", preset="veryfast", threads=2, logger=None,
                          ffmpeg_params=["-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
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
            "title": f"{meta['title']} | تحدي رقم {meta['episode']}"[:95],
            "description": (f"تحدي رقم {meta['episode']}\n\n"
                            "ثلاثة أسئلة: سهل، متوسط، صعب. لديك 8 ثوانٍ لكل سؤال!\n"
                            "كم أجبت بشكل صحيح من 3؟ اكتب نتيجتك في التعليقات.\n"
                            "تحدٍّ جديد كل يوم. اشترك في القناة حتى لا يفوتك.\n\n"
                            "#Shorts #تحدي #خمن_الدولة #أعلام #عواصم #معلومات_عامة"),
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


LOG_HEADER = ["date", "title", "format", "subject", "hook", "cta_style", "words", "total_sec", "idea_score",
              "video_id", "views", "viewed_vs_swiped_pct", "avg_view_pct", "likes", "comments", "subs_gained",
              "title_alts", "pinned_comment"]


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
            w.writerow([datetime.date.today().isoformat(), meta["title"], "game_" + meta["kind_name"],
                        meta["answers"], meta["hook_line"], meta["cta_style"], len(meta["script"].split()),
                        meta.get("audio_sec", ""), "", vid, "", "", "", "", "", "", "", meta.get("pinned_comment", "")])
    except Exception as e:
        print("Log skipped:", e)


def write_summary(meta, vid):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"### {meta['title']}\nhttps://youtube.com/shorts/{vid}\n\n"
                    f"- questions: {meta['answers']}\n- CTA style: {meta['cta_style']}\n"
                    f"- comment to pin by hand: {meta.get('pinned_comment', '')}\n")
    except Exception as e:
        print("Summary skipped:", e)


if __name__ == "__main__":
    if TEST:
        meta = make_script()
        meta["episode"] = 46
        print("Test render ->", build_video(meta))
        raise SystemExit
    creds = get_creds()
    meta = make_script()
    print("Title:", meta["title"], "| Questions:", meta["answers"])
    video_path = build_video(meta)
    vid = upload(video_path, meta, creds)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(SEP.join([meta["title"], "game_" + meta["kind_name"], meta["hook_line"], meta["answers"],
                          meta["cta_style"]]) + "\n")
    write_log(meta, vid)
    write_summary(meta, vid)
