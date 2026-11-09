"""
Stage 3 of the viral-clip pipeline.

Renders scored segments to clips in a chosen aspect ratio:
  - 16:9 (default) -> landscape, full frame kept
  - 9:16           -> vertical, dynamic speaker-tracking crop (YuNet + snap-on-cut)
  - 1:1            -> square
Captions are word-by-word CapCut style, burnt in via the ffmpeg ASS engine,
with selectable style presets. Optional --tighten removes filler words and
long silences (captions are re-timed automatically). Every clip gets a PNG
poster and a .meta.json with a ready-to-post title, description and hashtags.

No external/paid APIs. ffmpeg + OpenCV only. Usage:
    python render_clips.py [workdir] [--aspect 16:9|9:16|1:1]
                           [--out DIR] [--style NAME]
                           [--tighten] [--loudnorm] [--preview]
Output directory (default ./clips) can also be set with $CHOPIFY_OUT.
"""
import sys
import os
import re
import json
import argparse
import subprocess
import math
import urllib.request
from pathlib import Path

import branding
import tighten  # local module: filler/silence removal

# aspect -> (target_w, target_h, caption_font_size, caption_margin_v)
ASPECTS = {
    "16:9": (1920, 1080, 72, 95),
    "9:16": (1080, 1920, 96, 300),
    "1:1":  (1080, 1080, 82, 120),
}
DEFAULT_ASPECT = "9:16"

# --preview drafts: small, fast, no poster/meta. Caption font/margins scale down
# proportionally from the full-size ASPECTS values (height is the anchor).
PREVIEW = {
    "16:9": (854, 480),
    "9:16": (480, 854),
    "1:1":  (480, 480),
}

# Caption presets. Colors are #RRGGBB here; build_ass accepts both hex and the
# ASS strings returned by branding.load_brand. Font size and margin are measured
# at 1080px output height, so None means "use this aspect's base value".
CAPTION_STYLES = {
    "default": {
        "font": "Arial Black", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FFFF00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 3,
        "position": "bottom", "outline": 5, "shadow": 2, "background": "none",
        "background_color": "#000000", "background_opacity": 70,
        "animation": "none", "highlight_mode": "word", "emoji": False,
        "emphasis": False,
    },
    "clean": {
        "font": "Arial", "weight": 0, "color": "#FFFFFF",
        "highlight": "#FFB000", "emphasis_color": "#FFD400",
        "outline_color": "#101010", "uppercase": False, "words_per_cue": 3,
        "position": "bottom", "outline": 2, "shadow": 1, "background": "none",
        "animation": "none", "highlight_mode": "word",
    },
    "minimal": {
        "font": "Arial", "weight": 0, "color": "#FFFFFF",
        "highlight": "#D4EF83", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 5,
        "position": "bottom", "outline": 8, "shadow": 0, "background": "box",
        "background_color": "#111111", "background_opacity": 65,
        "animation": "none", "highlight_mode": "word",
    },
    "bold": {
        "font": "Impact", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FF7A00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": True, "words_per_cue": 2,
        "position": "bottom", "outline": 8, "shadow": 2, "background": "none",
        "animation": "pop", "highlight_mode": "word",
    },
    "boxed": {
        "font": "Arial Black", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FFFF00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": True, "words_per_cue": 3,
        "position": "bottom", "outline": 10, "shadow": 0, "background": "box",
        "background_color": "#000000", "background_opacity": 80,
        "animation": "none", "highlight_mode": "word",
    },
    "karaoke": {
        "font": "Arial Black", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FFFF00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 4,
        "position": "bottom", "outline": 5, "shadow": 2, "background": "none",
        "animation": "none", "highlight_mode": "karaoke",
    },
    "pop": {
        "font": "Impact", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FF7A00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": True, "words_per_cue": 2,
        "position": "bottom", "outline": 6, "shadow": 2, "background": "none",
        "animation": "pop", "highlight_mode": "word",
    },
    "neon": {
        "font": "Arial", "weight": 0, "color": "#FFFFFF",
        "highlight": "#00FFFF", "emphasis_color": "#00FFFF",
        "outline_color": "#00FFFF", "uppercase": True, "words_per_cue": 3,
        "position": "bottom", "outline": 3, "shadow": 0, "background": "none",
        "animation": "fade", "highlight_mode": "word",
    },
    "typewriter": {
        "font": "Courier New", "weight": 0, "color": "#FFFFFF",
        "highlight": "#00FF7F", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 4,
        "position": "bottom", "outline": 3, "shadow": 1, "background": "none",
        "animation": "typewriter", "highlight_mode": "word",
    },
    "retro": {
        "font": "Georgia", "weight": 0, "color": "#FFF6D5",
        "highlight": "#FFB000", "emphasis_color": "#FFD400",
        "outline_color": "#2B1A0A", "uppercase": False, "words_per_cue": 3,
        "position": "bottom", "outline": 2, "shadow": 6, "background": "none",
        "animation": "none", "highlight_mode": "word",
    },
    "hormozi": {
        "font": "Arial Black", "weight": -1, "color": "#FFFFFF",
        "highlight": "#FF0000", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 2,
        "position": "bottom", "outline": 5, "shadow": 2, "background": "none",
        "animation": "pop", "highlight_mode": "word",
    },
    "mrbeast": {
        "font": "Arial Black", "weight": -1, "color": "#FFFFFF",
        "highlight": "#00FF00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 2,
        "position": "bottom", "outline": 5, "shadow": 2, "background": "none",
        "animation": "pop", "highlight_mode": "word",
    },
    "podcast": {
        "font": "Arial", "weight": 0, "color": "#FFFFFF",
        "highlight": "#FFFF00", "emphasis_color": "#FFD400",
        "outline_color": "#000000", "uppercase": False, "words_per_cue": 4,
        "position": "bottom", "outline": 3, "shadow": 1, "background": "none",
        "animation": "none", "highlight_mode": "karaoke",
    },
}
DEFAULT_STYLE = "default"
ALIGNMENTS = {"bottom": 2, "center": 5, "top": 8}

# Automatic emphasis is deliberately small and deterministic. Two words per cue
# at most, so a line never turns into a wall of colour.
POWER_WORDS = frozenset("""
secret mistake mistakes never always free proven guarantee guaranteed fastest
best worst huge massive insane crazy shocking truth nobody everyone stop start
why how million billion thousand hundred percent money cash rich broke
""".split())

EMOJI_RULES = [
    (("money", "cash", "dollar", "profit", "revenue", "rich", "paid", "pay",
      "salary", "price", "cost", "income", "sale"), "💰"),
    (("grow", "growth", "scale", "audience", "follower", "subscriber", "view"), "📈"),
    (("idea", "secret", "tip", "hack", "strategy", "plan", "trick"), "💡"),
    (("time", "hour", "minute", "day", "week", "month", "year", "deadline"), "⏰"),
    (("love", "heart", "passion"), "❤️"),
    (("fire", "amazing", "incredible", "insane", "crazy"), "🔥"),
    (("win", "success", "best", "great", "awesome", "winner"), "🏆"),
    (("mistake", "wrong", "fail", "failure", "problem", "risk", "warning"), "⚠️"),
    (("think", "brain", "learn", "knowledge", "understand"), "🧠"),
    (("work", "job", "business", "client", "customer"), "💼"),
    (("video", "camera", "record", "content", "youtube", "tiktok", "podcast"), "🎬"),
]


def default_out():
    """Output dir: --out wins, then $CHOPIFY_OUT, then ./clips (no more hardcoded C:\\clips)."""
    return Path(os.environ.get("CHOPIFY_OUT") or "clips")


def get_style(name):
    if name not in CAPTION_STYLES:
        raise SystemExit(f"Unknown caption style '{name}'. "
                         f"Choose from: {', '.join(CAPTION_STYLES)}")
    return CAPTION_STYLES[name]


def resolve_caption(style_name=DEFAULT_STYLE, brand=None, caption=None):
    """Merge preset -> brand -> project override into one concrete style."""
    style = {"weight": -1, **branding.CAPTION_DEFAULTS, **get_style(style_name)}
    if brand:
        for key in branding.CAPTION_FIELDS:
            value = brand.get(key)
            if value is not None:
                style[key] = value
    if caption:
        style.update(branding.validate_caption(caption))
    return style


def ass_style_color(value):
    """Accept #RRGGBB and the ASS strings load_brand already produces."""
    if isinstance(value, str) and re.fullmatch(r"&H[0-9a-fA-F]{6,8}&", value):
        return value
    return branding.ass_color(value)


def ass_box_color(value, opacity):
    color = ass_style_color(value)[2:-1]
    bgr = color[2:] if len(color) == 8 else color
    alpha = round((100 - opacity) * 255 / 100)
    return f"&H{alpha:02X}{bgr}&"


def _plain_word(word):
    return re.sub(r"^[^\w$€£¥]+|[^\w%$€£¥]+$", "", str(word).strip())


def is_emphasized(word):
    plain = _plain_word(word)
    if not plain:
        return False
    if re.search(r"\d", plain) or re.search(r"[$€£¥]", plain):
        return True
    if plain.isupper() and len(plain) > 2 and any(c.isalpha() for c in plain):
        return True
    return plain.lower().strip(".,!?;:'\"") in POWER_WORDS


def _emoji_for(word):
    token = re.sub(r"[^a-z0-9']", "", str(word).lower())
    if not token:
        return None
    for stems, emoji in EMOJI_RULES:
        if any(token.startswith(stem) for stem in stems):
            return emoji
    return None


def _cue_highlight(group, enabled):
    """Indexes to emphasise, at most two per cue. Forced flags always count."""
    marked = [i for i, word in enumerate(group) if word.get("emphasis") is True]
    if enabled:
        marked += [i for i, word in enumerate(group)
                   if i not in marked and is_emphasized(word["word"])]
    return set(marked[:2])


def _cue_emoji(group):
    for index, word in enumerate(group):
        emoji = _emoji_for(word["word"])
        if emoji:
            return index, emoji
    return None, None


def _motion_tag(animation):
    if animation == "pop":
        return r"\fscx135\fscy135\t(0,120,\fscx100\fscy100)"
    if animation == "bounce":
        return (r"\fscx88\fscy118\t(0,90,\fscx112\fscy88)"
                r"\t(90,210,\fscx100\fscy100)")
    if animation == "fade":
        return r"\fad(150,0)"
    return ""

WORDS_PER_LINE = 3
DET_FPS = 5
DET_W = 640
EMA_ALPHA = 0.20
JUMP_FRAC = 0.22
MARGIN_FRAC = 0.18
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def sanitize(text, maxlen=70):
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:maxlen].strip("-") or "clip"


def ffprobe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def ffprobe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    duration = float(out)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid media duration for {path}")
    return duration


def padded_clip_bounds(start, end, words, duration, minimum=0.0,
                       before=0.25, after=0.35):
    """Pad clip bounds without crossing nearby words or the source limits."""
    earlier = [float(word["end"]) for word in words
               if float(word["end"]) <= start]
    later = [float(word["start"]) for word in words
             if float(word["start"]) >= end]
    left_pad = min(before, (start - max(earlier)) / 2) if earlier else before
    right_pad = min(after, (min(later) - end) / 2) if later else after
    return max(minimum, start - left_pad), min(duration, end + right_pad)


def audio_filter_chain(duration, loudnorm=False):
    fade_in = min(0.15, duration)
    fade_out = min(0.25, duration)
    filters = [
        f"afade=t=in:st=0:d={fade_in:.3f}",
        f"afade=t=out:st={max(0.0, duration - fade_out):.3f}:d={fade_out:.3f}",
    ]
    if loudnorm:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    return ",".join(filters)


def crop_for(W, H, tw, th):
    """Largest centred crop of the WxH source matching the target aspect."""
    tar = tw / float(th)
    src = W / float(H)
    if abs(src - tar) < 1e-3:
        cw, ch = W, H
    elif src > tar:                      # source wider -> crop width
        ch = H
        cw = int(round(H * tar))
    else:                                # source taller -> crop height
        cw = W
        ch = int(round(W / tar))
    cw -= cw % 2
    ch -= ch % 2
    x0 = max(0, (W - cw) // 2)
    y0 = max(0, (H - ch) // 2)
    return cw, ch, x0, y0


def _make_detector(workdir):
    import cv2
    model = Path(workdir).parent / "yunet.onnx"
    if not model.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(YUNET_URL, str(model))
        except Exception as e:
            print("  YuNet download failed -> Haar:", e, flush=True)
    if model.exists():
        try:
            return "yunet", cv2.FaceDetectorYN.create(str(model), "", (DET_W, 360), 0.6, 0.3, 5000)
        except Exception as e:
            print("  YuNet init failed -> Haar:", e, flush=True)
    return "haar", cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_track(source, start, dur, W, workdir):
    """Return all face centers per sampled frame and the fraction with faces."""
    import cv2
    det_path = Path(workdir) / "_det.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
             "-t", f"{dur:.3f}", "-vf", f"fps={DET_FPS},scale={DET_W}:-2",
             "-an", str(det_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True)
        kind, det = _make_detector(workdir)
        cap = cv2.VideoCapture(str(det_path))
        try:
            dw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or DET_W
            dh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360
            sx = W / float(dw)
            if kind == "yunet":
                det.setInputSize((int(dw), int(dh)))
            frames = []
            frames_with_faces = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = len(frames) / float(DET_FPS)
                if kind == "yunet":
                    _, faces = det.detect(frame)
                    centers = ([(float(face[0] + face[2] / 2.0) * sx)
                                for face in faces]
                               if faces is not None else [])
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = det.detectMultiScale(
                        gray, 1.1, 5, minSize=(36, 36))
                    centers = [float(x + width / 2.0) * sx
                               for x, _, width, _ in faces]
                centers.sort()
                frames.append((t, centers))
                frames_with_faces += bool(centers)
            share = frames_with_faces / len(frames) if frames else 0.0
            return {"frames": frames, "face_share": share}
        finally:
            cap.release()
    finally:
        try:
            det_path.unlink()
        except OSError:
            pass


def smooth_track(track, W, crop_w):
    half = crop_w / 2.0
    max_off = half - crop_w * MARGIN_FRAC
    jump = W * JUMP_FRAC
    track = [(time, (sum(centers) / len(centers) if centers else None))
             for time, centers in track]
    known = [c for _, c in track if c is not None]
    base = sorted(known)[len(known) // 2] if known else W / 2.0
    filled = []
    last = base
    for t, c in track:
        if c is None:
            c = last
        last = c
        filled.append((t, c))
    if not filled:
        return [(0.0, min(max(base, half), W - half))]
    out = []
    c = filled[0][1]
    prev_face = filled[0][1]
    for t, face in filled:
        if abs(face - prev_face) > jump:
            c = face
        else:
            c += EMA_ALPHA * (face - c)
        if face - c > max_off:
            c = face - max_off
        elif c - face > max_off:
            c = face + max_off
        prev_face = face
        out.append((t, min(max(c, half), W - half)))
    return out


def automatic_layout(detections, W, crop_w):
    """Choose blur-fit, split speakers, or tracked single-face framing."""
    frames = detections["frames"]
    if detections["face_share"] < 0.30:
        return "blur-fit", (0.25, 0.75)
    wide_pairs = [centers for _, centers in frames
                  if len(centers) >= 2 and centers[-1] - centers[0] > crop_w]
    sustained = max(3, math.ceil(len(frames) * 0.30))
    if len(wide_pairs) >= sustained:
        left = sorted(centers[0] for centers in wide_pairs)
        right = sorted(centers[-1] for centers in wide_pairs)
        return "split", (left[len(left) // 2] / W,
                         right[len(right) // 2] / W)
    return "single", (0.25, 0.75)


def build_sendcmd(track, crop_w, path):
    lines = []
    last = None
    for t, cx in track:
        x = int(round(cx - crop_w / 2.0))
        if last is None or abs(x - last) >= 2:
            lines.append(f"{t:.2f} crop x {x};")
            last = x
    if not lines:
        lines = ["0.0 crop x 0;"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def _first_caption_time(words, clip_start, clip_end):
    visible = [max(float(word["start"]), clip_start) - clip_start
               for word in words
               if not word.get("hidden")
               and float(word["end"]) > clip_start
               and float(word["start"]) < clip_end]
    return max(0.0, min(visible)) if visible else None


def _has_audio(source):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, check=True)
    return bool(result.stdout.strip())


def _pulse_filters(cue_time, tw, th):
    x = max(8, round(tw * 0.035))
    y = max(8, round(th * 0.04))
    height = max(4, round(th * 0.009))
    widths = [max(14, round(tw * ratio))
              for ratio in (0.012, 0.024, 0.036, 0.048, 0.036)]
    filters = []
    for i, width in enumerate(widths):
        begin = cue_time + i * 0.08
        end = begin + 0.08
        enable = f"between(t\\,{begin:.3f}\\,{end:.3f})"
        filters.append(
            f"drawbox=x={x}:y={y}:w={width}:h={height}:"
            f"color=0x56E0B0@0.95:t=fill:enable='{enable}'")
    return ",".join(filters)


def build_ass(words, clip_start, clip_end, path, tw, th, font_size, margin_v,
              style_name=DEFAULT_STYLE, brand=None, caption=None):
    st = resolve_caption(style_name, brand, caption)
    color = ass_style_color(st["color"])
    highlight = ass_style_color(st["highlight"])
    emphasis = ass_style_color(st["emphasis_color"])
    outline_color = ass_style_color(st["outline_color"])
    if st["font_size"] is not None:
        font_size = max(12, round(st["font_size"] * th / 1080))
    if st["margin"] is not None:
        margin_v = max(0, round(st["margin"] * th / 1080))
    outline_px = max(0, round(st["outline"] * th / 1080))
    shadow_px = max(0, round(st["shadow"] * th / 1080))
    alignment = ALIGNMENTS[st["position"]]
    if st["background"] == "box":
        border_style = 3
        outline_colour = ass_box_color(st["background_color"], st["background_opacity"])
        back_colour = outline_colour
    else:
        border_style = 1
        outline_colour = outline_color
        back_colour = "&H64000000"
    sub = [w for w in words
           if not w.get("hidden") and w["end"] > clip_start and w["start"] < clip_end]
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {tw}\nPlayResY: {th}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{st['font']},{font_size},{color},{color},{outline_colour},"
        f"{back_colour},{st['weight']},0,0,0,100,100,0,0,{border_style},"
        f"{outline_px},{shadow_px},{alignment},80,80,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    group_size = int(st["words_per_cue"] or WORDS_PER_LINE)
    groups = [sub[i:i + group_size] for i in range(0, len(sub), group_size)]
    animation = st["animation"]
    events = []
    for group in groups:
        emphasised = _cue_highlight(group, bool(st["emphasis"]))
        emoji_index, emoji = _cue_emoji(group) if st["emoji"] else (None, None)
        for i, word in enumerate(group):
            s0 = max(word["start"], clip_start) - clip_start
            en = min(clip_end, group[i + 1]["start"] if i + 1 < len(group)
                     else word["end"]) - clip_start
            if en <= s0:
                en = s0 + 0.12
            parts = []
            for j, item in enumerate(group):
                if animation == "typewriter" and j > i:
                    continue
                token = ass_escape(item["word"].strip())
                if st["uppercase"]:
                    token = token.upper()
                if j == emoji_index:
                    token += f" {{\\fnSegoe UI Emoji}}{emoji}{{\\fn{st['font']}}}"
                if j == i and st["highlight_mode"] != "none":
                    word_color = highlight
                elif st["highlight_mode"] == "karaoke" and j < i:
                    word_color = highlight
                elif st["highlight_mode"] != "none" and j in emphasised:
                    word_color = emphasis
                else:
                    word_color = color
                motion = _motion_tag(animation) if j == i else ""
                parts.append("{" + f"\\c{word_color}" + motion + "}" + token)
            events.append(
                f"Dialogue: 0,{ass_time(s0)},{ass_time(en)},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _fallback_metadata(seg, source_meta, content_profile, words):
    text = " ".join(str(word.get("word", "")).strip() for word in words).strip()
    text = text or str(seg.get("text") or seg.get("hook", "clip")).replace("-", " ")
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    if not words and not seg.get("text") and sentence:
        sentence = sentence[:1].upper() + sentence[1:]
    title = (sentence or "Clip")[:60].rstrip()
    tokens = re.findall(r"[A-Za-z0-9]{3,}", sentence.lower())
    stopwords = {"about", "after", "again", "because", "before", "could",
                 "every", "from", "have", "just", "more", "people", "that",
                 "their", "there", "these", "this", "those", "through",
                 "under", "what", "when", "where", "which", "with", "would",
                 "your"}
    keywords = []
    for token in tokens:
        if token not in stopwords and token not in keywords:
            keywords.append(token)
    hashtags = ["#shorts", "#reels"]
    hashtags.extend("#" + token for token in keywords[:2])
    if content_profile == "talk":
        hashtags.append("#podcast")
    elif len(hashtags) < 3:
        hashtags.append("#creator")
    hashtags = hashtags[:5]
    source_title = str((source_meta or {}).get("title") or "").strip()
    description = sentence or title
    if source_title:
        description += f"\nFrom: {source_title}"
    return {"title": title, "description": description, "hashtags": hashtags}


def _hosted_metadata(text, source_meta, content_profile):
    url = os.environ.get("CHOPIFY_HOSTED_LLM_URL")
    model = os.environ.get("CHOPIFY_HOSTED_LLM_MODEL")
    if not url:
        return None, "not_requested"
    if not model:
        return None, "failed"
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"},
                         "minItems": 3, "maxItems": 5},
        },
        "required": ["title", "description", "hashtags"],
    }
    messages = [
        {"role": "system", "content": (
            "Write accurate short-video metadata from the supplied source and "
            "clip transcript. Do not invent claims. Return only schema JSON. "
            "Title must be at most 60 characters; hashtags must number 3 to 5. "
            "Use #podcast only when the content profile is talk.")},
        {"role": "user", "content": json.dumps({
            "source_title": (source_meta or {}).get("title"),
            "content_profile": content_profile,
            "transcript": text,
        }, ensure_ascii=False)},
    ]
    payload = {
        "model": model, "messages": messages, "temperature": 0.2,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "clip_metadata", "strict": True,
                            "schema": schema},
        },
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("CHOPIFY_HOSTED_LLM_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw = result["choices"][0]["message"]["content"]
        generated = json.loads(raw) if isinstance(raw, str) else raw
        title = generated["title"].strip()
        description = generated["description"].strip()
        hashtags = generated["hashtags"]
        if (not title or len(title) > 60 or not description
                or not isinstance(hashtags, list) or not 3 <= len(hashtags) <= 5
                or not all(isinstance(tag, str) and re.fullmatch(
                    r"#[A-Za-z0-9_]+", tag.strip()) for tag in hashtags)):
            return None, "failed"
        hashtags = list(dict.fromkeys(tag.strip().lower() for tag in hashtags))
        if content_profile != "talk":
            hashtags = [tag for tag in hashtags if tag != "#podcast"]
        if not 3 <= len(hashtags) <= 5:
            return None, "failed"
        return {"title": title, "description": description,
                "hashtags": hashtags}, "ok"
    except (OSError, AttributeError, KeyError, TypeError, ValueError, IndexError,
            json.JSONDecodeError):
        return None, "failed"


def write_meta(seg, out_path, source_meta=None, content_profile="mixed",
               words=None):
    """Write ready-to-post metadata, using hosted generation only if configured."""
    words = words or []
    text = " ".join(str(word.get("word", "")).strip() for word in words).strip()
    generated, llm_status = _hosted_metadata(
        text, source_meta or {}, content_profile)
    meta = generated or _fallback_metadata(
        seg, source_meta or {}, content_profile, words)
    meta.update({
        "score": seg.get("overall"),
        "start": seg.get("start"),
        "end": seg.get("end"),
        "llm_status": llm_status,
    })
    Path(out_path).with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(out_path).with_suffix(".meta.json")


CAPTION_OPTION_MAP = {
    "words_per_cue": "words_per_cue",
    "uppercase": "uppercase",
    "caption_position": "position",
    "caption_animation": "animation",
    "caption_background": "background",
    "caption_outline": "outline",
    "caption_shadow": "shadow",
    "emoji": "emoji",
    "emphasis": "emphasis",
    "caption_font": "font",
    "caption_color": "color",
    "caption_highlight": "highlight",
    "caption_emphasis_color": "emphasis_color",
}


def add_caption_arguments(ap):
    """Add the shared caption flags to chopify.py and render_clips.py."""
    ap.add_argument("--caption-json", dest="caption_json", metavar="PATH",
                    help="load a full caption style JSON object")
    ap.add_argument("--words-per-cue", dest="words_per_cue", type=int, metavar="N",
                    help="words shown together in each caption cue (1-8)")
    ap.add_argument("--uppercase", action=argparse.BooleanOptionalAction, default=None,
                    help="force captions to uppercase (or --no-uppercase)")
    ap.add_argument("--caption-position", dest="caption_position",
                    choices=["bottom", "center", "top"], default=None)
    ap.add_argument("--caption-animation", dest="caption_animation",
                    choices=["none", "pop", "bounce", "fade", "typewriter"],
                    default=None, help="motion for the active word")
    ap.add_argument("--caption-background", dest="caption_background",
                    choices=["none", "box"], default=None,
                    help="draw a background box behind each cue")
    ap.add_argument("--caption-outline", dest="caption_outline", type=int,
                    metavar="N", help="outline width at 1080px height (0-12)")
    ap.add_argument("--caption-shadow", dest="caption_shadow", type=int,
                    metavar="N", help="shadow depth at 1080px height (0-12)")
    ap.add_argument("--emoji", action=argparse.BooleanOptionalAction, default=None,
                    help="add one relevant emoji per cue (or --no-emoji)")
    ap.add_argument("--emphasis", action=argparse.BooleanOptionalAction, default=None,
                    help="emphasise numbers, money and power words (or --no-emphasis)")
    ap.add_argument("--caption-font", dest="caption_font", metavar="FAMILY")
    ap.add_argument("--caption-color", dest="caption_color", metavar="#RRGGBB")
    ap.add_argument("--caption-highlight", dest="caption_highlight", metavar="#RRGGBB")
    ap.add_argument("--caption-emphasis-color", dest="caption_emphasis_color",
                    metavar="#RRGGBB")


def caption_from_args(args):
    """Build and validate a caption override from shared CLI flags."""
    supplied = getattr(args, "caption_json", None) is not None
    caption = {}
    path = getattr(args, "caption_json", None)
    if path:
        try:
            raw = Path(path).read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise ValueError(f"Could not read caption JSON: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Caption JSON is invalid: {exc}") from exc
        caption.update(branding.validate_caption(data))
    for dest, key in CAPTION_OPTION_MAP.items():
        value = getattr(args, dest, None)
        if value is not None:
            caption[key] = value
            supplied = True
    return branding.validate_caption(caption) if supplied else None


def _render_impl(seg, words, source, W, H, workdir, aspect, out_dir=None, style=DEFAULT_STYLE,
           do_tighten=False, loudnorm=False, preview=False, brand=None,
           layout="auto", centers=(0.25, 0.75), caption=None,
           source_offset=0.0, overlay="none", sound_effect="none",
           source_meta=None, content_profile="mixed",
           _center_crop_retry=False):
    from captions import write_srt
    out_dir = (Path(out_dir) if out_dir else default_out()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    resolved = resolve_caption(style, brand, caption)
    if layout not in {"auto", "single", "split", "blur-fit"} \
            or (layout == "split" and aspect != "9:16"):
        raise ValueError("Layout must be auto, single, split or blur-fit; split requires 9:16.")
    if len(centers) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in centers):
        raise ValueError("Speaker centers must be between 0 and 1.")
    if overlay not in {"none", "pulse"}:
        raise ValueError("Overlay must be none or pulse.")
    if sound_effect not in {"none", "ping"}:
        raise ValueError("Sound effect must be none or ping.")
    full_w, full_h, font_size, margin_v = ASPECTS[aspect]
    if preview:
        # fast 480p-class draft: smaller frame, smaller captions, no poster/meta
        tw, th = PREVIEW[aspect]
        scale = th / float(full_h)
        font_size = max(12, round(font_size * scale))
        margin_v = max(10, round(margin_v * scale))
    else:
        tw, th = full_w, full_h
    start = float(seg["start"])
    end = float(seg["end"])
    offset = float(source_offset)
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("Source offset must be a finite nonnegative number.")
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Invalid clip start/end times.")
    if start < offset - 1e-9:
        raise ValueError("Clip start precedes its source section.")
    start, end = padded_clip_bounds(
        start, end, words, offset + ffprobe_duration(source), minimum=offset)
    dur = end - start
    if dur <= 0:
        raise ValueError("Invalid clip start/end times.")
    if start < offset - 1e-9:
        raise ValueError("Clip start precedes its source section.")
    local_start = max(0.0, start - offset)
    cw, ch, x0, y0 = crop_for(W, H, tw, th)

    cap_words = words
    render_src = source
    if do_tighten:
        keeps = tighten.plan_cuts(words, start, end)
        render_src = out_dir / "_tight_source.mp4"
        tighten.apply_cuts(source, keeps, start, render_src,
                           source_offset=offset)
        cap_words, dur = tighten.remap_words(words, start, keeps)
        print(f"  tightened: {len(keeps)} keep-ranges, "
              f"{(end - start) - dur:.1f}s removed", flush=True)
        start = 0.0
        local_start = 0.0
    cue_time = _first_caption_time(cap_words, start, start + dur)
    build_ass(cap_words, start, start + dur, out_dir / "_caption.ass",
              tw, th, font_size, margin_v, style, brand, caption)

    needs_track = cw < W * 0.95
    detections = None
    if layout == "auto" and needs_track:
        try:
            detections = detect_track(render_src, local_start, dur, W, workdir)
        except Exception as exc:
            if _center_crop_retry:
                raise
            print(f"  face tracking failed ({exc}); retrying with center crop",
                  flush=True)
            return render(
                seg, words, source, W, H, workdir, aspect, out_dir=out_dir,
                style=style, do_tighten=do_tighten, loudnorm=loudnorm,
                preview=preview, brand=brand, layout="single", centers=centers,
                caption=caption, source_offset=source_offset, overlay=overlay,
                sound_effect=sound_effect, source_meta=source_meta,
                content_profile=content_profile, _center_crop_retry=True)
        layout, auto_centers = automatic_layout(detections, W, cw)
        if layout == "split":
            centers = auto_centers
    if layout == "split":
        # Keep every YUV420 panel even-sized. Older libass/ffmpeg builds can
        # crash on odd-height intermediate frames (854px drafts / 2 = 427).
        panel_h = (th // 4) * 2
        pcw, pch, _, py = crop_for(W, H, tw, panel_h)
        positions = [max(0, min(W - pcw, round(W * c - pcw / 2))) for c in centers]
        vf = (f"[0:v]split=2[a][b];[a]crop={pcw}:{pch}:{positions[0]}:{py},"
              f"scale={tw}:{panel_h}[top];[b]crop={pcw}:{pch}:{positions[1]}:{py},"
              f"scale={tw}:{panel_h}[bottom];[top][bottom]vstack=inputs=2,"
              f"pad={tw}:{th}:0:0,subtitles=_caption.ass")
        mode = "podcast split-screen"
    elif layout == "blur-fit":
        vf = (f"[0:v]split[bg][fg];"
              f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,"
              f"crop={tw}:{th},boxblur=20:10[background];"
              f"[fg]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
              f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2[foreground];"
              f"[background][foreground]overlay=(W-w)/2:(H-h)/2,"
              "subtitles=_caption.ass")
        mode = "blur-fit"
    elif needs_track and layout in {"auto", "single"} and not _center_crop_retry:
        detections = detections or detect_track(
            render_src, local_start, dur, W, workdir)
        track = smooth_track(detections["frames"], W, cw)
        build_sendcmd(track, cw, out_dir / "_crop.cmd")
        init_x = max(0, min(int(round(track[0][1] - cw / 2.0)), W - cw))
        vf = (f"sendcmd=f=_crop.cmd,crop={cw}:{ch}:{init_x}:{y0},"
              f"scale={tw}:{th},subtitles=_caption.ass")
        mode = "speaker-tracked"
    else:
        vf = f"crop={cw}:{ch}:{x0}:{y0},scale={tw}:{th},subtitles=_caption.ass"
        mode = "full-frame"

    suffix = ".preview" if preview else ""
    prefix = sanitize(str(seg["id"])) + "-" if seg.get("id") else ""
    out = out_dir / (prefix + sanitize(seg.get("hook", "clip")) + suffix + ".mp4")
    cmd = ["ffmpeg", "-y", "-ss", f"{local_start:.3f}", "-i", str(render_src)]
    if brand and brand.get("logo"):
        cmd += ["-i", brand["logo"]]
    use_sfx = sound_effect == "ping" and cue_time is not None
    source_has_audio = _has_audio(render_src) if use_sfx else False
    silence_input = None
    if use_sfx:
        if not source_has_audio:
            silence_input = 1 + int(bool(brand and brand.get("logo")))
            cmd += ["-f", "lavfi", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000"]
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=880:sample_rate=48000:duration=0.20"]
    cmd += ["-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]

    graph_mode = bool((brand and brand.get("logo")) or layout == "split"
                      or layout == "blur-fit"
                      or (overlay == "pulse" and cue_time is not None)
                      or use_sfx)
    if graph_mode:
        graph = vf if layout in {"split", "blur-fit"} else "[0:v]" + vf
        if overlay == "pulse" and cue_time is not None:
            graph += "," + _pulse_filters(cue_time, tw, th)
        if brand and brand.get("logo"):
            width = max(20, round(brand["logo_width"] * tw / 1920))
            graph += (f"[captioned];[1:v]scale={width}:-1[logo];"
                      "[captioned][logo]overlay=W-w-24:24:eof_action=repeat[v]")
        else:
            graph += "[v]"
        cmd[cmd.index("-c:v"):cmd.index("-c:v")] = [
            "-filter_complex", graph, "-map", "[v]"]
    else:
        cmd[cmd.index("-c:v"):cmd.index("-c:v")] = ["-vf", vf]

    if use_sfx:
        tone_index = 1 + int(bool(brand and brand.get("logo")))
        if not source_has_audio:
            tone_index += 1
        if silence_input is not None:
            base_audio = (f"[{silence_input}:a]atrim=duration={dur:.3f},"
                          "asetpts=PTS-STARTPTS[base]")
        else:
            base_filter = audio_filter_chain(dur, loudnorm)
            base_audio = (f"[0:a:0]asetpts=PTS-STARTPTS,"
                          f"{base_filter}[base]")
        delay_ms = max(0, round(cue_time * 1000))
        audio_graph = (
            base_audio + ";"
            f"[{tone_index}:a]volume=0.18,"
            "afade=t=in:st=0:d=0.005,afade=t=out:st=0.08:d=0.12,"
            f"adelay={delay_ms}:all=1[sfx];"
            "[base][sfx]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[audio_out]")
        graph_arg = cmd[cmd.index("-filter_complex") + 1]
        cmd[cmd.index("-filter_complex") + 1] = graph_arg + ";" + audio_graph
        cmd[cmd.index("-c:v"):cmd.index("-c:v")] = [
            "-map", "[audio_out]"]
    elif not use_sfx:
        cmd[cmd.index("-c:v"):cmd.index("-c:v")] = [
            "-af", audio_filter_chain(dur, loudnorm)]

    if graph_mode and not use_sfx:
        cmd[cmd.index("-c:v"):cmd.index("-c:v")] = ["-map", "0:a?"]
    print(">", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(out_dir),
                       stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError:
        if mode == "speaker-tracked" and not _center_crop_retry:
            print("  tracked crop failed; retrying with center crop", flush=True)
            return render(
                seg, words, source, W, H, workdir, aspect, out_dir=out_dir,
                style=style, do_tighten=do_tighten, loudnorm=loudnorm,
                preview=preview, brand=brand, layout="single", centers=centers,
                caption=caption, source_offset=source_offset, overlay=overlay,
                sound_effect=sound_effect, source_meta=source_meta,
                content_profile=content_profile, _center_crop_retry=True)
        raise
    write_srt(cap_words, start, start + dur, out.with_suffix(".srt"),
              group_size=resolved["words_per_cue"])

    for tmpf in (out_dir / "_caption.ass", out_dir / "_crop.cmd"):
        try:
            tmpf.unlink()
        except OSError:
            pass

    if do_tighten:
        try:
            render_src.unlink()
        except OSError:
            pass

    if preview:
        # drafts skip the poster and metadata - they exist to be eyeballed fast
        print(f"SAVED {out}  ({aspect} {tw}x{th} preview, {mode})", flush=True)
        return out

    # poster: a representative still saved next to the clip (<clip>.png)
    poster = out.with_suffix(".png")
    subprocess.run(["ffmpeg", "-y", "-ss", f"{dur * 0.4:.2f}", "-i", str(out),
                    "-frames:v", "1", "-q:v", "2", str(poster)],
                   check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)
    clip_words = [word for word in cap_words
                  if float(word["end"]) > start
                  and float(word["start"]) < start + dur]
    meta_path = write_meta(
        seg, out, source_meta=source_meta,
        content_profile=content_profile, words=clip_words)

    print(f"SAVED {out}  ({aspect} {tw}x{th}, {mode})  "
          f"+ poster {poster.name} + {meta_path.name}", flush=True)
    return out


def render(seg, words, source, W, H, workdir, aspect, out_dir=None,
           style=DEFAULT_STYLE, do_tighten=False, loudnorm=False, preview=False,
           brand=None, layout="auto", centers=(0.25, 0.75), caption=None,
           source_offset=0.0, overlay="none", sound_effect="none",
           source_meta=None, content_profile="mixed",
           _center_crop_retry=False):
    """Render one clip and remove per-render scratch files on every exit path."""
    cleanup_dir = (Path(out_dir) if out_dir else default_out()).resolve()
    try:
        return _render_impl(
            seg, words, source, W, H, workdir, aspect, out_dir=out_dir,
            style=style, do_tighten=do_tighten, loudnorm=loudnorm,
            preview=preview, brand=brand, layout=layout, centers=centers,
            caption=caption, source_offset=source_offset, overlay=overlay,
            sound_effect=sound_effect, source_meta=source_meta,
            content_profile=content_profile,
            _center_crop_retry=_center_crop_retry)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        tail = "\n".join(stderr.splitlines()[-12:])
        message = f"FFmpeg failed with exit code {exc.returncode}."
        if tail:
            message += "\nFFmpeg stderr tail:\n" + tail
        raise RuntimeError(message) from exc
    finally:
        for name in ("_caption.ass", "_crop.cmd", "_tight_source.mp4",
                     "_det.mp4"):
            try:
                (cleanup_dir / name).unlink()
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", nargs="?", default="work")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, choices=list(ASPECTS),
                    help="output aspect ratio (default 9:16)")
    ap.add_argument("--out", default=None,
                    help="output directory (default ./clips, or $CHOPIFY_OUT)")
    ap.add_argument("--style", default=DEFAULT_STYLE,
                    help="caption preset: " + ", ".join(CAPTION_STYLES))
    ap.add_argument("--tighten", action="store_true",
                    help="remove filler words (um, uh...) and long silences")
    ap.add_argument("--loudnorm", action="store_true",
                    help="normalize audio to -14 LUFS (platform standard)")
    ap.add_argument("--preview", action="store_true",
                    help="render fast 480p drafts (no poster/meta) to review "
                         "clip selection before the full render")
    ap.add_argument("--brand", help="path to reusable JSON brand preset")
    ap.add_argument("--layout", choices=["auto", "single", "split", "blur-fit"],
                    default="auto")
    ap.add_argument("--centers", type=float, nargs=2, default=[0.25, 0.75])
    add_caption_arguments(ap)
    args = ap.parse_args()
    from branding import load_brand
    brand = load_brand(args.brand) if args.brand else None
    try:
        caption = caption_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"Caption: {exc}")

    workdir = Path(args.workdir).resolve()
    out_dir = (Path(args.out) if args.out else default_out()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript = json.loads((workdir / "transcript.json").read_text(encoding="utf-8-sig"))
    segments = json.loads((workdir / "segments.json").read_text(encoding="utf-8-sig"))
    words = transcript["words"]

    source = Path(transcript["video"])
    if not source.exists():
        cands = sorted(workdir.glob("source.*"))
        source = cands[0] if cands else source
    source = source.resolve()

    W, H = ffprobe_dims(source)
    print(f"Source {source} {W}x{H}; {len(segments)} clips -> {args.aspect} "
          f"(style={args.style}, tighten={args.tighten}) -> {out_dir}", flush=True)
    selection_path = workdir / "selection.json"
    selection = (json.loads(selection_path.read_text(encoding="utf-8-sig"))
                 if selection_path.exists() else {})

    saved = []
    failed = 0
    for i, seg in enumerate(segments, 1):
        print(f"--- Clip {i}/{len(segments)}: "
              f"{seg.get('hook', '')[:60]!r} (overall {seg.get('overall')}) ---",
              flush=True)
        try:
            saved.append(str(render(
                {**seg, "id": f"{i:03}"}, words, source, W, H, workdir,
                args.aspect, out_dir=out_dir, style=args.style,
                do_tighten=args.tighten, loudnorm=args.loudnorm,
                preview=args.preview, brand=brand, layout=args.layout,
                centers=args.centers, caption=caption,
                source_meta={"title": transcript.get("title")},
                content_profile=selection.get("content_profile", "mixed"))))
        except Exception as e:
            print(f"ERROR rendering clip {i}: {e}", flush=True)
            failed += 1

    print("RENDERED", len(saved), "clips:", flush=True)
    for s in saved:
        print("  ", s, flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
