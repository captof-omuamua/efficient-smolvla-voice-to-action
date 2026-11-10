"""Drawing helpers for the v3 meters clip: layout, colours, fonts, text, eval-video frames, mp4 writer."""
import functools
import subprocess

import av
from PIL import Image, ImageDraw, ImageFont

from measured import SCENE, successes, video_path

# Layout of a 1920 x 1080 frame (full HD = a full 16:9 slide)
W, H = 1920, 1080
VID = 576  # each 360 x 360 eval video is scaled up 1.6x to 576 x 576
VID_Y = 146  # top edge of both videos
LEFT_X, RIGHT_X = 480 - VID // 2, 1440 - VID // 2  # left edges: each video centred in its half
GAP_X = W // 2  # centre of the gap between the two videos

# High-contrast colours on a near-black background
BG = (12, 14, 18)
WHITE, GREY, DIM = (255, 255, 255), (200, 205, 212), (62, 68, 78)
COLORS = {"Original": (255, 159, 67), "Tuned": (77, 196, 255)}  # orange vs blue: safe for colour-blind viewers

REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
NARROW = "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf"  # same height, narrower letters


@functools.lru_cache
def font(size, bold=False, narrow=False):
    """Load a font once per size and style (cached)."""
    return ImageFont.truetype(NARROW if narrow else BOLD if bold else REGULAR, size)


def text(draw, xy, s, size, color=WHITE, bold=False, anchor="la", narrow=False):
    """One line of text. anchor: 'la' left, 'ma' centred, 'ra' right; all measured from the top."""
    draw.text(xy, s, font=font(size, bold, narrow), fill=color, anchor=anchor)


def background(subtitle, headers, footer):
    """Everything that stays still: subtitle, a 'Name: numbers' header and a coloured border per video, footer."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    text(draw, (W // 2, 20), subtitle, 32, GREY, anchor="ma")
    for (name, rest), x in zip(headers, (LEFT_X, RIGHT_X)):
        # header centred over its video, the name in its side's colour
        first, f = f"{name}: ", font(44, True)
        left = x + VID / 2 - draw.textlength(first + rest, font=f) / 2
        text(draw, (left, 76), first, 44, COLORS[name], bold=True)
        text(draw, (left + draw.textlength(first, font=f), 76), rest, 44, WHITE, bold=True)
        # a coloured frame around the video ties it to its header
        draw.rectangle([x - 5, VID_Y - 5, x + VID + 4, VID_Y + VID + 4], outline=COLORS[name], width=5)
    # two footer lines in the narrow font, so the long caveat line fits at full height
    for row, (line, size, color) in enumerate(zip(footer, (28, 27), (WHITE, GREY))):
        assert draw.textlength(line, font=font(size, narrow=True)) <= W - 60, f"footer line {row + 1} is too wide"
        text(draw, (W // 2, 984 + 48 * row), line, size, color, anchor="ma", narrow=True)
    return img


def bar(draw, x, y, fraction, color, width=VID, height=16):
    """A meter: grey track, filled from the left up to `fraction` of its width."""
    draw.rectangle([x, y, x + width - 1, y + height - 1], fill=DIM)
    if fraction > 0:
        draw.rectangle([x, y, x + round(fraction * width) - 1, y + height - 1], fill=color)


def badge(draw, x, y, s, color):
    """A small pill with dark text, e.g. 'Task done'."""
    w = draw.textlength(s, font=font(30, True))
    draw.rounded_rectangle([x, y, x + w + 32, y + 46], radius=23, fill=color)
    text(draw, (x + 16, y + 8), s, 30, BG, bold=True)


def read_frames(path):
    """Decode every frame of an eval video, scaled up to VID x VID pixels."""
    with av.open(str(path)) as container:
        return [
            Image.fromarray(f.to_ndarray(format="rgb24")).resize((VID, VID), Image.LANCZOS)
            for f in container.decode(video=0)
        ]


def load_scene(setups):
    """The chosen episode's frames for each setup. Both runs must have solved it, in the same number of steps."""
    frames = {s["name"]: read_frames(video_path(s, *SCENE)) for s in setups}
    assert all(successes(s)[SCENE] for s in setups), "pick an episode that both runs solved"
    assert len({len(f) for f in frames.values()}) == 1, "pick an episode with equal step counts"
    return frames


class Mp4Writer:
    """Pipe RGB frames into ffmpeg's CPU H.264 encoder (libx264, never the GPU's NVENC)."""

    def __init__(self, path, fps):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",  # raw RGB frames in
            # RGB -> YUV with the HD colour matrix, and tag the file as HD colour so players show true colours
            "-vf", "scale=out_color_matrix=bt709,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-threads", "4",  # CPU encoder, high quality
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),  # plays in browsers, Teams, PowerPoint
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, image):
        self.proc.stdin.write(image.tobytes())

    def close(self):
        self.proc.stdin.close()
        assert self.proc.wait() == 0, "ffmpeg failed"
