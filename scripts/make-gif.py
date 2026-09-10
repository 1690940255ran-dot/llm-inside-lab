"""
把 CDP 抓到的 hero 帧拼成一张 GIF。

用法：
    python scripts/make-gif.py
默认会从 tmp/frames/ 读，输出 docs/demo.gif，宽度压到 960px，180ms/帧。

依赖：Pillow（pip install pillow 即可）
"""
import os, sys, glob
from PIL import Image

frames_dir = sys.argv[1] if len(sys.argv) > 1 else "tmp/frames"
out_path   = sys.argv[2] if len(sys.argv) > 2 else "docs/demo.gif"
target_w   = int(sys.argv[3]) if len(sys.argv) > 3 else 960
duration   = int(sys.argv[4]) if len(sys.argv) > 4 else 180

paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
if not paths:
    raise SystemExit(f"no frames in {frames_dir}; run scripts/capture-hero.mjs first")

# 用第一帧建一张统一调色板（避免每帧色调跳）
first = Image.open(paths[0]).convert("RGB")
ratio = target_w / first.width
target_h = round(first.height * ratio)
first = first.resize((target_w, target_h), Image.LANCZOS)
palette_img = first.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)

frames = []
for p in paths:
    img = Image.open(p).convert("RGB")
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    frames.append(img.quantize(palette=palette_img, dither=Image.Dither.NONE))

frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
    optimize=True,
    disposal=2,
)
print(f"gif saved: {out_path}  {target_w}x{target_h}  {len(frames)} frames  "
      f"{os.path.getsize(out_path)/1024:.0f} KB  {duration} ms/frame")