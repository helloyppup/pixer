import streamlit as st
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pickle
import math

# 可选：使用 skimage 进行 RGB 到 LAB 转换以提升颜色感知精度
try:
    from skimage.color import rgb2lab
    USE_LAB = False
except ImportError:
    USE_LAB = False

# ---------- 调色板加载与持久化 ----------
def save_palette_to_file(palette, filename='saved_palette.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(palette, f)

def load_palette_from_file(filename='saved_palette.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def load_local_palette():
    if os.path.exists("palette.json"):
        with open("palette.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return {item['name']: item['color'] for item in data}
    return {}

def load_palette(path_or_file):
    if hasattr(path_or_file, "read"):
        data = json.load(path_or_file)
    else:
        with open(path_or_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    if isinstance(data, list):
        data = {item['name']: item['color'] for item in data}
    elif not isinstance(data, dict):
        raise ValueError("不支持的调色板格式，请使用 dict 或 list。")
    palette = {}
    for name, hexcode in data.items():
        if not isinstance(hexcode, str) or not hexcode.startswith('#'):
            st.error(f"无效的色值: {name}: {hexcode}")
            continue
        try:
            rgb = tuple(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0,2,4))
        except ValueError:
            st.error(f"无法解析色值: {name}: {hexcode}")
            continue
        palette[name] = rgb
    return palette

# ---------- 颜色分析函数 ----------
def nearest_color(color, palette):
    if not palette:
        raise ValueError("调色板为空，请上传一个有效的调色板文件。")
    if isinstance(color, str):
        color = tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    distances = {}
    for name, pc in palette.items():
        pc_rgb = pc if isinstance(pc, tuple) else tuple(int(pc.lstrip('#')[i:i+2],16) for i in (0,2,4))
        distances[name] = np.linalg.norm(np.array(color) - np.array(pc_rgb))
    return min(distances, key=distances.get)

def predominant_color(region):
    pixels = region.reshape(-1,3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(colors[counts.argmax()])

# ---------- 绘图核心 ----------
def draw_grid_overlay(img, grid_size, line_color=(255,0,0,128)):
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, w, grid_size):
        draw.line([(x,0),(x,h)], fill=line_color, width=1)
    for y in range(0, h, grid_size):
        draw.line([(0,y),(w,y)], fill=line_color, width=1)
    return Image.alpha_composite(img.convert('RGBA'), overlay)

def basic_mosaic(img, grid_size, palette):
    w, h = img.size
    arr = np.array(img)
    out = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(out)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            box = arr[y:y+grid_size, x:x+grid_size]
            if not box.size:
                continue
            color = predominant_color(box)
            name = nearest_color(color, palette)
            if isinstance(palette[name], tuple):
                fill = palette[name]
            else:
                fill = tuple(int(palette[name].lstrip('#')[i:i+2],16) for i in (0,2,4))
            draw.rectangle([x, y, x+grid_size, y+grid_size], fill=fill)
    return out


def mapped_mosaic(img, grid_size, palette):
    w, h = img.size
    arr = np.array(img)
    out = Image.new('RGB', (w, h))
    draw = ImageDraw.Draw(out)
    color_count = {}
    # 内边距与可用尺寸
    padding = max(int(grid_size * 0.1), 2)
    avail = grid_size - 2 * padding
    # 计算统一字号
    sizes = []
    FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf")

    for name in palette:
        try:
            f_tmp = ImageFont.truetype(FONT_PATH, avail)
        except:
            f_tmp = ImageFont.load_default()
        bbox = f_tmp.getbbox(name)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= avail and th <= avail:
            sizes.append(avail)
        else:
            scale = min(avail / tw, avail / th)
            sizes.append(max(int(avail * scale), 6))
    uniform_size = min(sizes) if sizes else max(avail, 6)
    try:
        font = ImageFont.truetype(FONT_PATH, uniform_size)
    except:
        font = ImageFont.load_default()
    # 遍历绘制
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            box = arr[y:y+grid_size, x:x+grid_size]
            if not box.size:
                continue
            color = predominant_color(box)
            name = nearest_color(color, palette)
            if isinstance(palette[name], tuple):
                fill = palette[name]
            else:
                fill = tuple(int(palette[name].lstrip('#')[i:i+2],16) for i in (0,2,4))
            draw.rectangle([x, y, x+grid_size, y+grid_size], fill=fill)
            brightness = fill[0] * 0.299 + fill[1] * 0.587 + fill[2] * 0.114
            text_color = (0,0,0) if brightness > 186 else (255,255,255)
            bbox = font.getbbox(name)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x + padding + (avail - tw) / 2
            ty = y + padding + (avail - th) / 2
            draw.text((tx, ty), name, fill=text_color, font=font)
            color_count[name] = color_count.get(name, 0) + 1
    return out, color_count, font, padding, avail


def annotate_mapped(mosaic, grid_size, font, padding, avail):
    w, h = mosaic.size
    cols = w // grid_size
    rows = h // grid_size
    bbox = font.getbbox(str(max(rows - 1, cols - 1)))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin_left = tw + padding * 2
    margin_top = th + padding * 2
    annotated = Image.new('RGB', (margin_left + w, margin_top + h), (200,200,200))
    draw = ImageDraw.Draw(annotated)
    annotated.paste(mosaic, (margin_left, margin_top))
    # 5×5 淡灰色网格
    for x in range(0, w+1, grid_size * 5):
        draw.line([(margin_left + x, margin_top), (margin_left + x, margin_top + h)], fill=(200,200,200), width=1)
    for y in range(0, h+1, grid_size * 5):
        draw.line([(margin_left, margin_top + y), (margin_left + w, margin_top + y)], fill=(200,200,200), width=1)
    # 行坐标
    for r in range(rows):
        y = margin_top + r * grid_size + (grid_size - avail) / 2
        x_pos = (margin_left - padding - tw) / 2
        draw.text((x_pos, y), str(r), fill=(0,0,0), font=font)
    # 列坐标
    for c in range(cols):
        x = margin_left + c * grid_size + (grid_size - avail) / 2
        y_pos = (margin_top - padding - th) / 2
        draw.text((x, y_pos), str(c), fill=(0,0,0), font=font)
    return annotated


def append_legend(mosaic, count, palette, sq=20, pad=5, font_path="arial.ttf", fsize=14, bg=(255,255,255)):
    temp = Image.new('RGB', (1,1))
    td = ImageDraw.Draw(temp)
    try:
        font_leg = ImageFont.truetype(font_path, fsize)
    except:
        font_leg = ImageFont.load_default()
    items = sorted(count.items(), key=lambda x: -x[1])
    block_widths = []
    for name, cnt in items:
        text = f"{name}: {cnt}"
        bw = td.textbbox((0,0), text, font=font_leg)[2]
        block_widths.append(sq + pad + bw + pad)
    if not block_widths:
        return mosaic
    cell_w = max(block_widths)
    max_w = mosaic.width
    cols = max_w // cell_w or 1
    thg = td.textbbox((0,0), "Hg", font=font_leg)[3]
    line_h = max(sq, thg) + pad
    rows = (len(items) + cols - 1) // cols
    legend_h = rows * line_h + pad
    legend = Image.new('RGB', (max_w, legend_h), bg)
    d = ImageDraw.Draw(legend)
    r = max(2, sq // 5)
    for idx, (name, cnt) in enumerate(items):
        row, col = divmod(idx, cols)
        x0 = col * cell_w + pad
        y0 = row * line_h + pad
        color = palette[name] if isinstance(palette[name], tuple) else tuple(
            int(palette[name].lstrip('#')[i:i+2],16) for i in (0,2,4)
        )
        d.rounded_rectangle([x0, y0, x0+sq, y0+sq], radius=r, fill=color, outline=(0,0,0))
        text = f"{name}: {cnt}"
        txt_h = td.textbbox((0,0), text, font=font_leg)[3]
        d.text((x0+sq+pad, y0 + (sq - txt_h)//2), text, fill=(0,0,0), font=font_leg)
    final = Image.new('RGB', (mosaic.width, mosaic.height + legend_h), bg)
    final.paste(mosaic, (0,0))
    final.paste(legend, (0, mosaic.height))
    return final

# ---------- Streamlit 界面 ----------
st.title("图纸生成")
uploaded = st.file_uploader("上传图片", type=['png','jpg','jpeg'])
def init():
    st.session_state.grid_size = 20
if 'grid_size' not in st.session_state:
    init()
local_palette = load_local_palette()
palette_file = st.file_uploader("上传调色板 JSON", type=['json'])
if palette_file:
    try:
        palette = load_palette(palette_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    palette = local_palette

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.markdown("#### 调整网格大小")
    col1, col2 = st.columns([8, 2])
    with col2:
        st.number_input(
            "",
            min_value=1,
            value=st.session_state.grid_size,
            step=1,
            key='num_in',
            label_visibility='hidden',
            on_change=lambda: st.session_state.update(grid_size=st.session_state.num_in)
        )
    with col1:
        st.slider(
            "",
            min_value=1,
            max_value=min(img.size) // 3,
            value=st.session_state.grid_size,
            key='slider',
            label_visibility='collapsed',
            on_change=lambda: st.session_state.update(grid_size=st.session_state.slider)
        )
    gs = st.session_state.grid_size
    st.image(draw_grid_overlay(img, gs), caption=f"网格预览", use_container_width=True)
    if st.button("生成图纸"):
        basic = basic_mosaic(img, gs, palette)
        mapped, count, font, pad, avail = mapped_mosaic(img, gs, palette)
        annotated = annotate_mapped(mapped, gs, font, pad, avail)
        final_img = append_legend(annotated, count, palette)
        # 缩放以保证最小格尺寸
        min_cell = 10
        scale = math.ceil(min_cell / gs) if gs < min_cell else 1
        if scale > 1:
            basic = basic.resize((basic.width * scale, basic.height * scale), Image.NEAREST)
            final_img = final_img.resize((final_img.width * scale, final_img.height * scale), Image.NEAREST)
        st.subheader("预览")
        st.image(basic, use_container_width=True)
        st.subheader("图纸")
        st.image(final_img, use_container_width=True)
