
import numpy as np
from skimage.color import rgb2lab

if not hasattr(np, 'asscalar'):
    np.asscalar = lambda x: x.item()

import streamlit as st
import json
from PIL import Image, ImageDraw, ImageFont
import os
import pickle
import math
import io
import time




FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "GOTHIC.TTF")

Image.MAX_IMAGE_PIXELS = 10**9

FONT_SIZE=22

# ---------- 调色板加载与持久化 ----------
def save_palette_to_file(palette, filename='saved_palette.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(palette, f)

def load_palette_from_file(filename='saved_palette.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def load_local_palette(
    json_path: str = "palette.json",
    cache_path: str = "palette_lab_cache.json"
):
    """
    加载 palette.json，并缓存一次性计算好的 Lab 数组到 cache_path。
    返回：
      palette_dict: {name: "#RRGGBB"}
      names_list:   [name1, name2, …]
      lab_arr:      np.ndarray shape (N,3)，对应 names_list 的 Lab 值

    不会修改原来的 palette.json，只会创建/读取 cache_path。
    """
    # 如果缓存已经存在，直接读它
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        palette_dict = cache["palette_dict"]
        names_list   = cache["names_list"]
        lab_arr      = np.array(cache["lab_arr"], dtype=float)
        return palette_dict, names_list, lab_arr

    # 否则，先读原始 palette.json
    if not os.path.exists(json_path):
        # 没有 JSON 文件就返回空结构
        return {}, [], np.zeros((0,3), float)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 1) 去重，构建 palette_dict
    palette_dict = {}
    for item in data:
        base, hexcol = item["name"], item["color"]
        name, i = base, 1
        while name in palette_dict:
            name = f"{base}_{i}"
            i += 1
        palette_dict[name] = hexcol

    # 2) 构建 names_list 和归一化 RGB 数组
    names_list = list(palette_dict.keys())
    rgb = []
    for nm in names_list:
        hx = palette_dict[nm].lstrip("#")
        r = int(hx[0:2],16)/255.0
        g = int(hx[2:4],16)/255.0
        b = int(hx[4:6],16)/255.0
        rgb.append((r,g,b))
    rgb_arr = np.array(rgb, dtype=float)  # shape (N,3)

    # 3) 转到 Lab
    lab_arr = rgb2lab(rgb_arr[np.newaxis, :, :])[0]  # shape (N,3)

    # 4) 序列化到缓存文件，下次直接用
    cache = {
        "palette_dict": palette_dict,
        "names_list":   names_list,
        "lab_arr":      lab_arr.tolist()
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    return palette_dict, names_list, lab_arr

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
# def nearest_color(color, palette):
#     if not palette:
#         raise ValueError("调色板为空，请上传一个有效的调色板文件。")
#     if isinstance(color, str):
#         color = tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))
#     distances = {}
#     for name, pc in palette.items():
#         pc_rgb = pc if isinstance(pc, tuple) else tuple(int(pc.lstrip('#')[i:i+2],16) for i in (0,2,4))
#         distances[name] = np.linalg.norm(np.array(color) - np.array(pc_rgb))
#     return min(distances, key=distances.get)


def nearest_color(color, palette):
    """
    在 CIELAB 空间里，用向量化的 CIE76（欧氏距离）找出最接近输入色的调色板名称。

    参数：
      color: (R,G,B) 三元组，0–255 范围
      palette: load_local_palette() 返回的三元组
               (palette_dict, names_list, lab_arr)
        - palette_dict: {name: "#RRGGBB"}（绘图时用）
        - names_list:   [name1, name2, …]
        - lab_arr:      np.ndarray, shape (N,3)，对应 names_list 的 Lab 值
    返回：
      最接近输入色的 palette 中的 name（字符串）
    """
    palette_dict, names_list, lab_arr = palette

    if lab_arr.size == 0 or not names_list:
        raise ValueError("调色板为空，请先调用 load_local_palette() 并确认有内容。")

    # 1) 归一化 RGB 到 [0,1]
    rgb = np.array(color, dtype=float) / 255.0  # shape (3,)

    # 2) 转到 Lab，得到 shape (3,)
    lab = rgb2lab(rgb[np.newaxis, np.newaxis, :])[0, 0]

    # 3) 向量化 CIE76 距离计算
    dists = np.linalg.norm(lab_arr - lab, axis=1)

    # 4) 取最小值对应的索引
    idx = int(dists.argmin())

    return names_list[idx]



def predominant_max(region):
    pixels = region.reshape(-1,3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(colors[counts.argmax()])

# 中位数取色
def predominant_median(region):
    """
    对 region 的像素做中位数统计，
    返回三个通道的中位数作为代表色。
    """
    # print("调用")
    pixels = region.reshape(-1, 3)
    med = np.median(pixels, axis=0)
    return tuple(med.astype(int))

def predominant_mean(region):
    # print("调用平均")
    pixels = region.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    return tuple(mean.astype(int))


# ---------- 绘图核心 ----------
# def draw_grid_overlay(img, grid_size, line_color=(255,0,0,128)):
#     w, h = img.size
#     overlay = Image.new('RGBA', (w, h), (0,0,0,0))
#     draw = ImageDraw.Draw(overlay)
#     for x in range(0, w, grid_size):
#         draw.line([(x,0),(x,h)], fill=line_color, width=1)
#     for y in range(0, h, grid_size):
#         draw.line([(0,y),(w,y)], fill=line_color, width=1)
#     return Image.alpha_composite(img.convert('RGBA'), overlay)

def draw_grid_overlay(img, grid_size, line_color=(255,0,0,128)):
    w, h = img.size
    overlay = Image.new('RGBA', (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # 用 while + 浮点累加来画竖线
    x = 0.0
    while x < w:
        draw.line([(x, 0), (x, h)], fill=line_color, width=1)
        x += grid_size

    y = 0.0
    while y < h:
        draw.line([(0, y), (w, y)], fill=line_color, width=1)
        y += grid_size

    return Image.alpha_composite(img.convert('RGBA'), overlay)


def basic_mosaic(out_list):
    """
    img: PIL Image
    grid_size: float 或 int，都支持
    palette: 名称->颜色映射，颜色可以是 RGB tuple 或 "#RRGGBB" 字符串
    """

    return  draw_list(out_list,False)
    # w, h = img.size
    # arr = np.array(img)
    # out = Image.new('RGB', (w, h))
    # draw = ImageDraw.Draw(out)
    #
    # y = 0.0
    # while y < h:
    #     x = 0.0
    #     # 计算这一行真正要裁切的 y 范围
    #     y0 = int(y)
    #     y1 = min(h, int(math.ceil(y + grid_size)))
    #
    #     while x < w:
    #         x0 = int(x)
    #         x1 = min(w, int(math.ceil(x + grid_size)))
    #
    #         # 裁切当前块
    #         box = arr[y0:y1, x0:x1]
    #         if box.size == 0:
    #             x += grid_size
    #             continue
    #
    #         # 取主色、映射到调色板
    #         color = predominant_color(box)
    #         name  = nearest_color(color, palette)
    #
    #         # 解析填充值
    #         val = palette[name]
    #         if isinstance(val, tuple):
    #             fill = val
    #         else:
    #             fill = tuple(int(val.lstrip('#')[i:i+2], 16)
    #                          for i in (0, 2, 4))
    #
    #         # 用整数坐标绘制矩形
    #         draw.rectangle([x0, y0, x1, y1], fill=fill)
    #
    #         x += grid_size
    #     y += grid_size
    #
    # return out


def get_draw_list(img, grid_size, palette_tuple, predominant_color):
    """
    palette_tuple 是 load_local_palette() 返回的三元组：
      (palette_dict, names_list, lab_arr)
    """
    palette_dict, names_list, lab_arr = palette_tuple

    w, h = img.size
    arr = np.array(img)
    out_list = {}
    color_count = {}

    y = 0.0
    while y < h:
        x = 0.0
        y0 = int(y)
        y1 = min(h, int(math.ceil(y + grid_size)))

        while x < w:
            x0 = int(x)
            x1 = min(w, int(math.ceil(x + grid_size)))
            box = arr[y0:y1, x0:x1]
            if box.size == 0:
                x += grid_size
                continue

            # 取主色并映射
            color = predominant_color(box)
            name  = nearest_color(color, palette_tuple)

            # 解析填充色，一定要从 palette_dict 里拿
            val = palette_dict[name]
            if isinstance(val, tuple):
                fill = val
            else:
                fill = tuple(
                    int(val.lstrip('#')[i:i+2], 16)
                    for i in (0, 2, 4)
                )

            out_list[(x0, y0)] = [color, name, fill]
            color_count[name] = color_count.get(name, 0) + 1
            x += grid_size
        y += grid_size

    return out_list, color_count


def calculate_cell_size(out_list, font_path=FONT_PATH ):
    """
    计算绘制方格的边长：
    1. 使用 font_path 指定的字体，字号固定为 14。
    2. 遍历 out_list 中每个格子的 name，测量其文字宽高。
    3. 取最大宽度或高度（以较大值为准），并在此基础上增加 10% 内边距。
    4. 返回向上取整的正方形边长（整数像素）。

    参数:
        out_list: dict, 键为 (x,y) 或其他任意索引，值为 [color, name, fill]
        font_path: str, 字体文件路径

    返回:
        int, 方格边长 (像素)
    """

    # 加载 22 号字体
    font = ImageFont.truetype(font_path, FONT_SIZE)

    max_w = 0
    max_h = 0

    # 如果 out_list 是字典：遍历其 values；如果是二维列表，也可以 flatten
    iterable = out_list.values() if hasattr(out_list, 'values') else [
        cell for row in out_list for cell in row if cell is not None
    ]

    for entry in iterable:
        # entry 格式应为 [原色, name, fill]
        name = entry[1]
        # 获取文字的边界框 (x0, y0, x1, y1)
        bbox = font.getbbox(name)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > max_w:
            max_w = w
        if h > max_h:
            max_h = h

    # 以最大维度为基准，加 20% padding
    base = max(max_w, max_h)
    cell_size = int((base * 1.2) + 0.5)  # 四舍五入到最近整数

    return cell_size


def draw_list(out_list,isIndex=True,isLevel=False):
    # 测算每个方格的边长
    cell_size = calculate_cell_size(out_list, font_path=FONT_PATH)

    #  根据 out_list 键自动确定行列数
    x_coords = sorted({x for (x, y) in out_list.keys()})
    y_coords = sorted({y for (x, y) in out_list.keys()})
    cols = len(x_coords)
    rows = len(y_coords)

    #  建立新画布
    new_w = cols * cell_size
    new_h = rows * cell_size
    out_img = Image.new('RGB', (new_w, new_h))
    draw = ImageDraw.Draw(out_img)

    # 加载 22 号字体
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 6) 遍历每个格子，绘制色块＋文字
    for (orig_x, orig_y), (orig_color, name, fill) in out_list.items():
        # 计算在新画布上的行列索引
        col = x_coords.index(orig_x)
        row = y_coords.index(orig_y)

        if isLevel:
            col=(cols-1)-col

        x0 = col * cell_size
        y0 = row * cell_size

        # 6.1 绘制填充方块
        draw.rectangle(
            [x0, y0, x0 + cell_size, y0 + cell_size],
            fill=fill,
            outline=None
        )



        if isIndex:
            #决定文字颜色（明亮用黑，暗色用白）
            brightness = fill[0] * 0.299 + fill[1] * 0.587 + fill[2] * 0.114
            text_color = (0, 0, 0) if brightness > 186 else (255, 255, 255)

            #测量文字尺寸，居中绘制
            bbox = font.getbbox(name)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell_size - tw) / 2
            ty = y0 + (cell_size - th) / 2
            draw.text((tx, ty), name, fill=text_color, font=font)

    return  out_img,cell_size,rows,cols,font


def mapped_mosaic(out_list):
    # 1) 用 draw_list 拆分，得到 {(x,y): [orig_color, name, fill], ...}
    # out_list = draw_list(img, grid_size, palette,predominant_color)

    # # 2) 测算每个方格的边长
    # cell_size = calculate_cell_size(out_list, font_path=FONT_PATH)
    #
    # # 3) 根据 out_list 键自动确定行列数
    # x_coords = sorted({x for (x, y) in out_list.keys()})
    # y_coords = sorted({y for (x, y) in out_list.keys()})
    # cols = len(x_coords)
    # rows = len(y_coords)
    #
    # # 4) 建立新画布
    # new_w = cols * cell_size
    # new_h = rows * cell_size
    # out_img = Image.new('RGB', (new_w, new_h))
    # draw = ImageDraw.Draw(out_img)
    #
    # # 5) 加载 22 号字体
    # font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    #
    # color_count = {}
    #
    # # 6) 遍历每个格子，绘制色块＋文字
    # for (orig_x, orig_y), (orig_color, name, fill) in out_list.items():
    #     # 计算在新画布上的行列索引
    #     col = x_coords.index(orig_x)
    #     row = y_coords.index(orig_y)
    #     x0 = col * cell_size
    #     y0 = row * cell_size
    #
    #     # 6.1 绘制填充方块
    #     draw.rectangle(
    #         [x0, y0, x0 + cell_size, y0 + cell_size],
    #         fill=fill,
    #         outline=None
    #     )
    #
    #     # 6.2 决定文字颜色（明亮用黑，暗色用白）
    #     brightness = fill[0]*0.299 + fill[1]*0.587 + fill[2]*0.114
    #     text_color = (0,0,0) if brightness > 186 else (255,255,255)
    #
    #     # 6.3 测量文字尺寸，居中绘制
    #     bbox = font.getbbox(name)
    #     tw = bbox[2] - bbox[0]
    #     th = bbox[3] - bbox[1]
    #     tx = x0 + (cell_size - tw) / 2
    #     ty = y0 + (cell_size - th) / 2
    #     draw.text((tx, ty), name, fill=text_color, font=font)
    #
    #     # 6.4 更新计数
    #     color_count[name] = color_count.get(name, 0) + 1
    return draw_list(out_list)
    # return out_img, color_count, font, cell_size,cols, rows


def Level_mapped_mosaic(out_list):
    # # 1) 用 draw_list 拆分，得到 {(x,y): [orig_color, name, fill], ...}
    # # out_list = draw_list(img, grid_size, palette,,predominant_color)
    #
    # # 2) 测算每个方格的边长
    # cell_size = calculate_cell_size(out_list, font_path=FONT_PATH)
    #
    # # 3) 根据 out_list 键自动确定行列数
    # x_coords = sorted({x for (x, y) in out_list.keys()})
    # y_coords = sorted({y for (x, y) in out_list.keys()})
    # cols = len(x_coords)
    # rows = len(y_coords)
    #
    # # 4) 建立新画布
    # new_w = cols * cell_size
    # new_h = rows * cell_size
    # out_img = Image.new('RGB', (new_w, new_h))
    # draw = ImageDraw.Draw(out_img)
    #
    # # 5) 加载 22 号字体
    # font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    #
    #
    # # 6) 遍历每个格子，绘制色块＋文字
    # for (orig_x, orig_y), (orig_color, name, fill) in out_list.items():
    #     # 6.1 原列、行索引
    #     col = x_coords.index(orig_x)
    #     row = y_coords.index(orig_y)
    #
    #     # 6.2 水平翻转：新列索引 = (总列数 - 1) - 原列索引
    #     flipped_col = (cols - 1) - col
    #
    #     # 6.3 计算绘制起点
    #     x0 = flipped_col * cell_size
    #     y0 = row * cell_size
    #
    #     # 6.4 绘制填充方块
    #     draw.rectangle(
    #         [x0, y0, x0 + cell_size, y0 + cell_size],
    #         fill=fill,
    #         outline=None
    #     )
    #
    #     # 6.5 决定并绘制文字
    #     brightness = fill[0] * 0.299 + fill[1] * 0.587 + fill[2] * 0.114
    #     text_color = (0, 0, 0) if brightness > 186 else (255, 255, 255)
    #     bbox = font.getbbox(name)
    #     tw = bbox[2] - bbox[0]
    #     th = bbox[3] - bbox[1]
    #     tx = x0 + (cell_size - tw) / 2
    #     ty = y0 + (cell_size - th) / 2
    #     draw.text((tx, ty), name, fill=text_color, font=font)


    return draw_list(out_list,isLevel=True)

# def mapped_mosaic(img, grid_size, palette):
#     w, h = img.size
#     arr = np.array(img)
#     out = Image.new('RGB', (w, h))
#     draw = ImageDraw.Draw(out)
#     color_count = {}
#     # 内边距与可用尺寸
#     padding = max(int(grid_size * 0.1), 2)
#     avail = grid_size - 2 * padding
#     # 计算统一字号
#     sizes = []
#     FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "cambriaz.ttf")
#     FIXED_FONT_SIZE=22
#
#
#
#     for name in palette:
#         try:
#             f_tmp = ImageFont.truetype(FONT_PATH, avail)
#         except:
#             f_tmp = ImageFont.load_default()
#         bbox = f_tmp.getbbox(name)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#         if tw <= avail and th <= avail:
#             sizes.append(avail)
#         else:
#             scale = min(avail / tw, avail / th)
#             sizes.append(max(int(avail * scale), 6))
#     uniform_size = min(sizes) if sizes else max(avail, 6)
#     try:
#         font = ImageFont.truetype(FONT_PATH, uniform_size)
#     except:
#         font = ImageFont.load_default()
#
#
#     # 遍历绘制
#     for y in range(0, h, grid_size):
#         for x in range(0, w, grid_size):
#             box = arr[y:y+grid_size, x:x+grid_size]
#             if not box.size:
#                 continue
#             color = predominant_color(box)
#             name = nearest_color(color, palette)
#             if isinstance(palette[name], tuple):
#                 fill = palette[name]
#             else:
#                 fill = tuple(int(palette[name].lstrip('#')[i:i+2],16) for i in (0,2,4))
#             draw.rectangle([x, y, x+grid_size, y+grid_size], fill=fill)
#             brightness = fill[0] * 0.299 + fill[1] * 0.587 + fill[2] * 0.114
#             text_color = (0,0,0) if brightness > 186 else (255,255,255)
#             bbox = font.getbbox(name)
#             tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#             tx = x + padding + (avail - tw) / 2
#             ty = y + padding + (avail - th) / 2
#             draw.text((tx, ty), name, fill=text_color, font=font)
#             color_count[name] = color_count.get(name, 0) + 1
#     return out, color_count, font, padding, avail
def draw_5x5_grid(mosaic: Image.Image, cell_size: int,
                  line_color=(200, 200, 200, 200), line_width=4):
    # 1) 转成 RGBA，保留原图
    base = mosaic.convert("RGBA")
    w, h = base.size

    # 2) 新建透明图层叠加网格
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    step = cell_size * 5
    # 从 x=0,5*cell_size,10*cell_size... 到 w 画竖线
    for x in range(0, w+1, step):
        draw.line([(x, 0),(x, h)], fill=line_color, width=line_width)
    # 同理画横线
    for y in range(0, h+1, step):
        draw.line([(0, y),(w, y)], fill=line_color, width=line_width)

    # 3) 合成回 base 并转回原模式
    result = Image.alpha_composite(base, overlay)
    return result.convert(mosaic.mode)

def annotate_mapped(
        mosaic: Image.Image,
        cell_size: int,
        rows: int,
        cols: int,
        font: ImageFont.FreeTypeFont,
) -> Image.Image:
    # 新图尺寸：原中心图每边各加一格
    new_w = (cols + 2) * cell_size
    new_h = (rows + 2) * cell_size

    # 1) 背景 & 贴中心图
    annotated = Image.new('RGB', (new_w, new_h), (200, 200, 200))
    annotated.paste(mosaic, (cell_size, cell_size))

    draw = ImageDraw.Draw(annotated)

    # 2) 给中心区域每个格子画边框
    for r in range(rows):
        for c in range(cols):
            x0 = (c + 1) * cell_size
            y0 = (r + 1) * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], outline=(200,200,200), width=1)

    # 3) 在四周写坐标

    # 顶部：列号 0..cols-1
    for c in range(cols):
        label = str(c)
        tw, th = font.getbbox(label)[2:]
        # 该格左上角 x 坐标
        tx = (c + 1) * cell_size + (cell_size - tw) / 2
        ty = (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    # 底部：列号 0..cols-1
    for c in range(cols):
        label = str(c)
        tw, th = font.getbbox(label)[2:]
        tx = (c + 1) * cell_size + (cell_size - tw) / 2
        # y 在最下方留白区的垂直居中位置
        ty = (rows + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    # 左侧：行号 0..rows-1
    for r in range(rows):
        label = str(r)
        tw, th = font.getbbox(label)[2:]
        tx = (cell_size - tw) / 2
        ty = (r + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    # 右侧：行号 0..rows-1
    for r in range(rows):
        label = str(r)
        tw, th = font.getbbox(label)[2:]
        tx = (cols + 1) * cell_size + (cell_size - tw) / 2
        ty = (r + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    return annotated

# def annotate_mapped(mosaic, grid_size, font, padding, avail):
#     w, h = mosaic.size
#     cols = w // grid_size
#     rows = h // grid_size
#     bbox = font.getbbox(str(max(rows - 1, cols - 1)))
#     tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#     margin_left = tw + padding * 2
#     margin_top = th + padding * 2
#     annotated = Image.new('RGB', (margin_left + w, margin_top + h), (200,200,200))
#     draw = ImageDraw.Draw(annotated)
#     annotated.paste(mosaic, (margin_left, margin_top))
#     # 5×5 淡灰色网格
#     for x in range(0, w+1, grid_size * 5):
#         draw.line([(margin_left + x, margin_top), (margin_left + x, margin_top + h)], fill=(200,200,200), width=1)
#     for y in range(0, h+1, grid_size * 5):
#         draw.line([(margin_left, margin_top + y), (margin_left + w, margin_top + y)], fill=(200,200,200), width=1)
#     # 行坐标
#     for r in range(rows):
#         y = margin_top + r * grid_size + (grid_size - avail) / 2
#         x_pos = (margin_left - padding - tw) / 2
#         draw.text((x_pos, y), str(r), fill=(0,0,0), font=font)
#     # 列坐标
#     for c in range(cols):
#         x = margin_left + c * grid_size + (grid_size - avail) / 2
#         y_pos = (margin_top - padding - th) / 2
#         draw.text((x, y_pos), str(c), fill=(0,0,0), font=font)
#     return annotated


# def append_legend(mosaic, count, palette, sq=20, pad=5, font_path=FONT_PATH, fsize=14, bg=(255,255,255)):
#     temp = Image.new('RGB', (1,1))
#     td = ImageDraw.Draw(temp)
#     try:
#         font_leg = ImageFont.truetype(font_path, fsize)
#     except:
#         font_leg = ImageFont.load_default()
#
#     items = sorted(count.items(), key=lambda x: -x[1])
#     block_widths = []
#     for name, cnt in items:
#         text = f"{name}: {cnt}"
#         bw = td.textbbox((0,0), text, font=font_leg)[2]
#         block_widths.append(sq + pad + bw + pad)
#     if not block_widths:
#         return mosaic
#     cell_w = max(block_widths)
#     max_w = mosaic.width
#     cols = max_w // cell_w or 1
#     thg = td.textbbox((0,0), "Hg", font=font_leg)[3]
#     line_h = max(sq, thg) + pad
#     rows = (len(items) + cols - 1) // cols
#     legend_h = rows * line_h + pad
#     legend = Image.new('RGB', (max_w, legend_h), bg)
#     d = ImageDraw.Draw(legend)
#     r = max(2, sq // 5)
#     for idx, (name, cnt) in enumerate(items):
#         row, col = divmod(idx, cols)
#         x0 = col * cell_w + pad
#         y0 = row * line_h + pad
#         color = palette[name] if isinstance(palette[name], tuple) else tuple(
#             int(palette[name].lstrip('#')[i:i+2],16) for i in (0,2,4)
#         )
#         d.rounded_rectangle([x0, y0, x0+sq, y0+sq], radius=r, fill=color, outline=(0,0,0))
#         text = f"{name}: {cnt}"
#         txt_h = td.textbbox((0,0), text, font=font_leg)[3]
#         d.text((x0+sq+pad, y0 + (sq - txt_h)//2), text, fill=(0,0,0), font=font_leg)
#     final = Image.new('RGB', (mosaic.width, mosaic.height + legend_h), bg)
#     final.paste(mosaic, (0,0))
#     final.paste(legend, (0, mosaic.height))
#     return final

def append_legend(
    mosaic: Image.Image,
    count: dict,
    palette: dict,
    font_path=FONT_PATH,
    bg: tuple = (255, 255, 255)
) -> Image.Image:
    """
    在 mosaic 底部追加图例：
      - 色块边长为 mosaic 高度的 15%
      - 文字字号与色块边长保持一致
      - 色块和文字排成若干行列

    参数:
      - mosaic: PIL.Image 对象
      - count: {name: 次数}
      - palette: {name: RGB tuple 或 "#RRGGBB"}
      - pad: 图例内部元素间距（像素）
      - font_path: 字体文件路径
      - bg: 图例背景色

    返回:
      - 包含图例的新 Image 对象
    """
    # 计算色块大小和字体大小
    h = mosaic.height
    sq = int(h * 0.02)
    fsize = sq
    pad=6

    # 临时画布测文字
    temp = Image.new('RGB', (1, 1))
    td = ImageDraw.Draw(temp)
    try:
        font_leg = ImageFont.truetype(font_path, fsize)
    except:
        font_leg = ImageFont.load_default()

    # 排序并测每项宽度
    items = sorted(count.items(), key=lambda x: -x[1])
    block_widths = []
    for name, cnt in items:
        text = f"{name}: {cnt}"
        bw = td.textbbox((0, 0), text, font=font_leg)[2]
        block_widths.append(sq + pad + bw + pad)
    if not block_widths:
        return mosaic

    cell_w = max(block_widths)
    max_w = mosaic.width
    cols = max_w // cell_w or 1
    # 行高：取色块高度与文字高度中的较大值，再加间距
    th = td.textbbox((0,0), "Hg", font=font_leg)[3]
    line_h = max(sq, th) + pad
    rows = (len(items) + cols - 1) // cols
    legend_h = rows * line_h + pad

    # 创建图例画布
    legend = Image.new('RGB', (max_w, legend_h), bg)
    d = ImageDraw.Draw(legend)
    r = max(2, sq // 5)

    # 绘制色块和文字
    for idx, (name, cnt) in enumerate(items):
        row, col = divmod(idx, cols)
        x0 = col * cell_w + pad
        y0 = row * line_h + pad
        # 解析色值
        color = palette[name] if isinstance(palette[name], tuple) else tuple(
            int(palette[name].lstrip('#')[i:i+2], 16) for i in (0,2,4)
        )
        d.rounded_rectangle([x0, y0, x0+sq, y0+sq], radius=r, fill=color, outline=(0,0,0))
        text = f"{name}: {cnt}"
        txt_h = td.textbbox((0, 0), text, font=font_leg)[3]
        d.text((x0+sq+pad, y0 + (sq-txt_h)/2), text, fill=(0,0,0), font=font_leg)

    # 合并图例与主图
    final = Image.new('RGB', (mosaic.width, mosaic.height + legend_h), bg)
    final.paste(mosaic, (0, 0))
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
palette_file = st.file_uploader("上传调色板 JSON（不选择则采取默认值 mard：221）", type=['json'])
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

    method = st.radio(
        "选取主色算法",
        ("中位数:像素图推荐使用", "最大值:像素图，且图片质量较好时推荐使用","平均值:非像素图推荐使用"),
        index=0,
        key="color_method",
        horizontal = True
    )

    if st.session_state.color_method.startswith("中位数"):
        predominant_color = predominant_median
    elif st.session_state.color_method.startswith("最大值"):
        predominant_color = predominant_max
    else:
        predominant_color = predominant_mean

    col1, col2 = st.columns([8, 2])
    # 右侧精确输入
    with col2:
        st.number_input(
            "",
            min_value=1.0,
            value=float(st.session_state.grid_size),
            step=0.01,
            format="%.2f",
            key='num_in',
            label_visibility='hidden',
            on_change=lambda: st.session_state.update(grid_size=st.session_state.num_in)
        )

    # 左侧滑块
    with col1:
        st.slider(
            "",
            min_value=1.0,
            max_value=float(min(img.size) // 3),
            value=float(st.session_state.grid_size),
            step=0.01,
            format="%.2f",
            key='slider',
            label_visibility='hidden',
            on_change=lambda: st.session_state.update(grid_size=st.session_state.slider)
        )
    gs = st.session_state.grid_size
    st.image(draw_grid_overlay(img, gs), caption=f"网格预览", use_container_width=True)

    #     out_list, color_count=get_draw_list(img, gs, palette,predominant_color)
    #     basic,cell_size,rows,cols,font = draw_list(out_list,isIndex=False)
    #     out_img = draw_list(out_list)[0]
    #     out_img = draw_5x5_grid(out_img, cell_size)
    #     annotated = annotate_mapped(out_img, cell_size, rows,cols,font)
    #     final_img = append_legend(annotated,color_count, palette)
    #
    #     out_level_img = Level_mapped_mosaic(out_list)[0]
    #     out_level_img = draw_5x5_grid(out_level_img, cell_size)
    #     annotated = annotate_mapped(out_level_img, cell_size, rows, cols, font)
    #     level_img = append_legend(annotated, color_count, palette)
    #
    #     # 缩放以保证最小格尺寸
    #     min_cell = 10
    #     scale = math.ceil(min_cell / gs) if gs < min_cell else 1
    #     if scale > 1:
    #         basic = basic.resize((basic.width * scale, basic.height * scale), Image.NEAREST)
    #         final_img = final_img.resize((final_img.width * scale, final_img.height * scale), Image.NEAREST)
    #         level_img=level_img.resize((final_img.width * scale, final_img.height * scale), Image.NEAREST)
    #     st.subheader("预览")
    #     st.image(basic, use_container_width=True)
    #     # st.subheader("图纸")
    #     # # st.image(final_img, use_container_width=True)
    #     # # arr = np.array(final_img)
    #     # # st.image(arr, use_container_width=True)
    #     # st.image(final_img, use_container_width=True, output_format='JPEG')
    #     st.session_state["final_img"] = final_img
    #     st.session_state["level_img"] = level_img
    if st.button("生成图纸"):
        try:
            progress = st.progress(0)
            start=time.time()
            MAX_SECONDS=40
            # 步骤 1：分块并统计
            out_list, color_count = get_draw_list(img, gs, palette, predominant_color)
            elapsed = time.time() - start
            if elapsed > MAX_SECONDS:
                raise TimeoutError
            progress.progress(20)

            # 步骤 2：基本马赛克
            basic, cell_size, rows, cols, font = draw_list(out_list, isIndex=False)
            elapsed = time.time() - start
            if elapsed > MAX_SECONDS:
                raise TimeoutError
            progress.progress(40)

            # 步骤 3：生成图纸并注释
            final_img = draw_list(out_list)[0]
            final_img = draw_5x5_grid(final_img, cell_size)
            annotated = annotate_mapped(final_img, cell_size, rows, cols, font)
            final_img = append_legend(annotated, color_count, palette[0])
            elapsed = time.time() - start
            if elapsed > MAX_SECONDS:
                raise TimeoutError
            progress.progress(60)

            # 步骤 4：生成反转
            out_level_img = Level_mapped_mosaic(out_list)[0]
            out_level_img = draw_5x5_grid(out_level_img, cell_size)
            annotated = annotate_mapped(out_level_img, cell_size, rows, cols, font)
            level_img = append_legend(annotated, color_count, palette[0])
            elapsed = time.time() - start
            if elapsed > MAX_SECONDS:
                raise TimeoutError
            progress.progress(80)

            # 步骤 5：尺寸调整
            min_cell = 10
            scale = math.ceil(min_cell / gs) if gs < min_cell else 1
            if scale > 1:
                basic = basic.resize((basic.width * scale, basic.height * scale), Image.NEAREST)
                final_img = final_img.resize((final_img.width * scale, final_img.height * scale), Image.NEAREST)
                level_img = level_img.resize((level_img.width * scale, level_img.height * scale), Image.NEAREST)
            elapsed = time.time() - start
            if elapsed > MAX_SECONDS:
                raise TimeoutError
            progress.progress(100)

        except TimeoutError:
            st.error(f"⚠️ 处理已超过 {MAX_SECONDS} 秒，像素格数量可能过多，建议调大网格大小后再试。")
        else:
            # 成功完成
            st.success("✅ 生成完毕")
            st.subheader("预览")
            st.image(basic, use_container_width=True)
            # st.subheader("图纸")
            # # st.image(final_img, use_container_width=True)
            # # arr = np.array(final_img)
            # # st.image(arr, use_container_width=True)
            # st.image(final_img, use_container_width=True, output_format='JPEG')
            st.session_state["final_img"] = final_img
            st.session_state["level_img"] = level_img


    #如果缓存里有，就展示并给下载按钮
    if "final_img" in st.session_state:
        st.subheader("图纸（预览）")
        st.image(
            st.session_state["final_img"],
            use_container_width=True
        )

        # 准备原始大图二进制
        buf = io.BytesIO()
        st.session_state["final_img"].save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="⬇️ 下载原始大小图纸",
            data=buf,
            file_name=f"图纸_fullsize{time.time()}.png",
            mime="image/png"
        )

    if "level_img" in st.session_state:
        st.subheader("图纸（水平反转预览）")
        st.image(
            st.session_state["level_img"],
            use_container_width=True
        )

        # 准备原始大图二进制
        buf = io.BytesIO()
        st.session_state["level_img"].save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="⬇️ 下载原始大小图纸",
            data=buf,
            file_name=f"水平反转图纸_fullsize{time.time()}.png",
            mime="image/png"
        )