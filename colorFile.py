import json
import os

import numpy as np
from skimage.color import rgb2lab

import streamlit as st


def load_local_palette(
    json_path: str = "palette.json",
    cache_path: str = "palette_lab_cache.json",
    overwrite: bool = True,
):
    """
    加载 palette_1.json，并缓存一次性计算好的 Lab 数组到 cache_path。
    返回：
      palette_dict: {name: "#RRGGBB"}
      names_list:   [name1, name2, …]
      lab_arr:      np.ndarray shape (N,3)，对应 names_list 的 Lab 值

    不会修改原来的 palette_1.json，只会创建/读取 cache_path。
    """
    # 如果缓存已经存在，直接读它
    if not overwrite:
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
            hexs = cache["hexs"]
            names   = cache["names"]
            rgbs = np.array(cache["rgbs"],dtype=int)
            rgb_line=np.array(cache["rgb_line"],dtype=int)
            labs = np.array(cache["labs"], dtype=float)

            # print(hexs[56])

            return names,rgbs,labs,hexs,rgb_line

    # 否则，先读原始 palette_1.json
    if not os.path.exists(json_path):
        # 没有 JSON 文件就返回空结构
        return {}, [], np.zeros((0,3), float)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    rgb_line,rgbs=color_to_rgb(data)
    labs=color_to_lab(rgbs)
    names=list(data.keys())
    hexs=[data[n]["hex"] for n in data]
    write_to_cach(cache_path, names,hexs, rgbs, labs,rgb_line)



    return names,rgbs,labs,hexs,rgb_line

def color_to_nplist(colors):
    names = list(colors.keys())  # ["C1","C2",…]
    hexs = [colors[n]["hex"] for n in names]
    rgbs = [tuple(colors[n]["rgb"]) for n in names]
    labs = [tuple(colors[n]["lab"]) for n in names]

    # 定义 structured dtype
    dt = np.dtype([
        ("name", "U5"),  # 最长 5 字符的 Unicode
        ("hex", "U7"),  # "#rrggbb"
        ("rgb", "i1", (3,)),  # 3 个 8-bit 整数
        ("lab", "f8", (3,)),  # 3 个 64-bit 浮点
    ])

    arr = np.zeros(len(names), dtype=dt)
    arr["name"] = names
    arr["hexs"] = hexs
    arr["rgbs"] = rgbs
    arr["labs"] = labs

    return arr


def srgb_to_linear(rgb):
    # sRGB逆Gamma校正
    linear = np.where(rgb <= 0.04045,
                      rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def color_to_rgb(data: dict):
    srgb = []
    for meta in data.values():
        hex_str = meta["hex"].lstrip("#")
        vals = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        # vals = [int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4)]
        srgb.append(vals)
    # print(srgb)
    srgb_arr = np.array(srgb, dtype=int)
    rgb_line  = srgb_to_linear(srgb_arr)

    # 这里把线性 RGB 再映射回 0–255 整数，用于绘图
    rgb8 = ( (rgb_line * 255).round().astype(int) ).tolist()

    return rgb_line, srgb_arr

def color_to_lab(rgb_list):
    rgb_list = rgb_list.astype(float) / 255.0
    # 步骤3: 转换到Lab空间
    return rgb2lab(rgb_list[np.newaxis, :, :])[0]


def write_to_cach(cache_path,name,hex,rgb,lab,rgb_line):

    # 序列化到缓存文件，下次直接用
    rgb = rgb.tolist()
    rgb_line = rgb_line.tolist()
    lab = lab.tolist()

    cache = {
        "names": name,
        "rgbs": rgb,
        "hexs": hex,
        "labs":lab,
        "rgb_line":rgb_line,
    }



    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

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