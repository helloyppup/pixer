import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab
from colorfix import build_color_mapping,merge_similar_colors

import streamlit as st


# 画可视化网格
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


def get_draw_list(img, grid_size, palette, predominant_color):

    w, h = img.size
    # cols = math.ceil(w / grid_size)
    # rows = math.ceil(h / grid_size)
    arr = np.array(img)
    cols=int(w//grid_size)
    rows=int(h//grid_size)

    coords, colors = [], []


    margin_frac = 0.2
    inset = int(grid_size * margin_frac)


    # 取色
    for j in range(rows):
        y0 = int(round(j * grid_size))
        y1 = int(round(min(h, (j + 1) * grid_size)))
        for i in range(cols):
            x0 = int(round(i * grid_size))
            x1 = int(round(min(w, (i + 1) * grid_size)))

            # 先取整格
            full_box = arr[y0:y1, x0:x1]
            if full_box.size == 0 :
                continue

            # 计算内缩后的子区域坐标
            xi0, yi0 = x0 + inset, y0 + inset
            xi1, yi1 = x1 - inset, y1 - inset
            # 如果内缩后有效面积足够，否则用整格
            if xi1 > xi0 and yi1 > yi0:
                sub = arr[yi0:yi1, xi0:xi1]
            else:
                sub = full_box

            # 主色提取仍然调用 predominant_color，只是传 sub 而不是 full_box
            coords.append((x0, y0))
            c = tuple(predominant_color(sub))
            colors.append(c)


    # 映射到lab空间
    colors_arr = np.array(colors, dtype=float) / 255.0
    colors_lab = rgb2lab(colors_arr.reshape(-1, 1, 3)).reshape(-1, 3)

    # 去相近色
    colors=merge_similar_colors(colors_lab)  #lab
    # print(f"去重颜色{colors}")

    # 批量去重＋映射

    mapping = build_color_mapping(colors, palette)


    # print(print(type(colors[0])))
    # print(coords)


    # 最终填充 out_list 和 color_count
    out_list, color_count = {}, {}
    for coord,col in zip(coords, colors):
        # print(f"coord is {coord}  {type(coord)}\n")
        # print(f"col is {col}  {type(col) }\n")

        name, hex,rgb = mapping[col]
        # print(f"mapping:{mapping[col]}")
        out_list[coord] = [col, name, hex,rgb]
        color_count[name] = color_count.get(name, 0) + 1

    return out_list, color_count


def calculate_cell_size(out_list, padding=0.2,font_path=None,font_size=None ):
    """

    :param out_list: 要输出的文字
    :param font_path:
    :return:
    """
    font_path = st.session_state.FONT_PATH
    font_size = st.session_state.FONT_SIZE
    # 加载 22 号字体
    font = ImageFont.truetype(font_path, font_size)

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
    padding=1+padding
    cell_size = int((base * padding) + 0.5)  # 四舍五入到最近整数

    return cell_size


def draw_list(out_list,isIndex=True,isLevel=False):
    # 测算每个方格的边长
    cell_size = calculate_cell_size(out_list, font_path=st.session_state.FONT_PATH)

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

    # 加载 字体
    font = ImageFont.truetype(st.session_state.FONT_PATH, st.session_state.FONT_SIZE)

    # 遍历每个格子，绘制色块＋文字
    #print(f"out_list: {out_list}")
    for (orig_x, orig_y), (orig_color, name, hex,rgb) in out_list.items():
        # print(orig_color)

        # 计算在新画布上的行列索引
        col = x_coords.index(orig_x)
        row = y_coords.index(orig_y)

        if isLevel:
            col=(cols-1)-col

        x0 = col * cell_size
        y0 = row * cell_size

        # print(f"RGB:{hex}:{rgb}")
        rgb=tuple(rgb)
        # if name=="C9":
        #     print(rgb)


        # 6.1 绘制填充方块
        draw.rectangle(
            [x0, y0, x0 + cell_size, y0 + cell_size],
            fill=rgb,
            outline=None
        )
        # pixel = out_img.getpixel((x0 + 1, y0 + 1))
        # print(f"{pixel},{rgb}")





        if isIndex:
            # 如果 fill 是 "#RRGGBB" 或 "#RGB" 格式的字符串，就先把它解析成 (R, G, B)
            if isinstance(rgb, str) and rgb.startswith('#'):
                hexstr = rgb.lstrip('#')
                if len(hexstr) == 3:  # 短格式 "#RGB"
                    fill_rgb = tuple(int(c * 2, 16) for c in hexstr)
                elif len(hexstr) == 6:  # 长格式 "#RRGGBB"
                    fill_rgb = tuple(int(hexstr[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    raise ValueError(f"Unsupported hex color: {rgb}")
            else:
                fill_rgb = rgb  # 原本就是 (R, G, B) 三元组

            #决定文字颜色（明亮用黑，暗色用白）
            brightness = fill_rgb[0] * 0.299 + fill_rgb[1] * 0.587 + fill_rgb[2] * 0.114
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
    return draw_list(out_list)

def Level_mapped_mosaic(out_list):
    return draw_list(out_list,isLevel=True)

def draw_5x5_grid(mosaic: Image.Image, cell_size: int,
                  line_color=(200, 200, 200, 200), line_width=4):
    # 转成 RGBA，保留原图
    base = mosaic.convert("RGBA")
    w, h = base.size

    #新建透明图层叠加网格
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    step = cell_size * 5
    # 从 x=0,5*cell_size,10*cell_size... 到 w 画竖线
    for x in range(0, w+1, step):
        draw.line([(x, 0),(x, h)], fill=line_color, width=line_width)
    # 同理画横线
    for y in range(0, h+1, step):
        draw.line([(0, y),(w, y)], fill=line_color, width=line_width)

    #合成回 base 并转回原模式
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

    # 顶部：列号 0..cols
    for c in range(cols):
        label = str(c+1)
        tw, th = font.getbbox(label)[2:]
        # 该格左上角 x 坐标
        tx = (c + 1) * cell_size + (cell_size - tw) / 2
        ty = (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    # 底部：列号 1..cols
    for c in range(cols):
        label = str(c+1)
        tw, th = font.getbbox(label)[2:]
        tx = (c + 1) * cell_size + (cell_size - tw) / 2
        # y 在最下方留白区的垂直居中位置
        ty = (rows + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0,0,0), font=font)

    # 左侧：行号 1..rows
    for r in range(rows):
        label = str(r + 1)
        tw, th = font.getbbox(label)[2:]
        tx = (cell_size - tw) / 2
        ty = (r + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    # 右侧：行号 1..rows
    for r in range(rows):
        label = str(r + 1)
        tw, th = font.getbbox(label)[2:]
        tx = (cols + 1) * cell_size + (cell_size - tw) / 2
        ty = (r + 1) * cell_size + (cell_size - th) / 2
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

    return annotated


def append_legend(
    mosaic: Image.Image,
    count: dict,
    palette: dict,
    font_path=None,
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
    font_path = st.session_state.FONT_PATH
    # 计算色块大小和字体大小
    h = mosaic.height
    sq = int(h * 0.03)
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
    # print("items:", items)
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
    # print(f"items: {items}")
    for idx, (name, cnt) in enumerate(items):
        row, col = divmod(idx, cols)
        x0 = col * cell_w + pad
        y0 = row * line_h + pad
        # 解析色值
        # print(f"{name}: {cnt}")
        d.text((x0, y0), name, font=font_leg)

        try:
            # print(f"palette: {palette}")
            names, rgbs, labs, hexs,_ =palette
            # print(f"names: {names}")
            if name in names:
                idx = names.index(name)
                color = tuple(rgbs[idx])
        except ValueError:
            raise KeyError(f"不存在这个name {name!r}")


        d.rounded_rectangle([x0, y0, x0+sq, y0+sq], radius=r, fill=color, outline=(0,0,0))
        text = f"{name}: {cnt}"
        txt_h = td.textbbox((0, 0), text, font=font_leg)[3]
        d.text((x0+sq+pad, y0 + (sq-txt_h)/2), text, fill=(0,0,0), font=font_leg)

    # 合并图例与主图
    final = Image.new('RGB', (mosaic.width, mosaic.height + legend_h), bg)
    final.paste(mosaic, (0, 0))
    final.paste(legend, (0, mosaic.height))
    return final