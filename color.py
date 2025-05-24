import json
import os

from PIL import Image, ImageDraw
import math

def draw_color_grid(hex_colors, columns=15, square_size=50, output_path='color_grid.png'):
    """
    根据十六进制颜色列表绘制方块网格，每行 columns 个方块，保存为 PNG。

    :param hex_colors: List[str]，颜色的十六进制字符串，如 ['#FF0000', '00FF00', '0000FF']
    :param columns: int，每行方块数量，默认为 15
    :param square_size: int，每个方块的边长（像素），默认为 50
    :param output_path: str，输出文件路径，默认为 'color_grid.png'
    """
    # 确保所有颜色都以 '#' 开头
    hex_colors = [c if c.startswith('#') else '#' + c for c in hex_colors]

    total = len(hex_colors)
    rows = math.ceil(total / columns)

    # 计算画布尺寸
    width = columns * square_size
    height = rows * square_size

    # 创建 RGBA 画布（可支持透明背景）
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    for idx, color in enumerate(hex_colors):
        row = idx // columns
        col = idx % columns
        x0 = col * square_size
        y0 = row * square_size
        x1 = x0 + square_size
        y1 = y0 + square_size
        draw.rectangle([x0, y0, x1, y1], fill=color)

    # 保存为 PNG
    img.save(output_path)
    print(f'已保存：{output_path}')

if __name__ == '__main__':

    if os.path.exists("palette_1.json"):
        with open("palette_1.json", "r", encoding="utf-8") as f:
            data = json.load(f)

    colors = []
    for c in data:
        colors.append(c['color'])

    draw_color_grid(colors, columns=15, square_size=60, output_path='my_colors.png')
