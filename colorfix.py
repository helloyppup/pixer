import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from scipy.spatial import cKDTree
from skimage.color import rgb2lab, deltaE_ciede2000
from colorFile import srgb_to_linear
import streamlit as st
from collections import defaultdict

def nearest_color(color, palette):
    # print(time.time())
    names,rgbs,labs,hexs = palette

    # print(f"类型{type(lab_arr)}")

    # 输入颜色 Gamma 校正
    rgb = np.array(color, dtype=float) / 255.0
    rgb_linear = srgb_to_linear(rgb)  # 确保此函数已定义（参考前文）
    lab = rgb2lab(rgb_linear[np.newaxis, np.newaxis, :])[0, 0]


    # 计算 CIEDE2000 距离
    input_lab = LabColor(lab[0], lab[1], lab[2])
    dists = [delta_e_cie2000(input_lab, p_lab) for p_lab in labs]
    idx = np.argmin(dists)

    return names[idx]

def build_color_mapping(colors, palette,k=20):
    """

    :param colors: lab颜色，被映射的颜色
    :param palette: 处理的色板
    :param k: 粗筛候选数（默认 20）
    :return:
    """
    names,rgbs,labs,hexs,rgb_line=palette
    # rgbs=srgb_to_linear(rgbs)
    # print(rgbs)

    # 处理输入的 Lab 颜色为np数组
    colors_lab = np.array(colors, dtype=float).reshape(-1, 3)
    # print(colors_lab)

    mapping = {}

    # 首先直接对比有没有一样的hex


    # 构建 KD 树进行粗筛
    tree = cKDTree(labs)
    _, idxs = tree.query(colors_lab, k=k)

    # print(f"idxs: {idxs}")

    for orig_lab, lab_vec, neigh in zip(colors, colors_lab, idxs):
        # 将原始 Lab 转换为浮点元组作为键
        key = tuple(float(x) for x in orig_lab)

        # 处理候选索引
        neigh = np.atleast_1d(neigh)
        lab_in = np.tile(lab_vec, (len(neigh), 1))
        lab_cand = labs[neigh]

        # 计算 ΔE2000 并找到最优候选
        de2000 = deltaE_ciede2000(lab_in[np.newaxis, :, :], lab_cand[np.newaxis, :, :],kL=st.session_state.KL/10,
                                  kC=st.session_state.KC/10,
                                  kH=st.session_state.KH/10)
        best_idx = np.argmin(de2000[0])
        sel_idx = neigh[best_idx]

        # 获取对应的调色板名称和 RGB
        name = names[sel_idx]
        # print(name)

        # fill_rgb = tuple(map(int, palette_rgb[sel_idx]))
        hex = hexs[sel_idx]
        rgb = rgbs[sel_idx]
        # if name == "C9":
        #     print(sel_idx)
        #     print(hex)
        #     print(rgbs[sel_idx])
        #     print(rgb)
        mapping[key] = (name, hex,rgb)

    # print(f"mapping: {mapping}")
    return mapping


def merge_similar_colors(lab_colors, l_thresh=None, a_thresh=None, b_thresh=None):
    l_thresh = st.session_state.L_THRESH
    a_thresh = st.session_state.A_THRESH
    b_thresh = st.session_state.B_THRESH

    print(f"执行降噪{l_thresh},{a_thresh},{b_thresh}")
    n = len(lab_colors)
    processed = np.zeros(n, dtype=bool)
    result = np.zeros_like(lab_colors)

    # 哈希分桶优化查找速度（各通道按阈值分桶）
    bucket_map = defaultdict(list)
    for i, (L, a, b) in enumerate(lab_colors):
        bucket_key = (int(L // l_thresh), int(a // a_thresh), int(b // b_thresh))
        bucket_map[bucket_key].append(i)

    for i in range(n):
        if processed[i]:
            continue

        # 获取当前颜色值
        L_curr, a_curr, b_curr = lab_colors[i]

        # 收集所有可能相邻的哈希桶（3x3x3=27个桶）
        l_bucket = int(L_curr // l_thresh)
        a_bucket = int(a_curr // a_thresh)
        b_bucket = int(b_curr // b_thresh)
        candidate_indices = []

        for dl in (-1, 0, 1):
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    bucket_key = (l_bucket + dl, a_bucket + da, b_bucket + db)
                    candidate_indices.extend(bucket_map.get(bucket_key, []))

        # 去重后筛选符合条件的颜色索引
        candidate_indices = list(set(candidate_indices))
        group = []
        for j in candidate_indices:
            if not processed[j]:
                Lj, aj, bj = lab_colors[j]
                if (abs(Lj - L_curr) <= l_thresh/10 and
                        abs(aj - a_curr) <= a_thresh/10 and
                        abs(bj - b_curr) <= b_thresh/10):
                    group.append(j)

        # 计算中位数并更新结果
        if group:
            median_color = np.median(lab_colors[group], axis=0)
            for idx in group:
                result[idx] = median_color
                processed[idx] = True

    return [tuple(r) for r in result]


def predominant_max(region):
    pixels = region.reshape(-1,3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(colors[counts.argmax()])


def quantized_mode(region, q1=32, q2=4, top_m=5, noise_frac=0.01):
    """
        两阶段量化 + 众数 + 噪声剔除主色提取

        参数
        ----
        region : ndarray, shape (H, W, 3)
            输入图像区域，dtype 通常为 uint8，RGB 通道值 0–255。
        q1 : int
            第一阶段量化步长（粗量化），建议 16–64。
        q2 : int
            第二阶段量化步长（细量化），建议 2–8。
        top_m : int
            第一阶段保留的主色桶数（从高到低选取前 top_m 个桶）。
        noise_frac : float
            噪声阈值（小于 region 像素总数 * noise_frac 的桶将被当作噪声剔除）。

        返回
        ----
        color : tuple of int
            最终提取的主色 RGB 三元组。
        """
    # 展平到 (N,3)
    pixels = region.reshape(-1, 3)
    total = pixels.shape[0]
    if total == 0:
        raise ValueError("Empty region provided to quantized_mode_robust.")

    # 第一阶段：粗量化
    buckets1 = (pixels // q1) * q1
    keys1, cnt1 = np.unique(buckets1, axis=0, return_counts=True)

    # 噪声剔除
    keep_mask = cnt1 > total * noise_frac
    if not np.any(keep_mask):
        # 如果所有桶都被剔除，则退回保留所有出现过的桶
        keep_mask = cnt1 > 0
    keys1_filt = keys1[keep_mask]
    cnt1_filt = cnt1[keep_mask]

    # 选取出现次数最多的 top_m 个桶
    m = min(top_m, cnt1_filt.size)
    top_idxs = cnt1_filt.argsort()[-m:]
    chosen_buckets = keys1_filt[top_idxs]

    # 第二阶段：在 top_m 桶对应的像素里再细量化
    # 构造 mask：像素量化后的值属于 chosen_buckets
    mask = (buckets1[:, None, :] == chosen_buckets[None, :, :]).all(axis=2).any(axis=1)
    sub = pixels[mask]

    # 对 sub 再做细量化并计数
    buckets2 = (sub // q2) * q2
    keys2, cnt2 = np.unique(buckets2, axis=0, return_counts=True)

    # 兜底：如果没有任何候选（极端情况），退回均值色
    if cnt2.size == 0:
        mean = np.mean(pixels, axis=0)
        best = np.atleast_1d(mean.astype(int))
        return tuple(best.tolist())

    # 选出第二阶段出现次数最多的桶
    best_idx = cnt2.argmax()
    best = keys2[best_idx]

    # 确保 best 是一维可迭代
    best = np.atleast_1d(best).astype(int)
    return tuple(best.tolist())

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