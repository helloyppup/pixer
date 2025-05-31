import io
import math
import time

import cv2
import numpy as np
import  streamlit as st
from PIL import Image

from colorFile import  load_local_palette,load_palette
from colorfix import predominant_median,predominant_max,quantized_mode,predominant_mean

from draw import *

def draw():
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



        # —— 创建滑块，指定 key ——
    with st.sidebar:
        st.header("去杂色阈值")
        st.slider("L（亮度） 通道强度", 10, 400, key="L_THRESH")
        st.slider("A（红绿） 通道强度", 10, 400, key="A_THRESH")
        st.slider("B（蓝黄） 通道强度", 10, 400, key="B_THRESH")

        st.header("映射算法")
        st.slider("L（亮度）影响", 0, 20,  key="KL")
        st.slider("A（红绿）影响", 0, 20,  key="KC")
        st.slider("B（蓝黄）影响", 0, 20,  key="KH")

        st.header("降噪参数")
        st.slider("邻域直径 (d)", 0, 20,  key="d")
        st.slider("颜色融合度", 0, 150,  key="sigmaColor")
        st.slider("空间融合度", 0, 150,  key="sigmaSpace")


    if uploaded:
        # uploaded=denoised_test(uploaded)

        img = Image.open(uploaded).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]

        denoised_img_np = cv2.bilateralFilter(img_np, d=st.session_state.d, sigmaColor=st.session_state.sigmaColor, sigmaSpace=st.session_state.sigmaSpace)
        denoised_img = Image.fromarray(denoised_img_np[:, :, ::-1])

        st.image(denoised_img, caption="降噪图", use_container_width=True,output_format='PNG')





        st.markdown("#### 调整网格大小")

        method = st.radio(
            "选取主色算法",
            ("中位数:像素图推荐使用", "最大值:像素图，且图片质量较好时推荐使用","量化众数","平均值:非像素图推荐使用"),
            index=0,
            key="color_method",
            horizontal = True,
            help="像素图不要选择最后一个"
        )

        if st.session_state.color_method.startswith("中位数"):
            predominant_color = predominant_median
        elif st.session_state.color_method.startswith("最大值"):
            predominant_color = predominant_max
        elif st.session_state.color_method.startswith("量化众数"):
            predominant_color = quantized_mode
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
                max_value=float(min(img.size) // 20),
                value=float(st.session_state.grid_size),
                step=0.01,
                format="%.2f",
                key='slider',
                label_visibility='hidden',
                on_change=lambda: st.session_state.update(grid_size=st.session_state.slider)
            )
        gs = st.session_state.grid_size
        st.image(draw_grid_overlay(denoised_img, gs), caption=f"网格预览", use_container_width=True,output_format='PNG')
        if st.button("生成图纸",use_container_width=True,type="primary"):
            try:
                progress = st.progress(0)
                start=time.time()
                MAX_SECONDS=60
                # 步骤 1：分块并统计
                # print(type(palette[-1]))

                out_list, color_count = get_draw_list(denoised_img, gs, palette, predominant_color)
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
                final_img = append_legend(annotated, color_count, palette)
                elapsed = time.time() - start
                if elapsed > MAX_SECONDS:
                    raise TimeoutError
                progress.progress(60)

                # 步骤 4：生成反转
                out_level_img = Level_mapped_mosaic(out_list)[0]
                out_level_img = draw_5x5_grid(out_level_img, cell_size)
                annotated = annotate_mapped(out_level_img, cell_size, rows, cols, font)
                level_img = append_legend(annotated, color_count, palette)
                elapsed = time.time() - start
                if elapsed > MAX_SECONDS:
                    raise TimeoutError
                progress.progress(80)

                # test = draw_list(out_list, isIndex=False)[0]
                # st.image(test, caption=f"采样测试", use_container_width=True)

                # 步骤 5：尺寸调整
                min_cell = 10
                scale = math.ceil(min_cell / gs) if gs < min_cell else 1
                if scale > 1:
                    basic = basic.resize((basic.width * scale, basic.height * scale), Image.NEAREST)
                    final_img = final_img.resize((final_img.width * scale, final_img.height * scale), Image.NEAREST)
                    level_img = level_img.resize((level_img.width * scale, level_img.height * scale), Image.NEAREST)
                    # test_img=test.resize((basic.width * scale, basic.height * scale), Image.NEAREST)
                elapsed = time.time() - start
                if elapsed > MAX_SECONDS:
                    raise TimeoutError
                progress.progress(100)

            except TimeoutError:
                st.error(f"⚠️ 处理已超过 {MAX_SECONDS} 秒，像素格数量可能过多，建议调大网格大小后再试。")
            else:
                # 成功完成
                # x0 = 0 * cell_size
                # y0 = 0 * cell_size
                # px = basic.getpixel((x0 + cell_size // 2, y0 + cell_size // 2))
                # print(px)
                st.success("✅ 生成完毕")
                st.subheader("预览")
                st.image(basic, use_container_width=True,output_format='PNG')
                # st.subheader("图纸")
                # # st.image(final_img, use_container_width=True)
                # # arr = np.array(final_img)
                # # st.image(arr, use_container_width=True)
                # st.image(final_img, use_container_width=True, output_format='JPEG')
                st.session_state["final_img"] = final_img
                st.session_state["level_img"] = level_img

        if "final_img" in st.session_state and "level_img" in st.session_state:
            # 获取图像对象
            img1 = st.session_state["final_img"]
            img2 = st.session_state["level_img"]

            # 对齐高度
            max_height = max(img1.height, img2.height)

            # 创建新画布
            new_width = img1.width + 5 + img2.width  # 5px黑条宽度
            combined = Image.new("RGB", (new_width, max_height), color=(255, 255, 255))

            # 粘贴第一张图
            combined.paste(img1, (0, (max_height - img1.height) // 2))

            # 添加黑色分割条（网页3的间隔条思路）
            combined.paste(Image.new("RGB", (5, max_height), (0, 0, 0)), (img1.width, 0))

            # 粘贴第二张图
            combined.paste(img2, (img1.width + 5, (max_height - img2.height) // 2))

            # 显示预览
            st.subheader("拼接图纸预览")
            st.image(combined, use_container_width=True)

            # 生成下载按钮（网页6的保存逻辑）
            buf = io.BytesIO()
            combined.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label="⬇️ 下载拼接图纸",
                data=buf,
                file_name=f"combined_drawing_{time.time()}.png",
                mime="image/png"
            )


        #如果缓存里有，就展示并给下载按钮
        if "final_img" in st.session_state:
            st.subheader("图纸（预览）")
            st.image(
                st.session_state["final_img"],
                use_container_width=True,
                output_format='PNG'
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
                use_container_width=True,
                output_format='PNG'
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