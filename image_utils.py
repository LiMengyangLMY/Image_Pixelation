from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import cv2

"""
函数说明：
1.load_color_database(file_path)：加载指定位置的数据集并返回一个dataframe；
2.get_base_coefficient(rgb_value):计算原有基础系数；
3.get_scene_coefficient(target_val, db_val):计算RGB增减系数；
4.find_nearest_color(target_rgb, color_database):在颜色数据集中找到指定RGB颜色的相近色号；
"""

def load_color_database(file_path):
    """加载颜色数据，返回颜色数据框"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"颜色数据库文件 {file_path} 不存在")
    
    color_data = pd.read_csv(file_path)
    color_data = color_data.dropna(subset=['R', 'G', 'B', 'num'])
    color_data['R'] = pd.to_numeric(color_data['R'], errors='coerce').fillna(0).astype(int)
    color_data['G'] = pd.to_numeric(color_data['G'], errors='coerce').fillna(0).astype(int)
    color_data['B'] = pd.to_numeric(color_data['B'], errors='coerce').fillna(0).astype(int)
    color_data['R'] = color_data['R'].clip(0, 255) #clip(0,255)表示数值约束再0-255之间
    color_data['G'] = color_data['G'].clip(0, 255)
    color_data['B'] = color_data['B'].clip(0, 255)
    return color_data

def get_base_coefficient(rgb_value):
    """原有基础系数计算"""
    if rgb_value < 0:
        return 0.2
    elif rgb_value > 50:
        return 1.0
    else:
        return 0.3 + (rgb_value / 50) * 0.5

def get_scene_coefficient(target_val, db_val):
    """RGB增减相关系数"""
    if target_val < 80 and db_val < target_val:
        return 0.7
    elif target_val > 170 and db_val > target_val:
        return 0.7
    else:
        return 1.0

def find_nearest_color(target_rgb, color_database):
    """找到与目标RGB值最接近的颜色编号"""
    tr, tg, tb = target_rgb
    color_data = color_database.copy()
    
    r_base = get_base_coefficient(tr)
    g_base = get_base_coefficient(tg)
    b_base = get_base_coefficient(tb)
    
    color_data['r_coeff'] = color_data['R'].apply(lambda x: r_base * get_scene_coefficient(tr, x))
    color_data['g_coeff'] = color_data['G'].apply(lambda x: g_base * get_scene_coefficient(tg, x))
    color_data['b_coeff'] = color_data['B'].apply(lambda x: b_base * get_scene_coefficient(tb, x))
    
    color_data['distance'] = np.sqrt(
        (color_data['r_coeff'] * (color_data['R'] - tr))**2 + 
        (color_data['g_coeff'] * (color_data['G'] - tg))**2 + 
        (color_data['b_coeff'] * (color_data['B'] - tb))**2
    )
    
    nearest_color = color_data.loc[color_data['distance'].idxmin()]
    return nearest_color['num'], (nearest_color['R'], nearest_color['G'], nearest_color['B'])

from sklearn.cluster import KMeans



import numpy as np
from PIL import Image

def image_to_color_array(input_path, color_db_path, scale_factor=0.03):
    """
    将图片转换为 4 维数组：[颜色编码, R, G, B]
    
    返回：
        color_array: shape = (H, W, 4)
    """
    color_database = load_color_database(color_db_path)
    color_code_count = {}

    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        raise Exception(f"打开图片失败: {e}")

    original_width, original_height = img.size
    target_width = max(1, int(original_width * scale_factor))
    target_height = max(1, int(original_height * scale_factor))
    #压缩图片关键步骤
    img_resized = img.resize((target_width, target_height), Image.NEAREST)

    # 初始化 4 维数组
    color_array = np.empty((target_height, target_width, 4), dtype=object)

    for y in range(target_height):
        for x in range(target_width):
            r, g, b = img_resized.getpixel((x, y))
            color_code, nearest_rgb = find_nearest_color((r, g, b), color_database)
            nr, ng, nb = nearest_rgb

            color_array[y, x] = [color_code, nr, ng, nb]

            if color_code in color_code_count:
                color_code_count[color_code]["count"] += 1
            else:
                color_code_count[color_code] = {
                    "count": 1,
                    "r": int(nr),
                    "g": int(ng),
                    "b": int(nb)
                }
    return color_array,color_code_count


from PIL import Image, ImageDraw, ImageFont

def visualize_color_array(color_array, pixel_scale):
    """
    将 4 维数组可视化为图片
    
    参数：
        color_array: (H, W, 4) -> [color_code, R, G, B]
        pixel_scale: 单个像素块边长
    """
    h, w, _ = color_array.shape
    output_width = int(w * pixel_scale)
    output_height = int(h * pixel_scale)

    output_img = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(output_img)

    font_size = max(int(pixel_scale / 2), 1)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for y in range(h):
        for x in range(w):
            color_code, r, g, b = color_array[y, x]

            draw.rectangle(
                [
                    x * pixel_scale,
                    y * pixel_scale,
                    (x + 1) * pixel_scale - 1,
                    (y + 1) * pixel_scale - 1
                ],
                fill=(r, g, b)
            )

            text_color = (255 - r, 255 - g, 255 - b)

            draw.text(
                (x * pixel_scale + pixel_scale // 2,
                 y * pixel_scale + pixel_scale // 2),
                str(color_code),
                fill=text_color,
                font=font,
                anchor="mm"
            )

    return output_img

import numpy as np
from sklearn.cluster import KMeans

def reduce_color_array(color_array, color_code_count, target_cluster_count):
    """
    基于 color_code_count 的 RGB 聚类，减少颜色种类数量
    """
    target_cluster_count = int(target_cluster_count)
    codes = list(color_code_count.keys())
    actual_color_count = len(codes)
    if target_cluster_count >= actual_color_count:
        raise ValueError(
            f"目标簇数 ({target_cluster_count}) ≥ 实际颜色数 ({actual_color_count})"
        )

    # ========= 1. 构建 RGB 数据（来自 color_code_count） =========
    rgb_array = np.array(
        [[v["r"], v["g"], v["b"]] for v in color_code_count.values()],
        dtype=int
    )

    # 对应的颜色编码顺序
    code_list = list(color_code_count.keys())
    print(target_cluster_count)
    
    # ========= 2. KMeans 聚类（仅 RGB） =========
    kmeans = KMeans(n_clusters=target_cluster_count, random_state=0)
    labels = kmeans.fit_predict(rgb_array)
    
    # ========= 3. 簇内选择代表颜色（按 count 最大） =========
    cluster_to_rep = {}

    for cluster_id in range(target_cluster_count):
        # 找到该簇内的颜色编码
        cluster_codes = [
            code_list[i]
            for i in range(len(code_list))
            if labels[i] == cluster_id
        ]

        # 按出现次数选最大者
        rep_code = max(
            cluster_codes,
            key=lambda c: color_code_count[c]["count"]
        )

        rep_info = color_code_count[rep_code]

        cluster_to_rep[cluster_id] = {
            "code": rep_code,
            "r": rep_info["r"],
            "g": rep_info["g"],
            "b": rep_info["b"]
        }

    # ========= 4. 构建旧 code -> 新 code / rgb 映射 =========
    code_replace_map = {}

    for i, old_code in enumerate(code_list):
        cluster_id = labels[i]
        rep = cluster_to_rep[cluster_id]
        code_replace_map[old_code] = rep

    # ========= 5. 替换 color_array =========
    h, w, _ = color_array.shape
    new_color_array = np.empty_like(color_array, dtype=object)

    for y in range(h):
        for x in range(w):
            old_code, _, _, _ = color_array[y, x]
            rep = code_replace_map[old_code]

            new_color_array[y, x] = [
                rep["code"],
                int(rep["r"]),
                int(rep["g"]),
                int(rep["b"])
            ]
    
    return new_color_array


def cartoon_color_array(color_array):
    iso_limit=2
    stark_diff=135
    """
    针对 color_array 的极速清理函数
    参数:
        color_array: (H, W, 4) 结构的 numpy 数组 -> [color_code, R, G, B]
        iso_limit: 5x5邻域内同色数量低于此值则判定为孤立点
        stark_diff: 细节保护阈值，RGB距离超过此值则不处理（保护瞳孔）.值越大颜色越纯。
    返回:
        new_array: 格式与输入完全一致的 (H, W, 4) 数组
    """
    h, w, _ = color_array.shape
    # 显式使用 np.copy 确保不修改原始数据
    new_array = np.copy(color_array)
    
    # 提取代码层用于极速 4-邻域 比较
    codes = color_array[:, :, 0]
    
    # 5x5 检查半径
    radius = 2 
    
    # 遍历非边缘区域
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            current_code = codes[y, x]
            
            # 【提速核心】边缘跳过：
            # 如果中心点与上下左右 4 个格子的 color_code 完全一致，说明在色块内部，直接跳过
            if (current_code == codes[y-1, x] and current_code == codes[y+1, x] and 
                current_code == codes[y, x-1] and current_code == codes[y, x+1]):
                continue
            
            # 获取 5x5 邻域数据块 (25, 4)
            patch = color_array[y-2:y+3, x-2:x+3].reshape(-1, 4)
            
            # 移除中心点 (索引为 12)，剩余 24 个邻居
            neighbors = np.delete(patch, 12, axis=0)
            neighbor_codes = neighbors[:, 0]
            
            # --- 判定 1: 空间孤立检查 ---
            # 统计邻域内有多少个格子的 code 与当前一致
            if np.sum(neighbor_codes == current_code) < iso_limit:
                
                # --- 判定 2: 细节保护 (NumPy 向量化计算 RGB 距离) ---
                current_rgb = color_array[y, x, 1:4].astype(float)
                neighbor_rgbs = neighbors[:, 1:4].astype(float)
                
                # 一次性计算当前像素与 24 个邻居的欧氏距离
                dists = np.linalg.norm(neighbor_rgbs - current_rgb, axis=1)
                
                # 如果该像素与周围所有颜色反差都极大 (all > stark_diff)，判定为瞳孔细节，保留
                if np.all(dists > stark_diff):
                    continue
                
                # --- 判定 3: 杂色替换 (邻域共识投票) ---
                # 统计邻域内出现频率最高的 color_code
                vals, counts = np.unique(neighbor_codes, return_counts=True)
                most_common_code = vals[np.argmax(counts)]
                
                # 从邻域数据中提取该 code 对应的第一组完整 [code, R, G, B]
                replace_idx = np.where(neighbor_codes == most_common_code)[0][0]
                new_array[y, x] = neighbors[replace_idx]

    return new_array


def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=20):
    # 生成颜色数组
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array, pixel_scale)
    return output_path, output_img, set(color_array[:, :, 0].flatten())

def reduce_image_colors(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=1, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 聚类减少颜色
    color_array = reduce_color_array(color_array, color_code_count, target_color_count)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array, pixel_scale)
    return output_path, output_img

def clean_cartoon_image(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    color_array = cartoon_color_array(color_array)
    output_img = visualize_color_array(color_array, pixel_scale)

    return output_path, output_img