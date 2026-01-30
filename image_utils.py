from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np

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

def clean_cartoon_artifacts(color_array, color_code_count, isolation_threshold=8, frequency_threshold=0.001):
    """
    清理卡通图像杂色的独立函数
    :param color_array: image_to_color_array 返回的 (H, W, 4) 数组
    :param color_code_count: 颜色计数统计字典
    :param isolation_threshold: 邻域判定阈值（1-8），越高越严格
    :param frequency_threshold: 全局频率阈值，低于此比例的颜色将被剔除
    """
    h, w, _ = color_array.shape
    new_array = color_array.copy()
    total_pixels = h * w

    # --- 步骤 1: 空间过滤 (消除孤立噪点) ---
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current_code = color_array[y, x, 0]
            
            # 获取周围 8 个格子的颜色编码
            neighbors = [
                color_array[y-1, x-1, 0], color_array[y-1, x, 0], color_array[y-1, x+1, 0],
                color_array[y, x-1, 0],                           color_array[y, x+1, 0],
                color_array[y+1, x-1, 0], color_array[y+1, x, 0], color_array[y+1, x+1, 0]
            ]
            
            # 如果当前颜色在邻域内出现次数极少
            if neighbors.count(current_code) < (8 - isolation_threshold + 1):
                # 找到邻域内最频繁的颜色
                most_common_code = max(set(neighbors), key=neighbors.count)
                
                # 从原 count 字典或邻域像素中获取该颜色的 RGB 信息进行替换
                for ny in range(y-1, y+2):
                    for nx in range(x-1, x+2):
                        if color_array[ny, nx, 0] == list(set(neighbors))[0]: # 简化逻辑，取邻域代表
                             new_array[y, x] = color_array[ny, nx].copy()
                             break

    # --- 步骤 2: 频率过滤 (剔除极少数出现的杂色) ---
    valid_codes = {code for code, info in color_code_count.items() 
                   if (info['count'] / total_pixels) >= frequency_threshold}
    
    if len(valid_codes) < len(color_code_count):
        for y in range(h):
            for x in range(w):
                code = new_array[y, x, 0]
                if code not in valid_codes:
                    # 寻找 RGB 距离最近的有效颜色进行替换
                    curr_rgb = np.array([new_array[y, x, 1], new_array[y, x, 2], new_array[y, x, 3]])
                    best_code = None
                    min_dist = float('inf')
                    
                    for vc in valid_codes:
                        v_info = color_code_count[vc]
                        dist = np.linalg.norm(curr_rgb - np.array([v_info['r'], v_info['g'], v_info['b']]))
                        if dist < min_dist:
                            min_dist = dist
                            best_code = vc
                    
                    # 更新数组值
                    rep = color_code_count[best_code]
                    new_array[y, x] = [best_code, int(rep['r']), int(rep['g']), int(rep['b'])]

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
    color_array = clean_cartoon_artifacts(color_array, color_code_count)
    output_img = visualize_color_array(color_array, pixel_scale)
    return output_path, output_img