from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans

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

def reduce_color_array(color_array, color_code_count, target_cluster_count):
    """
    基于 KMeans 聚类减少颜色种类数量，并同步更新统计字典
    """
    target_cluster_count = int(target_cluster_count)
    codes = list(color_code_count.keys())
    actual_color_count = len(codes)
    
    # 如果目标簇数大于等于实际颜色数，不进行聚类，直接返回
    if target_cluster_count >= actual_color_count:
        return color_array, color_code_count

    # 1. 构建 RGB 数据
    rgb_array = np.array(
        [[v["r"], v["g"], v["b"]] for v in color_code_count.values()],
        dtype=int
    )
    code_list = list(color_code_count.keys())
    
    # 2. KMeans 聚类
    kmeans = KMeans(n_clusters=target_cluster_count, random_state=0)
    labels = kmeans.fit_predict(rgb_array)
    
    # 3. 计算映射关系与更新统计
    cluster_to_rep = {}      # 簇ID -> 代表色信息
    new_color_code_count = {} # 新的统计字典
    
    for cluster_id in range(target_cluster_count):
        # 找到该簇内的所有原始编码
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_codes = [code_list[i] for i in cluster_indices]

        # 选取簇内出现次数（count）最多的编码作为代表色
        rep_code = max(cluster_codes, key=lambda c: color_code_count[c]["count"])
        rep_info = color_code_count[rep_code]

        # 记录代表色信息
        rep_data = {
            "code": rep_code,
            "r": int(rep_info["r"]),
            "g": int(rep_info["g"]),
            "b": int(rep_info["b"])
        }
        cluster_to_rep[cluster_id] = rep_data
        
        # 汇总该簇所有颜色的计数到代表色中
        new_color_code_count[rep_code] = {
            "count": sum(color_code_count[c]["count"] for c in cluster_codes),
            "r": rep_data["r"],
            "g": rep_data["g"],
            "b": rep_data["b"]
        }

    # 4. 构建旧 code -> 新信息的完整映射表
    code_replace_map = {}
    for i, old_code in enumerate(code_list):
        cluster_id = labels[i]
        code_replace_map[old_code] = cluster_to_rep[cluster_id]

    # 5. 应用替换到 color_array
    h, w, _ = color_array.shape
    new_color_array = np.empty_like(color_array, dtype=object)

    for y in range(h):
        for x in range(w):
            old_code = color_array[y, x, 0]
            rep = code_replace_map[old_code]
            new_color_array[y, x] = [rep["code"], rep["r"], rep["g"], rep["b"]]
    
    return new_color_array, new_color_code_count


    """
    针对 color_array 的清理函数（去噪），在处理过程中同步更新 color_code_count
    """
    iso_limit = 2      # 孤立点判定阈值
    stark_diff = 65    # 细节保护阈值
    
    h, w, _ = color_array.shape
    new_array = np.copy(color_array)
    
    # 深拷贝一份统计字典用于实时更新
    updated_counts = {k: v.copy() for k, v in color_code_count.items()}
    
    # 提取代码层用于比较
    codes = color_array[:, :, 0]
    radius = 2 
    
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            current_code = codes[y, x]
            
            # 内部点跳过（上下左右颜色一致则不处理）
            if (current_code == codes[y-1, x] and current_code == codes[y+1, x] and 
                current_code == codes[y, x-1] and current_code == codes[y, x+1]):
                continue
            
            # 获取 5x5 邻域
            patch = color_array[y-2:y+3, x-2:x+3].reshape(-1, 4)
            neighbors = np.delete(patch, 12, axis=0) # 移除中心点
            neighbor_codes = neighbors[:, 0]
            
            # 判定是否为孤立点
            if np.sum(neighbor_codes == current_code) < iso_limit:
                current_rgb = color_array[y, x, 1:4].astype(float)
                neighbor_rgbs = neighbors[:, 1:4].astype(float)
                dists = np.linalg.norm(neighbor_rgbs - current_rgb, axis=1)
                
                # 如果与周围颜色反差极大，判定为细节（如瞳孔），保留
                if np.all(dists > stark_diff):
                    continue
                
                # 投票选取邻域内出现频率最高的颜色
                vals, freq = np.unique(neighbor_codes, return_counts=True)
                most_common_code = vals[np.argmax(freq)]
                
                if current_code != most_common_code:
                    # --- 核心：在处理过程中更新计数 ---
                    updated_counts[current_code]["count"] -= 1
                    updated_counts[most_common_code]["count"] += 1
                    
                    # 执行替换：选取邻域中该 code 对应的第一个 RGB 值
                    replace_idx = np.where(neighbor_codes == most_common_code)[0][0]
                    new_array[y, x] = neighbors[replace_idx]

    # 移除由于清理导致计数归零的颜色
    final_counts = {k: v for k, v in updated_counts.items() if v["count"] > 0}
    
    return new_array, final_counts


    """
    将 4 维数组可视化为带坐标轴、网格和图例的专业图纸
    参数：
        color_array: (H, W, 4) -> [color_code, R, G, B]
        pixel_scale: 单个像素块边长（建议 30-40 以便容纳文字）
    """
    h, w, _ = color_array.shape
    
    # --- 1. 参数配置 ---
    margin = int(pixel_scale * 1.5)  # 坐标轴留白空间
    legend_height = 80              # 底部图例高度
    grid_color = (200, 200, 200)    # 细网格颜色
    major_grid_color = (100, 100, 100) # 5x5粗网格颜色
    
    # 最终画布尺寸
    canvas_w = w * pixel_scale + 2 * margin
    canvas_h = h * pixel_scale + 2 * margin + legend_height
    
    output_img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(output_img)

    # 加载字体
    try:
        # 尝试加载中文字体（如果是Windows环境）或Arial
        font_code = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.4))
        font_axis = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.5))
        font_legend = ImageFont.truetype("arial.ttf", 16)
    except:
        font_code = font_axis = font_legend = ImageFont.load_default()

    # --- 2. 统计颜色信息 (用于图例) ---
    color_stats = {}
    for y in range(h):
        for x in range(w):
            code, r, g, b = color_array[y, x]
            if code not in color_stats:
                color_stats[code] = {"rgb": (int(r), int(g), int(b)), "count": 0}
            color_stats[code]["count"] += 1
    
    # --- 3. 绘制主图区域 ---
    for y in range(h):
        for x in range(w):
            color_code, r, g, b = color_array[y, x]
            
            # 填充色块
            rect_l = margin + x * pixel_scale
            rect_t = margin + y * pixel_scale
            rect_r = rect_l + pixel_scale
            rect_b = rect_t + pixel_scale
            
            draw.rectangle([rect_l, rect_t, rect_r, rect_b], fill=(r, g, b))
            
            # 绘制颜色编码 (计算对比色防止看不清)
            text_color = (255, 255, 255) if (r*0.299 + g*0.587 + b*0.114) < 128 else (0, 0, 0)
            draw.text(((rect_l + rect_r)/2, (rect_t + rect_b)/2), 
                      str(color_code), fill=text_color, font=font_code, anchor="mm")

    # --- 4. 绘制网格线 ---
    # 绘制垂直线
    for x in range(w + 1):
        line_x = margin + x * pixel_scale
        width = 2 if x % 5 == 0 else 1
        color = major_grid_color if x % 5 == 0 else grid_color
        draw.line([(line_x, margin), (line_x, margin + h * pixel_scale)], fill=color, width=width)
        
        # 绘制横向坐标序号 (顶部和底部)
        if x < w:
            axis_x = margin + x * pixel_scale + pixel_scale // 2
            draw.text((axis_x, margin // 2), str(x + 1), fill="black", font=font_axis, anchor="mm")
            draw.text((axis_x, margin + h * pixel_scale + margin // 2), str(x + 1), fill="black", font=font_axis, anchor="mm")

    # 绘制水平线
    for y in range(h + 1):
        line_y = margin + y * pixel_scale
        width = 2 if y % 5 == 0 else 1
        color = major_grid_color if y % 5 == 0 else grid_color
        draw.line([(margin, line_y), (margin + w * pixel_scale, line_y)], fill=color, width=width)
        
        # 绘制纵向坐标序号 (左侧和右侧)
        if y < h:
            axis_y = margin + y * pixel_scale + pixel_scale // 2
            draw.text((margin // 2, axis_y), str(y + 1), fill="black", font=font_axis, anchor="mm")
            draw.text((margin + w * pixel_scale + margin // 2, axis_y), str(y + 1), fill="black", font=font_axis, anchor="mm")

    # --- 5. 绘制底部图例 ---
    legend_start_y = margin * 2 + h * pixel_scale
    current_legend_x = margin
    
    for code in sorted(color_stats.keys()):
        info = color_stats[code]
        # 绘制小色块
        box_size = 20
        draw.rectangle([current_legend_x, legend_start_y, 
                        current_legend_x + box_size, legend_start_y + box_size], 
                       fill=info["rgb"], outline="black")
        
        # 绘制文本: 编码 (数量)
        legend_text = f"{code}  ({info['count']})"
        draw.text((current_legend_x + box_size + 5, legend_start_y + box_size // 2), 
                  legend_text, fill="black", font=font_legend, anchor="lm")
        
        # 移动到下一个图例位置（简单水平排列）
        current_legend_x += 100 
        if current_legend_x > canvas_w - 100: # 换行处理
            current_legend_x = margin
            legend_start_y += 30

    return output_img

import numpy as np

def find_nearest_non_rare_color(target_rgb, non_rare_rgb_list):
    """
    辅助函数：在非稀有色预存列表中找到与目标 RGB 最接近的颜色。
    """
    min_dist = float('inf')
    best_match = None
    for item in non_rare_rgb_list:
        dist = np.linalg.norm(target_rgb - item['rgb'])
        if dist < min_dist:
            min_dist = dist
            best_match = item
    
    if best_match:
        return [
            best_match['code'], 
            int(best_match['rgb'][0]), 
            int(best_match['rgb'][1]), 
            int(best_match['rgb'][2])
        ]
    return None

def reduce_color_Pro_array(color_array, color_code_count, dist_threshold=200):
    """
    逻辑进阶版：
    (1) 过滤邻域：替换时跳过其他稀有色，确保只替换为非稀有色。
    (2) 强制清零：若邻域无合适颜色，强制全局匹配，确保稀有色最终数量为 0。
    (3) 实时更新：同步更新 color_code_count。
    """
    h, w, _ = color_array.shape
    total_pixels = h * w
    threshold_limit = total_pixels * 0.02
    
    # 1. 统计与初始化
    final_counts = {k: v.copy() for k, v in color_code_count.items()}
    
    # 确定初始稀有色集合
    rare_codes = {code for code, info in final_counts.items() if info["count"] < threshold_limit}
    non_rare_codes = {code for code, info in final_counts.items() if info["count"] >= threshold_limit}
    
    # 安全兜底：如果没找到非稀有色（阈值太高），取出现频率最高的颜色作为基准
    if not non_rare_codes:
        most_common = max(final_counts.keys(), key=lambda k: final_counts[k]['count'])
        non_rare_codes = {most_common}

    # 预存非稀有色的 RGB 列表以加速匹配
    non_rare_rgb_list = [
        {
            'code': code, 
            'rgb': np.array([final_counts[code]['r'], final_counts[code]['g'], final_counts[code]['b']], dtype=float)
        } for code in non_rare_codes
    ]

    new_array = np.copy(color_array)
    # 8个方向的偏移量
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    for y in range(h):
        for x in range(w):
            current_code = new_array[y, x, 0]
            
            # 仅处理稀有色
            if current_code in rare_codes:
                current_rgb = new_array[y, x, 1:4].astype(float)
                
                # 寻找合法的邻居（必须是非稀有色）
                valid_neighbor_data = []
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n_pixel = new_array[ny, nx]
                        n_code = n_pixel[0]
                        
                        # 逻辑(1)：跳过另一个稀有色
                        if n_code in rare_codes:
                            continue
                        
                        dist = np.linalg.norm(current_rgb - n_pixel[1:4].astype(float))
                        valid_neighbor_data.append({
                            'code': n_code, 
                            'dist': dist, 
                            'full': n_pixel
                        })
                
                # 确定替换目标
                target_replace_data = None
                
                # 判定：如果邻域内没有非稀有色，或者邻域颜色距离都太远
                if not valid_neighbor_data or all(d['dist'] > dist_threshold for d in valid_neighbor_data):
                    # 逻辑(2)的一部分：强制去全局非稀有色库找，确保一定能换掉
                    target_replace_data = find_nearest_non_rare_color(current_rgb, non_rare_rgb_list)
                else:
                    # 在非稀有邻居中找最接近的一个
                    target_replace_data = min(valid_neighbor_data, key=lambda d: d['dist'])['full']
                
                # 执行替换与计数更新
                if target_replace_data is not None:
                    new_code = target_replace_data[0]
                    # 更新图像数组
                    new_array[y, x] = target_replace_data
                    
                    # 逻辑(3)：更新计数
                    final_counts[current_code]["count"] -= 1
                    final_counts[new_code]["count"] += 1

    # 最后移除所有计数为 0 的颜色（逻辑(2)的最终保证）
    final_counts = {k: v for k, v in final_counts.items() if v["count"] > 0}
    
    return new_array, final_counts

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



    iso_limit=2
    stark_diff=65
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


def visualize_color_array(color_array, color_code_count, pixel_scale=30):
    """
    将 4 维数组可视化。
    
    参数：
        color_array: (H, W, 4) -> [color_code, R, G, B]
        color_code_count: 颜色统计字典，由上游处理函数提供
        pixel_scale: 单个像素块边长
    """
    h, w, _ = color_array.shape
    
    # --- 1. 参数与动态布局配置 ---
    margin = int(pixel_scale * 1.5)  # 坐标轴数字留白
    
    # 动态计算图例高度
    legend_box_width = 120           # 每个图例占用的宽度
    legend_line_height = 40          # 每行图例的高度
    cols_per_row = max(1, (w * pixel_scale) // legend_box_width) # 每行显示的图例数
    num_colors = len(color_code_count)
    num_rows = (num_colors + cols_per_row - 1) // cols_per_row   # 计算需要多少行
    
    legend_padding = 40              # 图例与主图的间距
    legend_total_height = num_rows * legend_line_height + legend_padding
    
    canvas_w = w * pixel_scale + 2 * margin
    canvas_h = h * pixel_scale + 2 * margin + legend_total_height
    
    output_img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(output_img)

    # 加载字体
    try:
        font_code = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.4))
        font_axis = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.5))
        font_legend = ImageFont.truetype("arial.ttf", 16)
    except:
        font_code = font_axis = font_legend = ImageFont.load_default()

    # --- 2. 绘制主图区域 ---
    for y in range(h):
        for x in range(w):
            color_code, r, g, b = color_array[y, x]
            
            rect_l = margin + x * pixel_scale
            rect_t = margin + y * pixel_scale
            rect_r = rect_l + pixel_scale
            rect_b = rect_t + pixel_scale
            
            # 填充色块
            draw.rectangle([rect_l, rect_t, rect_r, rect_b], fill=(r, g, b))
            
            # 智能文字对比色 (亮度算法)
            text_color = (255, 255, 255) if (r*0.299 + g*0.587 + b*0.114) < 128 else (0, 0, 0)
            draw.text(((rect_l + rect_r)/2, (rect_t + rect_b)/2), 
                      str(color_code), fill=text_color, font=font_code, anchor="mm")

    # --- 3. 绘制网格与坐标轴 ---
    grid_color = (220, 220, 220)
    major_grid_color = (120, 120, 120)

    # 垂直方向
    for x in range(w + 1):
        lx = margin + x * pixel_scale
        draw.line([(lx, margin), (lx, margin + h * pixel_scale)], 
                  fill=major_grid_color if x % 5 == 0 else grid_color, 
                  width=2 if x % 5 == 0 else 1)
        if x < w:
            ax = margin + x * pixel_scale + pixel_scale // 2
            draw.text((ax, margin // 2), str(x + 1), fill="black", font=font_axis, anchor="mm")
            draw.text((ax, margin + h * pixel_scale + margin // 2), str(x + 1), fill="black", font=font_axis, anchor="mm")

    # 水平方向
    for y in range(h + 1):
        ly = margin + y * pixel_scale
        draw.line([(margin, ly), (margin + w * pixel_scale, ly)], 
                  fill=major_grid_color if y % 5 == 0 else grid_color, 
                  width=2 if y % 5 == 0 else 1)
        if y < h:
            ay = margin + y * pixel_scale + pixel_scale // 2
            draw.text((margin // 2, ay), str(y + 1), fill="black", font=font_axis, anchor="mm")
            draw.text((margin + w * pixel_scale + margin // 2, ay), str(y + 1), fill="black", font=font_axis, anchor="mm")

    # --- 4. 动态绘制图例 ---
    legend_start_y = margin * 2 + h * pixel_scale + 10
    sorted_codes = sorted(color_code_count.keys())
    
    for i, code in enumerate(sorted_codes):
        info = color_code_count[code]
        row = i // cols_per_row
        col = i % cols_per_row
        
        item_x = margin + col * legend_box_width
        item_y = legend_start_y + row * legend_line_height
        
        # 颜色块
        box_s = 22
        draw.rectangle([item_x, item_y, item_x + box_s, item_y + box_s], 
                       fill=(int(info['r']), int(info['g']), int(info['b'])), 
                       outline="black")
        
        # 文字说明: 编码 (数量)
        legend_txt = f"{code} ({info['count']})"
        draw.text((item_x + box_s + 6, item_y + box_s // 2), 
                  legend_txt, fill="black", font=font_legend, anchor="lm")

    return output_img

def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=20):
    # 生成颜色数组
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array,color_code_count, pixel_scale)
    return output_path, output_img, set(color_array[:, :, 0].flatten())

def reduce_image_colors(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=1, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 聚类减少颜色
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array, color_code_count,pixel_scale)
    return output_path, output_img

def reduce_image_colors_Pro(input_path, output_path, color_db_path, scale_factor=0.03,target_color_count=1,pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    color_array, color_code_count = reduce_color_Pro_array(color_array, color_code_count)
    output_img = visualize_color_array(color_array,color_code_count, pixel_scale)

    return output_path, output_img