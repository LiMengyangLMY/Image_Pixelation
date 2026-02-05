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
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from skimage import color # 必须安装 scikit-image

def load_color_database(file_path):
    """加载颜色数据，并预先计算 Lab 坐标以优化搜索效率"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"颜色数据库文件 {file_path} 不存在")
    
    color_data = pd.read_csv(file_path)
    color_data = color_data.dropna(subset=['R', 'G', 'B', 'num'])
    
    # 清洗数据
    for col in ['R', 'G', 'B']:
        color_data[col] = pd.to_numeric(color_data[col], errors='coerce').fillna(0).astype(int).clip(0, 255)
    
    # --- 核心改进：预计算 Lab ---
    # 将 RGB 转换为 [0, 1] 范围
    rgbs = color_data[['R', 'G', 'B']].values.reshape(-1, 1, 3) / 255.0
    # 转换为 Lab 空间
    labs = color.rgb2lab(rgbs).reshape(-1, 3)
    
    color_data['L'] = labs[:, 0]
    color_data['a_lab'] = labs[:, 1]
    color_data['b_lab'] = labs[:, 2]
    return color_data

def find_nearest_color(target_rgb, color_database):
    """基于 CIELAB 感知距离寻找最接近色号"""
    tr, tg, tb = target_rgb
    
    # 1. 目标 RGB 转 Lab
    target_rgb_norm = np.array([[target_rgb]], dtype=float) / 255.0
    target_lab = color.rgb2lab(target_rgb_norm).reshape(3)
    
    # 2. 直接从数据库提取预存的 Lab 坐标
    db_labs = color_database[['L', 'a_lab', 'b_lab']].values
    
    # 3. 计算欧氏距离（在 Lab 空间即为 Delta E，反映真实感官差异）
    # 替换了原有的 base_coefficient 等不准确的系数
    distances = np.linalg.norm(db_labs - target_lab, axis=1)
    
    nearest_row = color_database.iloc[np.argmin(distances)]
    return nearest_row['num'], (int(nearest_row['R']), int(nearest_row['G']), int(nearest_row['B']))

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



def find_nearest_non_rare_color_lab(target_lab, non_rare_lab_list):
    """
    辅助函数：在非稀有色 Lab 预存列表中找到与目标 Lab 最接近的颜色。
    """
    min_dist = float('inf')
    best_match = None
    
    for item in non_rare_lab_list:
        # 直接计算 Lab 空间的欧式距离 (Delta E)
        dist = np.linalg.norm(target_lab - item['lab'])
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
    逻辑进阶版 (Lab 空间修正版)：
    (1) 使用 CIELAB 计算颜色距离，避免偏色。
    (2) 阈值建议：dist_threshold 设为 30-50 左右效果最佳。
    """
    h, w, _ = color_array.shape
    total_pixels = h * w
    threshold_limit = total_pixels * 0.02
    
    # 1. 统计与预处理：预计算所有色号的 Lab 值以提速
    final_counts = {k: v.copy() for k, v in color_code_count.items()}
    
    # 预计算所有已知编码的 Lab 值
    code_to_lab = {}
    for code, info in final_counts.items():
        rgb_norm = np.array([[[info['r'], info['g'], info['b']]]], dtype=float) / 255.0
        code_to_lab[code] = color.rgb2lab(rgb_norm).reshape(3)

    # 确定稀有色和非稀有色集合
    rare_codes = {code for code, info in final_counts.items() if info["count"] < threshold_limit}
    non_rare_codes = {code for code, info in final_counts.items() if info["count"] >= threshold_limit}
    
    if not non_rare_codes:
        most_common = max(final_counts.keys(), key=lambda k: final_counts[k]['count'])
        non_rare_codes = {most_common}

    # 预存非稀有色的 Lab 信息
    non_rare_lab_list = [
        {
            'code': code, 
            'lab': code_to_lab[code],
            'rgb': np.array([final_counts[code]['r'], final_counts[code]['g'], final_counts[code]['b']])
        } for code in non_rare_codes
    ]

    new_array = np.copy(color_array)
    #directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    directions = [ (-1,0), (0,-1), (0,1),  (1,0)]
    for y in range(h):
        for x in range(w):
            current_code = new_array[y, x, 0]
            
            # 仅处理稀有色
            if current_code in rare_codes:
                current_lab = code_to_lab[current_code]
                
                # 寻找合法的邻居（必须是非稀有色）
                valid_neighbor_data = []
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n_code = new_array[ny, nx, 0]
                        
                        # 跳过稀有色邻居
                        if n_code in rare_codes:
                            continue
                        
                        # 计算 Lab 距离 (Delta E)
                        dist = np.linalg.norm(current_lab - code_to_lab[n_code])
                        valid_neighbor_data.append({
                            'code': n_code, 
                            'dist': dist, 
                            'full': new_array[ny, nx]
                        })
                
                # 确定替换目标
                target_replace_data = None
                
                # 如果邻域内没有非稀有色，或者邻域颜色距离都太远 (保护瞳孔细节)
                if not valid_neighbor_data or all(d['dist'] > dist_threshold for d in valid_neighbor_data):
                    # 全局匹配最接近的非稀有色
                    target_replace_data = find_nearest_non_rare_color_lab(current_lab, non_rare_lab_list)
                else:
                    # 邻域匹配：选最接近的非稀有色邻居
                    target_replace_data = min(valid_neighbor_data, key=lambda d: d['dist'])['full']
                
                # 执行替换与计数更新
                if target_replace_data is not None:
                    new_code = target_replace_data[0]
                    new_array[y, x] = target_replace_data
                    
                    final_counts[current_code]["count"] -= 1
                    final_counts[new_code]["count"] += 1

    # 移除所有计数为 0 的颜色
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
    return output_path, output_img, set(color_array[:, :, 0].flatten()),color_array

def reduce_image_colors(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=1, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 聚类减少颜色
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array, color_code_count,pixel_scale)
    return output_path, output_img,color_array

def reduce_image_colors_Pro(input_path, output_path, color_db_path, scale_factor=0.03,target_color_count=1,pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    color_array, color_code_count = reduce_color_Pro_array(color_array, color_code_count)
    output_img = visualize_color_array(color_array,color_code_count, pixel_scale)

    return output_path, output_img,color_array