from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import cv2
import sqlite3
from sklearn.cluster import KMeans
from skimage import color 
import json

# 全局变量存储颜色信息
_COLOR_CACHE = None
def init_color_cache(db_path='./data/Color/colors.db'):
    """在程序启动或切换数据库时调用一次"""
    global _COLOR_CACHE
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT num, R, G, B, lab_l, lab_a, lab_b FROM colors")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        _COLOR_CACHE = None
        return

    db_data = np.array(rows, dtype=object)
    db_labs = db_data[:, 4:7].astype(np.float32)
    
    _COLOR_CACHE = {
        'full_rows': rows,
        'labs': db_labs
    }

def find_nearest_color(target_lab):
    """基于内存缓存寻找最近色"""
    global _COLOR_CACHE
    if _COLOR_CACHE is None:
        return None, (0, 0, 0), (0, 0, 0)

    db_labs = _COLOR_CACHE['labs']
    distances = np.linalg.norm(db_labs - target_lab, axis=1)
    min_idx = np.argmin(distances)
    
    best_match = _COLOR_CACHE['full_rows'][min_idx]
    code = str(best_match[0])
    rgb = (int(best_match[1]), int(best_match[2]), int(best_match[3]))
    lab = (float(best_match[4]), float(best_match[5]), float(best_match[6]))
    
    return code, rgb, lab

def reduce_color_array(color_array, color_code_count, target_cluster_count):

    try:
        target_cluster_count = int(target_cluster_count)
    except (ValueError, TypeError):
        # 如果转换失败，维持原样返回，避免崩溃
        return color_array, color_code_count

    current_codes = list(color_code_count.keys())
    num_current_colors = len(current_codes)

    
    if target_cluster_count >= num_current_colors or num_current_colors == 0:
        return color_array, color_code_count
    
    # 3. 准备聚类数据，确保字段名一致且为数值型
    # 显式将所有 key 转为字符串，确保后续映射逻辑统一
    lab_data = []
    ordered_codes = []
    for code, info in color_code_count.items():
        lab_data.append([
            float(info.get("l_lab", 0)), 
            float(info.get("a_lab", 0)), 
            float(info.get("b_lab", 0))
        ])
        ordered_codes.append(str(code))
    
    lab_data = np.array(lab_data, dtype=np.float32)
    
    # 4. 执行 KMeans 聚类
    # n_clusters 必须为 int，已在步骤1保证
    kmeans = KMeans(
        n_clusters=target_cluster_count, 
        random_state=0, 
        n_init=10
    ).fit(lab_data)
    labels = kmeans.labels_

    # 5. 构建代码替换映射表 (Old Code -> New Representative Code)
    code_replace_map = {}
    new_counts = {}
    
    for cluster_id in range(target_cluster_count):
        # 找到属于该簇的所有原始颜色索引
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        if not indices:
            continue
            
        cluster_codes = [ordered_codes[i] for i in indices]
        
        # 策略：选择该簇中在原图中出现频率最高（count最大）的颜色作为代表色
        rep_code = max(cluster_codes, key=lambda c: color_code_count[c]["count"])
        rep_info = color_code_count[rep_code].copy()
        
        # 累加该簇所有颜色的总计数
        rep_info["count"] = sum(color_code_count[c]["count"] for c in cluster_codes)
        new_counts[rep_code] = rep_info
        
        # 建立映射关系
        for c in cluster_codes:
            code_replace_map[str(c)] = str(rep_code)

    # 6. 全局更新 color_array
    # 使用 NumPy 处理，避免 Python 原生嵌套循环在处理大数据量时的类型判定问题
    h, w = color_array.shape[:2]
    # 先扁平化并确保全是字符串
    flat_array = color_array.flatten().astype(str)
    
    # 执行替换：如果 code 在 map 中则替换，否则保留原样
    # 使用 np.vectorize 或者列表推导式处理字符串对象数组
    replaced_flat = np.array([code_replace_map.get(x, x) for x in flat_array], dtype=object)
    
    # 重新塑造回原始形状 (h, w, 1)
    color_array = replaced_flat.reshape((h, w, 1))

    return color_array, new_counts

def reduce_color_Pro_array(color_array, color_code_count, dist_threshold=200):
    """
    逻辑进阶版 (Lab 空间修正版)：
    (1) 使用 CIELAB 计算颜色距离，自动合并出现频率极低的“杂色”。
    (2) 策略：稀有色优先向邻域内的非稀有色合并；若邻域无合适颜色，则全局寻找最接近的非稀有色。
    (3) 阈值建议：dist_threshold 建议设为 30-50 左右。
    """
    if not color_code_count:
        return color_array, color_code_count

    h, w, _ = color_array.shape
    total_pixels = h * w
    # 稀有色判定标准：出现频率低于总像素的 2%
    threshold_limit = total_pixels * 0.02
    
    # 1. 初始化计数器与 Lab 缓存（确保 Key 为字符串以防报错）
    final_counts = {str(k): v.copy() for k, v in color_code_count.items()}
    
    # 预提取所有编码的 Lab 值（直接从 color_code_count 获取，避免重复计算）
    code_to_lab = {}
    for code, info in final_counts.items():
        code_to_lab[code] = np.array([
            float(info.get('l_lab', 0)), 
            float(info.get('a_lab', 0)), 
            float(info.get('b_lab', 0))
        ], dtype=np.float32)

    # 2. 确定稀有色和非稀有色集合
    rare_codes = {code for code, info in final_counts.items() if info["count"] < threshold_limit}
    non_rare_codes = {code for code, info in final_counts.items() if info["count"] >= threshold_limit}
    
    # 极端情况处理：如果没有非稀有色，则选出现次数最多的颜色作为基准
    if not non_rare_codes:
        most_common = max(final_counts.keys(), key=lambda k: final_counts[k]['count'])
        non_rare_codes = {most_common}
        rare_codes.discard(most_common)

    # 预存非稀有色的 Lab 信息，用于全局快速检索
    non_rare_lab_list = [
        {'code': code, 'lab': code_to_lab[code]} 
        for code in non_rare_codes
    ]

    # 3. 遍历图像执行合并逻辑
    new_array = np.copy(color_array)
    # 定义邻域检查方向 (上下左右)
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    
    for y in range(h):
        for x in range(w):
            current_code = str(new_array[y, x, 0])
            
            # 仅处理稀有色：尝试将其替换为更主流的颜色
            if current_code in rare_codes:
                current_lab = code_to_lab[current_code]
                
                # 步骤 A: 寻找合法的邻居（必须是非稀有色）
                valid_neighbor_data = []
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n_code = str(new_array[ny, nx, 0])
                        
                        # 邻居也是稀有色，跳过（不向稀有色合并，防止杂色堆积）
                        if n_code in rare_codes:
                            continue
                        
                        # 计算 Lab 距离
                        dist = np.linalg.norm(current_lab - code_to_lab[n_code])
                        valid_neighbor_data.append({'code': n_code, 'dist': dist})
                
                # 确定替换目标
                target_code = None
                
                # 优先尝试邻域匹配：邻居存在且色差在阈值范围内
                if valid_neighbor_data:
                    best_neighbor = min(valid_neighbor_data, key=lambda d: d['dist'])
                    if best_neighbor['dist'] <= dist_threshold:
                        target_code = best_neighbor['code']
                
                # 步骤 B: 如果邻域不合适，执行全局匹配（寻找最接近的主色）
                if target_code is None:
                    # 在非稀有色集合中找最近的一个
                    global_match = min(non_rare_lab_list, key=lambda d: np.linalg.norm(current_lab - d['lab']))
                    target_code = global_match['code']
                
                # 执行替换与计数更新
                if target_code and target_code != current_code:
                    new_array[y, x, 0] = target_code
                    final_counts[current_code]["count"] -= 1
                    final_counts[target_code]["count"] += 1

    # 4. 清理：移除所有计数归零的稀有色
    final_counts = {k: v for k, v in final_counts.items() if v["count"] > 0}
    
    return new_array, final_counts


def image_to_color_array(input_path, color_db_path, scale_factor=0.03):
    init_color_cache(color_db_path)
    color_code_count = {}

    img = Image.open(input_path).convert('RGB')
    target_width = max(1, int(img.size[0] * scale_factor))
    target_height = max(1, int(img.size[1] * scale_factor))
    img_resized = img.resize((target_width, target_height), Image.NEAREST)

    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_np) 

    color_array = np.empty((target_height, target_width, 1), dtype=object)

    for y in range(target_height):
        for x in range(target_width):
            pixel_lab = img_lab[y, x]
            code, rgb, lab = find_nearest_color(pixel_lab)
            color_array[y, x, 0] = code

            if code in color_code_count:
                color_code_count[code]["count"] += 1
            else:
                color_code_count[code] = {
                    "count": 1,
                    "r_rgb": rgb[0], "g_rgb": rgb[1], "b_rgb": rgb[2],
                    "l_lab": lab[0], "a_lab": lab[1], "b_lab": lab[2] 
                }
    return color_array, color_code_count

def visualize_color_array(color_array, color_code_count, pixel_scale=30):
    pixel_scale = int(pixel_scale) 
    h, w = color_array.shape[:2]
    
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

    # 加载字体 (增加默认字体备选方案)
    try:
        # 尝试加载系统字体，若失败则使用默认
        font_code = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.4))
        font_axis = ImageFont.truetype("arial.ttf", int(pixel_scale * 0.5))
        font_legend = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_code = font_axis = font_legend = ImageFont.load_default()

    # --- 2. 绘制主图区域 ---
    for y in range(h):
        for x in range(w):
            # 从 color_array 获取编码，从 color_code_count 获取颜色
            color_code = str(color_array[y, x, 0])
            info = color_code_count.get(color_code)
            if not info: continue
            
            r, g, b = info['r_rgb'], info['g_rgb'], info['b_rgb']
            
            rect_l = margin + x * pixel_scale
            rect_t = margin + y * pixel_scale
            rect_r = rect_l + pixel_scale
            rect_b = rect_t + pixel_scale
            
            # 填充色块
            draw.rectangle([rect_l, rect_t, rect_r, rect_b], fill=(int(r), int(g), int(b)))
            
            # 智能文字对比色 (亮度算法)
            text_color = (255, 255, 255) if (r*0.299 + g*0.587 + b*0.114) < 128 else (0, 0, 0)
            draw.text(((rect_l + rect_r)/2, (rect_t + rect_b)/2), 
                      color_code, fill=text_color, font=font_code, anchor="mm")

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
    sorted_codes = sorted(
        color_code_count.keys(), 
        key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))
    )
    for i, code in enumerate(sorted_codes):
        info = color_code_count[code]
        row = i // cols_per_row
        col = i % cols_per_row
        
        item_x = margin + col * legend_box_width
        item_y = legend_start_y + row * legend_line_height
        
        # 颜色块
        box_s = 22
        draw.rectangle([item_x, item_y, item_x + box_s, item_y + box_s], 
                       fill=(int(info['r_rgb']), int(info['g_rgb']), int(info['b_rgb'])), 
                       outline="black")
        
        # 文字说明: 编码 (数量)
        legend_txt = f"{code} ({info['count']})"
        draw.text((item_x + box_s + 6, item_y + box_s // 2), 
                  legend_txt, fill="black", font=font_legend, anchor="lm")

    return output_img

def save_drawing_to_sqlite(drawing_id, color_array, color_code_count):
    db_dir = './data/DrawingData'
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        
    db_path = os.path.join(db_dir, f"{drawing_id}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS grid')
    cursor.execute('DROP TABLE IF EXISTS metadata')
    
    cursor.execute('CREATE TABLE grid (r INTEGER, c INTEGER, color_id TEXT)')
    
    rows, cols = color_array.shape[:2]
    flattened_data = []
    for r in range(rows):
        for c in range(cols):
            flattened_data.append((r, c, str(color_array[r, c, 0])))
    
    cursor.executemany('INSERT INTO grid VALUES (?, ?, ?)', flattened_data)

    cursor.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')
    cursor.execute('INSERT INTO metadata VALUES (?, ?)', ('color_code_count', json.dumps(color_code_count)))
    cursor.execute('INSERT INTO metadata VALUES (?, ?)', ('dimensions', json.dumps([rows, cols])))

    conn.commit()
    conn.close()

def load_drawing_from_sqlite(drawing_id):
    db_path = os.path.join('./data/DrawingData', f"{drawing_id}.db")
    if not os.path.exists(db_path):
        return None, None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT value FROM metadata WHERE key='dimensions'")
        rows_count, cols_count = json.loads(cursor.fetchone()[0])
        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        color_code_count = json.loads(cursor.fetchone()[0])
        cursor.execute("SELECT r, c, color_id FROM grid")
        grid_data = cursor.fetchall()

        color_array = np.empty((rows_count, cols_count, 1), dtype=object)
        for r, c, color_id in grid_data:
            color_array[r, c, 0] = color_id
        return color_array, color_code_count
    except Exception as e:
        print(f"解析失败: {e}")
        return None, None
    finally:
        conn.close()

#——————————————————————————————————————————#
# ---------------- 封装接口 ----------------#
#——————————————————————————————————————————#
def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    output_img = visualize_color_array(color_array, color_code_count, pixel_scale)
    return output_path, output_img, set(color_array.flatten()), color_array, color_code_count

def reduce_image_colors(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=10, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    output_img = visualize_color_array(color_array, color_code_count, pixel_scale)
    return output_path, output_img, color_array, color_code_count

def reduce_image_colors_Pro(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=10, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 先粗筛聚类，再精筛杂色
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    color_array, color_code_count = reduce_color_Pro_array(color_array, color_code_count)
    output_img = visualize_color_array(color_array, color_code_count, pixel_scale)
    return output_path, output_img, color_array, color_code_count