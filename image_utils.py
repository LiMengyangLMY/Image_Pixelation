from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import cv2
import sqlite3
from sklearn.cluster import KMeans
from skimage import color 
import json

#全局变量存储颜色信息
_COLOR_CACHE = None
#加载/初始化颜色信息
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

    # 预先转换为 NumPy 数组，避免后续重复转换
    db_data = np.array(rows, dtype=object)
    # 提取 Lab 矩阵并转为 float32 (计算更快)
    db_labs = db_data[:, 4:7].astype(np.float32)
    
    # 存储为字典结构方便检索
    _COLOR_CACHE = {
        'full_rows': rows,
        'labs': db_labs
    }

#找与输入Lab最近的颜色，返回Lab
def find_nearest_color(target_lab):
    """
    基于内存缓存寻找最近色
    返回: (色号, (R, G, B), (L, a, b))
    """
    global _COLOR_CACHE
    if _COLOR_CACHE is None:
        return None, (0, 0, 0), (0, 0, 0)

    db_labs = _COLOR_CACHE['labs']
    # 向量化计算欧氏距离 (Delta E)
    distances = np.linalg.norm(db_labs - target_lab, axis=1)
    min_idx = np.argmin(distances)
    
    # 获取原始行数据: (num, R, G, B, lab_l, lab_a, lab_b)
    best_match = _COLOR_CACHE['full_rows'][min_idx]
    
    code = best_match[0]
    rgb = (int(best_match[1]), int(best_match[2]), int(best_match[3]))
    lab = (float(best_match[4]), float(best_match[5]), float(best_match[6]))
    
    return code, rgb, lab

# 聚类函数：减少颜色种类
def reduce_color_array(color_array, color_code_count, target_cluster_count):
    if int(target_cluster_count) >= len(color_code_count):
        return color_array, color_code_count

    # 构建 Lab 数据集进行聚类
    lab_data = np.array([[v["L"], v["a"], v["b"]] for v in color_code_count.values()])
    code_list = list(color_code_count.keys())

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=int(target_cluster_count), random_state=0).fit(lab_data)
    labels = kmeans.labels_

    # 映射旧色号到新色号 (取簇内 count 最高的)
    code_replace_map = {}
    new_counts = {}
    
    for cluster_id in range(int(target_cluster_count)):
        indices = [i for i, l in enumerate(labels) if l == cluster_id]
        cluster_codes = [code_list[i] for i in indices]
        rep_code = max(cluster_codes, key=lambda c: color_code_count[c]["count"])
        rep_info = color_code_count[rep_code]
        
        new_counts[rep_code] = {
            "count": sum(color_code_count[c]["count"] for c in cluster_codes),
            "L": rep_info["L"], "a": rep_info["a"], "b": rep_info["b"]
        }
        for c in cluster_codes:
            code_replace_map[c] = rep_code

    # 更新 color_array (矢量化操作)
    for y in range(color_array.shape[0]):
        for x in range(color_array.shape[1]):
            old_code = color_array[y, x, 0]
            new_code = code_replace_map[old_code]
            info = new_counts[new_code]
            color_array[y, x] = [new_code, info["L"], info["a"], info["b"]]

    return color_array, new_counts

#从非稀有色中找目标颜色的近似颜色
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

#功能:减少颜色种类→去除杂色
def reduce_color_Pro_array(color_array, color_code_count, dist_threshold=15):
    """基于新结构的 Lab 距离合并颜色"""
    if not color_code_count:
        return color_array, color_code_count

    # 按频率排序
    sorted_codes = sorted(color_code_count.keys(), 
                          key=lambda c: color_code_count[c]['count'], 
                          reverse=True)
    code_replace_map = {}
    processed_codes = []

    for curr_code in sorted_codes:
        info = color_code_count[curr_code]
        curr_lab = np.array([info['l_lab'], info['a_lab'], info['b_lab']])
        
        found = False
        for rep_code in processed_codes:
            rep_info = color_code_count[rep_code]
            rep_lab = np.array([rep_info['l_lab'], rep_info['a_lab'], rep_info['b_lab']])
            
            # 在感知均匀的 Lab 空间计算距离
            if np.linalg.norm(curr_lab - rep_lab) < dist_threshold:
                code_replace_map[curr_code] = rep_code
                found = True
                break
        
        if not found:
            processed_codes.append(curr_code)
            code_replace_map[curr_code] = curr_code

    # 构建新的统计字典（结构保持一致）
    new_counts = {}
    for old, new in code_replace_map.items():
        if new not in new_counts:
            new_counts[new] = color_code_count[new].copy()
            new_counts[new]['count'] = 0
        new_counts[new]['count'] += color_code_count[old]['count']

    # 更新 color_array (使用 NumPy 批量更新)
    grid_codes = color_array[:, :, 0]
    for old, new in code_replace_map.items():
        if old != new:
            mask = (grid_codes == old)
            info = new_counts[new]
            color_array[mask] = [new, info['l_lab'], info['a_lab'], info['b_lab']]

    return color_array, new_counts

#将图纸压缩→转换为Lab数组→找最近的色号
def image_to_color_array(input_path, color_db_path, scale_factor=0.03):
    init_color_cache(color_db_path)
    color_code_count = {}

    img = Image.open(input_path).convert('RGB')
    target_width = max(1, int(img.size[0] * scale_factor))
    target_height = max(1, int(img.size[1] * scale_factor))
    img_resized = img.resize((target_width, target_height), Image.NEAREST)

    # 图片整图转为 Lab 矩阵 (H, W, 3)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_np) 

    # 为了方便后续处理，color_array 建议只存 code
    color_array = np.empty((target_height, target_width, 1), dtype=object)

    for y in range(target_height):
        for x in range(target_width):
            pixel_lab = img_lab[y, x]
            
            # 获取所有信息
            code, rgb, lab = find_nearest_color(pixel_lab)

            color_array[y, x, 0] = code

            if code in color_code_count:
                color_code_count[code]["count"] += 1
            else:
                # 同时存储 RGB 和 Lab
                color_code_count[code] = {
                    "count": 1,
                    "r_rgb": rgb[0], "g_rgb": rgb[1], "b_rgb": rgb[2],
                    "l_lab": lab[0], "a_lab": lab[1], "b_lab": lab[2] 
                }
    return color_array, color_code_count

#可视化color_array数组，生成image
def visualize_color_array(color_array, color_code_count, pixel_size):
    h, w, _ = color_array.shape
    img = Image.new("RGB", (w * pixel_size, h * pixel_size), "white")
    draw = ImageDraw.Draw(img)

    # 预转换所有用到的 Lab 到 RGB 以提速
    rgb_lookup = {}
    for code, info in color_code_count.items():
        # Lab 转 RGB (skimage 期望 [1, 1, 3] 形状)
        lab_pixel = np.array([[[info["L"], info["a"], info["b"]]]])
        rgb_pixel = (color.lab2rgb(lab_pixel).flatten() * 255).astype(int)
        rgb_lookup[code] = tuple(rgb_pixel)

    for y in range(h):
        for x in range(w):
            code = color_array[y, x, 0]
            fill_color = rgb_lookup.get(code, (255, 255, 255))
            draw.rectangle(
                [x * pixel_size, y * pixel_size, (x + 1) * pixel_size, (y + 1) * pixel_size],
                fill=fill_color
            )
    return img

#将图纸信息存为数据库信息，便于用户保存图纸源信息。
#目的：（1）避免加载导致信息丢失（2）关闭网页可再次加载图纸信息
def save_drawing_to_sqlite(drawing_id, color_array, color_code_count):
    """
    将图纸数组和统计信息保存到独立的 SQLite 文件中
    """
    db_path = os.path.join('./data/DrawingData', f"{drawing_id}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 创建网格数据表 (存储 r, c, color_id)
    cursor.execute('CREATE TABLE IF NOT EXISTS grid (r INTEGER, c INTEGER, color_id TEXT)')
    
    # 准备批量插入数据 (将 numpy 数组展平)
    rows, cols, _ = color_array.shape
    flattened_data = []
    for r in range(rows):
        for c in range(cols):
            flattened_data.append((r, c, str(color_array[r, c, 0])))
    
    cursor.executemany('INSERT INTO grid VALUES (?, ?, ?)', flattened_data)

    # 2. 创建元数据表 (存储统计信息 JSON)
    cursor.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                  ('color_code_count', json.dumps(color_code_count)))
    cursor.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', 
                  ('dimensions', json.dumps([rows, cols])))

    conn.commit()
    conn.close()

#加载图纸信息，返回color_array, color_code_count
def load_drawing_from_sqlite(drawing_id):
    db_path = os.path.join('./data/DrawingData', f"{drawing_id}.db")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"未找到 ID 为 {drawing_id} 的图纸数据库文件。")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. 从 metadata 表获取元数据
        cursor.execute("SELECT value FROM metadata WHERE key='dimensions'")
        rows_count, cols_count = json.loads(cursor.fetchone()[0])

        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        color_code_count = json.loads(cursor.fetchone()[0])

        # 2. 从 grid 表获取像素数据
        cursor.execute("SELECT r, c, color_id FROM grid")
        grid_data = cursor.fetchall()

        # 3. 还原为 color_array
        # 创建一个全空对象数组，形状为 (行, 列, 1)
        color_array = np.empty((rows_count, cols_count, 1), dtype=object)

        # 填充数据
        for r, c, color_id in grid_data:
            color_array[r, c, 0] = color_id

    except Exception as e:
        print(f"解析图纸数据库失败: {e}")
        return None, None
    finally:
        conn.close()

    return color_array, color_code_count

def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=20):
    # 生成颜色数组
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array,color_code_count, pixel_scale)
    return output_path, output_img, set(color_array[:, :, 0].flatten()),color_array,color_code_count


def reduce_image_colors(input_path, output_path, color_db_path, scale_factor=0.03, target_color_count=1, pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    # 聚类减少颜色
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    # 使用自定义的像素块尺寸进行渲染
    output_img = visualize_color_array(color_array, color_code_count,pixel_scale)
    return output_path, output_img,color_array,color_code_count


def reduce_image_colors_Pro(input_path, output_path, color_db_path, scale_factor=0.03,target_color_count=1,pixel_scale=20):
    color_array, color_code_count = image_to_color_array(input_path, color_db_path, scale_factor)
    color_array, color_code_count = reduce_color_array(color_array, color_code_count, target_color_count)
    color_array, color_code_count = reduce_color_Pro_array(color_array, color_code_count)
    output_img = visualize_color_array(color_array,color_code_count, pixel_scale)

    return output_path, output_img,color_array,color_code_count