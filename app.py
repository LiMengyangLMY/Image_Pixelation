#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file, jsonify
import os
import uuid
import json
import sqlite3
import pandas as pd
import numpy as np
import colorsys
from werkzeug.utils import secure_filename
from skimage import color as sk_color
from PIL import Image

# image_utils 导入
from image_utils import (
    save_drawing_to_sqlite, 
    reduce_image_colors_Pro, 
    process_image_with_color_code, 
    reduce_image_colors, 
    visualize_color_array,
    load_drawing_from_sqlite
)
# db_manager 导入
from db_manager import update_pixel_in_db, batch_update_in_db, get_db_path

from file_manager import limit_files
#————————————————————————————————#
#             全局变量            #
#————————————————————————————————#
# 全局配置路径
COLOR_DB_DIR = './data/Color/'
if not os.path.exists(COLOR_DB_DIR):
    os.makedirs(COLOR_DB_DIR)

# 默认选中的数据库文件（初始设为一个默认库）
current_color_db = os.path.join(COLOR_DB_DIR, 'colors.db')

# 核心状态维护
current_state = {
    'grid': None,               # NumPy array (H, W, 1) 存储 ID
    'palette': {},              # { 'ID': [R, G, B] }
    'pixel_size': 20,
    'color_code_count': {}      # { 'ID': { 'count': n, 'r_rgb': r, ... 'l_lab': l ... } }
}

temp_result_data = {"color_array": None, "pixel_size": 20, "color_code_count": {}}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

#————————————————————————————————————————#
#                  首页
#————————————————————————————————————————#
@app.route('/')
def home():
    db_files = [f for f in os.listdir(COLOR_DB_DIR) if f.endswith('.db')]
    current_name = os.path.basename(current_color_db)
    return render_template('index.html', db_files=db_files, current=current_name)

#————————————————————————————————————————#
#              图片处理页面
#————————————————————————————————————————#
@app.route('/image_conversion', methods=['GET', 'POST'])
def image_conversion():
    global temp_result_data
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="请选择文件")

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', error="文件格式不正确")

        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"{unique_id}_{filename}"
        output_filename = f"output_{unique_id}.png"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        file.save(input_path)

        try:
            target_width_cells = int(request.form.get('target_width', 40))
            pixel_size = int(request.form.get('pixel_size', 20))
            
            with Image.open(input_path) as img:
                orig_w, _ = img.size
                calc_scale_factor = target_width_cells / orig_w
            
            is_reduce_on = request.form.get('reduce_colors') == 'on'
            is_reduce_Pro_on = request.form.get('reduce_colors_Pro') == 'on'
            target_count = int(request.form.get('color_count', 16))

            # 执行转换逻辑
            if is_reduce_Pro_on:
                _, processed_img, color_array, color_code_count = reduce_image_colors_Pro(
                    input_path, output_path, current_color_db,
                    scale_factor=calc_scale_factor, target_color_count=target_count, pixel_scale=pixel_size
                )
            elif is_reduce_on:
                _, processed_img, color_array, color_code_count = reduce_image_colors(
                    input_path, output_path, current_color_db,
                    scale_factor=calc_scale_factor, target_color_count=target_count, pixel_scale=pixel_size
                )
            else:
                _, processed_img, _, color_array, color_code_count = process_image_with_color_code(
                    input_path, output_path, current_color_db,
                    scale_factor=calc_scale_factor, pixel_scale=pixel_size
                )

            # 存入临时区
            temp_result_data['color_array'] = color_array
            temp_result_data['pixel_size'] = pixel_size
            temp_result_data['color_code_count'] = color_code_count

            processed_img.save(output_path)
            limit_files(app.config['UPLOAD_FOLDER'])
            limit_files(app.config['OUTPUT_FOLDER'])
            return render_template('image_conversion.html', success=True,
                                 original_image=url_for('static', filename=f'uploads/{input_filename}'),
                                 processed_image=url_for('static', filename=f'outputs/{output_filename}'))
        except Exception as e:
            return render_template('image_conversion.html', error=f"处理失败: {e}")
    

    return render_template('image_conversion.html')


#————————————————————————————————————————#
#            颜色数据库管理页面
#————————————————————————————————————————#
@app.route('/colors')
def view_colors():
    return render_template('colors.html')


#————————————————————————————————————————#
#          保存图纸、图纸源数据
#————————————————————————————————————————#
# 下载可视化图片
@app.route('/download_file/<filename>')
def download_file(filename):
    """
    使用 visualize_color_array 生成的图片文件下载
    对应前端：url_for('download_file', filename=...)
    """
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path):
        # 如果是即时生成的修改结果，可以先调用 visualize_color_array 再发送
        # 这里假设文件已存在于输出目录
        return "文件不存在", 404
        
    return send_file(file_path, as_attachment=True)

# 保存图纸源数据到本地数据库
@app.route('/api/save_modified', methods=['POST'])
def save_modified():
    # 尝试从 temp_result_data 获取数据（如果 current_state 为空）
    grid_to_save = current_state.get('grid')
    counts_to_save = current_state.get('color_code_count')

    if grid_to_save is None and temp_result_data.get('color_array') is not None:
        grid_to_save = temp_result_data['color_array']
        counts_to_save = temp_result_data['color_code_count']

    if grid_to_save is None:
        return jsonify({"status": "error", "msg": "没有可保存的数据"}), 400

    try:
        drawing_id = request.json.get('drawing_id') or str(uuid.uuid4())[:8]
        # 确保是 numpy 数组格式
        color_array = np.array(grid_to_save, dtype=object)
        save_drawing_to_sqlite(drawing_id, color_array, counts_to_save)
        
        # 关键：更新当前的上下文 ID，以便跳转后 draw_page 能找到它
        global CURRENT_DRAWING_ID
        CURRENT_DRAWING_ID = drawing_id 
        
        return jsonify({"status": "success", "msg": "图纸源数据已保存", "drawing_id": drawing_id})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

#下载图纸源数据对应的可视化图片

    # 从数据库读取
    grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
    
    if grid_array is None:
        return "无可下载的数据", 404

    # --- 核心修复：强制转换为 NumPy 数组 ---
    # 确保 grid_array 是 (H, W, 1) 或 (H, W) 的 NumPy 结构
    grid_array = np.array(grid_array)

    from image_utils import _COLOR_CACHE
    # 确保缓存存在，否则提供备选逻辑
    if _COLOR_CACHE and 'full_rows' in _COLOR_CACHE:
        palette_map = {str(row[0]): [int(row[1]), int(row[2]), int(row[3])] for row in _COLOR_CACHE['full_rows']}
    else:
        # 备选：如果缓存失效，从当前图纸统计信息中构建一个临时调色盘
        palette_map = {str(k): [v['r_rgb'], v['g_rgb'], v['b_rgb']] for k, v in color_code_count.items()}
    
    # 渲染图片
    try:
        img = visualize_color_array(grid_array, palette_map, pixel_scale=20) 
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"export_{CURRENT_DRAWING_ID}.png")
        img.save(output_path)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"生成图片失败: {str(e)}", 500

@app.route('/download_modified')
def download_modified():
    # 1. 从数据库加载数据
    grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
    
    if grid_array is None:
        return "无可下载的数据", 404

    grid_array = np.array(grid_array)
    if grid_array.ndim == 2:
        grid_array = grid_array[:, :, np.newaxis]

    from image_utils import _COLOR_CACHE, init_color_cache
    
    if _COLOR_CACHE is None:
        init_color_cache(current_color_db) 

    try:
        img = visualize_color_array(grid_array, color_code_count, pixel_scale=20) 
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"export_{CURRENT_DRAWING_ID}.png")
        img.save(output_path)
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        print(f"渲染下载图纸时出错: {e}")
        return f"生成图片失败: {str(e)}", 500

#————————————————————————————————————————#
#          交互修改图纸
#————————————————————————————————————————#
CURRENT_DRAWING_ID = 'last_converted'
#进入draw_page页面
@app.route('/draw_page')
def draw_page():
    global temp_result_data
    
    if temp_result_data.get('color_array') is not None:
        save_drawing_to_sqlite(CURRENT_DRAWING_ID, temp_result_data['color_array'], temp_result_data['color_code_count'])
        temp_result_data['color_array'] = None

    grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
    if grid_array is None:
        return "图纸数据不存在，请先处理图片"

    from image_utils import _COLOR_CACHE, init_color_cache # 引入 init_color_cache
    if _COLOR_CACHE is None:
        init_color_cache(current_color_db) 
        from image_utils import _COLOR_CACHE 
    palette = {}
    if _COLOR_CACHE:
        for row in _COLOR_CACHE['full_rows']:
            color_id = str(row[0])
            palette[color_id] = {
                "r_rgb": int(row[1]),
                "g_rgb": int(row[2]),
                "b_rgb": int(row[3]),
                "l_lab": float(row[4]),
                "a_lab": float(row[5]),
                "b_lab": float(row[6])
            }

    drawings_dir = './data/DrawingData'
    drawings = [f[:-3] for f in os.listdir(drawings_dir) if f.endswith('.db')] if os.path.exists(drawings_dir) else []

    return render_template('draw_page.html', 
                            grid=grid_array.tolist(), 
                            palette=palette,         
                            color_counts=color_code_count,
                            drawings=drawings)

# 新建空白图纸
@app.route('/api/create_blank', methods=['POST'])
def create_blank():
    data = request.json
    drawing_id = data.get('drawing_id')
    width = int(data.get('width', 40))
    height = int(data.get('height', 40))
    
    # 逻辑：创建一个全白（或调色盘第一个颜色）的矩阵
    # 这里我们创建一个由 ID '1' 填充的矩阵，并构建初始元数据
    default_id = "1"
    new_grid = np.full((height, width, 1), default_id, dtype=object)
    
    # 初始化统计信息：总像素数全归于默认色
    initial_counts = {
        default_id: {
            "count": width * height,
            "r_rgb": 255, "g_rgb": 255, "b_rgb": 255,
            "l_lab": 100, "a_lab": 0, "b_lab": 0
        }
    }
    
    try:
        # 直接调用工具函数写入新数据库文件
        save_drawing_to_sqlite(drawing_id, new_grid, initial_counts)
        return jsonify({"status": "success", "drawing_id": drawing_id})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

# 裁剪图纸
@app.route('/api/crop_drawing', methods=['POST'])
def crop_drawing():
    data = request.json  # 接收前端传来的坐标：x1, y1, x2, y2
    drawing_id = CURRENT_DRAWING_ID # 或从 session 获取
    
    try:
        # 先从数据库读取当前完整数据
        grid_array, color_code_count = load_drawing_from_sqlite(drawing_id)
        
        # 执行 NumPy 裁剪操作
        cropped_grid = grid_array[data['y1']:data['y2']+1, data['x1']:data['x2']+1]
        
        # 重新计算裁剪区域内的颜色统计
        new_counts = {}
        for code in cropped_grid.flatten():
            code = str(code)
            if code in new_counts:
                new_counts[code]['count'] += 1
            else:
                # 从原统计中继承颜色元数据
                info = color_code_count.get(code, {
                    "r_rgb": 0, "g_rgb": 0, "b_rgb": 0,
                    "l_lab": 0, "a_lab": 0, "b_lab": 0
                }).copy()
                info['count'] = 1
                new_counts[code] = info
        
        # 将裁剪后的数据覆盖写入原数据库文件
        save_drawing_to_sqlite(drawing_id, cropped_grid, new_counts)
        
        return jsonify({"status": "success", "new_counts": new_counts})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

# 加载已有图纸接口
@app.route('/api/load_drawing', methods=['POST'])
def load_drawing_api():
    global CURRENT_DRAWING_ID
    drawing_id = request.json.get('drawing_id')
    
    if os.path.exists(get_db_path(drawing_id)):
        CURRENT_DRAWING_ID = drawing_id # 切换当前操作的图纸上下文
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "msg": "图纸文件不存在"}), 404

#单格更新
@app.route('/api/update_pixel', methods=['POST'])
def update_pixel():
    data = request.json
    try:
        r, c = int(data['r']), int(data['c'])
        new_id = str(data['new_id'])
        
        l_lab = data.get('l_lab', 0)
        a_lab = data.get('a_lab', 0)
        b_lab = data.get('b_lab', 0)
        r_rgb = data.get('r_rgb', 0)
        g_rgb = data.get('g_rgb', 0)
        b_rgb = data.get('b_rgb', 0)
        
        # 调用更新函数，传入所有必要参数
        new_counts = update_pixel_in_db(
            CURRENT_DRAWING_ID, r, c, new_id,
            l_lab=l_lab, a_lab=a_lab, b_lab=b_lab,
            r_rgb=r_rgb, g_rgb=g_rgb, b_rgb=b_rgb
        )
        
        if new_counts is not None:
            return jsonify({"status": "success", "new_counts": new_counts})
        else:
            return jsonify({"status": "error", "msg": "更新失败或颜色未改变"}), 400
            
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500
#同色替换
@app.route('/api/batch_update', methods=['POST'])
def batch_update():
    data = request.json
    old_id = str(data['old_id'])
    new_id = str(data['new_id'])
    
    new_counts = batch_update_in_db(CURRENT_DRAWING_ID, old_id, new_id)
    
    if new_counts is not None:
        return jsonify({"status": "success", "new_counts": new_counts})
    else:
        # 如果返回 None，可能是 ID 相同或更新过程中出现异常
        return jsonify({"status": "success", "msg": "无需更新或更新未执行"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)