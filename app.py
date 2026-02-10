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

# 从修复后的 image_utils 导入
from image_utils import (
    save_drawing_to_sqlite, 
    reduce_image_colors_Pro, 
    process_image_with_color_code, 
    reduce_image_colors, 
    visualize_color_array
)

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

@app.route('/')
def home():
    db_files = [f for f in os.listdir(COLOR_DB_DIR) if f.endswith('.db')]
    current_name = os.path.basename(current_color_db)
    return render_template('index.html', db_files=db_files, current=current_name)

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
            return render_template('image_conversion.html', success=True,
                                 original_image=url_for('static', filename=f'uploads/{input_filename}'),
                                 processed_image=url_for('static', filename=f'outputs/{output_filename}'))
        except Exception as e:
            return render_template('image_conversion.html', error=f"处理失败: {e}")
    
    return render_template('image_conversion.html')

@app.route('/draw_page')
def draw_page():
    global current_state, temp_result_data
    
    if temp_result_data.get('color_array') is not None:
        # 同步数据，统一为 list 方便前端渲染
        current_state['grid'] = temp_result_data['color_array'].tolist()
        current_state['pixel_size'] = temp_result_data['pixel_size']
        current_state['color_code_count'] = temp_result_data['color_code_count']
        
        # 加载调色盘信息用于交互
        df_colors = pd.read_csv(current_color_db)
        current_state['palette'] = {
            str(row['num']): [int(row['R']), int(row['G']), int(row['B'])] 
            for _, row in df_colors.iterrows()
        }

    if current_state['grid'] is None:
        return "请先处理图片再进入绘图页"

    return render_template('draw_page.html', 
                           grid=current_state['grid'], 
                           palette=current_state['palette'],
                           color_counts=current_state['color_code_count'])

@app.route('/api/batch_update', methods=['POST'])
def batch_update():
    global current_state
    data = request.json
    old_id, new_id = str(data['old_id']), str(data['new_id'])
    
    if old_id == new_id or old_id not in current_state['color_code_count']:
        return jsonify({"status": "success", "new_counts": current_state['color_code_count']})

    # 1. 更新网格数据 (注意 grid 现在是 [ [ [ID], [ID] ] ] 结构)
    grid_np = np.array(current_state['grid'], dtype=object)
    mask = (grid_np[:, :, 0] == old_id)
    grid_np[mask, 0] = new_id
    current_state['grid'] = grid_np.tolist()

    # 2. 合并统计信息
    old_info = current_state['color_code_count'].pop(old_id)
    if new_id in current_state['color_code_count']:
        current_state['color_code_count'][new_id]['count'] += old_info['count']
    else:
        # 如果新 ID 不在当前统计中（手动替换为库中其他色），需要从调色盘补全
        new_rgb = current_state['palette'].get(new_id, [0,0,0])
        current_state['color_code_count'][new_id] = {
            'count': old_info['count'],
            'r_rgb': new_rgb[0], 'g_rgb': new_rgb[1], 'b_rgb': new_rgb[2],
            'l_lab': old_info.get('l_lab', 0), 'a_lab': old_info.get('a_lab', 0), 'b_lab': old_info.get('b_lab', 0)
        }

    return jsonify({"status": "success", "new_counts": current_state['color_code_count']})


    if current_state['grid'] is None:
        return "数据为空", 400

    try:
        color_array = np.array(current_state['grid'], dtype=object)
        # 调用 image_utils 的渲染函数，此时字段名已对齐
        img = visualize_color_array(
            color_array, 
            current_state['color_code_count'], 
            current_state['pixel_size']
        )
        
        temp_path = os.path.join(app.config['OUTPUT_FOLDER'], "modified_draw.png")
        img.save(temp_path)
        return send_file(temp_path, as_attachment=True, download_name="final_design.png")
    except Exception as e:
        return f"生成失败: {e}", 500

@app.route('/api/manage_db', methods=['POST'])
def manage_db():
    global current_color_db, temp_result_data
    data = request.json
    action = data.get('action')
    filename = data.get('filename')

    # 确保文件名以 .db 结尾
    if not filename.endswith('.db'):
        filename += '.db'
    
    file_path = os.path.join(COLOR_DB_DIR, filename)

    try:
        if action == 'create':
            if os.path.exists(file_path):
                return jsonify({"status": "error", "msg": "数据库已存在"}), 400
            
            # 创建新数据库并初始化表结构
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS colors (
                    num TEXT PRIMARY KEY,
                    R INTEGER,
                    G INTEGER,
                    B INTEGER,
                    lab_l REAL,
                    lab_a REAL,
                    lab_b REAL
                )
            ''')
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "msg": f"数据库 {filename} 创建成功"})

        elif action == 'select':
            if not os.path.exists(file_path):
                return jsonify({"status": "error", "msg": "数据库不存在"}), 404
            
            current_color_db = file_path
            # 切换库后必须清除内存中的处理缓存，防止颜色编号对不上
            temp_result_data = {"color_array": None, "pixel_size": 20, "color_code_count": {}}
            # 同时调用 image_utils 的初始化函数刷新内存缓存
            from image_utils import init_color_cache
            init_color_cache(current_color_db)
            
            return jsonify({"status": "success", "msg": f"已切换至数据库: {filename}"})

        elif action == 'delete':
            if filename == 'colors.db': # 保护默认库
                return jsonify({"status": "error", "msg": "不能删除默认数据库"}), 400
            
            if os.path.exists(file_path):
                os.remove(file_path)
                # 如果删除的是当前选中的库，切回默认库
                if current_color_db == file_path:
                    current_color_db = os.path.join(COLOR_DB_DIR, 'colors.db')
                return jsonify({"status": "success", "msg": f"数据库 {filename} 已删除"})
            else:
                return jsonify({"status": "error", "msg": "找不到该文件"}), 404

    except Exception as e:
        return jsonify({"status": "error", "msg": f"操作失败: {str(e)}"}), 500

    return jsonify({"status": "error", "msg": "未知操作"}), 400

@app.route('/colors')
def view_colors():
    return render_template('colors.html')


#————————————————————————————————————————#
#          保存图纸、图纸源数据
#————————————————————————————————————————#
# 保存/下载可视化图片
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
    """
    使用 save_drawing_to_sqlite 保存图纸源信息
    """
    global current_state
    if current_state['grid'] is None:
        return jsonify({"status": "error", "msg": "没有可保存的数据"}), 400

    try:
        # 生成或获取图纸 ID
        drawing_id = request.json.get('drawing_id') or str(uuid.uuid4())[:8]
        
        # 将当前内存中的 grid (list) 转回 numpy 数组以符合工具类要求
        color_array = np.array(current_state['grid'], dtype=object)
        color_code_count = current_state['color_code_count']
        
        # 调用工具类函数保存到 ./data/DrawingData/{id}.db
        save_drawing_to_sqlite(drawing_id, color_array, color_code_count)
        
        return jsonify({
            "status": "success", 
            "msg": f"图纸源数据已保存", 
            "drawing_id": drawing_id
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)