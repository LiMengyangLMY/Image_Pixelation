#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file,jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from image_utils import reduce_image_colors_Pro,process_image_with_color_code, reduce_image_colors, load_color_database
import PIL.Image 
import pandas as pd
from flask import render_template
import numpy as np
from sklearn.cluster import KMeans
import colorsys
from skimage import color as sk_color

#全局变量
current_state = {
    'grid': None,       # 2D list: 存储每个格子的颜色 ID 或 RGB
    'palette': {},      # dict: { 'ID': [R, G, B] }
    'pixel_size': 20
}
temp_result_data = {"color_array": None, "pixel_size": 20}
last_color_data = []
os.environ["OMP_NUM_THREADS"] = "1"
# 初始化Flask应用
app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大上传16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    """主导航页面"""
    return render_template('index.html')


@app.route('/image_conversion', methods=['GET', 'POST'])
def image_conversion():
    global last_color_data # 引用全局变量
    global temp_result_data
    if request.method == 'POST':
        # 1. 初始化变量，防止后续引用报错
        input_path = None 
        
        if 'file' not in request.files:
            return render_template('index.html', error="请选择文件")

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', error="文件格式不正确")

        # 2. 保存文件并获取路径
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"{unique_id}_{filename}"
        output_filename = f"output_{unique_id}_{os.path.splitext(filename)[0]}.png"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        file.save(input_path)

        # 3. 参数处理：计算放缩比例
        try:
            # 用户输入的宽度格子数
            target_width_cells = int(request.form.get('target_width', 40))
            # 每个格子的显示像素大小
            pixel_size = int(request.form.get('pixel_size', 20))
            
            with PIL.Image.open(input_path) as img:
                orig_w, _ = img.size
                # 关键计算：根据目标格子数计算放缩因子
                # 例如：原图1000px，目标40格，scale = 40/1000 = 0.04
                calc_scale_factor = target_width_cells / orig_w
            
            color_db_path = 'color_data.csv'
        except Exception as e:
            return render_template('index.html', error=f"参数或图片解析失败: {e}")

        # 4. 执行图片处理
        reduce_is_checked = request.form.get('reduce_colors') == 'on'
        
        try:
            # 获取互斥选项的状态
            is_reduce_on = request.form.get('reduce_colors') == 'on'
            is_reduce_Pro_on = request.form.get('reduce_colors_Pro') == 'on'
            
            if is_reduce_Pro_on:
                # 执行减少颜色Pro模式
                target_count = int(request.form.get('color_count', 16))
                _, processed_img,color_array = reduce_image_colors_Pro(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    target_color_count=target_count,
                    pixel_scale=pixel_size
                )
            elif is_reduce_on:
                # 执行减少颜色模式
                target_count = int(request.form.get('color_count', 16))
                _, processed_img ,color_array = reduce_image_colors(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    target_color_count=target_count,
                    pixel_scale=pixel_size
                )
            else:
                # 默认基础处理
                _, processed_img, _ ,color_array = process_image_with_color_code(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    pixel_scale=pixel_size
                )

            if isinstance(color_array, np.ndarray):
                last_color_data = color_array.tolist()
            else:
                last_color_data = color_array

            temp_result_data['color_array'] = color_array
            temp_result_data['pixel_size'] = pixel_size

            processed_img.save(output_path)
        except Exception as e:
            return render_template('image_conversion.html', error=f"处理失败: {e}")

        return render_template(
            'image_conversion.html',
            success=True,
            original_image=url_for('static', filename=f'uploads/{input_filename}'),
            processed_image=url_for('static', filename=f'outputs/{output_filename}')
        )
    
    return render_template('image_conversion.html')


# 下载处理后的图片
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )




@app.route('/colors')
def view_colors():
    df = pd.read_csv('color_data.csv')
    # 转换为 Lab 空间以供显示
    rgbs = df[['R', 'G', 'B']].values.reshape(-1, 1, 3) / 255.0
    labs = sk_color.rgb2lab(rgbs).reshape(-1, 3)
    
    # 构建分组字典
    grouped_data = {}
    for i, row in df.iterrows():
        # 提取前缀，如 "A1" -> "A"
        prefix = ''.join([c for c in str(row['num']) if c.isalpha()]) or "Other"
        
        if prefix not in grouped_data:
            grouped_data[prefix] = []
            
        l_val = colorsys.rgb_to_hls(row['R']/255, row['G']/255, row['B']/255)[1]
        
        grouped_data[prefix].append({
            'num': row['num'],
            'r': row['R'], 'g': row['G'], 'b': row['B'],
            'lab': [round(x, 1) for x in labs[i]],
            'hex': '#{:02x}{:02x}{:02x}'.format(int(row['R']), int(row['G']), int(row['B'])),
            'text_color': '#FFFFFF' if l_val < 0.5 else '#000000'
        })
    
    return render_template('colors.html', grouped_data=dict(sorted(grouped_data.items())))

@app.route('/update_color', methods=['POST'])
def update_color():
    data = request.json
    df = pd.read_csv('color_data.csv')
    # 根据 num 更新对应的 R, G, B
    idx = df[df['num'] == data['num']].index
    if not idx.empty:
        df.loc[idx, ['R', 'G', 'B']] = [int(data['r']), int(data['g']), int(data['b'])]
        df.to_csv('color_data.csv', index=False)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 404

@app.route('/draw_page')
def draw_page():
    global current_state, temp_result_data
    
    # 1. 只有点击进入此页面时才从临时区搬运数据到全局状态
    if temp_result_data.get('color_array') is not None:
        color_array = temp_result_data['color_array']
        # 存入 4 维数组结构: [编号, R, G, B]
        current_state['grid'] = color_array.tolist() if hasattr(color_array, 'tolist') else color_array
        current_state['pixel_size'] = temp_result_data.get('pixel_size', 20)
        
        # 2. 核心修改：加载 palette 供侧边栏颜色选择使用
        # 强制将编号转为字符串并去空格，防止匹配失败
        try:
            df_colors = pd.read_csv('color_data.csv')
            current_state['palette'] = {
                str(row['num']).strip(): [int(row['R']), int(row['G']), int(row['B'])] 
                for _, row in df_colors.iterrows()
            }
        except Exception as e:
            print(f"加载颜色库失败: {e}")
            current_state['palette'] = {}

    # 检查是否有数据
    if current_state['grid'] is None:
        return "请先在'图片颜色转换'页面处理图片再进入绘图页"

    # 3. 必须在 render_template 中显式传递 palette 变量
    return render_template('draw_page.html', 
                           grid=current_state['grid'], 
                           palette=current_state['palette'])


@app.route('/api/update_pixel', methods=['POST'])
def update_pixel():
    data = request.json # {r, c, new_color_id}
    r, c = data['r'], data['c']
    current_state['grid'][r][c] = data['new_id']
    return jsonify({"status": "success"})

@app.route('/api/batch_update', methods=['POST'])
def batch_update():
    data = request.json # {old_id, new_id}
    grid = np.array(current_state['grid'])
    grid[grid == data['old_id']] = data['new_id']
    current_state['grid'] = grid.tolist()
    # 注意：此处更新了 color_code_count 的逻辑体现为 grid 中 ID 的分布改变
    return jsonify({"status": "success"})

@app.route('/download_modified')
def download_modified():
    img = visualize_color_array(current_state['grid'], current_state['palette'], current_state['pixel_size'])
    temp_path = os.path.join(app.config['OUTPUT_FOLDER'], "modified_draw.png")
    img.save(temp_path)
    return send_file(temp_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)