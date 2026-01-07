#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, redirect, url_for, send_file
from PIL import Image, ImageDraw, ImageFont
import os
import pandas as pd
import numpy as np
import uuid
from werkzeug.utils import secure_filename

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

# 你的核心处理代码（保持不变）
def load_color_database(file_path):
    """加载颜色数据，返回颜色数据框"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"颜色数据库文件 {file_path} 不存在")
    
    color_data = pd.read_csv(file_path)
    color_data = color_data.dropna(subset=['R', 'G', 'B', 'num'])
    color_data['R'] = pd.to_numeric(color_data['R'], errors='coerce').fillna(0).astype(int)
    color_data['G'] = pd.to_numeric(color_data['G'], errors='coerce').fillna(0).astype(int)
    color_data['B'] = pd.to_numeric(color_data['B'], errors='coerce').fillna(0).astype(int)
    color_data['R'] = color_data['R'].clip(0, 255)
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

def process_image_with_color_code(input_path, output_path, color_db_path, scale_factor=0.03, pixel_scale=50):
    """改进的图片处理函数"""
    color_database = load_color_database(color_db_path)
    
    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        raise Exception(f"打开图片失败: {e}")
    
    original_width, original_height = img.size
    target_width = max(1, int(original_width * scale_factor))
    target_height = max(1, int(original_height * scale_factor))

    img_resized = img.resize((target_width, target_height), Image.NEAREST)
    output_width = target_width * pixel_scale
    output_height = target_height * pixel_scale
    output_img = Image.new("RGB", (output_width, output_height), color="white")
    draw = ImageDraw.Draw(output_img)

    # 字体加载（兼容不同系统）
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", pixel_scale // 2)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", pixel_scale // 2)
        except IOError:
            font = ImageFont.load_default()

    # 处理每个像素
    for y in range(target_height):
        for x in range(target_width):
            pixel_r, pixel_g, pixel_b = img_resized.getpixel((x, y))
            color_code, nearest_rgb = find_nearest_color((pixel_r, pixel_g, pixel_b), color_database)
            nr, ng, nb = nearest_rgb
            
            # 绘制像素块
            draw.rectangle(
                [x * pixel_scale, y * pixel_scale,
                 (x + 1) * pixel_scale - 1, (y + 1) * pixel_scale - 1],
                fill=(nr, ng, nb)
            )

            # 绘制颜色编号
            text_color = (255 - nr, 255 - ng, 255 - nb)
            font_size = min(pixel_scale // 2, 12)
            
            try:
                current_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                current_font = ImageFont.load_default()
            
            draw.text(
                (x * pixel_scale + pixel_scale // 2, y * pixel_scale + pixel_scale // 2),
                str(color_code),
                fill=text_color,
                font=current_font,
                anchor="mm"
            )
    
    # 保存结果
    output_img.save(output_path)
    return output_path

# 辅助函数：检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Flask路由
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('index.html', error="请选择要上传的图片文件")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="未选择文件")
        
        if file and allowed_file(file.filename):
            # 生成唯一文件名
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            input_filename = f"{unique_id}_{filename}"
            output_filename = f"output_{unique_id}_{os.path.splitext(filename)[0]}.png"
            
            # 保存上传的文件
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            file.save(input_path)
            
            # 获取用户输入的参数
            try:
                scale_factor = float(request.form.get('scale_factor', 0.03))
                pixel_scale = int(request.form.get('pixel_scale', 20))
                color_db_path = request.form.get('color_db_path', 'color_data.csv')
            except ValueError:
                return render_template('index.html', error="参数格式错误，请输入有效的数字")
            
            # 处理图片
            try:
                result_path = process_image_with_color_code(
                    input_path, 
                    output_path, 
                    color_db_path,
                    scale_factor=scale_factor,
                    pixel_scale=pixel_scale
                )
                
                # 返回处理结果
                return render_template(
                    'index.html', 
                    success=True,
                    original_image=url_for('static', filename=f'uploads/{input_filename}'),
                    processed_image=url_for('static', filename=f'outputs/{output_filename}')
                )
            except Exception as e:
                return render_template('index.html', error=f"处理图片时出错: {str(e)}")
    
    # GET请求时显示表单
    return render_template('index.html')

# 下载处理后的图片
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)