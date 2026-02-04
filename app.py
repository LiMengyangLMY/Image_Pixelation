#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from image_utils import reduce_image_colors_Pro,process_image_with_color_code, reduce_image_colors, load_color_database
import PIL.Image 
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

@app.route('/', methods=['GET', 'POST'])
def index():
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
                _, processed_img = reduce_image_colors_Pro(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    target_color_count=target_count,
                    pixel_scale=pixel_size
                )
            elif is_reduce_on:
                # 执行减少颜色模式
                target_count = int(request.form.get('color_count', 16))
                _, processed_img = reduce_image_colors(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    target_color_count=target_count,
                    pixel_scale=pixel_size
                )
            else:
                # 默认基础处理
                _, processed_img, _ = process_image_with_color_code(
                    input_path, output_path, color_db_path,
                    scale_factor=calc_scale_factor,
                    pixel_scale=pixel_size
                )
            
            processed_img.save(output_path)
        except Exception as e:
            return render_template('index.html', error=f"处理失败: {e}")

        return render_template(
            'index.html',
            success=True,
            original_image=url_for('static', filename=f'uploads/{input_filename}'),
            processed_image=url_for('static', filename=f'outputs/{output_filename}')
        )

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