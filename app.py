#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from image_utils import process_image_with_color_code, reduce_image_colors, load_color_database
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

# 辅助函数：检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Flask路由
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # ========= 1. 文件校验 =========
        if 'file' not in request.files:
            return render_template('index.html', error="请选择要上传的图片文件")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="未选择文件")

        if not (file and allowed_file(file.filename)):
            return render_template('index.html', error="不支持的文件格式")

        # ========= 2. 保存文件 =========
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"{unique_id}_{filename}"
        output_filename = f"output_{unique_id}_{os.path.splitext(filename)[0]}.png"

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        file.save(input_path)

        # ========= 3. 读取参数 =========
        try:
            scale_factor = float(request.form.get('scale_factor', 0.03))
            color_db_path = request.form.get('color_db_path', 'color_data.csv')
        except ValueError:
            return render_template('index.html', error="参数格式错误")

        reduce_colors = request.form.get('reduce_colors') == 'on'
        color_count = None

        if reduce_colors:
            try:
                color_count = int(request.form.get('color_count'))
            except (TypeError, ValueError):
                return render_template('index.html', error="请输入有效的颜色数量")

        # ========= 4. Stage 1：生成未减少颜色的图 =========
        try:
            from image_utils import process_image_with_color_code

            output_path,full_image, used_color_codes = process_image_with_color_code(
                input_path,
                output_path,        
                color_db_path=color_db_path,
                scale_factor=scale_factor
            )
        except Exception as e:
            return render_template('index.html', error=f"图片处理失败：{e}")

        # ========= 5. Stage 2：可选减少颜色 =========
        try:
            if reduce_colors:
                from image_utils import reduce_image_colors, load_color_database

                output_path,reduced_img = reduce_image_colors(
                    input_path,
                    output_path,        
                    color_db_path,
                    scale_factor=scale_factor,
                    target_color_count=color_count
                )
                reduced_img.save(output_path)
            else:
                full_image.save(output_path)

        except Exception as e:
            return render_template('index.html', error=f"减少颜色时出错：{e}")

        # ========= 6. 返回结果 =========
        return render_template(
            'index.html',
            success=True,
            original_image=url_for('static', filename=f'uploads/{input_filename}'),
            processed_image=url_for('static', filename=f'outputs/{output_filename}')
        )

    # ========= GET =========
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