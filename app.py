#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file
import os
import uuid
from werkzeug.utils import secure_filename
from image_utils import process_image_with_color_code

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