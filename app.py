# -*- coding: utf-8 -*-
#=====================================================
#       Flask Web应用 + 图片颜色编码处理 (Day 4)
#=====================================================
from flask import Flask, render_template, request, url_for, send_file, jsonify, redirect, flash, session
import os
import uuid
import json
import sqlite3
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_mail import Mail, Message
import random
import threading
import socket
from apscheduler.schedulers.background import BackgroundScheduler # 定时任务库
from dotenv import load_dotenv
load_dotenv()

# 解决部分环境下中文主机名导致发邮件报错的问题
try:
    socket.gethostname = lambda: "localhost"
except:
    pass

# 导入自定义模块
from image_utils import (
    save_drawing_to_sqlite, 
    reduce_image_colors_Pro, 
    process_image_with_color_code, 
    reduce_image_colors, 
    visualize_color_array,
    load_drawing_from_sqlite,
    init_color_cache
)
import image_utils
from db_manager import (
    COLOR_DB_PATH,
    create_blank_drawing_logic,
    update_pixel_in_db, 
    batch_update_in_db, 
    get_db_path,
    crop_drawing_logic,
    save_verification_code,
    verify_code_logic,
    update_password_by_email,
    undo_logic 
)
from file_manager import limit_files, run_auto_clean # [新增] 引入后台清理逻辑

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback-key-for-dev')

# 路径配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
COLOR_DB_DIR = './data/Color/'
if not os.path.exists(COLOR_DB_DIR):
    os.makedirs(COLOR_DB_DIR)

current_color_db = os.path.join(COLOR_DB_DIR, 'colors.db')

# ——————————————————————————————————————#
#          邮件配置 (SMTP)              #
# ——————————————————————————————————————#
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
# ！！！注意不要直接暴露密钥！！！
mail_user = os.getenv('MAIL_USERNAME')
mail_pass = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USERNAME'] = mail_user
app.config['MAIL_PASSWORD'] = mail_pass
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

mail = Mail(app)

# ————————————————————————————————#
#             全局变量             #
# ————————————————————————————————#
current_state = {
    'grid': None,
    'palette': {},
    'pixel_size': 20,
    'color_code_count': {}
}
# 注意：temp_result_data 依然用于暂存“转换后、进入画图页前”的数据
# 建议后续结合 Redis 优化，目前结合 Session 使用
temp_result_data = {"color_array": None, "pixel_size": 20, "color_code_count": {}}

# ————————————————————————————————#
#             后台任务             #
# ————————————————————————————————#
# 仅在主进程启动定时任务，防止 Debug 模式下重复运行
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    scheduler = BackgroundScheduler()
    # 每 60 分钟执行一次自动清理
    scheduler.add_job(func=run_auto_clean, trigger="interval", minutes=60)
    scheduler.start()
    print("⏰ 后台自动清理任务已启动 (Common用户过期文件)...")

# ————————————————————————————————#
#          工具函数                #
# ————————————————————————————————#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_verification_code():
    return str(random.randint(100000, 999999))

# ————————————————————————————————#
#          用户鉴权配置            #
# ————————————————————————————————#
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

class User(UserMixin):
    def __init__(self, id, username, email, user_level):
        self.id = id
        self.username = username
        self.email = email
        self.user_level = user_level

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('./data/users.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, username, email, user_level, vip_expire_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        cursor.execute("SELECT id, username, email, user_level FROM users WHERE id = ?", (user_id,))
        row = list(cursor.fetchone()) + [None] if cursor.fetchone() else None

    if row:
        user_id_val, username, email, level, expire_at = row
        # VIP 自动过期逻辑
        if level == 'vip' and expire_at:
            try:
                expire_dt = datetime.strptime(expire_at, '%Y-%m-%d %H:%M:%S')
                if datetime.now() > expire_dt:
                    cursor.execute("UPDATE users SET user_level = 'common' WHERE id = ?", (user_id_val,))
                    conn.commit()
                    level = 'common'
            except ValueError:
                pass
        conn.close()
        return User(user_id_val, username, email, level)
    conn.close()
    return None

# ————————————————————————————————#
#          API 路由 (Auth)         #
# ————————————————————————————————#
@app.route('/login_page')
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    account = data.get('account')
    password = data.get('password')

    conn = sqlite3.connect('./data/users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password_hash, user_level FROM users WHERE username = ? OR email = ?", (account, account))
    user_row = cursor.fetchone()
    conn.close()

    if user_row and check_password_hash(user_row[3], password):
        user_obj = User(user_row[0], user_row[1], user_row[2], user_row[4])
        login_user(user_obj)
        # 登录成功，清除可能存在的旧 session drawing_id
        session.pop('drawing_id', None)
        return jsonify({"status": "success", "msg": "登录成功", "username": user_obj.username})
    
    return jsonify({"status": "error", "msg": "用户名或密码错误"}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    vcode = data.get('vcode')

    if not all([username, email, password, vcode]):
        return jsonify({"status": "error", "msg": "资料填写不完整"}), 400

    if not verify_code_logic(email, vcode):
        return jsonify({"status": "error", "msg": "验证码错误或已过期"}), 400

    pwd_hash = generate_password_hash(password)
    try:
        conn = sqlite3.connect('./data/users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password_hash, user_level) VALUES (?, ?, ?, 'common')", 
                       (username, email, pwd_hash))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "msg": "注册成功，请登录"})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "msg": "用户名或邮箱已存在"}), 400

@app.route('/api/send_email_code', methods=['POST'])
def send_email_code():
    data = request.json
    email = data.get('email')
    if not email: return jsonify({"status": "error", "msg": "请输入邮箱地址"}), 400
    code = generate_verification_code()
    try:
        save_verification_code(email, code)
        msg = Message(subject="【拼豆图纸】注册验证码", recipients=[email])
        msg.body = f"您的注册验证码是：{code}，有效期为5分钟。如非本人操作请忽略。"
        mail.send(msg)
        return jsonify({"status": "success", "msg": "验证码已发送，请查收！"})
    except Exception as e:
        return jsonify({"status": "error", "msg": f"邮件发送失败: {str(e)}"}), 500

@app.route('/api/activate_vip', methods=['POST'])
def api_activate_vip():
    data = request.json
    username = data.get('username')
    code = data.get('code')
    conn = sqlite3.connect('./data/users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT code, valid_days FROM vip_codes WHERE code = ? AND is_used = 0", (code,))
    row = cursor.fetchone()
    if row:
        valid_days = row[1]
        expire_date = (datetime.now() + timedelta(days=valid_days)).strftime('%Y-%m-%d %H:%M:%S')
        try:
            cursor.execute("UPDATE users SET user_level = 'vip', vip_expire_at = ? WHERE username = ?", (expire_date, username))
            cursor.execute("UPDATE vip_codes SET is_used = 1, used_by = ? WHERE code = ?", (username, code))
            conn.commit()
            return jsonify({"status": "success", "msg": f"激活成功！有效期至 {expire_date}"})
        except Exception as e:
            conn.rollback()
            return jsonify({"status": "error", "msg": str(e)}), 500
        finally:
            conn.close()
    return jsonify({"status": "error", "msg": "无效卡密"}), 400

@app.route('/api/reset_password', methods=['POST'])
def api_reset_password():
    data = request.json
    email = data.get('email')
    vcode = data.get('vcode')
    new_password = data.get('password')
    if not all([email, vcode, new_password]):
        return jsonify({"status": "error", "msg": "请填写完整信息"}), 400
    if not verify_code_logic(email, vcode):
        return jsonify({"status": "error", "msg": "验证码错误或已过期"}), 400
    pwd_hash = generate_password_hash(new_password)
    if update_password_by_email(email, pwd_hash):
        return jsonify({"status": "success", "msg": "密码已重置"})
    else:
        return jsonify({"status": "error", "msg": "该邮箱未注册"}), 404

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login_page'))

# ————————————————————————————————#
#             业务路由             #
# ————————————————————————————————#

@app.route('/')
@login_required
def home():
    db_files = [f for f in os.listdir(COLOR_DB_DIR) if f.endswith('.db')]
    current_name = os.path.basename(current_color_db)
    return render_template('index.html', db_files=db_files, current=current_name)

@app.route('/image_conversion', methods=['GET', 'POST'])
@login_required
def image_conversion():
    global temp_result_data
    if request.method == 'POST':
        if 'file' not in request.files: return render_template('index.html', error="请选择文件")
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename): return render_template('index.html', error="文件格式不正确")

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

            if is_reduce_Pro_on:
                _, processed_img, color_array, color_code_count = reduce_image_colors_Pro(
                    input_path, output_path, current_color_db, scale_factor=calc_scale_factor, target_color_count=target_count, pixel_scale=pixel_size
                )
            elif is_reduce_on:
                _, processed_img, color_array, color_code_count = reduce_image_colors(
                    input_path, output_path, current_color_db, scale_factor=calc_scale_factor, target_color_count=target_count, pixel_scale=pixel_size
                )
            else:
                _, processed_img, _, color_array, color_code_count = process_image_with_color_code(
                    input_path, output_path, current_color_db, scale_factor=calc_scale_factor, pixel_scale=pixel_size
                )

            # 暂存数据 (注意：并发高时建议放入 Session 或 Redis)
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

@app.route('/colors')
def view_colors():
    return render_template('colors.html')

@app.route('/download_file/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.exists(file_path): return "文件不存在", 404
    return send_file(file_path, as_attachment=True)

@app.route('/api/save_modified', methods=['POST'])
@login_required
def save_modified():
    # 1. 获取 Drawing ID
    req_id = request.json.get('drawing_id')
    # 如果没有传 ID，生成一个新的 (通常用于从转换页第一次保存)
    if not req_id:
        req_id = f"{str(uuid.uuid4())[:8]}"
        
    # 2. 存入 Session，确保后续操作针对此图
    session['drawing_id'] = req_id

    # 3. 获取数据
    grid_to_save = current_state.get('grid')
    counts_to_save = current_state.get('color_code_count')
    if grid_to_save is None and temp_result_data.get('color_array') is not None:
        grid_to_save = temp_result_data['color_array']
        counts_to_save = temp_result_data['color_code_count']

    if grid_to_save is None:
        return jsonify({"status": "error", "msg": "没有可保存的数据"}), 400

    try:
        color_array = np.array(grid_to_save, dtype=object)
        # save_drawing_to_sqlite 会自动根据 user.id 存入专属目录
        save_drawing_to_sqlite(req_id, color_array, counts_to_save)
        
        return jsonify({
            "status": "success", 
            "msg": "图纸已保存", 
            "drawing_id": req_id
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/download_modified')
@login_required
def download_modified():
    drawing_id = session.get('drawing_id')
    if not drawing_id: return "请先打开或保存一张图纸", 400

    grid_array, color_code_count = load_drawing_from_sqlite(drawing_id)
    if grid_array is None: return "数据丢失", 404

    grid_array = np.array(grid_array)
    if grid_array.ndim == 2: grid_array = grid_array[:, :, np.newaxis]
    if image_utils._COLOR_CACHE is None: init_color_cache(current_color_db) 

    try:
        img = visualize_color_array(grid_array, color_code_count, pixel_scale=20) 
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"export_{drawing_id}.png")
        img.save(output_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"生成图片失败: {str(e)}", 500

# ————————————————————————————————#
#             绘图页交互           #
# ————————————————————————————————#
@app.route('/draw_page')
@login_required
def draw_page():
    global temp_result_data
    
    # 1. 确定 Drawing ID (优先级: URL参数 > Session > 新建)
    current_id = request.args.get('id')
    if not current_id:
        current_id = session.get('drawing_id')
    
    # 2. 场景 A: 从转换页面刚刚跳转过来，有临时数据
    if temp_result_data.get('color_array') is not None:
        if not current_id:
            current_id = f"{str(uuid.uuid4())[:8]}" # 生成新 ID
        
        # 立即落盘到用户专属目录
        save_drawing_to_sqlite(current_id, temp_result_data['color_array'], temp_result_data['color_code_count'])
        
        # 清空临时区并绑定 Session
        temp_result_data['color_array'] = None
        session['drawing_id'] = current_id

    # 3. 场景 B: 没有 ID，创建空白图
    if not current_id:
        current_id = f"{str(uuid.uuid4())[:8]}"
        create_blank_drawing_logic(current_id, 40, 40)
        session['drawing_id'] = current_id

    # 4. 获取历史图纸列表 (仅 VIP 可见)
    drawings = []
    if current_user.user_level == 'vip':
        user_dir = os.path.join('./data/DrawingData', f"user_{current_user.id}")
        if os.path.exists(user_dir):
            drawings = [f[:-3] for f in os.listdir(user_dir) if f.endswith('.db')]

    # 5. 加载数据
    grid_array, color_code_count = load_drawing_from_sqlite(current_id)
    if grid_array is None:
        # 如果 ID 无效，重置为新的空白图
        current_id = f"{str(uuid.uuid4())[:8]}"
        session['drawing_id'] = current_id
        grid_array, color_code_count = create_blank_drawing_logic(current_id, 40, 40)

    # 准备调色盘
    if image_utils._COLOR_CACHE is None:
        image_utils.init_color_cache(COLOR_DB_PATH)
    
    palette = {}
    if image_utils._COLOR_CACHE:
        for row in image_utils._COLOR_CACHE['full_rows']:
            color_id = str(row[0])
            palette[color_id] = {
                "r_rgb": int(row[1]), "g_rgb": int(row[2]), "b_rgb": int(row[3]),
                "l_lab": float(row[4]), "a_lab": float(row[5]), "b_lab": float(row[6])
            }

    return render_template('draw_page.html', 
                            grid=grid_array.tolist(), 
                            palette=palette,         
                            color_counts=color_code_count,
                            drawings=drawings, # VIP 才会有列表
                            current_id=current_id,
                            user_level=current_user.user_level)

@app.route('/api/create_blank', methods=['POST'])
@login_required
def create_blank():
    data = request.json
    drawing_id = data.get('drawing_id')
    width = int(data.get('width', 40))
    height = int(data.get('height', 40))

    try:
        create_blank_drawing_logic(drawing_id, width, height)
        session['drawing_id'] = drawing_id
        return jsonify({"status": "success", "drawing_id": drawing_id})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/crop_drawing', methods=['POST'])
@login_required
def crop_drawing():
    data = request.json
    drawing_id = session.get('drawing_id')
    if not drawing_id: return jsonify({"status": "error", "msg": "会话丢失"}), 400
    try:
        new_counts = crop_drawing_logic(drawing_id, data['x1'], data['y1'], data['x2'], data['y2'])
        return jsonify({"status": "success", "new_counts": new_counts})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/load_drawing', methods=['POST'])
@login_required
def load_drawing_api():
    drawing_id = request.json.get('drawing_id')
    if get_db_path(drawing_id) and os.path.exists(get_db_path(drawing_id)):
        session['drawing_id'] = drawing_id 
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "msg": "图纸文件不存在"}), 404

@app.route('/api/update_pixel', methods=['POST'])
@login_required
def update_pixel():
    data = request.json
    drawing_id = session.get('drawing_id')
    if not drawing_id: return jsonify({"status": "error", "msg": "会话丢失"}), 400

    try:
        r, c = int(data['r']), int(data['c'])
        new_id = str(data['new_id'])
        
        new_counts = update_pixel_in_db(
            drawing_id, r, c, new_id,
            l_lab=data.get('l_lab', 0), a_lab=data.get('a_lab', 0), b_lab=data.get('b_lab', 0),
            r_rgb=data.get('r_rgb', 0), g_rgb=data.get('g_rgb', 0), b_rgb=data.get('b_rgb', 0)
        )
        if new_counts is not None:
            return jsonify({"status": "success", "new_counts": new_counts})
        else:
            return jsonify({"status": "error", "msg": "无需更新"}), 400
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/batch_update', methods=['POST'])
@login_required
def batch_update():
    data = request.json
    drawing_id = session.get('drawing_id')
    if not drawing_id: return jsonify({"status": "error", "msg": "会话丢失"}), 400

    old_id = str(data['old_id'])
    new_id = str(data['new_id'])
    new_counts = batch_update_in_db(drawing_id, old_id, new_id)
    return jsonify({"status": "success", "new_counts": new_counts})
@app.route('/api/undo', methods=['POST'])

# 输入rgb找相近色
@app.route('/api/find_nearest_color', methods=['POST'])
@login_required
def api_find_nearest_color():
    data = request.json
    try:
        r = float(data.get('r', 0))
        g = float(data.get('g', 0))
        b = float(data.get('b', 0))
        
        # 归一化为 0-1 范围，符合 skimage rgb2lab 的输入要求
        img_np = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
        # 调用 image_utils 中的 skimage color 模块
        img_lab = image_utils.color.rgb2lab(img_np)
        target_lab = img_lab[0, 0]
        
        # 寻找相近色
        code, rgb, lab = image_utils.find_nearest_color(target_lab)
        
        if code is None:
            return jsonify({"status": "error", "msg": "颜色缓存未初始化"}), 500
            
        return jsonify({
            "status": "success", 
            "code": code, 
            "rgb": rgb, 
            "lab": lab
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@login_required
def undo_action():
    drawing_id = session.get('drawing_id')
    if not drawing_id: return jsonify({"status": "error", "msg": "会话丢失"}), 400
    
    if undo_logic(drawing_id):
        return jsonify({"status": "success", "msg": "撤销成功"})
    else:
        return jsonify({"status": "error", "msg": "没有可撤销的步骤"}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)