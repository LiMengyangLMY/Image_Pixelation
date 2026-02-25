#=====================================================
#       Flask Web应用 + 图片颜色编码处理
#=====================================================
from flask import Flask, render_template, request, url_for, send_file, jsonify, redirect, flash
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
    init_color_cache,
    _COLOR_CACHE
)
from db_manager import (
    create_blank_drawing_logic,
    update_pixel_in_db, 
    batch_update_in_db, 
    get_db_path,
    crop_drawing_logic,
    save_verification_code,
    verify_code_logic,
    update_password_by_email
)
from file_manager import limit_files

app = Flask(__name__)
# 设置一个随机的密钥，用于加密 Session
app.secret_key = 'wo-de-pin-dou-xiang-mu-secret-key' 

# 路径配置
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
COLOR_DB_DIR = './data/Color/'
if not os.path.exists(COLOR_DB_DIR):
    os.makedirs(COLOR_DB_DIR)

# 默认颜色数据库
current_color_db = os.path.join(COLOR_DB_DIR, 'colors.db')

# ——————————————————————————————————————#
#          邮件配置 (SMTP)              #
# ——————————————————————————————————————#
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '1364527938@qq.com'
app.config['MAIL_PASSWORD'] = 'klmwjzlnsgsngeab'  # 授权码
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

mail = Mail(app)

# ————————————————————————————————#
#             全局变量 (注意并发)  #
# ————————————————————————————————#
# 注意：在多用户环境下，使用全局变量存储状态是不安全的。
# 建议后续改为将 drawing_id 存储在用户的 session 中。
current_state = {
    'grid': None,
    'palette': {},
    'pixel_size': 20,
    'color_code_count': {}
}
temp_result_data = {"color_array": None, "pixel_size": 20, "color_code_count": {}}
CURRENT_DRAWING_ID = 'last_converted'

# ————————————————————————————————#
#          工具函数                #
# ————————————————————————————————#
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_verification_code():
    return str(random.randint(100000, 999999))

def send_async_email(app_obj, msg):
    with app_obj.app_context():
        try:
            mail.send(msg)
            print(f"邮件已发送至 {msg.recipients}")
        except Exception as e:
            print(f"邮件发送失败: {e}")

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

# 统一的 user_loader (包含了 VIP 过期检查逻辑)
@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('./data/users.db')
    cursor = conn.cursor()
    # 注意：这里假设数据库中有 vip_expire_at 字段，如果没有请先执行 init_user_db.py 中的升级脚本
    try:
        cursor.execute("SELECT id, username, email, user_level, vip_expire_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
    except sqlite3.OperationalError:
        # 容错：如果数据库表结构还没更新
        cursor.execute("SELECT id, username, email, user_level FROM users WHERE id = ?", (user_id,))
        row = list(cursor.fetchone()) + [None] if cursor.fetchone() else None

    if row:
        user_id_val, username, email, level, expire_at = row
        
        # --- VIP 自动过期逻辑 ---
        if level == 'vip' and expire_at:
            try:
                expire_dt = datetime.strptime(expire_at, '%Y-%m-%d %H:%M:%S')
                if datetime.now() > expire_dt:
                    # 已过期，降级
                    cursor.execute("UPDATE users SET user_level = 'common' WHERE id = ?", (user_id_val,))
                    conn.commit()
                    level = 'common'
            except ValueError:
                pass # 时间格式错误忽略
        
        conn.close()
        return User(user_id_val, username, email, level)
    
    conn.close()
    return None

# ————————————————————————————————#
#          身份验证路由            #
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

    if not email:
        return jsonify({"status": "error", "msg": "请输入邮箱地址"}), 400

    # 生成验证码
    code = generate_verification_code()
    
    # 存入数据库
    try:
        save_verification_code(email, code)
    except Exception as e:
        return jsonify({"status": "error", "msg": f"数据库错误: {str(e)}"}), 500

    # 构造邮件
    msg = Message(subject="【拼豆图纸】注册验证码", recipients=[email])
    msg.body = f"您的注册验证码是：{code}，有效期为5分钟。如非本人操作请忽略。"
    
    try:
        print(f"正在向 {email} 发送邮件...")
        mail.send(msg)
        print("✅ 发送成功！")
        return jsonify({"status": "success", "msg": "验证码已发送，请查收！"})
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        # 这里会把具体错误返回给网页，让你直接看到
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
    vcode = data.get('vcode')       # 前端传来的验证码
    new_password = data.get('password')
    
    # 1. 基础校验
    if not all([email, vcode, new_password]):
        return jsonify({"status": "error", "msg": "请填写完整信息"}), 400

    # 2. 校验验证码 (关键步骤)
    if not verify_code_logic(email, vcode):
        return jsonify({"status": "error", "msg": "验证码错误或已过期"}), 400

    # 3. 生成新密码哈希
    pwd_hash = generate_password_hash(new_password)
    
    # 4. 更新数据库
    if update_password_by_email(email, pwd_hash):
        return jsonify({"status": "success", "msg": "密码已重置，请使用新密码登录"})
    else:
        return jsonify({"status": "error", "msg": "该邮箱未注册"}), 404

@app.route('/logout')
@login_required
def logout():
    logout_user()
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
    if not os.path.exists(file_path):
        return "文件不存在", 404
    return send_file(file_path, as_attachment=True)

@app.route('/api/save_modified', methods=['POST'])
def save_modified():
    global CURRENT_DRAWING_ID
    # 尝试从 temp_result_data 获取数据
    grid_to_save = current_state.get('grid')
    counts_to_save = current_state.get('color_code_count')

    if grid_to_save is None and temp_result_data.get('color_array') is not None:
        grid_to_save = temp_result_data['color_array']
        counts_to_save = temp_result_data['color_code_count']

    if grid_to_save is None:
        return jsonify({"status": "error", "msg": "没有可保存的数据"}), 400

    try:
        drawing_id = request.json.get('drawing_id') or str(uuid.uuid4())[:8]
        # 确保是 numpy 数组
        color_array = np.array(grid_to_save, dtype=object)
        save_drawing_to_sqlite(drawing_id, color_array, counts_to_save)
        
        CURRENT_DRAWING_ID = drawing_id 
        return jsonify({"status": "success", "msg": "图纸源数据已保存", "drawing_id": drawing_id})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/download_modified')
def download_modified():
    # 1. 从数据库加载数据
    grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
    
    if grid_array is None:
        return "无可下载的数据", 404

    grid_array = np.array(grid_array)
    # 处理维度问题，visualize 需要 (H,W,1)
    if grid_array.ndim == 2:
        grid_array = grid_array[:, :, np.newaxis]

    if _COLOR_CACHE is None:
        init_color_cache(current_color_db) 

    try:
        img = visualize_color_array(grid_array, color_code_count, pixel_scale=20) 
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"export_{CURRENT_DRAWING_ID}.png")
        img.save(output_path)
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return f"生成图片失败: {str(e)}", 500

# ————————————————————————————————#
#             绘图页交互           #
# ————————————————————————————————#
@app.route('/draw_page')
def draw_page():
    global temp_result_data, CURRENT_DRAWING_ID
    
    # 1. 尝试保存处理后的临时数据
    if temp_result_data.get('color_array') is not None:
        save_drawing_to_sqlite(CURRENT_DRAWING_ID, temp_result_data['color_array'], temp_result_data['color_code_count'])
        temp_result_data['color_array'] = None

    # 2. 扫描已有图纸
    drawings_dir = './data/DrawingData'
    drawings = [f[:-3] for f in os.listdir(drawings_dir) if f.endswith('.db')] if os.path.exists(drawings_dir) else []

    # 3. 加载当前
    grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
    
    # 4. 自动处理空数据
    if grid_array is None:
        if drawings:
            CURRENT_DRAWING_ID = drawings[0]
            grid_array, color_code_count = load_drawing_from_sqlite(CURRENT_DRAWING_ID)
        else:
            new_id = "default_blank"
            # 修正：此处删除了多余的 current_color_db 参数
            grid_array, color_code_count = create_blank_drawing_logic(new_id, 40, 40)
            CURRENT_DRAWING_ID = new_id
            drawings = [new_id]

    # 5. 准备调色盘
    if _COLOR_CACHE is None:
        init_color_cache(current_color_db) 
        
    palette = {}
    if _COLOR_CACHE:
        for row in _COLOR_CACHE['full_rows']:
            color_id = str(row[0])
            palette[color_id] = {
                "r_rgb": int(row[1]), "g_rgb": int(row[2]), "b_rgb": int(row[3]),
                "l_lab": float(row[4]), "a_lab": float(row[5]), "b_lab": float(row[6])
            }

    # 确保 grid 是 list 格式传给前端 JSON
    grid_list = grid_array.tolist() if grid_array is not None else []

    return render_template('draw_page.html', 
                            grid=grid_list, 
                            palette=palette,         
                            color_counts=color_code_count,
                            drawings=drawings,
                            current_id=CURRENT_DRAWING_ID)

@app.route('/api/create_blank', methods=['POST'])
def create_blank():
    global CURRENT_DRAWING_ID
    data = request.json
    drawing_id = data.get('drawing_id')
    width = int(data.get('width', 40))
    height = int(data.get('height', 40))

    try:
        # 修正：删除了 current_color_db 参数，防止报错
        create_blank_drawing_logic(drawing_id, width, height)
        CURRENT_DRAWING_ID = drawing_id
        
        return jsonify({
            "status": "success", 
            "drawing_id": drawing_id,
            "message": "已成功创建空白图纸"
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/crop_drawing', methods=['POST'])
def crop_drawing():
    data = request.json
    drawing_id = CURRENT_DRAWING_ID
    try:
        new_counts = crop_drawing_logic(
            drawing_id, 
            data['x1'], data['y1'], 
            data['x2'], data['y2']
        )
        return jsonify({"status": "success", "new_counts": new_counts})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/load_drawing', methods=['POST'])
def load_drawing_api():
    global CURRENT_DRAWING_ID
    drawing_id = request.json.get('drawing_id')
    
    if os.path.exists(get_db_path(drawing_id)):
        CURRENT_DRAWING_ID = drawing_id 
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "msg": "图纸文件不存在"}), 404

@app.route('/api/update_pixel', methods=['POST'])
def update_pixel():
    data = request.json
    try:
        r, c = int(data['r']), int(data['c'])
        new_id = str(data['new_id'])
        
        # 提取可选参数
        l_lab = data.get('l_lab', 0)
        a_lab = data.get('a_lab', 0)
        b_lab = data.get('b_lab', 0)
        r_rgb = data.get('r_rgb', 0)
        g_rgb = data.get('g_rgb', 0)
        b_rgb = data.get('b_rgb', 0)
        
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

@app.route('/api/batch_update', methods=['POST'])
def batch_update():
    data = request.json
    old_id = str(data['old_id'])
    new_id = str(data['new_id'])
    
    new_counts = batch_update_in_db(CURRENT_DRAWING_ID, old_id, new_id)
    
    # 注意：如果 batch_update 返回 None (例如没有颜色被替换)，前端也应该收到 success
    if new_counts is not None:
        return jsonify({"status": "success", "new_counts": new_counts})
    else:
        return jsonify({"status": "success", "msg": "无需更新或更新未执行"})

@app.route('/api/undo', methods=['POST'])
def undo_action():
    from db_manager import undo_logic
    drawing_id = CURRENT_DRAWING_ID
    
    if undo_logic(drawing_id, 10):
        return jsonify({"status": "success", "msg": "撤销成功"})
    else:
        return jsonify({"status": "error", "msg": "没有可撤销的步骤"}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)