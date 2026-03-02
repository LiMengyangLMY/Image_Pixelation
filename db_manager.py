import sqlite3
import json
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from flask_login import current_user
from image_utils import save_drawing_to_sqlite, load_drawing_from_sqlite

# ————————————————— 配置路径 —————————————————
DB_ROOT = './data/DrawingData'
COLOR_DB_ROOT = './data/Color'
DEFAULT_COLOR_DB_PATH = os.path.join(COLOR_DB_ROOT, 'colors.db')
USER_DB_PATH = './data/users.db'

# ————————————————— 核心：路径路由 —————————————————
def get_user_dir(user_id):
    """确保用户的专属画图目录存在"""
    user_dir = os.path.join(DB_ROOT, f"user_{user_id}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def get_db_path(drawing_id):
    """画图数据路径：基于当前登录用户的 ID"""
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        user_dir = get_user_dir(current_user.id)
        return os.path.join(user_dir, f"{drawing_id}.db")
    
    public_dir = os.path.join(DB_ROOT, 'public_temp')
    if not os.path.exists(public_dir): os.makedirs(public_dir)
    return os.path.join(public_dir, f"{drawing_id}.db")

def get_undo_limit():
    """根据用户等级返回撤销步数"""
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if current_user.user_level == 'vip':
            return 10  
    return 5           

# ————————————————— 用户颜色库管理逻辑 —————————————————
def get_user_color_db_dir(user_id):
    """确保用户的专属颜色库目录存在"""
    user_color_dir = os.path.join(COLOR_DB_ROOT, f"user_{user_id}")
    if not os.path.exists(user_color_dir):
        os.makedirs(user_color_dir)
    return user_color_dir

def get_color_db_path(db_name):
    """获取指定的颜色库路径，带有空表自动修复功能"""
    import os
    import shutil
    import sqlite3
    
    # 1. 如果前端传的是默认库
    if db_name == 'colors.db' or not db_name:
        return DEFAULT_COLOR_DB_PATH
    
    # 2. 如果前端传的是专属库
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        user_color_dir = get_user_color_db_dir(current_user.id)
        db_path = os.path.join(user_color_dir, db_name)
        # 核心逻辑：判断是否需要从默认库复制初始化
        needs_init = False
        
        # 场景 A: 文件根本不存在，或者文件大小为 0 (空壳)
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            needs_init = True
        else:
            # 场景 B: 文件存在，但里面可能没有 colors 表 (损坏或意外生成)
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='colors'")
                if not cursor.fetchone():
                    needs_init = True
                conn.close()
            except Exception:
                needs_init = True # 只要读取报错，一律重置
                
        # 执行初始化/修复操作
        if needs_init:
            shutil.copy2(DEFAULT_COLOR_DB_PATH, db_path)
            
        return db_path
    else:
        raise PermissionError("请先登录再操作自定义颜色库。")

def verify_color_db_permission(db_name):
    """核心拦截器：校验是否有权限对该颜色库进行修改/删除"""
    if db_name == 'colors.db' or not db_name:
        raise PermissionError("系统默认颜色库 (colors.db) 受到保护，不允许任何增删改查或删除表的操作！")

def create_user_color_db(db_name, copy_from_default=True):
    """用户新建自己的颜色库"""
    verify_color_db_permission(db_name)
    
    db_path = get_color_db_path(db_name)
    if os.path.exists(db_path):
        raise FileExistsError(f"颜色库 {db_name} 已存在！")
        
    if copy_from_default:
        shutil.copy2(DEFAULT_COLOR_DB_PATH, db_path)
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE colors 
                          (num TEXT PRIMARY KEY, R INTEGER, G INTEGER, B INTEGER, 
                           lab_l REAL, lab_a REAL, lab_b REAL)''')
        conn.commit()
        conn.close()
    return db_path

def delete_user_color_db(db_name):
    """用户删除自己的颜色库"""
    verify_color_db_permission(db_name)
    
    db_path = get_color_db_path(db_name)
    if os.path.exists(db_path):
        os.remove(db_path)
    else:
        raise FileNotFoundError("找不到要删除的颜色库。")

# ————————————————— 辅助函数 —————————————————
def get_color_info_from_db(color_id, active_color_db='colors.db'):
    """
    仅从指定的 active_color_db 查询颜色。
    如果在该库没找到，直接返回 0 (不再去 colors.db 兜底)。
    """
    target_db_path = get_color_db_path(active_color_db)
    
    try:
        conn = sqlite3.connect(target_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT R, G, B, lab_l, lab_a, lab_b FROM colors WHERE num = ?", (str(color_id),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "r_rgb": int(row[0]), "g_rgb": int(row[1]), "b_rgb": int(row[2]),
                "l_lab": float(row[3]), "a_lab": float(row[4]), "b_lab": float(row[5])
            }
    except Exception:
        pass
        
    # 如果没找到，严格返回全 0（透明/黑）
    return {"r_rgb": 0, "g_rgb": 0, "b_rgb": 0, "l_lab": 0, "a_lab": 0, "b_lab": 0}

# ————————————————— 快照与撤销 —————————————————
def save_snapshot(drawing_id):
    max_snapshots = get_undo_limit()
    base_path = get_db_path(drawing_id)
    if not os.path.exists(base_path): return

    for i in range(max_snapshots - 1, 0, -1):
        old_snap = f"{base_path}.snap_{i-1}"
        new_snap = f"{base_path}.snap_{i}"
        if os.path.exists(old_snap):
            shutil.copy2(old_snap, new_snap)

    shutil.copy2(base_path, f"{base_path}.snap_0")
    
    for i in range(max_snapshots, 15): 
        excess_snap = f"{base_path}.snap_{i}"
        if os.path.exists(excess_snap):
            os.remove(excess_snap)

def undo_logic(drawing_id):
    max_snapshots = get_undo_limit()
    base_path = get_db_path(drawing_id)
    snap_0 = f"{base_path}.snap_0"
    if not os.path.exists(snap_0): return False

    shutil.copy2(snap_0, base_path)
    os.remove(snap_0)
    for i in range(1, max_snapshots):
        this_snap = f"{base_path}.snap_{i}"
        prev_snap = f"{base_path}.snap_{i-1}"
        if os.path.exists(this_snap):
            os.rename(this_snap, prev_snap)
    return True

# ————————————————— 图纸操作逻辑 —————————————————
def create_blank_drawing_logic(drawing_id, width=40, height=40, active_color_db='colors.db'):
    default_id = "H2"
    h2_data = {"count": width * height, "r_rgb": 255, "g_rgb": 255, "b_rgb": 255, "l_lab": 100, "a_lab": 0, "b_lab": 0}
    
    color_info = get_color_info_from_db(default_id, active_color_db)
    if color_info['r_rgb'] != 0: h2_data.update(color_info)

    new_grid = np.full((height, width, 1), default_id, dtype=object)
    initial_counts = {default_id: h2_data}
    
    save_drawing_to_sqlite(drawing_id, new_grid, initial_counts)
    return new_grid, initial_counts

def crop_drawing_logic(drawing_id, x1, y1, x2, y2, active_color_db='colors.db'):
    grid_array, color_code_count = load_drawing_from_sqlite(drawing_id)
    if grid_array is None: raise Exception("无法加载图纸")

    cropped_grid = grid_array[y1:y2+1, x1:x2+1]
    if cropped_grid.shape == grid_array.shape: return color_code_count

    new_counts = {}
    for code in cropped_grid.flatten():
        code = str(code)
        if code in new_counts:
            new_counts[code]['count'] += 1
        else:
            info = color_code_count.get(code, get_color_info_from_db(code, active_color_db)).copy()
            info['count'] = 1
            new_counts[code] = info

    save_snapshot(drawing_id)
    save_drawing_to_sqlite(drawing_id, cropped_grid, new_counts)
    return new_counts

def update_pixel_in_db(drawing_id, r, c, new_id, l_lab=0, a_lab=0, b_lab=0, r_rgb=0, g_rgb=0, b_rgb=0, active_color_db='colors.db'):
    new_id = str(new_id).strip()
    db_path = get_db_path(drawing_id)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('BEGIN TRANSACTION;')
        cursor.execute('SELECT color_id FROM grid WHERE r = ? AND c = ?', (int(r), int(c)))
        res = cursor.fetchone()
        if not res: return None
        old_id = str(res[0])

        if old_id == new_id: return None

        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_data = json.loads(cursor.fetchone()[0])
        conn.close() 
        
        save_snapshot(drawing_id)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE grid SET color_id = ? WHERE r = ? AND c = ?', (new_id, int(r), int(c)))
        
        if old_id in meta_data:
            meta_data[old_id]['count'] = max(0, meta_data[old_id]['count'] - 1)
        
        if new_id in meta_data:
            meta_data[new_id]['count'] += 1
        else:
            if r_rgb == 0 and g_rgb == 0:
                meta_data[new_id] = get_color_info_from_db(new_id, active_color_db)
                meta_data[new_id]['count'] = 1
            else:
                meta_data[new_id] = {
                    "count": 1, "r_rgb": int(r_rgb), "g_rgb": int(g_rgb), "b_rgb": int(b_rgb),
                    "l_lab": float(l_lab), "a_lab": float(a_lab), "b_lab": float(b_lab)
                }
        
        cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))
        meta_data = {k: v for k, v in meta_data.items() if v.get('count', 0) > 0}
        conn.commit()
        return meta_data
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

def batch_update_in_db(drawing_id, old_id, new_id, active_color_db='colors.db'):
    old_id, new_id = str(old_id), str(new_id)
    if old_id == new_id: return None
    
    save_snapshot(drawing_id)
    
    db_path = get_db_path(drawing_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE grid SET color_id = ? WHERE color_id = ?', (new_id, old_id))
        change_count = cursor.rowcount
        
        if change_count > 0:
            cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
            meta_data = json.loads(cursor.fetchone()[0])
            
            if old_id in meta_data: del meta_data[old_id]
            
            if new_id in meta_data:
                meta_data[new_id]['count'] += change_count
            else:
                info = get_color_info_from_db(new_id, active_color_db)
                info['count'] = change_count
                meta_data[new_id] = info
                
            cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))
            meta_data = {k: v for k, v in meta_data.items() if v.get('count', 0) > 0}
            conn.commit()
            return meta_data
        return None
    finally:
        conn.close()

# ————————————————— 用户与验证码逻辑 —————————————————
def save_verification_code(email, code):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    expire_at = (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("DELETE FROM email_verify WHERE email = ?", (email,))
    cursor.execute("INSERT INTO email_verify (email, code, expire_at) VALUES (?, ?, ?)", 
                   (email, code, expire_at))
    conn.commit()
    conn.close()

def verify_code_logic(email, code):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("SELECT id FROM email_verify WHERE email = ? AND code = ? AND expire_at > ?", (email, code, now_str))
    row = cursor.fetchone()
    if row:
        cursor.execute("DELETE FROM email_verify WHERE email = ?", (email,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def update_password_by_email(email, new_password_hash):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_password_hash, email))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"重置密码失败: {e}")
        return False
    finally:
        conn.close()