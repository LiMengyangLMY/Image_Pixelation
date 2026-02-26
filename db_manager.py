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
COLOR_DB_PATH = './data/Color/colors.db'
USER_DB_PATH = './data/users.db'

# ————————————————— 核心：路径路由 —————————————————
def get_user_dir(user_id):
    """确保用户的专属目录存在"""
    user_dir = os.path.join(DB_ROOT, f"user_{user_id}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def get_db_path(drawing_id):
    """
    Day 4 重构：路径严格基于当前登录用户的 ID。
    格式：./data/DrawingData/user_{id}/{drawing_id}.db
    """
    # 1. 正常登录用户
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        user_dir = get_user_dir(current_user.id)
        return os.path.join(user_dir, f"{drawing_id}.db")
    
    # 2. 异常兜底（理论上 app.py 的 @login_required 会拦截，但为了代码健壮性）
    # 如果未登录，存入 public_temp
    public_dir = os.path.join(DB_ROOT, 'public_temp')
    if not os.path.exists(public_dir): os.makedirs(public_dir)
    return os.path.join(public_dir, f"{drawing_id}.db")

def get_undo_limit():
    """根据用户等级返回撤销步数"""
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        if current_user.user_level == 'vip':
            return 10  # VIP 尊享 10 步
    return 5           # 普通用户 5 步

# ————————————————— 辅助函数 —————————————————
def get_color_info_from_master_db(color_id):
    """从主颜色库补全丢失的颜色信息"""
    try:
        conn = sqlite3.connect(COLOR_DB_PATH)
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
    return {"r_rgb": 0, "g_rgb": 0, "b_rgb": 0, "l_lab": 0, "a_lab": 0, "b_lab": 0}

# ————————————————— 快照与撤销 —————————————————
def save_snapshot(drawing_id):
    """保存快照 (带自动清理多余步数)"""
    max_snapshots = get_undo_limit()
    base_path = get_db_path(drawing_id)
    if not os.path.exists(base_path): return

    # 1. 滚动旧快照 (例如 4->5, 3->4...)
    for i in range(max_snapshots - 1, 0, -1):
        old_snap = f"{base_path}.snap_{i-1}"
        new_snap = f"{base_path}.snap_{i}"
        if os.path.exists(old_snap):
            shutil.copy2(old_snap, new_snap)

    # 2. 保存当前状态为 snap_0
    shutil.copy2(base_path, f"{base_path}.snap_0")
    
    # 3. 清理超出当前权限的旧快照 (例如 VIP 降级后清理第6-10步)
    for i in range(max_snapshots, 15): 
        excess_snap = f"{base_path}.snap_{i}"
        if os.path.exists(excess_snap):
            os.remove(excess_snap)

def undo_logic(drawing_id):
    """执行撤销"""
    max_snapshots = get_undo_limit()
    base_path = get_db_path(drawing_id)
    snap_0 = f"{base_path}.snap_0"
    
    if not os.path.exists(snap_0): return False

    # 恢复数据
    shutil.copy2(snap_0, base_path)
    
    # 移除已使用的 snap_0，快照队列前移
    os.remove(snap_0)
    for i in range(1, max_snapshots):
        this_snap = f"{base_path}.snap_{i}"
        prev_snap = f"{base_path}.snap_{i-1}"
        if os.path.exists(this_snap):
            os.rename(this_snap, prev_snap)
    return True

# ——————————————————————————————————————————————————#
#                      图纸操作逻辑 
# ——————————————————————————————————————————————————#
def create_blank_drawing_logic(drawing_id, width=40, height=40):
    default_id = "H2"
    h2_data = {"count": width * height, "r_rgb": 255, "g_rgb": 255, "b_rgb": 255, "l_lab": 100, "a_lab": 0, "b_lab": 0}
    
    # 尝试从颜色库获取白色信息
    color_info = get_color_info_from_master_db(default_id)
    if color_info['r_rgb'] != 0: h2_data.update(color_info)

    new_grid = np.full((height, width, 1), default_id, dtype=object)
    initial_counts = {default_id: h2_data}
    
    # save_drawing_to_sqlite 会调用 get_db_path，自动路由到 user_{id}
    save_drawing_to_sqlite(drawing_id, new_grid, initial_counts)
    return new_grid, initial_counts

def crop_drawing_logic(drawing_id, x1, y1, x2, y2):
    grid_array, color_code_count = load_drawing_from_sqlite(drawing_id)
    if grid_array is None: raise Exception("无法加载图纸")

    cropped_grid = grid_array[y1:y2+1, x1:x2+1]
    if cropped_grid.shape == grid_array.shape: return color_code_count

    # 重新统计颜色
    new_counts = {}
    for code in cropped_grid.flatten():
        code = str(code)
        if code in new_counts:
            new_counts[code]['count'] += 1
        else:
            info = color_code_count.get(code, get_color_info_from_master_db(code)).copy()
            info['count'] = 1
            new_counts[code] = info

    save_snapshot(drawing_id)
    save_drawing_to_sqlite(drawing_id, cropped_grid, new_counts)
    return new_counts

def update_pixel_in_db(drawing_id, r, c, new_id, l_lab=0, a_lab=0, b_lab=0, r_rgb=0, g_rgb=0, b_rgb=0):
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
        conn.close() # 暂时关闭以释放锁，因为 save_snapshot 要复制文件
        
        # 1. 保存快照
        save_snapshot(drawing_id)
        
        # 2. 重新连接写入
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE grid SET color_id = ? WHERE r = ? AND c = ?', (new_id, int(r), int(c)))
        
        # 更新计数
        if old_id in meta_data:
            meta_data[old_id]['count'] = max(0, meta_data[old_id]['count'] - 1)
        
        if new_id in meta_data:
            meta_data[new_id]['count'] += 1
        else:
            # 补全元数据
            if r_rgb == 0 and g_rgb == 0:
                meta_data[new_id] = get_color_info_from_master_db(new_id)
                meta_data[new_id]['count'] = 1
            else:
                meta_data[new_id] = {
                    "count": 1, "r_rgb": int(r_rgb), "g_rgb": int(g_rgb), "b_rgb": int(b_rgb),
                    "l_lab": float(l_lab), "a_lab": float(a_lab), "b_lab": float(b_lab)
                }
        
        cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))
        conn.commit()
        return meta_data
    except Exception as e:
        if conn: conn.rollback()
        raise e
    finally:
        if conn: conn.close()

def batch_update_in_db(drawing_id, old_id, new_id):
    old_id, new_id = str(old_id), str(new_id)
    if old_id == new_id: return None
    
    save_snapshot(drawing_id) # 快照
    
    db_path = get_db_path(drawing_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE grid SET color_id = ? WHERE color_id = ?', (new_id, old_id))
        change_count = cursor.rowcount
        
        if change_count > 0:
            cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
            meta_data = json.loads(cursor.fetchone()[0])
            
            # 更新计数
            if old_id in meta_data: del meta_data[old_id]
            
            if new_id in meta_data:
                meta_data[new_id]['count'] += change_count
            else:
                info = get_color_info_from_master_db(new_id)
                info['count'] = change_count
                meta_data[new_id] = info
                
            cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))
            conn.commit()
            return meta_data
        return None
    finally:
        conn.close()

# ——————————————————————————————————————————————————#
#                     用户与验证码逻辑                
# ——————————————————————————————————————————————————#
# 保存验证码到数据库
def save_verification_code(email, code):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    # 5分钟有效期
    expire_at = (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
    
    # 先删除旧验证码，防止积压
    cursor.execute("DELETE FROM email_verify WHERE email = ?", (email,))
    cursor.execute("INSERT INTO email_verify (email, code, expire_at) VALUES (?, ?, ?)", 
                   (email, code, expire_at))
    
    conn.commit()
    conn.close()

# 校验验证码
def verify_code_logic(email, code):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute("""
        SELECT id FROM email_verify 
        WHERE email = ? AND code = ? AND expire_at > ?
    """, (email, code, now_str))
    
    row = cursor.fetchone()
    
    if row:
        # 验证成功，立即失效防止重放
        cursor.execute("DELETE FROM email_verify WHERE email = ?", (email,))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

# 通过邮箱重置密码
def update_password_by_email(email, new_password_hash):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", 
                       (new_password_hash, email))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"重置密码失败: {e}")
        return False
    finally:
        conn.close()