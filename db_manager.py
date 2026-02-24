import sqlite3
import json
import numpy as np
import os
import shutil
from image_utils import save_drawing_to_sqlite,load_drawing_from_sqlite
#路径设置
USER_DB_PATH = './data/users.db'
DB_DIR = './data/DrawingData'
COLOR_DB_PATH = './data/Color/colors.db'
#快照数，即可撤回的步骤数
MAX_UNDO_STEPS = 10
#获取地址
def get_db_path(drawing_id):
    return os.path.join(DB_DIR, f"{drawing_id}.db")


def get_color_info_from_master_db(color_id):
    """辅助函数：从主颜色库获取颜色的 RGB/Lab 信息"""
    try:
        conn = sqlite3.connect(COLOR_DB_PATH)
        cursor = conn.cursor()
        # 假设 colors 表结构是: num, R, G, B, lab_l, lab_a, lab_b
        cursor.execute("SELECT R, G, B, lab_l, lab_a, lab_b FROM colors WHERE num = ?", (str(color_id),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "r_rgb": int(row[0]), "g_rgb": int(row[1]), "b_rgb": int(row[2]),
                "l_lab": float(row[3]), "a_lab": float(row[4]), "b_lab": float(row[5])
            }
    except Exception as e:
        print(f"无法从主库获取颜色 {color_id}: {e}")
    
    # 如果找不到，返回默认黑色或错误标识
    return {"r_rgb": 0, "g_rgb": 0, "b_rgb": 0, "l_lab": 0, "a_lab": 0, "b_lab": 0}


#————————————————————————————————————————#
#          撤销功能（快照功能）            #
#————————————————————————————————————————# 
#滚动备份0-max_snapshots-1（共max_snapshots个备份）
def save_snapshot(drawing_id,max_snapshots=MAX_UNDO_STEPS):
    base_path = get_db_path(drawing_id)
    if not os.path.exists(base_path):
        return

    # 滚动位移旧快照
    for i in range(max_snapshots-1, 0, -1):
        old_snap = f"{base_path}.snap_{i-1}"
        new_snap = f"{base_path}.snap_{i}"
        if os.path.exists(old_snap):
            shutil.copy2(old_snap, new_snap)

    # 创建当前状态的 snapshot_0
    shutil.copy2(base_path, f"{base_path}.snap_0")

#将 snapshot_0 恢复为当前数据库，并位移后续快照
def undo_logic(drawing_id,max_snapshots=MAX_UNDO_STEPS):
    base_path = get_db_path(drawing_id)
    snap_0 = f"{base_path}.snap_0"
    
    if not os.path.exists(snap_0):
        return False

    # 1. 恢复当前数据库
    shutil.copy2(snap_0, base_path)
    
    # 2. 移除已使用的快照，并将后面的向前移动
    os.remove(snap_0)
    for i in range(1, max_snapshots):
        this_snap = f"{base_path}.snap_{i}"
        prev_snap = f"{base_path}.snap_{i-1}"
        if os.path.exists(this_snap):
            os.rename(this_snap, prev_snap)
            
    return True

#————————————————————————————————————————#
#                新建空白图纸             #
#————————————————————————————————————————# 
#新建空白图纸
def create_blank_drawing_logic(drawing_id, width=40, height=40):
    default_id = "H2"
    h2_data = {
        "count": width * height,
        "r_rgb": 255, "g_rgb": 255, "b_rgb": 255,
        "l_lab": 100, "a_lab": 0, "b_lab": 0
    }
    
    # 1. 查询颜色库获取 H2 属性
    try:
        with sqlite3.connect(current_color_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT R, G, B, lab_l, lab_a, lab_b FROM colors WHERE num = ?", (default_id,))
            row = cursor.fetchone()
            if row:
                h2_data.update({
                    "r_rgb": int(row[0]), "g_rgb": int(row[1]), "b_rgb": int(row[2]),
                    "l_lab": float(row[3]), "a_lab": float(row[4]), "b_lab": float(row[5])
                })
    except Exception as e:
        print(f"查询颜色库失败，使用默认白色填充: {e}")

    # 2. 生成数据矩阵
    new_grid = np.full((height, width, 1), default_id, dtype=object)
    initial_counts = {default_id: h2_data}
    
    # 3. 保存到数据库
    save_drawing_to_sqlite(drawing_id, new_grid, initial_counts)
    
    return new_grid, initial_counts

#————————————————————————————————————————#
#          交互修改图纸源数据              #
#————————————————————————————————————————# 
#剪裁图纸
def crop_drawing_logic(drawing_id, x1, y1, x2, y2):
    """封装裁剪的核心逻辑"""
    # 1. 从数据库读取当前完整数据
    grid_array, color_code_count = load_drawing_from_sqlite(drawing_id)
    if grid_array is None:
        raise Exception("无法加载图纸数据")

    # 2. 执行 NumPy 裁剪操作
    cropped_grid = grid_array[y1:y2+1, x1:x2+1]
    
    # 3. 检查是否有实际变化 (可选：防止空操作产生快照)
    if cropped_grid.shape == grid_array.shape:
        return color_code_count

    # 4. 重新计算裁剪区域内的颜色统计
    new_counts = {}
    for code in cropped_grid.flatten():
        code = str(code)
        if code in new_counts:
            new_counts[code]['count'] += 1
        else:
            info = color_code_count.get(code, {
                "r_rgb": 0, "g_rgb": 0, "b_rgb": 0,
                "l_lab": 0, "a_lab": 0, "b_lab": 0
            }).copy()
            info['count'] = 1
            new_counts[code] = info

    # 5. 修改前执行快照保存
    save_snapshot(drawing_id)

    # 6. 将裁剪后的数据覆盖写入原数据库文件
    save_drawing_to_sqlite(drawing_id, cropped_grid, new_counts)
    
    return new_counts

#单格更换
def update_pixel_in_db(drawing_id, r, c, new_id, l_lab=0, a_lab=0, b_lab=0, r_rgb=0, g_rgb=0, b_rgb=0):
    new_id = str(new_id).strip()
    db_path = get_db_path(drawing_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('BEGIN TRANSACTION;')

        # 获取旧ID
        cursor.execute('SELECT color_id FROM grid WHERE r = ? AND c = ?', (int(r), int(c)))
        result = cursor.fetchone()
        if not result: return None
        old_id = str(result[0]).strip()

        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_data = json.loads(cursor.fetchone()[0])

        if old_id == new_id:
            conn.commit()
            return meta_data

        save_snapshot(drawing_id)
        # 更新网格
        cursor.execute('UPDATE grid SET color_id = ? WHERE r = ? AND c = ?', (new_id, int(r), int(c)))

        # 更新Metadata: 旧颜色 -1
        if old_id in meta_data:
            meta_data[old_id]['count'] = max(0, meta_data[old_id]['count'] - 1)
            # 可选：如果减到0是否删除？通常保留以便撤销，或者前端过滤
            if meta_data[old_id]['count'] == 0:
                pass 

        # 更新Metadata: 新颜色 +1
        if new_id in meta_data:
            meta_data[new_id]['count'] += 1
        else:
            # =================================================
            # [增强修复]：如果前端传来的全是 0，尝试从主库补全数据
            # =================================================
            if r_rgb == 0 and g_rgb == 0 and b_rgb == 0:
                # 认为是数据缺失，从主库查
                fetched_info = get_color_info_from_master_db(new_id)
                meta_data[new_id] = {
                    "count": 1,
                    "r_rgb": fetched_info['r_rgb'], "g_rgb": fetched_info['g_rgb'], "b_rgb": fetched_info['b_rgb'],
                    "l_lab": fetched_info['l_lab'], "a_lab": fetched_info['a_lab'], "b_lab": fetched_info['b_lab']
                }
            else:
                # 前端数据正常，使用前端数据
                meta_data[new_id] = {
                    "count": 1,
                    "r_rgb": int(r_rgb), "g_rgb": int(g_rgb), "b_rgb": int(b_rgb),
                    "l_lab": float(l_lab), "a_lab": float(a_lab), "b_lab": float(b_lab)
                }

        cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', 
                       (json.dumps(meta_data), 'color_code_count'))
        
        conn.commit()
        return meta_data

    except Exception as e:
        conn.rollback()
        print(f"数据库更新失败: {e}")
        return None
    finally:
        conn.close()

#批量换颜色
def batch_update_in_db(drawing_id, old_id, new_id):

    old_id, new_id = str(old_id), str(new_id)
    if old_id == new_id: return None

    save_snapshot(drawing_id)
    db_path = get_db_path(drawing_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('BEGIN TRANSACTION;')
        
        # 1. 执行网格替换
        cursor.execute('UPDATE grid SET color_id = ? WHERE color_id = ?', (new_id, old_id))
        change_count = cursor.rowcount

        if change_count == 0:
            conn.commit()
            return None

        # 2. 读取元数据
        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_row = cursor.fetchone()
        if not meta_row: return None # 保护机制
        
        meta_data = json.loads(meta_row[0])

        if old_id in meta_data:
            # 移除旧颜色计数 (逻辑修正：如果只替换了部分，不应该直接 pop，但这里是 UPDATE WHERE color_id=old 也就是全部替换，所以 pop 是安全的)
            # 但为了安全起见，我们把 count 设为 0 或者删除
            del meta_data[old_id]

            # 更新新颜色
            if new_id in meta_data:
                meta_data[new_id]['count'] += change_count
            else:
                # =================================================
                # [致命错误修复]：不要复制 old_info！去主库查！
                # =================================================
                color_info = get_color_info_from_master_db(new_id)
                
                meta_data[new_id] = {
                    "count": change_count,
                    "r_rgb": color_info['r_rgb'],
                    "g_rgb": color_info['g_rgb'],
                    "b_rgb": color_info['b_rgb'],
                    "l_lab": color_info['l_lab'],
                    "a_lab": color_info['a_lab'],
                    "b_lab": color_info['b_lab']
                }

            cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))

        conn.commit()
        return meta_data

    except Exception as e:
        conn.rollback()
        print(f"同色替换失败: {e}")
        return None
    finally:
        conn.close()

#————————————————————————————————————————#
#                USER管理                #
#————————————————————————————————————————#     
# 新增用户
def add_user(username, password_hash):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                       (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print("用户名已存在")
        return False
    finally:
        conn.close()

# 修改密码
def update_password(username, new_password_hash):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", 
                       (new_password_hash, username))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"修改密码失败: {e}")
        return False
    finally:
        conn.close()

# 获取用户信息：(id, username, password_hash, user_level)
def get_user_info(username):
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash, user_level FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

#设置用户等级：(common 或 vip)
def set_user_level(username, level):
    if level not in ['common', 'vip']:
        return False
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET user_level = ? WHERE username = ?", (level, username))
        conn.commit()
        return True
    finally:
        conn.close()

