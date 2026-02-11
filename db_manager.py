import sqlite3
import json
import numpy as np
import os

DB_DIR = './data/DrawingData'

#获取地址
def get_db_path(drawing_id):
    return os.path.join(DB_DIR, f"{drawing_id}.db")


# 增加颜色库的路径定义
COLOR_DB_PATH = './data/Color/colors.db'

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

# 
def batch_update_in_db(drawing_id, old_id, new_id):
    old_id, new_id = str(old_id), str(new_id)
    if old_id == new_id: return None

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