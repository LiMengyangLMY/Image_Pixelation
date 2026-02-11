import sqlite3
import json
import numpy as np
import os

DB_DIR = './data/DrawingData'

#获取地址
def get_db_path(drawing_id):
    return os.path.join(DB_DIR, f"{drawing_id}.db")

#单格替换+更新数据库
# db_manager.py

def update_pixel_in_db(drawing_id, r, c, new_id, l_lab=0, a_lab=0, b_lab=0, r_rgb=0, g_rgb=0, b_rgb=0):
    # 1. 统一 ID 格式（去除空格）
    new_id = str(new_id).strip()
    
    db_path = get_db_path(drawing_id)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('BEGIN TRANSACTION;')

        # 2. 获取旧像素信息
        cursor.execute('SELECT color_id FROM grid WHERE r = ? AND c = ?', (int(r), int(c)))
        result = cursor.fetchone()
        
        # 3. 读取现有元数据
        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_row = cursor.fetchone()
        if not meta_row:
            return None
        meta_data = json.loads(meta_row[0])

        if not result:
            return meta_data
        
        old_id = str(result[0]).strip()

        # 如果颜色没变，直接返回
        if old_id == new_id:
            conn.commit()
            return meta_data

        # 4. 更新物理网格表
        cursor.execute('UPDATE grid SET color_id = ? WHERE r = ? AND c = ?', (new_id, int(r), int(c)))

        # 5. 更新 Metadata 逻辑
        # A. 旧颜色计数减 1
        if old_id in meta_data:
            meta_data[old_id]['count'] = max(0, meta_data[old_id]['count'] - 1)
            # 如果计数归零，建议保留 Key 以维持色板完整，或根据需求 del
            if meta_data[old_id]['count'] == 0:
                pass 

        # B. 新颜色计数加 1
        if new_id in meta_data:
            meta_data[new_id]['count'] += 1
            # 可选：如果传入了值，也可以在此更新该颜色的最新 Lab/RGB
        else:
            # 如果是一个图纸中从未出现过的新颜色，创建对应条目
            meta_data[new_id] = {
                "count": 1,
                "r_rgb": int(r_rgb), "g_rgb": int(g_rgb), "b_rgb": int(b_rgb),
                "l_lab": float(l_lab), "a_lab": float(a_lab), "b_lab": float(b_lab)
            }

        # 6. 写回数据库并提交
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


#同色替换+更新数据库
def batch_update_in_db(drawing_id, old_id, new_id):
    # 确保 ID 为字符串格式
    old_id, new_id = str(old_id), str(new_id)
    if old_id == new_id:
        return None

    db_path = f'./data/DrawingData/{drawing_id}.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. 开启事务保证原子性
        cursor.execute('BEGIN TRANSACTION;')

        # 2. 执行物理替换
        cursor.execute('UPDATE grid SET color_id = ? WHERE color_id = ?', (new_id, old_id))
        # 获取受影响的行数（即被替换的方格数量）
        change_count = cursor.rowcount

        # 如果没有方格被替换，直接提交并返回
        if change_count == 0:
            conn.commit()
            return None

        # 3. 读取元数据进行逻辑加减
        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_data = json.loads(cursor.fetchone()[0])

        if old_id in meta_data:
            # 提取旧颜色的元数据信息
            old_info = meta_data.pop(old_id)
            
            # 更新新颜色的统计逻辑
            if new_id in meta_data:
                meta_data[new_id]['count'] += change_count
            else:
                # 如果新 ID 不在当前统计中，创建它并继承颜色属性（或根据业务逻辑初始化）
                meta_data[new_id] = {
                    "count": change_count,
                    "r_rgb": old_info.get('r_rgb', 0),
                    "g_rgb": old_info.get('g_rgb', 0),
                    "b_rgb": old_info.get('b_rgb', 0),
                    "l_lab": old_info.get('l_lab', 0),
                    "a_lab": old_info.get('a_lab', 0),
                    "b_lab": old_info.get('b_lab', 0)
                }

            # 4. 写回更新后的元数据
            cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))

        # 5. 提交事务
        conn.commit()
        return meta_data

    except Exception as e:
        conn.rollback()
        print(f"同色替换失败: {e}")
        return None
    finally:
        conn.close()
