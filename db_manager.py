import sqlite3
import json
import numpy as np
import os

DB_DIR = './data/DrawingData'

#获取地址
def get_db_path(drawing_id):
    return os.path.join(DB_DIR, f"{drawing_id}.db")

#单格替换+更新数据库
def update_pixel_in_db(drawing_id, r, c, new_id):
    """
    单点像素修改：在事务中同步更新 grid 像素和 metadata 计数（逻辑加减）
    """
    new_id = str(new_id)
    db_path = f'./data/DrawingData/{drawing_id}.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 1. 开启事务
        cursor.execute('BEGIN TRANSACTION;')

        # 2. 获取该坐标原有的颜色 ID (用于后续统计加减)
        cursor.execute('SELECT color_id FROM grid WHERE r = ? AND c = ?', (r, c))
        result = cursor.fetchone()
        if not result:
            return None
        old_id = str(result[0])

        # 如果颜色没变，直接跳过
        if old_id == new_id:
            conn.commit()
            return None

        # 3. 更新 grid 表物理像素
        cursor.execute('UPDATE grid SET color_id = ? WHERE r = ? AND c = ?', (new_id, r, c))

        # 4. 读取 metadata 进行逻辑加减
        cursor.execute("SELECT value FROM metadata WHERE key='color_code_count'")
        meta_data = json.loads(cursor.fetchone()[0])

        # 旧颜色计数 -1
        if old_id in meta_data:
            meta_data[old_id]['count'] -= 1
            # 如果某颜色数量归零，根据业务需求决定是否保留 Key
            if meta_data[old_id]['count'] <= 0:
                # 这里选择保留 Key 以防止丢失颜色元数据，仅将 count 设为 0
                meta_data[old_id]['count'] = 0

        # 新颜色计数 +1
        if new_id in meta_data:
            meta_data[new_id]['count'] += 1
        else:
            # 如果是调色盘中新选用的颜色，初始化其统计信息
            # 注意：实际开发中应从 palette 数据库获取该 ID 的 R/G/B/Lab
            meta_data[new_id] = {
                "count": 1,
                "r_rgb": 0, "g_rgb": 0, "b_rgb": 0, # 建议此处通过其它逻辑补全颜色信息
                "l_lab": 0, "a_lab": 0, "b_lab": 0
            }

        # 5. 写回更新后的元数据
        cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(meta_data), 'color_code_count'))

        # 6. 提交事务
        conn.commit()
        return meta_data

    except Exception as e:
        conn.rollback()
        print(f"单点更新失败: {e}")
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
