import pandas as pd
import sqlite3
import os
import numpy as np
from skimage import color

def migrate_and_store_lab():
    # 路径设置
    db_path = './data/Color/colors.db'

    if not os.path.exists(db_path):
        print("错误：找不到数据库文件！")
    else:
        # 1. 连接并读取现有数据
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM colors", conn)
        
        print("正在计算 Lab 空间值...")
        
        # 2. 计算 Lab 信息
        # 将 RGB 归一化到 [0, 1]
        rgb_values = df[['R', 'G', 'B']].values.astype(np.float32) / 255.0
        # 转换维度符合 skimage 要求 (N, 1, 3)
        rgb_reshaped = rgb_values.reshape(-1, 1, 3)
        # 执行转换
        lab_values = color.rgb2lab(rgb_reshaped).reshape(-1, 3)

        # 3. 将结果填回 DataFrame
        # 注意：使用 lab_l, lab_a, lab_b 避免与蓝色通道 'B' 冲突
        df['lab_l'] = lab_values[:, 0]
        df['lab_a'] = lab_values[:, 1]
        df['lab_b'] = lab_values[:, 2]

        # 4. 覆盖写入原数据库
        # 使用 if_exists='replace' 会根据现在的 df 重新定义表结构
        df.to_sql('colors', conn, if_exists='replace', index=False)

        # 5. 重新创建索引以保证性能
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_num ON colors(num)")
        
        conn.commit()
        conn.close()
        
        print("✅ 转换成功！")
        print("现在数据库包含以下列：", df.columns.tolist())
        print(df[['num', 'R', 'G', 'B', 'lab_l', 'lab_a', 'lab_b']].head())

if __name__ == "__main__":
    migrate_and_store_lab()
    db_path = './data/Color/colors.db'
    conn = sqlite3.connect(db_path)

    # 执行查询
    query = "SELECT * FROM colors LIMIT 10"
    df = pd.read_sql_query(query, conn)

    # 打印结果
    print("--- colors.db 数据预览 ---")
    print(df)

    conn.close()