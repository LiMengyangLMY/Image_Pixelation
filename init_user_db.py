import sqlite3
import os

# 路径配置
DB_DIR = './data'
USER_DB_PATH = os.path.join(DB_DIR, 'users.db')

def init_user_database():
    """一键初始化用户系统所需的所有数据表"""
    # 确保 data 目录存在
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        print(f"创建目录: {DB_DIR}")

    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()

    # 1. 创建 users 表 (基础信息 + 用户等级 + 邮箱 + 安全问题)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            user_level TEXT NOT NULL DEFAULT 'common',
            security_question TEXT,
            security_answer TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. 创建 vip_codes 表 (卡密管理)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vip_codes (
            code TEXT PRIMARY KEY,
            is_used INTEGER DEFAULT 0,
            used_by TEXT,
            valid_days INTEGER DEFAULT 31
        )
    ''')

    # 3. 创建 email_verify 表 (临时验证码存储)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_verify (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            code TEXT NOT NULL,
            expire_at DATETIME NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
    print(f"--- 用户系统数据库初始化完成 ---")
    print(f"位置: {USER_DB_PATH}")
    print(f"包含表: users, vip_codes, email_verify")


import sqlite3

DB_PATH = "./data/users.db"

def add_vip_expire_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1️⃣ 查询当前表结构
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]

    print("当前字段：", columns)

    # 2️⃣ 判断字段是否已经存在
    if "vip_expire_at" in columns:
        print("字段 vip_expire_at 已存在，无需添加。")
    else:
        print("字段不存在，开始添加...")
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN vip_expire_at INTEGER
        """)
        conn.commit()
        print("字段添加成功！")

    # 3️⃣ 再次验证
    cursor.execute("PRAGMA table_info(users)")
    new_columns = [col[1] for col in cursor.fetchall()]

    print("修改后的字段：", new_columns)

    if "vip_expire_at" in new_columns:
        print("验证成功：字段已存在。")
    else:
        print("验证失败：字段未添加成功。")

    conn.close()


if __name__ == "__main__":
    add_vip_expire_column()
