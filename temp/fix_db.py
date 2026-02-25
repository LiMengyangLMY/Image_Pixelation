import sqlite3
import os

# ——————————————————————————————————————————————
# 核心修复：自动获取 data/users.db 的绝对路径
# ——————————————————————————————————————————————
# 1. 获取当前脚本 (temp/fix_db.py) 的绝对路径
current_script_path = os.path.abspath(__file__)
# 2. 获取脚本所在的目录 (temp/)
script_dir = os.path.dirname(current_script_path)
# 3. 获取项目根目录 (即 temp 的上一级: Image_Pixelation/)
project_root = os.path.dirname(script_dir)
# 4. 拼接出数据库的准确路径
DB_PATH = os.path.join(project_root, 'data', 'users.db')

def fix_database():
    print(f"🔍 正在寻找数据库...")
    print(f"   -> 目标路径: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        print(f"❌ 错误：依然找不到数据库文件！")
        print(f"   请确认你的 users.db 是否真的在 data 文件夹里？")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 检查 email_verify 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='email_verify'")
    if cursor.fetchone():
        print("✅ 检测通过：email_verify 表已存在，无需修复。")
    else:
        print("⚠️ 检测到表缺失，正在创建 email_verify 表...")
        cursor.execute('''
            CREATE TABLE email_verify (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                code TEXT NOT NULL,
                expire_at DATETIME NOT NULL
            )
        ''')
        conn.commit()
        print("✅ 修复成功：表已创建完毕！")

    conn.close()

if __name__ == "__main__":
    fix_database()