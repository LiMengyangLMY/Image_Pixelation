# -*- coding: utf-8 -*-
import sqlite3
import os

# 数据库路径 (根据你的项目结构)
DB_PATH = './data/users.db'

def view_users():
    print("=" * 60)
    print(f"🔍 正在读取用户数据库: {DB_PATH}")
    print("=" * 60)

    if not os.path.exists(DB_PATH):
        print(f"❌ 错误：找不到数据库文件！请先运行 init_user_db.py 初始化。")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 查询核心字段
        cursor.execute("SELECT id, username, email, user_level, password_hash, vip_expire_at FROM users")
        rows = cursor.fetchall()

        if not rows:
            print("📭 数据库是空的，没有任何用户。")
        else:
            # 打印表头
            # 使用格式化字符串对齐输出
            header = "{:<4} | {:<12} | {:<20} | {:<8} | {:<20}".format("ID", "用户名", "邮箱", "等级", "密码哈希(前10位)")
            print(header)
            print("-" * 80)

            for row in rows:
                user_id, name, email, level, pwd_hash, vip_date = row
                
                # 密码只显示前10位用于确认是否存在
                short_hash = pwd_hash[:10] + "..." if pwd_hash else "无密码"
                
                # 处理 None 值
                name = str(name)
                email = str(email)
                level = str(level)
                
                print("{:<4} | {:<12} | {:<20} | {:<8} | {:<20}".format(
                    user_id, name, email, level, short_hash
                ))
                
                if level == 'vip' and vip_date:
                    print(f"     ╚═ 👑 VIP过期时间: {vip_date}")

        print("=" * 60)
        conn.close()

    except Exception as e:
        print(f"❌ 读取失败: {e}")

if __name__ == "__main__":
    view_users()