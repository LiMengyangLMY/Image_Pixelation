import sqlite3
from datetime import datetime, timedelta
import os

# 配置用户数据库路径
USER_DB_PATH = './data/users.db'

def set_user_vip(email):
    if not os.path.exists(USER_DB_PATH):
        print(f"❌ 找不到数据库文件: {USER_DB_PATH}")
        return

    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # 设置 VIP 有效期为 365 天后
        expire_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
        
        # 执行更新操作
        cursor.execute("""
            UPDATE users 
            SET user_level = 'vip', vip_expire_at = ? 
            WHERE email = ?
        """, (expire_date, email))
        
        # 检查是否有行被更新
        if cursor.rowcount > 0:
            print(f"✅ 成功！已将用户 {email} 设置为 VIP。")
            print(f"📅 有效期至: {expire_date}")
        else:
            print(f"⚠️ 找不到邮箱为 {email} 的用户，请确认该账号已在系统中注册。")
            
        conn.commit()
    except Exception as e:
        print(f"❌ 数据库操作出错: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    target_email = "13021327429@163.com"
    print(f"正在尝试将 {target_email} 设为 VIP...")
    set_user_vip(target_email)