import os
import time
import sqlite3

# 配置路径
DB_ROOT = './data/DrawingData'
USER_DB_PATH = './data/users.db'

#限制静态资源文件夹（如 uploads/outputs）的文件数量，防止无限增长
def limit_files(folder_path, max_files=20):
    if not os.path.exists(folder_path):
        return
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    if len(files) > max_files:
        # 按修改时间排序，删除旧的
        files.sort(key=os.path.getmtime)
        for i in range(len(files) - max_files):
            try:
                os.remove(files[i])
            except OSError:
                pass

#自动清理过期文件
def run_auto_clean():
    """
    【后台定时任务】
    1. 查找所有普通用户 (Common)。
    2. 扫描其专属目录。
    3. 删除修改时间超过 5 小时的 .db 图纸及快照。
    """
    print(f"[{time.strftime('%H:%M:%S')}] 🧹 执行过期图纸清理任务...")
    
    # 1. 获取普通用户 ID 列表
    common_user_ids = []
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE user_level = 'common'")
        rows = cursor.fetchall()
        common_user_ids = [row[0] for row in rows]
        conn.close()
    except Exception as e:
        print(f"❌ [清理失败] 无法读取用户数据库: {e}")
        return

    # 2. 遍历清理
    now = time.time()
    expire_seconds = 5 * 3600  # 5小时过期
    cleaned_count = 0
    
    for user_id in common_user_ids:
        user_dir = os.path.join(DB_ROOT, f"user_{user_id}")
        if not os.path.exists(user_dir):
            continue
            
        # 遍历该用户目录下的所有文件
        for f in os.listdir(user_dir):
            # 仅处理数据库文件和快照文件
            if f.endswith('.db') or '.snap_' in f:
                f_path = os.path.join(user_dir, f)
                try:
                    mtime = os.path.getmtime(f_path)
                    # 检查是否超时
                    if now - mtime > expire_seconds:
                        os.remove(f_path)
                        cleaned_count += 1
                except Exception:
                    pass
    
    if cleaned_count > 0:
        print(f"✅ [清理完成] 共移除了 {cleaned_count} 个过期文件。")
    else:
        print("✅ [清理完成] 暂无过期文件。")