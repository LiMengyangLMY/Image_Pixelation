import os

def limit_files(folder_path, max_files=20):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    #print(f"当前文件夹 {folder_path} 共有文件: {len(files)} 个")
    if len(files) > max_files:
        # 按修改时间排序，删除最旧的
        files.sort(key=os.path.getmtime)
        for i in range(len(files) - max_files):
            os.remove(files[i])
    