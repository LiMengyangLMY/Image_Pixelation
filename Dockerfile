# 1. 使用官方 Python 3.9 轻量级镜像作为基础
FROM python:3.9-slim

# 2. 设置容器内的工作目录为 /app
WORKDIR /app

# 3. 替换为阿里云源以加速下载，并安装 OpenCV/skimage 所需的 C++ 底层依赖与字体 (极其关键的一步)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 4. 复制本地的 requirements.txt 到容器里
COPY requirements.txt .

# 5. 使用阿里云镜像源，高速安装你刚才导出的那一长串 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 6. 将当前目录下的所有代码和静态文件，全部复制到容器内
COPY . .

# 7. 告诉服务器，我们的程序要用 5000 端口
EXPOSE 5000

# 8. 声明挂载点：保护你的 users.db 和 DrawingData 用户专属图纸目录不丢失
VOLUME ["/app/data"]

# 9. 启动命令：用 Gunicorn 替代 Flask 测试服务器，并开启 --preload 防止后台清理任务重复执行
CMD ["gunicorn", "--workers", "3", "--preload", "--bind", "0.0.0.0:5000", "app:app"]