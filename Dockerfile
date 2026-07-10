FROM python:3.11-slim

# 系统依赖(OpenCV 需要 libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 装 Python 依赖(利用 Docker 缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷代码 + 模型 + 文档
COPY . .

# 数据 / 日志目录挂 volume
VOLUME ["/app/model", "/app/data", "/app/logs"]

EXPOSE 8080

CMD ["python", "app.py"]
