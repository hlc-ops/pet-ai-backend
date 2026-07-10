#!/bin/bash
# 一键冒烟测试

BASE=${BASE:-http://localhost:8080}

echo "===== 1. 健康检查 ====="
curl -s $BASE/api/health | python -m json.tool

echo ""
echo "===== 2. 查看配置 ====="
curl -s $BASE/api/config | python -m json.tool

echo ""
echo "===== 3. 上传视频 ====="
if [ -z "$1" ]; then
    echo "用法: ./test_upload.sh <video_file>"
    exit 1
fi

curl -F "video=@$1" \
     -F "cameraId=c002" \
     -F "petId=p002" \
     $BASE/api/kennels/A02/stream-video | python -m json.tool

echo ""
echo "===== 4. 查看日志(实时) ====="
echo "在另一个终端跑:"
echo "  tail -f logs/ai-backend.log"
