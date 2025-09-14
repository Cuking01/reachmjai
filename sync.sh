#!/bin/bash

# ---------- 用户配置区 ----------
LOCAL_PATH="."                      # 本地要同步的文件夹
REMOTE_USER="root"                  # 远程服务器用户名
REMOTE_HOST="region-42.seetacloud.com"  # 远程服务器地址
REMOTE_PATH="/root/reachjai"        # 远程目标路径
PORT="51953"                        # SSH端口
PASSWORD="eilO1ra7I7n4"                    # 密码（明文存储，仅用于学习环境）
# -------------------------------

# 检查依赖工具
check_dependencies() {
    if ! command -v rsync &> /dev/null; then
        echo "错误：rsync 未安装，请执行 'sudo apt install rsync'"
        exit 1
    fi
    if ! command -v sshpass &> /dev/null; then
        echo "错误：sshpass 未安装，请执行 'sudo apt install sshpass'"
        exit 1
    fi
}

# 同步函数
sync_files() {
    echo -e "\n=== 开始同步 ==="
    echo "本地路径: $LOCAL_PATH"
    echo "远程目标: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH} (端口: $PORT)"

    # 使用rsync + sshpass（指定端口）
    sshpass -p "$PASSWORD" rsync -avz --progress \
        -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
        "$LOCAL_PATH/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

    if [ $? -eq 0 ]; then
        echo -e "\n[成功] 同步完成！"
    else
        echo -e "\n[失败] 同步错误，请检查："
        echo "1. 密码是否正确"
        echo "2. 服务器是否开放了端口 $PORT"
        echo "3. 运行手动测试命令："
        echo "   sshpass -p '$PASSWORD' ssh -p $PORT $REMOTE_USER@$REMOTE_HOST"
    fi
}

# 主逻辑
main() {
    check_dependencies
    while true; do
        read -p "按 Enter 开始同步，输入 Q 退出: " input
        case "$input" in
            Q|q) echo "退出脚本"; exit 0;;
            *) sync_files;;
        esac
    done
}

main
