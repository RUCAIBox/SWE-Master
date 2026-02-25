#!/bin/bash

# 批量LSP工具评估脚本
# 用于批量处理多个JSONL文件

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
INPUT_DIR=""
OUTPUT_DIR=""
LIMIT_LSP=2
MAX_WORKERS=5
MODEL="openai/nbgexp"
API_BASE="xx"
API_KEY="xx"
FILE_PATTERN="*.jsonl"

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项]

批量LSP工具评估脚本 - 批量处理目录中的多个JSONL文件

必需参数:
    -d, --dir DIRECTORY     输入目录路径（包含JSONL文件）

可选参数:
    -o, --output DIR        输出目录路径（默认自动生成）
    -p, --pattern PATTERN   文件匹配模式（默认: *.jsonl）
    -l, --limit-lsp NUM     LSP工具最小使用次数阈值（默认: 2）
    -w, --max-workers NUM   并行处理的最大线程数（默认: 5）
    -m, --model NAME        LLM模型名称（默认: openai/nbgexp）
    --api-base URL          API基础URL
    --api-key KEY           API密钥
    -h, --help              显示此帮助信息

示例:
    # 处理目录中所有JSONL文件
    $0 -d /path/to/input/dir

    # 指定输出目录和参数
    $0 -d /path/to/input/dir -o /path/to/output/dir -l 3 -w 10

    # 处理特定模式的文件
    $0 -d /path/to/input/dir -p "*-filtered.jsonl"

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--pattern)
            FILE_PATTERN="$2"
            shift 2
            ;;
        -l|--limit-lsp)
            LIMIT_LSP="$2"
            shift 2
            ;;
        -w|--max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        --api-base)
            API_BASE="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 必须指定输入目录 (-d)${NC}"
    show_help
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_SCRIPT="$SCRIPT_DIR/run_llm_judge.sh"

# 检查运行脚本是否存在
if [ ! -f "$RUN_SCRIPT" ]; then
    echo -e "${RED}错误: 运行脚本不存在: $RUN_SCRIPT${NC}"
    exit 1
fi

# 查找所有匹配的文件
mapfile -t FILES < <(find "$INPUT_DIR" -type f -name "$FILE_PATTERN" | sort)

if [ ${#FILES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 在目录 $INPUT_DIR 中未找到匹配 '$FILE_PATTERN' 的文件${NC}"
    exit 1
fi

# 显示配置信息
echo -e "${GREEN}==================== 批量处理配置 ====================${NC}"
echo -e "输入目录:     $INPUT_DIR"
echo -e "输出目录:     ${OUTPUT_DIR:-自动生成}"
echo -e "文件模式:     $FILE_PATTERN"
echo -e "找到文件数:   ${#FILES[@]}"
echo -e "LSP阈值:      $LIMIT_LSP"
echo -e "并行线程数:   $MAX_WORKERS"
echo -e "模型:         $MODEL"
echo -e "${GREEN}=====================================================${NC}"
echo ""

# 列出将要处理的文件
echo -e "${BLUE}将处理以下文件:${NC}"
for i in "${!FILES[@]}"; do
    echo -e "  $((i+1)). ${FILES[$i]}"
done
echo ""

# 确认执行
read -p "是否继续执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消${NC}"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)

# 处理每个文件
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=${#FILES[@]}

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILE_NUM=$((i+1))
    
    echo ""
    echo -e "${GREEN}==================== 处理文件 $FILE_NUM/$TOTAL_COUNT ====================${NC}"
    echo -e "文件: $FILE"
    echo ""
    
    # 构建命令
    CMD="bash \"$RUN_SCRIPT\" -i \"$FILE\" --limit-lsp $LIMIT_LSP --max-workers $MAX_WORKERS --model \"$MODEL\" --api-base \"$API_BASE\" --api-key \"$API_KEY\""
    
    if [ -n "$OUTPUT_DIR" ]; then
        FILENAME=$(basename "$FILE")
        OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
        CMD="$CMD -o \"$OUTPUT_FILE\""
    fi
    
    # 执行命令
    if eval $CMD; then
        echo -e "${GREEN}✓ 文件 $FILE_NUM/$TOTAL_COUNT 处理成功${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ 文件 $FILE_NUM/$TOTAL_COUNT 处理失败${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# 计算执行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# 显示总结
echo ""
echo -e "${GREEN}==================== 处理完成 ====================${NC}"
echo -e "总文件数:     $TOTAL_COUNT"
echo -e "成功:         ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "失败:         ${RED}$FAIL_COUNT${NC}"
echo -e "总耗时:       ${HOURS}时${MINUTES}分${SECONDS}秒"
echo -e "${GREEN}================================================${NC}"

# 退出码
if [ $FAIL_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi