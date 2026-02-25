#!/bin/bash

# LSP工具评估脚本
# 使用LLM评估LSP工具在代码问题解决中的效果

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
INPUT_FILE="./R2E-Gym/results/ood/glm-rollout-r2e-gym_06_1_2200_128k/glm-ood-lsp-r2e-gym_06_1_2200_128k.jsonl"
OUTPUT_FILE=""
LIMIT_LSP=2
MAX_WORKERS=10
MODEL="openai/nbgexp"
# MODEL="openai/deepseek-reasoner"
API_BASE="xx"
API_KEY="xx"


# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项]

LSP工具评估脚本 - 使用LLM评估LSP工具在代码问题解决中的效果

必需参数:
    -i, --input FILE        输入JSONL文件路径

可选参数:
    -o, --output FILE       输出JSONL文件路径（默认自动生成）
    -l, --limit-lsp NUM     LSP工具最小使用次数阈值（默认: 2）
    -w, --max-workers NUM   并行处理的最大线程数（默认: 5）
    -m, --model NAME        LLM模型名称（默认: openai/nbgexp）
    --api-base URL          API基础URL
    --api-key KEY           API密钥
    -h, --help              显示此帮助信息

示例:
    # 基本使用
    $0 -i /path/to/input.jsonl

    # 指定输出路径和参数
    $0 -i /path/to/input.jsonl -o /path/to/output.jsonl -l 3 -w 10

    # 使用不同的模型
    $0 -i /path/to/input.jsonl -m openai/deepseek-reasoner --api-base https://api.deepseek.com/v1

环境变量:
    LLM_API_KEY             可以通过环境变量设置API密钥
    LLM_API_BASE            可以通过环境变量设置API基础URL

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
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

# 检查环境变量
if [ -n "$LLM_API_KEY" ]; then
    API_KEY="$LLM_API_KEY"
fi

if [ -n "$LLM_API_BASE" ]; then
    API_BASE="$LLM_API_BASE"
fi

# 验证必需参数
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}错误: 必须指定输入文件 (-i)${NC}"
    show_help
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}错误: 输入文件不存在: $INPUT_FILE${NC}"
    exit 1
fi

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到python3命令${NC}"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/filter_sft_ood_data.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: Python脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# 构建命令
CMD="python3 \"$PYTHON_SCRIPT\" -i \"$INPUT_FILE\" --limit-lsp $LIMIT_LSP --max-workers $MAX_WORKERS --model \"$MODEL\" --api-base \"$API_BASE\" --api-key \"$API_KEY\""

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD -o \"$OUTPUT_FILE\""
fi

# 显示配置信息
echo -e "${GREEN}==================== 配置信息 ====================${NC}"
echo -e "输入文件:     $INPUT_FILE"
echo -e "输出文件:     ${OUTPUT_FILE:-自动生成}"
echo -e "LSP阈值:      $LIMIT_LSP"
echo -e "并行线程数:   $MAX_WORKERS"
echo -e "模型:         $MODEL"
echo -e "API Base:     $API_BASE"
echo -e "${GREEN}================================================${NC}"
echo ""

# 确认执行
read -p "是否继续执行? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}已取消${NC}"
    exit 0
fi

# 执行Python脚本
echo -e "${GREEN}开始执行...${NC}"
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo -e "${GREEN}执行成功!${NC}"
else
    echo -e "${RED}执行失败!${NC}"
    exit 1
fi