#!/bin/bash

# LSP Tool Evaluation Script
# Uses LLM to evaluate the effectiveness of LSP tools in solving code-related issues.

set -e  # Exit immediately if a command exits with a non-zero status

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default parameters
INPUT_FILE="./R2E-Gym/results/ood/glm-rollout-r2e-gym_06_1_2200_128k/glm-ood-lsp-r2e-gym_06_1_2200_128k.jsonl"
OUTPUT_FILE=""
LIMIT_LSP=2
MAX_WORKERS=10
MODEL="openai/nbgexp"
# MODEL="openai/deepseek-reasoner"
API_BASE="xx"
API_KEY="xx"


# Display help information
show_help() {
    cat << EOF
Usage: $0 [options]

LSP Tool Evaluation Script - Evaluates the effectiveness of LSP tools in solving code-related issues using LLMs.

Required Arguments:
    -i, --input FILE        Path to the input JSONL file

Optional Arguments:
    -o, --output FILE       Path to the output JSONL file (default: auto-generated)
    -l, --limit-lsp NUM     Threshold for minimum LSP tool usage count (default: 2)
    -w, --max-workers NUM   Maximum number of threads for parallel processing (default: 5)
    -m, --model NAME        LLM model name (default: openai/nbgexp)
    --api-base URL          API base URL
    --api-key KEY           API key
    -h, --help              Display this help message

Examples:
    # Basic usage
    $0 -i /path/to/input.jsonl

    # Specify output path and parameters
    $0 -i /path/to/input.jsonl -o /path/to/output.jsonl -l 3 -w 10

    # Use a different model
    $0 -i /path/to/input.jsonl -m openai/deepseek-reasoner --api-base https://api.deepseek.com/v1

Environment Variables:
    LLM_API_KEY             API key can be set via this environment variable
    LLM_API_BASE            API base URL can be set via this environment variable

EOF
}

# Parse command line arguments
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
            echo -e "${RED}Error: Unknown parameter '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check environment variables
if [ -n "$LLM_API_KEY" ]; then
    API_KEY="$LLM_API_KEY"
fi

if [ -n "$LLM_API_BASE" ]; then
    API_BASE="$LLM_API_BASE"
fi

# Validate required arguments
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file must be specified (-i)${NC}"
    show_help
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file does not exist: $INPUT_FILE${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 command not found${NC}"
    exit 1
fi

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/filter_sft_ood_data.py"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Build command
CMD="python3 \"$PYTHON_SCRIPT\" -i \"$INPUT_FILE\" --limit-lsp $LIMIT_LSP --max-workers $MAX_WORKERS --model \"$MODEL\" --api-base \"$API_BASE\" --api-key \"$API_KEY\""

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD -o \"$OUTPUT_FILE\""
fi

# Display configuration information
echo -e "${GREEN}==================== Configuration ====================${NC}"
echo -e "Input File:      $INPUT_FILE"
echo -e "Output File:     ${OUTPUT_FILE:-Auto-generated}"
echo -e "LSP Threshold:   $LIMIT_LSP"
echo -e "Max Workers:     $MAX_WORKERS"
echo -e "Model:           $MODEL"
echo -e "API Base:        $API_BASE"
echo -e "${GREEN}======================================================${NC}"
echo ""

# Confirm execution
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cancelled${NC}"
    exit 0
fi

# Execute Python script
echo -e "${GREEN}Starting execution...${NC}"
eval $CMD

# Check execution results
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Execution successful!${NC}"
else
    echo -e "${RED}Execution failed!${NC}"
    exit 1
fi