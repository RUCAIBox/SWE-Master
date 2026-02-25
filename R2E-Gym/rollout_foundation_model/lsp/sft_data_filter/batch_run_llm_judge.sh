#!/bin/bash

# Batch LSP Tool Evaluation Script
# Used for batch processing multiple JSONL files

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default parameters
INPUT_DIR=""
OUTPUT_DIR=""
LIMIT_LSP=2
MAX_WORKERS=5
MODEL="openai/nbgexp"
API_BASE="xx"
API_KEY="xx"
FILE_PATTERN="*.jsonl"

# Display help information
show_help() {
    cat << EOF
Usage: $0 [options]

Batch LSP Tool Evaluation Script - Process multiple JSONL files in a directory

Required Arguments:
    -d, --dir DIRECTORY     Path to input directory (containing JSONL files)

Optional Arguments:
    -o, --output DIR        Path to output directory (default: auto-generated)
    -p, --pattern PATTERN   File matching pattern (default: *.jsonl)
    -l, --limit-lsp NUM     Threshold for minimum LSP tool usage count (default: 2)
    -w, --max-workers NUM   Maximum number of threads for parallel processing (default: 5)
    -m, --model NAME        LLM model name (default: openai/nbgexp)
    --api-base URL          API base URL
    --api-key KEY           API key
    -h, --help              Display this help message

Examples:
    # Process all JSONL files in a directory
    $0 -d /path/to/input/dir

    # Specify output directory and parameters
    $0 -d /path/to/input/dir -o /path/to/output/dir -l 3 -w 10

    # Process files with a specific pattern
    $0 -d /path/to/input/dir -p "*-filtered.jsonl"

EOF
}

# Parse command line arguments
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
            echo -e "${RED}Error: Unknown parameter '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory must be specified (-d)${NC}"
    show_help
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_SCRIPT="$SCRIPT_DIR/run_llm_judge.sh"

# Check if the run script exists
if [ ! -f "$RUN_SCRIPT" ]; then
    echo -e "${RED}Error: Run script not found: $RUN_SCRIPT${NC}"
    exit 1
fi

# Find all matching files
mapfile -t FILES < <(find "$INPUT_DIR" -type f -name "$FILE_PATTERN" | sort)

if [ ${#FILES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No files matching '$FILE_PATTERN' found in directory $INPUT_DIR${NC}"
    exit 1
fi

# Display configuration information
echo -e "${GREEN}==================== Batch Processing Configuration ====================${NC}"
echo -e "Input Directory:  $INPUT_DIR"
echo -e "Output Directory: ${OUTPUT_DIR:-Auto-generated}"
echo -e "File Pattern:     $FILE_PATTERN"
echo -e "Files Found:      ${#FILES[@]}"
echo -e "LSP Threshold:    $LIMIT_LSP"
echo -e "Max Workers:      $MAX_WORKERS"
echo -e "Model:            $MODEL"
echo -e "${GREEN}========================================================================${NC}"
echo ""

# List files to be processed
echo -e "${BLUE}The following files will be processed:${NC}"
for i in "${!FILES[@]}"; do
    echo -e "  $((i+1)). ${FILES[$i]}"
done
echo ""

# Confirm execution
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cancelled${NC}"
    exit 0
fi

# Record start time
START_TIME=$(date +%s)

# Process each file
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=${#FILES[@]}

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILE_NUM=$((i+1))
    
    echo ""
    echo -e "${GREEN}==================== Processing File $FILE_NUM/$TOTAL_COUNT ====================${NC}"
    echo -e "File: $FILE"
    echo ""
    
    # Build command
    CMD="bash \"$RUN_SCRIPT\" -i \"$FILE\" --limit-lsp $LIMIT_LSP --max-workers $MAX_WORKERS --model \"$MODEL\" --api-base \"$API_BASE\" --api-key \"$API_KEY\""
    
    if [ -n "$OUTPUT_DIR" ]; then
        FILENAME=$(basename "$FILE")
        OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
        CMD="$CMD -o \"$OUTPUT_FILE\""
    fi
    
    # Execute command
    if eval $CMD; then
        echo -e "${GREEN}✓ File $FILE_NUM/$TOTAL_COUNT processed successfully${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ File $FILE_NUM/$TOTAL_COUNT processing failed${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# Calculate execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Display summary
echo ""
echo -e "${GREEN}==================== Processing Completed ====================${NC}"
echo -e "Total Files:     $TOTAL_COUNT"
echo -e "Success:         ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failure:         ${RED}$FAIL_COUNT${NC}"
echo -e "Total Duration:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo -e "${GREEN}=============================================================${NC}"

# Exit code
if [ $FAIL_COUNT -eq 0 ]; then
    exit 0
else
    exit 1
fi