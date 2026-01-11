#!/bin/bash
# Quick launcher script for humanizing the APR paper

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: ./humanize_my_paper.sh YOUR_API_KEY [api_provider]"
    echo ""
    echo "Examples:"
    echo "  ./humanize_my_paper.sh sk_xxx_your_key rephrasy"
    echo "  ./humanize_my_paper.sh your_key writehuman"
    echo ""
    echo "Supported providers: rephrasy (default), writehuman, undetectable, humanizerpro"
    exit 1
fi

API_KEY=$1
API_PROVIDER=${2:-rephrasy}

echo "================================================"
echo "Anti-GPTZero Paper Humanizer"
echo "================================================"
echo "Input:    lam_main_latest.tex"
echo "Output:   lam_main_latest_humanized.tex"
echo "Provider: $API_PROVIDER"
echo "================================================"
echo ""

python humanize_paper.py \
    lam_main_latest.tex \
    --api "$API_PROVIDER" \
    --api-key "$API_KEY" \
    --output lam_main_latest_humanized.tex \
    --delay 3.0 \
    --max-retries 3 \
    --verbose

echo ""
echo "Done! Check lam_main_latest_humanized.tex for the humanized paper."
echo "Report saved to: lam_main_latest_humanized_report.txt"
