#!/bin/bash
set -e  # 有錯就停止

CONTEXT_PATH=$1
TEST_PATH=$2
OUTPUT_PATH=$3

TEST_DIR=$(dirname "$TEST_PATH")
TEST_BASENAME=$(basename "$TEST_PATH" .json)
TEST_FORMAT="${TEST_DIR}/${TEST_BASENAME}_step1.json"

python step1_mc_preprocess.py \
    --context_file $CONTEXT_PATH \
    --test_file $TEST_PATH \
    --output_file $TEST_FORMAT

python mc_inference_script.py \
    --model_path ./hw1_mc_roberta_wwm_ext_e3 \
    --test_file $TEST_FORMAT \
    --output_file step1_predictions.json \
    --batch_size 16

python qa_inference_script.py \
	--model_path ./hw1_qa_model_lert_base \
	--test_file step1_predictions.json\
	--output_file $OUTPUT_PATH
