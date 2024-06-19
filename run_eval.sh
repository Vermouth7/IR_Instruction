export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


# GPU_NUM=4
# MODEL_PATH="/data1/chh/models/yutaozhu94/INTERS-Falcon-1b"
# TOKENIZER_PATH="/data1/chh/models/yutaozhu94/INTERS-Falcon-1b"
RESULT_PATH="./results"
EVAL_DATA_PATH="./dataset_curation"


# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pointwise

# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pairwise

# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method listwise


# MODEL_PATH="/data1/chh/models/yutaozhu94/INTERS-LLaMA-7b-chat"
# TOKENIZER_PATH="/data1/chh/models/yutaozhu94/INTERS-LLaMA-7b-chat"


# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pointwise

# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pairwise

# torchrun --nproc_per_node 8 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method listwise



# MODEL_PATH="/data1/chh/models/yutaozhu94/INTERS-Mistral-7b"
# TOKENIZER_PATH="/data1/chh/models/yutaozhu94/INTERS-Mistral-7b"


# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pointwise

# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pairwise

# torchrun --nproc_per_node 8 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method listwise




MODEL_PATH="/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct"
TOKENIZER_PATH="/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct"

# torchrun --nproc_per_node 1 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pointwise

# torchrun --nproc_per_node 2 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 1024 \
#   --batch_size 1 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pairwise

# torchrun --nproc_per_node 8 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method listwise


MODEL_PATH="/data1/chh/models/mistralai/Mistral-7B-Instruct-v0.2"
TOKENIZER_PATH="/data1/chh/models/mistralai/Mistral-7B-Instruct-v0.2"


torchrun --nproc_per_node 1 rerank.py \
  --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
  --output_dir ${RESULT_PATH} \
  --model_name_or_path ${MODEL_PATH} \
  --tokenizer_name_or_path ${TOKENIZER_PATH} \
  --dataset_cache_dir hf_cache/dataset/ \
  --use_flash_attention_2 False \
  --max_length 2048 \
  --batch_size 4 \
  --with_description True \
  --dataset_name my_dataset \
  --rerank_method pointwise

# torchrun --nproc_per_node 4 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method pairwise

# torchrun --nproc_per_node 8 rerank.py \
#   --eval_data ${EVAL_DATA_PATH}/my_dataset.jsonl \
#   --output_dir ${RESULT_PATH} \
#   --model_name_or_path ${MODEL_PATH} \
#   --tokenizer_name_or_path ${TOKENIZER_PATH} \
#   --dataset_cache_dir hf_cache/dataset/ \
#   --use_flash_attention_2 False \
#   --max_length 2048 \
#   --batch_size 4 \
#   --with_description True \
#   --dataset_name my_dataset \
#   --rerank_method listwise