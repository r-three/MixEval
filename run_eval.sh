python -m mix_eval.evaluate \
    --model_name llama_3_8b_instruct \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --batch_size 20 \
    --max_gpu_memory 5GiB \
    --output_dir results/ \
    --judge_model_id meta-llama/Llama-3.1-8B-Instruct \
    --inference_only \
    
python -m mix_eval.compute_metrics \
    --benchmark mixeval_hard \
    --version 2024-06-01 \
    --model_response_dir results/ \
    --models_to_eval llama_3_8b_instruct \
    --judge_model_id meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 1 \
    --max_gpu_memory 5GiB \


