from vllm import LLM, SamplingParams
import torch.distributed as dist
import torch
import multiprocessing as mp
from typing import List, Dict

def run_inference(model_name: str, messages: List[Dict], queue: mp.Queue):
    """Run inference in a separate process"""
    try:
        llm = LLM(
            model=model_name,
            tokenizer_mode="auto",
            enable_prefix_caching=True,
            tensor_parallel_size=4
        )
        
        outputs = llm.chat(messages, sampling_params=SamplingParams(temperature=0.5))
        queue.put(('success', outputs[0].outputs[0].text))
        
    except Exception as e:
        queue.put(('error', str(e)))
    finally:
        if 'llm' in locals():
            if hasattr(llm, 'engine'):
                del llm.engine
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.cuda.empty_cache()

def main():
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    
    # First inference
    print("\nStarting first inference...")
    q1 = mp.Queue()
    p1 = mp.Process(target=run_inference, args=(model_name, messages, q1))
    p1.start()
    status1, result1 = q1.get()
    p1.join()
    
    if status1 == 'error':
        print(f"First inference failed: {result1}")
        return
        
    print(f"First result: {result1}")
    
    # Second inference
    print("\nStarting second inference...")
    q2 = mp.Queue()
    p2 = mp.Process(target=run_inference, args=(model_name, messages, q2))
    p2.start()
    status2, result2 = q2.get()
    p2.join()
    
    if status2 == 'error':
        print(f"Second inference failed: {result2}")
        return
        
    print(f"Second result: {result2}")

if __name__ == "__main__":
    main()