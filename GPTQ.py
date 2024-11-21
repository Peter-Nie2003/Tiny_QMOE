from transformers import AutoModelForCausalLM, AutoTokenizer,GPTQConfig
from optimum.gptq import GPTQQuantizer
import torch

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True,torch_dtype=torch.float16)

quantizer = GPTQQuantizer(bits=8,dataset="c4")

quantized_model = quantizer.quantize_model(model, tokenizer)

quantized_model.save_pretrained("llama3.2-1B-quantized")
tokenizer.save_pretrained("llama3.2-1B-quantized")
