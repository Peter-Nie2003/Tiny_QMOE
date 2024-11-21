from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
quantized_model = AutoModelForCausalLM.from_pretrained("llama3.2-1B-quantized", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("llama3.2-1B-quantized")


generator = pipeline("text-generation", model=quantized_model, tokenizer=tokenizer)


output = generator("Once upon a time, in a faraway land,", max_length=50, num_return_sequences=1)
print(output)