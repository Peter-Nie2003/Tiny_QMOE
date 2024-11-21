from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load the tokenizer and model
# Load model directly


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare input text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text
output = model.generate(**inputs, max_length=50)  
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print generated text
print(generated_text)
