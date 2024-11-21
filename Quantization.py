import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# Load the Llama 3.2-1B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

class Quantizer(nn.Module):
    def configure(self, bits):
        if bits == 1.5:
            self.maxq = torch.tensor(-1)  
        else:
            self.maxq = torch.tensor(2 ** int(bits) - 1)

    def find_params(self, x):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        xmin = x.min()
        xmax = x.max()

        if self.maxq < 0:  
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.unsqueeze(0)
        self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.maxq < 0: 
            return (x > self.scale / 2).float() * self.scale + (x < self.zero / 2).float() * self.zero
        q = torch.clamp(torch.round(x / self.scale) + self.zero, 0, self.maxq)
        return self.scale * (q - self.zero)

quantizer = Quantizer()
quantizer.configure(6)

def quantize_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            with torch.no_grad(): 
              quantizer.find_params(param.data)
              param.data = quantizer.quantize(param.data)

quantize_model_weights(model)

# Save the quantized model
#model.save_pretrained("quantized-llama-3.2-1B")  # Specify the path to save
#tokenizer.save_pretrained("quantized-llama-3.2-1B")


input_text = "Tonight i will go to play basketball"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=50) 

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:", generated_text)

def check_zero_ratio(model):
    zero_ratio = sum((param == 0).float().mean().item() for param in model.parameters() if param.requires_grad) / len([param for param in model.parameters() if param.requires_grad])
    print("Zero ratio in model weights:", zero_ratio)

check_zero_ratio(model)