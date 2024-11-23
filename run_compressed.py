import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CompressedLinear(nn.Module):
    def __init__(self, in_features, out_features, compression_table, compressed_weight, bias=None, sequence_length=4):
        super(CompressedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compression_table = compression_table
        self.compressed_weight = compressed_weight
        self.sequence_length = sequence_length
        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def decompress_weights(self):
        decompression_table = {idx: seq for seq, idx in self.compression_table.items()}
        decompressed_weight = []
        compressed_data = self.compressed_weight
        i = 0
        while i < len(compressed_data):
            codeword = compressed_data[i]
            i += 1
            if codeword == 0xFFFF:
                # Read raw values
                raw_values = compressed_data[i:i + self.sequence_length].astype(np.uint8)
                decompressed_weight.extend(raw_values)
                i += self.sequence_length
            else:
                sequence = decompression_table[codeword]
                decompressed_weight.extend(sequence)
        decompressed_weight = np.array(decompressed_weight, dtype=np.uint8)
        # Convert to appropriate dtype and reshape
        weight_tensor = torch.from_numpy(decompressed_weight.astype(np.float32))
        weight_tensor = weight_tensor.view(self.out_features, self.in_features)
        return weight_tensor

    def forward(self, input):
        weight = self.decompress_weights().to(input.device)
        if self.bias is not None:
            bias = self.bias.to(input.device)
            return nn.functional.linear(input, weight, bias)
        else:
            return nn.functional.linear(input, weight)

def load_compression_table(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def replace_linear_layers(model, compressed_weights_dir, compression_tables_dir, sequence_length=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_path = name.replace('.', '_')
            compressed_weight_path = os.path.join(compressed_weights_dir, f'compressed_weights_{layer_path}.npy')
            compression_table_path = os.path.join(compression_tables_dir, f'compression_table_{layer_path}.pkl')
            bias_path = os.path.join(compressed_weights_dir, f'bias_{layer_path}.npy')

            if os.path.exists(compressed_weight_path) and os.path.exists(compression_table_path):
                # Load compressed weight and compression table
                compressed_weight = np.load(compressed_weight_path)
                compression_table = load_compression_table(compression_table_path)

                # Load bias if it exists
                if module.bias is not None and os.path.exists(bias_path):
                    bias = np.load(bias_path)
                else:
                    bias = None

                # Create a new CompressedLinear layer
                compressed_linear = CompressedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    compression_table=compression_table,
                    compressed_weight=compressed_weight,
                    bias=bias,
                    sequence_length=sequence_length
                )

                # Replace the module in the model
                parent_module = model
                sub_names = name.split('.')
                for sub_name in sub_names[:-1]:
                    parent_module = getattr(parent_module, sub_name)
                setattr(parent_module, sub_names[-1], compressed_linear)
            else:
                print(f"Compression data not found for layer: {name}")

def run_inference(model, tokenizer, input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt').to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run compressed model with text input.')
    parser.add_argument('--compressed_weights_dir', type=str, required=True, help='Directory containing compressed weights')
    parser.add_argument('--compression_tables_dir', type=str, required=True, help='Directory containing compression tables')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--input_text', type=str, required=True, help='Input text for the model')
    parser.add_argument('--output_text_file', type=str, default='output.txt', help='File to save the output text')
    args = parser.parse_args()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Replace linear layers with compressed versions
    replace_linear_layers(model, args.compressed_weights_dir, args.compression_tables_dir, sequence_length=4)

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run inference
    output_text = run_inference(model, tokenizer, args.input_text)

    # Save the output text
    with open(args.output_text_file, 'w') as f:
        f.write(output_text)

    print(f"Generated text saved to {args.output_text_file}")

    '''python your_script.py \
    --compressed_weights_dir path_to_compressed_weights \
    --compression_tables_dir path_to_compression_tables \
    --model_name_or_path your_model_name_or_path \
    --input_text "Your input text here" \
    --output_text_file output.txt'''

# import numpy as np
# import pickle
# import torch
# import torch.nn as nn
# from scipy.spatial import KDTree
# from transformers import AutoTokenizer, AutoModelForCausalLM

# def decompress_model(compressed_weights, compression_table, model_template, sequence_length=8):
#     reverse_table = {idx: seq for seq, idx in compression_table.items()}
#     decompressed_weights = []

#     for idx in compressed_weights:
#         sequence = reverse_table[idx]
#         decompressed_weights.extend(sequence)

#     decompressed_weights = np.array(decompressed_weights, dtype=np.uint8)

#     # Reconstruct the model's parameters
#     offset = 0
#     for param in model_template.parameters():
#         num_elements = param.numel()
#         param_data = decompressed_weights[offset:offset + num_elements].reshape(param.shape)
#         # Convert to the appropriate dtype (e.g., float32) from uint8
#         # Here, we'll assume the quantized weights can be scaled back to float32
#         # You may need to adjust this depending on how your quantization works
#         param.data = torch.from_numpy(param_data.astype(np.float32)).to(param.device)
#         offset += num_elements

#     return model_template

# def run_inference(model, tokenizer, input_text):
#     model.eval()
#     inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=50)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return generated_text

# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='Run compressed model with text input.')
#     parser.add_argument('--compressed_weights', type=str, required=True, help='Path to compressed weights file (.npy)')
#     parser.add_argument('--compression_table', type=str, required=True, help='Path to compression table file (.pkl)')
#     parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path')
#     parser.add_argument('--input_text', type=str, required=True, help='Input text for the model')
#     parser.add_argument('--output_text_file', type=str, default='output.txt', help='File to save the output text')
#     args = parser.parse_args()

#     # Load the tokenizer and model template
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     model_template = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

#     # Load compressed weights
#     compressed_weights = np.load(args.compressed_weights)

#     # Load compression table
#     with open(args.compression_table, 'rb') as f:
#         compression_table = pickle.load(f)

#     # Decompress and reconstruct the model
#     model = decompress_model(compressed_weights, compression_table, model_template)

#     # Move model to appropriate device (CPU or GPU)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # Run inference
#     output_text = run_inference(model, tokenizer, args.input_text)

#     # Save the output text
#     with open(args.output_text_file, 'w') as f:
#         f.write(output_text)

#     print(f"Generated text saved to {args.output_text_file}")