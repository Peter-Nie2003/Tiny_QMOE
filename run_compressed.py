import numpy as np
import pickle
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from transformers import AutoTokenizer, AutoModelForCausalLM

def decompress_model(compressed_weights, compression_table, model_template, sequence_length=8):
    reverse_table = {idx: seq for seq, idx in compression_table.items()}
    decompressed_weights = []

    for idx in compressed_weights:
        sequence = reverse_table[idx]
        decompressed_weights.extend(sequence)

    decompressed_weights = np.array(decompressed_weights, dtype=np.uint8)

    # Reconstruct the model's parameters
    offset = 0
    for param in model_template.parameters():
        num_elements = param.numel()
        param_data = decompressed_weights[offset:offset + num_elements].reshape(param.shape)
        # Convert to the appropriate dtype (e.g., float32) from uint8
        # Here, we'll assume the quantized weights can be scaled back to float32
        # You may need to adjust this depending on how your quantization works
        param.data = torch.from_numpy(param_data.astype(np.float32)).to(param.device)
        offset += num_elements

    return model_template

def run_inference(model, tokenizer, input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run compressed model with text input.')
    parser.add_argument('--compressed_weights', type=str, required=True, help='Path to compressed weights file (.npy)')
    parser.add_argument('--compression_table', type=str, required=True, help='Path to compression table file (.pkl)')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Pretrained model name or path')
    parser.add_argument('--input_text', type=str, required=True, help='Input text for the model')
    parser.add_argument('--output_text_file', type=str, default='output.txt', help='File to save the output text')
    args = parser.parse_args()

    # Load the tokenizer and model template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_template = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load compressed weights
    compressed_weights = np.load(args.compressed_weights)

    # Load compression table
    with open(args.compression_table, 'rb') as f:
        compression_table = pickle.load(f)

    # Decompress and reconstruct the model
    model = decompress_model(compressed_weights, compression_table, model_template)

    # Move model to appropriate device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run inference
    output_text = run_inference(model, tokenizer, args.input_text)

    # Save the output text
    with open(args.output_text_file, 'w') as f:
        f.write(output_text)

    print(f"Generated text saved to {args.output_text_file}")