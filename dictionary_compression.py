import numpy as np
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

def find_frequent_sequences(quantized_model, sequence_length=4, top_k=2**16 - 1):
    sequence_counts = Counter()
    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy().astype(np.uint8)
        sequences = (
            tuple(weights[i:i + sequence_length])
            for i in range(len(weights) - sequence_length + 1)
        )
        sequence_counts.update(sequences)
    most_frequent = sequence_counts.most_common(top_k)
    compression_table = {seq: idx + 1 for idx, (seq, _) in enumerate(most_frequent)}
    # Reserve codeword 0xFFFF (65535) for special purposes
    return compression_table

def compress_model(quantized_model, compression_table, sequence_length=4):
    compressed_files = []
    param_index = 0
    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy().astype(np.uint8)
        weights_length = len(weights)
        compressed_param = []
        i = 0
        while i <= weights_length - sequence_length:
            sequence = tuple(weights[i:i + sequence_length])
            if sequence in compression_table:
                compressed_param.append(compression_table[sequence])
                i += sequence_length
            else:
                # Store raw values with special codeword
                compressed_param.append(0xFFFF)
                compressed_param.extend(sequence)
                i += sequence_length
        # Handle remaining weights
        remaining_weights = weights[i:]
        if remaining_weights.size > 0:
            compressed_param.append(0xFFFF)
            compressed_param.extend(remaining_weights)
        compressed_param = np.array(compressed_param, dtype=np.uint16)
        filename = f'compressed_weights_param_{param_index}.npy'
        np.save(filename, compressed_param)
        compressed_files.append(filename)
        param_index += 1
    return compressed_files

def main():
    import os

    # Path to your quantized model
    model_path = "llama3.2-1B-quantized"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the quantized model directly
    quantized_model = AutoModelForCausalLM.from_pretrained(model_path)

    # Check if compressed files exist
    compressed_files = [f for f in os.listdir() if f.startswith('compressed_weights_param_') and f.endswith('.npy')]
    if not compressed_files or not os.path.exists('compression_table.pkl'):
        # Step 2: Find most frequent sequences
        compression_table = find_frequent_sequences(quantized_model)
        
        # Step 3: Compress the model
        compressed_files = compress_model(quantized_model, compression_table)
        
        with open('compression_table.pkl', 'wb') as f:
            pickle.dump(compression_table, f)
        
        print("Model compressed and saved.")
    else:
        with open('compression_table.pkl', 'rb') as f:
            compression_table = pickle.load(f)
        print("Loaded existing compressed files and compression table.")

if __name__ == '__main__':
    main()