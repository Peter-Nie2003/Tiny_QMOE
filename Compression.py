import numpy as np
from collections import Counter
import pickle
from scipy.spatial import KDTree
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_frequent_sequences(quantized_model, sequence_length=8, top_k=2**16):
    sequence_counts = Counter()
    max_unique_sequences = top_k * 10  # Adjust as needed based on available memory

    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy()  # Keep original dtype
        sequences = (
            tuple(weights[i:i + sequence_length])
            for i in range(len(weights) - sequence_length + 1)
        )
        for seq in sequences:
            sequence_counts[seq] += 1
            if len(sequence_counts) > max_unique_sequences:
                sequence_counts = Counter(dict(sequence_counts.most_common(top_k)))
    most_frequent = sequence_counts.most_common(top_k)
    compression_table = {seq: idx for idx, (seq, _) in enumerate(most_frequent)}
    return compression_table

def build_kdtree(compression_table):
    sequences = np.array(list(compression_table.keys()), dtype=np.uint8)
    tree = KDTree(sequences)
    return tree

def compress_model(quantized_model, compression_table, tree, sequence_length=8):
    compressed_files = []
    param_index = 0
    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy()  # Keep original dtype
        weights_length = len(weights)
        compressed_param = []
        i = 0
        while i <= weights_length - sequence_length:
            sequence = tuple(weights[i:i + sequence_length])
            if sequence in compression_table:
                compressed_param.append(compression_table[sequence])
            else:
                distance, idx = tree.query([sequence])
                nearest_sequence = tuple(tree.data[idx[0]])
                compressed_param.append(compression_table[nearest_sequence])
            i += sequence_length
        # Handle remaining weights
        if i < weights_length:
            remaining_sequence = weights[i:]
            padded_sequence = np.pad(remaining_sequence, (0, sequence_length - len(remaining_sequence)), 'constant')
            sequence = tuple(padded_sequence)
            if sequence in compression_table:
                compressed_param.append(compression_table[sequence])
            else:
                distance, idx = tree.query([sequence])
                nearest_sequence = tuple(tree.data[idx[0]])
                compressed_param.append(compression_table[nearest_sequence])
        compressed_param = np.array(compressed_param, dtype=np.uint16)
        filename = f'compressed_weights_param_{param_index}.npy'
        np.save(filename, compressed_param)
        compressed_files.append(filename)
        param_index += 1
    return compressed_files

def decompress_model_from_files(compressed_files, compression_table, sequence_length=8):
    reverse_table = {idx: seq for seq, idx in compression_table.items()}
    decompressed_weights = []

    for filename in compressed_files:
        compressed_weights = np.load(filename)
        for idx in compressed_weights:
            sequence = reverse_table[idx]
            decompressed_weights.extend(sequence)
    return np.array(decompressed_weights)  # Keep original dtype

def reconstruct_model(decompressed_weights, model_template):
    offset = 0
    for param in model_template.parameters():
        num_elements = param.numel()
        param_data = decompressed_weights[offset:offset + num_elements].reshape(param.shape)
        param.data = torch.tensor(param_data, dtype=param.dtype).to(param.device)
        offset += num_elements
    return model_template

def decompress_model_from_files(compressed_files, compression_table, sequence_length=8):
    reverse_table = {idx: seq for seq, idx in compression_table.items()}
    decompressed_weights = []

    for filename in compressed_files:
        compressed_weights = np.load(filename)
        for idx in compressed_weights:
            sequence = reverse_table[idx]
            decompressed_weights.extend(sequence)
    return np.array(decompressed_weights, dtype=np.uint8)

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
        
        # Build KDTree for nearest neighbor search
        sequences = np.array(list(compression_table.keys()))
        tree = KDTree(sequences)

        # Step 3: Compress the model
        compressed_files = compress_model(quantized_model, compression_table, tree)

        # Save the compression table
        with open('compression_table.pkl', 'wb') as f:
            pickle.dump(compression_table, f)
        
        print("Model compressed and saved.")
    else:
        # Load the compression table
        with open('compression_table.pkl', 'rb') as f:
            compression_table = pickle.load(f)
        print("Loaded existing compressed files and compression table.")

    # Decompress and reconstruct the model
    decompressed_weights = decompress_model_from_files(compressed_files, compression_table)
    quantized_model = reconstruct_model(decompressed_weights, quantized_model)

    print("Decompression successful.")

if __name__ == '__main__':
    main()