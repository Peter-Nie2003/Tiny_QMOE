import numpy as np
from collections import Counter
import pickle
from scipy.spatial import KDTree
import torch

def find_frequent_sequences(quantized_model, sequence_length=8, top_k=2**16):
    sequence_counts = Counter()
    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy().astype(np.uint8)
        sequences = [
            tuple(weights[i:i + sequence_length]) 
            for i in range(len(weights) - sequence_length + 1)
        ]
        sequence_counts.update(sequences)
    most_frequent = sequence_counts.most_common(top_k)
    compression_table = {seq: idx for idx, (seq, _) in enumerate(most_frequent)}
    return compression_table

def build_kdtree(compression_table):
    sequences = np.array(list(compression_table.keys()), dtype=np.uint8)
    tree = KDTree(sequences)
    return tree

def compress_model(quantized_model, compression_table, tree, sequence_length=8):
    compressed_weights = []
    for param in quantized_model.parameters():
        weights = param.flatten().detach().cpu().numpy().astype(np.uint8)
        i = 0
        while i <= len(weights) - sequence_length:
            sequence = weights[i:i + sequence_length]
            sequence_tuple = tuple(sequence)
            if sequence_tuple in compression_table:
                compressed_weights.append(compression_table[sequence_tuple])
            else:
                # Use KD-Tree to find the nearest sequence
                distance, idx = tree.query(sequence)
                nearest_sequence = tree.data[idx]
                compressed_weights.append(compression_table[tuple(nearest_sequence)])
            i += sequence_length
        # Handle remaining weights if necessary
        if i < len(weights):
            remaining_sequence = weights[i:]
            padded_sequence = np.pad(remaining_sequence, (0, sequence_length - len(remaining_sequence)), 'constant')
            sequence_tuple = tuple(padded_sequence)
            if sequence_tuple in compression_table:
                compressed_weights.append(compression_table[sequence_tuple])
            else:
                distance, idx = tree.query(padded_sequence)
                nearest_sequence = tree.data[idx]
                compressed_weights.append(compression_table[tuple(nearest_sequence)])
    return np.array(compressed_weights, dtype=np.uint16)

def decompress_model(compressed_weights, compression_table, sequence_length=8):
    reverse_table = {idx: seq for seq, idx in compression_table.items()}
    decompressed_weights = []
    for idx in compressed_weights:
        sequence = reverse_table[idx]
        decompressed_weights.extend(sequence)
    return np.array(decompressed_weights, dtype=np.uint8)

def reconstruct_model(decompressed_weights, model_template):
    offset = 0
    for param in model_template.parameters():
        num_elements = param.numel()
        param.data = torch.tensor(
            decompressed_weights[offset:offset + num_elements].reshape(param.shape), dtype=torch.uint8
        ).to(param.device)
        offset += num_elements
    return model_template

def main():
    # Define or load your model here
    model = ...  # Replace with your model definition or loading code

    # Step 1: Quantize the model
    quantizer = GPTQQuantizer(bits=8)
    quantized_model = quantizer.quantize_model(model)
    
    # Step 2: Find most frequent sequences
    compression_table = find_frequent_sequences(quantized_model)
    
    # Build KDTree for nearest neighbor search
    tree = build_kdtree(compression_table)
    
    # Step 3: Compress the model
    compressed_weights = compress_model(quantized_model, compression_table, tree)
    
    # Save compressed weights and table
    np.save('compressed_weights.npy', compressed_weights)
    with open('compression_table.pkl', 'wb') as f:
        pickle.dump(compression_table, f)
    
    print("Model compressed and saved.")
    
    # To decompress:
    decompressed_weights = decompress_model(compressed_weights, compression_table)
    
    # Reconstruct the quantized model's parameters
    quantized_model = reconstruct_model(decompressed_weights, quantized_model)
    
    print("Decompression successful.")

if __name__ == '__main__':
    main()