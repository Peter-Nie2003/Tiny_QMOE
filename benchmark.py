import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
import argparse
import torch.nn as nn
import time
import pandas as pd

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
    import pickle
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
                
                compressed_weight = np.load(compressed_weight_path)
                compression_table = load_compression_table(compression_table_path)

                
                if module.bias is not None and os.path.exists(bias_path):
                    bias = np.load(bias_path)
                else:
                    bias = None

                
                compressed_linear = CompressedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    compression_table=compression_table,
                    compressed_weight=compressed_weight,
                    bias=bias,
                    sequence_length=sequence_length
                )

                parent_module = model
                sub_names = name.split('.')
                for sub_name in sub_names[:-1]:
                    parent_module = getattr(parent_module, sub_name)
                setattr(parent_module, sub_names[-1], compressed_linear)
            else:
                print(f"Compression data not found for layer: {name}")

def create_prompt(question, options, answer=None):
    """
    Create a prompt for the model given a question and options.
    If answer is provided, include it in the prompt.

    Args:
        question: The question string.
        options: A list of options.
        answer: The correct answer string (optional).

    Returns:
        prompt: The prompt string.
    """
    prompt = f"{question}\n"
    option_labels = ['A', 'B', 'C', 'D']
    for label, option in zip(option_labels, options):
        prompt += f"{label}. {option}\n"
    prompt += "Answer:"
    if answer is not None:
        prompt += f" {answer}\n\n"
    else:
        prompt += " "
    return prompt


def evaluate_mmlu(model, tokenizer, device, few_shot=False):
    model.eval()
    model.to(device)

    total_questions = 0
    total_correct = 0
    total_latency = 0

    # Get all available configs (subjects)
    configs = get_dataset_config_names('hendrycks_test')

    for subject in configs:
        print(f"\nEvaluating subject: {subject}")

        # Load the dataset for the specific subject
        dataset = load_dataset('hendrycks_test', subject, cache_dir="MMLU_data", ignore_verifications=True)
        print(f"Available splits for {subject}: {list(dataset.keys())}")

        # Load validation and test splits
        validation_dataset = dataset['validation'] if 'validation' in dataset else None
        test_dataset = dataset['test'] if 'test' in dataset else None

        if test_dataset is None:
            print(f"No 'test' split found for subject: {subject}")
            continue

        if few_shot and validation_dataset is None:
            print(f"No 'validation' split found for subject: {subject}, cannot perform few-shot evaluation.")
            continue

        if few_shot:
            validation_examples = [example for example in validation_dataset]
            if len(validation_examples) < 5:
                print(f"Warning: Validation dataset for {subject} has less than 5 examples.")
                validation_examples *= (5 // len(validation_examples)) + 1  # Replicate as needed
            import random
            random.seed(42)

        # Now process each example in the test_dataset
        for idx, example in enumerate(test_dataset):

            try:
                # For few-shot evaluation, construct the few-shot context
                if few_shot:
                    # Sample 5 examples from validation_examples
                    few_shot_examples = random.sample(validation_examples, 5)
                    # Construct the few-shot context
                    few_shot_prompt = ""
                    for few_shot_example in few_shot_examples:
                        # Get question, options, correct answer
                        fs_question = few_shot_example['question']
                        fs_options = few_shot_example['choices']
                        fs_correct_answer = few_shot_example['answer']

                        # Map the correct answer to the text of the option
                        if isinstance(fs_correct_answer, int):
                            fs_correct_answer_index = fs_correct_answer
                        else:
                            answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                            fs_correct_answer_index = answer_mapping.get(fs_correct_answer.strip(), -1)

                        # Get the correct option text
                        if 0 <= fs_correct_answer_index < len(fs_options):
                            fs_correct_option = fs_options[fs_correct_answer_index]
                        else:
                            # Skip if invalid
                            continue

                        # Create the prompt for this example, including the answer
                        fs_prompt = create_prompt(fs_question, fs_options, fs_correct_option)

                        few_shot_prompt += fs_prompt
                else:
                    few_shot_prompt = ""

                # Now construct the prompt for the test question
                question = example['question']
                options = example['choices']
                correct_answer = example['answer']

                # Adjust answer mapping if necessary
                if isinstance(correct_answer, int):
                    correct_answer_index = correct_answer
                else:
                    answer_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                    correct_answer_index = answer_mapping.get(correct_answer.strip(), -1)

                # Create the prompt for the test question (without the answer)
                test_prompt = create_prompt(question, options)

                # Combine the few-shot context and the test prompt
                prompt = few_shot_prompt + test_prompt

                # Proceed with the rest of the code: computing log-likelihoods for each option
                start_time = time.time()
                # For each option, compute the log-likelihood
                option_scores = []
                for opt_idx, option in enumerate(options):
                    # Prepare input text
                    input_text = prompt + f" {option}"

                    # Tokenize the entire input
                    inputs = tokenizer(input_text, return_tensors='pt').to(device)

                    # Identify the position of the option tokens in the input_ids
                    option_tokens = tokenizer.encode(" " + option, add_special_tokens=False)
                    option_length = len(option_tokens)
                    start_position = inputs['input_ids'].shape[1] - option_length

                    # Pass through the model
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits

                    # Get the logits for the option tokens
                    logits_option = logits[:, start_position - 1:-1, :]
                    log_probs = torch.nn.functional.log_softmax(logits_option, dim=-1)

                    # Gather the log probabilities of the target tokens
                    option_tokens_tensor = torch.tensor(option_tokens).unsqueeze(0).to(device)
                    target_log_probs = log_probs.gather(2, option_tokens_tensor.unsqueeze(-1)).squeeze(-1)

                    # Sum the log probabilities
                    total_log_prob = target_log_probs.sum().item()

                    option_scores.append(total_log_prob)

                end_time = time.time()

                # Record latency
                inference_time = end_time - start_time
                total_latency += inference_time
                # Select the option with the highest score
                predicted_index = int(np.argmax(option_scores))
                predicted_answer_index = predicted_index

                # Check if the prediction is correct
                if predicted_answer_index == correct_answer_index:
                    total_correct += 1
                total_questions += 1
                print(f"\nTotal questions: {total_questions}")
                print(f"Total correct: {total_correct}")
                print(f"Latency: {inference_time:.4f} seconds")
                if total_questions % 100 == 0:
                    print(f"Processed {total_questions} questions...")

            except Exception as e:
                print(f"An error occurred while processing example {idx + 1}: {e}")
                continue

    accuracy = total_correct / total_questions if total_questions > 0 else 0
    average_latency = total_latency / total_questions if total_questions > 0 else 0
    print(f"\nTotal questions: {total_questions}")
    print(f"Total correct: {total_correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Inference Latency: {average_latency:.4f} seconds")

    return accuracy


def evaluate_arc_challenge(model, tokenizer, device, few_shot=False):
    """
    Evaluate the LLM on the ARC-Challenge dataset.

    Args:
        model: The LLM model.
        tokenizer: Tokenizer corresponding to the LLM.
        device: Device to perform evaluation on (CPU/GPU).
        few_shot: Whether to perform few-shot evaluation.

    Returns:
        accuracy: The accuracy of the model on the ARC-Challenge dataset.
    """
    model.eval()
    model.to(device)

    # Load the ARC-Challenge dataset
    dataset = load_dataset('ai2_arc', 'ARC-Challenge', cache_dir="ARC_Challenge_data")
    test_dataset = dataset['test']

    total_questions = 0
    total_correct = 0
    total_latency = 0

    for idx, example in enumerate(test_dataset):
        try:
            question = example['question']
            options = example['choices']['text']
            correct_answer_label = example['answerKey']

            # Map the correct answer label (e.g., 'A', 'B') to the index
            label_to_index = {label: idx for idx, label in enumerate(example['choices']['label'])}
            correct_answer_index = label_to_index[correct_answer_label]

            # Construct a few-shot prompt if required
            if few_shot:
                few_shot_examples = random.sample(test_dataset, 5)
                few_shot_prompt = ""
                for few_shot_example in few_shot_examples:
                    fs_question = few_shot_example['question']
                    fs_options = few_shot_example['choices']['text']
                    fs_correct_label = few_shot_example['answerKey']
                    fs_correct_index = label_to_index[fs_correct_label]
                    fs_correct_option = fs_options[fs_correct_index]
                    fs_prompt = (
                        f"Question: {fs_question}\n"
                        + "".join([f"{label}. {opt}\n" for label, opt in zip(example['choices']['label'], fs_options)])
                        + f"Answer: {fs_correct_option}\n\n"
                    )
                    few_shot_prompt += fs_prompt
            else:
                few_shot_prompt = ""

            # Construct the test prompt
            prompt = (
                few_shot_prompt
                + f"Question: {question}\n"
                + "".join([f"{label}. {opt}\n" for label, opt in zip(example['choices']['label'], options)])
                + "Answer:"
            )

            # Tokenize and generate the model's response
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
            outputs = model.generate(
                inputs['input_ids'],
                max_length=512,
                temperature=0.7,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()

            # Decode the response and map it back to the option index
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = response.split("Answer:")[-1].strip()
            predicted_index = label_to_index.get(predicted_answer, -1)

            # Compare the predicted answer index with the correct answer index
            if predicted_index == correct_answer_index:
              total_correct += 1
            total_questions += 1

            # Record latency
            inference_time = end_time - start_time
            total_latency += inference_time

            # Log details for this inference
            current_accuracy = total_correct / total_questions if total_questions > 0 else 0
            print(f"Question {idx + 1}:")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {options[correct_answer_index]}")
            print(f"Inference Time: {inference_time:.4f} seconds")
            print(f"Total Questions: {total_questions}")
            print(f"Total Correct: {total_correct}")
            print(f"Current Accuracy: {current_accuracy * 100:.2f}%\n")

        except Exception as e:
            print(f"An error occurred at question {idx + 1}: {e}")
            continue

    # Compute final metrics
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    average_latency = total_latency / total_questions if total_questions > 0 else 0
    print(f"\nFinal Results:")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Inference Latency: {average_latency:.4f} seconds")

    return accuracy


def evaluate_arc_easy(model, tokenizer, device, few_shot=False):
    """
    Evaluate the LLM on the ARC-Challenge dataset.

    Args:
        model: The LLM model.
        tokenizer: Tokenizer corresponding to the LLM.
        device: Device to perform evaluation on (CPU/GPU).
        few_shot: Whether to perform few-shot evaluation.

    Returns:
        accuracy: The accuracy of the model on the ARC-Challenge dataset.
    """
    model.eval()
    model.to(device)

    # Load the ARC-Challenge dataset
    dataset = load_dataset('ai2_arc', 'ARC-Easy', cache_dir="ARC_Easy_data")
    test_dataset = dataset['test']

    total_questions = 0
    total_correct = 0
    total_latency = 0

    for idx, example in enumerate(test_dataset):
        try:
            question = example['question']
            options = example['choices']['text']
            correct_answer_label = example['answerKey']

            # Map the correct answer label (e.g., 'A', 'B') to the index
            label_to_index = {label: idx for idx, label in enumerate(example['choices']['label'])}
            correct_answer_index = label_to_index[correct_answer_label]

            # Construct a few-shot prompt if required
            if few_shot:
                few_shot_examples = random.sample(test_dataset, 5)
                few_shot_prompt = ""
                for few_shot_example in few_shot_examples:
                    fs_question = few_shot_example['question']
                    fs_options = few_shot_example['choices']['text']
                    fs_correct_label = few_shot_example['answerKey']
                    fs_correct_index = label_to_index[fs_correct_label]
                    fs_correct_option = fs_options[fs_correct_index]
                    fs_prompt = (
                        f"Question: {fs_question}\n"
                        + "".join([f"{label}. {opt}\n" for label, opt in zip(example['choices']['label'], fs_options)])
                        + f"Answer: {fs_correct_option}\n\n"
                    )
                    few_shot_prompt += fs_prompt
            else:
                few_shot_prompt = ""

            # Construct the test prompt
            prompt = (
                few_shot_prompt
                + f"Question: {question}\n"
                + "".join([f"{label}. {opt}\n" for label, opt in zip(example['choices']['label'], options)])
                + "Answer:"
            )

            # Tokenize and generate the model's response
            start_time = time.time()
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(device)
            outputs = model.generate(
                inputs['input_ids'],
                max_length=512,
                temperature=0.7,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()

            # Decode the response and map it back to the option index
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = response.split("Answer:")[-1].strip()
            predicted_index = label_to_index.get(predicted_answer, -1)

            # Compare the predicted answer index with the correct answer index
            if predicted_index == correct_answer_index:
              total_correct += 1
            total_questions += 1

            # Record latency
            inference_time = end_time - start_time
            total_latency += inference_time

            # Log details for this inference
            current_accuracy = total_correct / total_questions if total_questions > 0 else 0
            print(f"Question {idx + 1}:")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {options[correct_answer_index]}")
            print(f"Inference Time: {inference_time:.4f} seconds")
            print(f"Total Questions: {total_questions}")
            print(f"Total Correct: {total_correct}")
            print(f"Current Accuracy: {current_accuracy * 100:.2f}%\n")

        except Exception as e:
            print(f"An error occurred at question {idx + 1}: {e}")
            continue

    # Compute final metrics
    accuracy = total_correct / total_questions if total_questions > 0 else 0
    average_latency = total_latency / total_questions if total_questions > 0 else 0
    print(f"\nFinal Results:")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Inference Latency: {average_latency:.4f} seconds")

    return accuracy





def main():
    parser = argparse.ArgumentParser(description='Evaluate quantized and compressed models on benchmarks.')
    parser.add_argument('--model_name', type=str, default='llama3.2-1B-quantized', help='Model name or path')
    parser.add_argument('--compressed_model', action='store_true', help='Use compressed model')
    parser.add_argument('--compressed_weights', type=str, default='llama3.2-1B_Compressed', help='Directory containing compressed weights')
    parser.add_argument('--compression_tables', type=str, default='llama3.2-1B_Compressed', help='Directory containing compression tables')
    parser.add_argument('--sequence_length', type=int, default=4, help='Sequence length for compression')
    parser.add_argument('--benchmark', type=str, choices=['MMLU', 'ARC_Challenge','ARC_Easy'], default='MMLU', help='Benchmark to evaluate the model')
    parser.add_argument('--few_shot', action='store_true', help='Use few-shot evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir="./")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # Load the model
    if args.compressed_model:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        replace_linear_layers(model, args.compressed_weights, args.compression_tables, sequence_length=args.sequence_length)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", cache_dir="./")

    # Evaluate the model on the chosen benchmark
    if args.benchmark == 'MMLU':
      accuracy = evaluate_mmlu(model, tokenizer, device=device, few_shot=args.few_shot)
      print(f"MMLU Accuracy: {accuracy * 100:.2f}%")
    elif args.benchmark == 'ARC_Challenge':
      accuracy = evaluate_arc_challenge(model, tokenizer, device=device, few_shot=args.few_shot)
      print(f"ARC Challenge Accuracy: {accuracy * 100:.2f}%")
    elif args.benchmark == 'ARC_Easy':
      accuracy = evaluate_arc_easy(model, tokenizer, device=device, few_shot=args.few_shot)
      print(f"ARC Easy Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()