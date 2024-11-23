import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import sys

#python benchmark.py --model_type original --model_path path_to_original_model --input_text "Your input text here"

def run_original_model(model_path, input_text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    num_tokens_output = len(tokenizer.tokenize(generated_text))
    inference_time = end_time - start_time

    return generated_text, num_tokens_output, inference_time

def run_quantized_model(model_path, input_text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Ensure the model is quantized (if using a specific quantization method, adjust accordingly)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    num_tokens_output = len(tokenizer.tokenize(generated_text))
    inference_time = end_time - start_time

    return generated_text, num_tokens_output, inference_time

def run_compressed_model(model_path, input_text):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load compressed weights
    compressed_weights = np.load('compressed_weights.npy')

    # Load compression table
    with open('compression_table.pkl', 'rb') as f:
        compression_table = pickle.load(f)

    # Load the model template
    model_template = AutoModelForCausalLM.from_pretrained(model_path)

    # Decompress and reconstruct the model
    model = decompress_model(compressed_weights, compression_table, model_template)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)
    end_time = time.time()

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    num_tokens_output = len(tokenizer.tokenize(generated_text))
    inference_time = end_time - start_time

    return generated_text, num_tokens_output, inference_time

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
        param.data = torch.from_numpy(param_data.astype(param.dtype)).to(param.device)
        offset += num_elements

    return model_template

def evaluate_quality(generated_text, reference_text):
    # Placeholder for quality evaluation
    # You can implement specific evaluation metrics here
    # For now, we'll just return 0
    quality_score = 0.0
    return quality_score

def evaluate_mmlu(model, tokenizer, k_shot=5):
    from datasets import load_dataset
    mmlu = load_dataset('hendrycks_test')
    subjects = mmlu['train'].features.keys()
    total_correct = 0
    total_questions = 0

    for subject in subjects:
        train_examples = mmlu['train'][subject][:k_shot]
        few_shot_prompt = ''
        for ex in train_examples:
            few_shot_prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"

        test_examples = mmlu['test'][subject]
        for ex in test_examples:
            prompt = few_shot_prompt + f"Question: {ex['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 10)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = generated_text.strip().split('Answer:')[-1].strip()
            correct_answer = ex['answer'].strip()
            if predicted_answer == correct_answer:
                total_correct += 1
            total_questions += 1

    accuracy = total_correct / total_questions
    return accuracy

def evaluate_open_rewrite(model, tokenizer):
    from datasets import load_dataset
    from rouge_score import rouge_scorer

    dataset = load_dataset('open_rewrite_eval')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_score = 0
    total_examples = 0

    for ex in dataset['test']:
        input_text = ex['input']
        reference_text = ex['reference']
        inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = scorer.score(reference_text, generated_text)['rougeL'].fmeasure
        total_score += score
        total_examples += 1

    average_score = total_score / total_examples
    return average_score

def evaluate_tldr9(model, tokenizer):
    from datasets import load_dataset
    from rouge_score import rouge_scorer

    dataset = load_dataset('tldr_dataset')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    total_score = 0
    total_examples = 0

    for ex in dataset['test']:
        input_text = ex['article']
        reference_summary = ex['summary']
        prompt = f"{input_text}\n\nTL;DR:"
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50)
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = scorer.score(reference_summary, generated_summary)['rougeL'].fmeasure
        total_score += score
        total_examples += 1

    average_score = total_score / total_examples
    return average_score

def evaluate_ifeval(model, tokenizer):
    import torch
    import csv
    import os
    from rouge_score import rouge_scorer

    # Path to the IFEval dataset CSV file (adjust accordingly)
    dataset_path = 'path_to_ifeval_dataset/ifeval.csv'  # Update with your dataset path

    total = 0
    total_rougeL = 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Ensure the dataset exists
    if not os.path.exists(dataset_path):
        print("IFEval dataset not found at the specified path.")
        return None

    with open(dataset_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            instruction = row['instruction']
            input_text = row.get('input', '')
            reference_output = row['output']

            # Prepare the prompt
            if input_text:
                prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
            else:
                prompt = f"Instruction: {instruction}\nResponse:"

            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Compute Rouge-L score
            score = scorer.score(reference_output, generated_text)['rougeL'].fmeasure
            total_rougeL += score
            total += 1

            if total >= 100:  # Limit to first 100 samples
                break

    average_rougeL = total_rougeL / total if total > 0 else 0.0
    return average_rougeL

def evaluate_gsm8k(model, tokenizer):
    import torch
    from datasets import load_dataset
    import re

    # Load the GSM8K dataset
    gsm8k = load_dataset('gsm8k', 'main')

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; remove or adjust for full evaluation
    max_samples = 100

    for sample in gsm8k['test']:
        question = sample['question'].strip()
        answer = sample['answer'].strip()

        # Prepare the prompt with zero-shot CoT
        prompt = question + "\nAnswer: Let's think step by step."

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the final answer from the generated text
        # The final answer is usually after '####'
        gen_answer_match = re.search(r'####\s*(.*)', generated_text)
        if gen_answer_match:
            gen_final_answer = gen_answer_match.group(1).strip()
        else:
            # As a fallback, try to find the last number in the generated text
            gen_numbers = re.findall(r'\d+\.?\d*', generated_text)
            gen_final_answer = gen_numbers[-1] if gen_numbers else None

        # Extract the final answer from the reference answer
        ref_answer_match = re.search(r'####\s*(.*)', answer)
        if ref_answer_match:
            ref_final_answer = ref_answer_match.group(1).strip()
        else:
            ref_numbers = re.findall(r'\d+\.?\d*', answer)
            ref_final_answer = ref_numbers[-1] if ref_numbers else None

        if gen_final_answer and ref_final_answer:
            # Compare the numerical answers
            if gen_final_answer == ref_final_answer:
                correct += 1

        total += 1
        if total >= max_samples:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_math(model, tokenizer):
    import torch
    import json
    import os
    import re

    # Path to the MATH dataset directory (adjust accordingly)
    dataset_dir = 'path_to_math_dataset/test'  # Update with your dataset path

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; adjust as needed
    max_samples = 100

    # Collect all problem file paths
    problem_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                problem_files.append(os.path.join(root, file))

    for problem_file in problem_files[:max_samples]:
        with open(problem_file, 'r') as f:
            problem = json.load(f)

        question = problem['problem']
        solution = problem['solution']

        # Prepare the prompt
        prompt = question + "\nAnswer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the final answer from the generated text
        # Assuming the answer is between '\boxed{}' in LaTeX
        gen_answer_match = re.search(r'\\boxed\{([^}]*)\}', generated_text)
        if gen_answer_match:
            gen_final_answer = gen_answer_match.group(1).strip()
        else:
            # Fallback: extract the last numerical expression
            gen_numbers = re.findall(r'\d+\.?\d*', generated_text)
            gen_final_answer = gen_numbers[-1] if gen_numbers else None

        # Extract the final answer from the reference solution
        ref_answer_match = re.search(r'\\boxed\{([^}]*)\}', solution)
        if ref_answer_match:
            ref_final_answer = ref_answer_match.group(1).strip()
        else:
            ref_numbers = re.findall(r'\d+\.?\d*', solution)
            ref_final_answer = ref_numbers[-1] if ref_numbers else None

        if gen_final_answer and ref_final_answer:
            # Compare the numerical answers
            if gen_final_answer == ref_final_answer:
                correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Benchmark different versions of Llama-3.2-1B.')
    parser.add_argument('--model_type', type=str, required=True, choices=['original', 'quantized', 'compressed'],
                        help='Type of model to benchmark.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--input_text', type=str, required=True, help='Input text to the model.')
    parser.add_argument('--reference_text', type=str, default='', help='Reference text for quality evaluation.')
    args = parser.parse_args()

    if args.model_type == 'original':
        generated_text, num_tokens_output, inference_time = run_original_model(args.model_path, args.input_text)
    elif args.model_type == 'quantized':
        generated_text, num_tokens_output, inference_time = run_quantized_model(args.model_path, args.input_text)
    elif args.model_type == 'compressed':
        generated_text, num_tokens_output, inference_time = run_compressed_model(args.model_path, args.input_text)
    else:
        print("Invalid model type selected.")
        sys.exit(1)

    # Evaluate quality (placeholder)
    quality_score = evaluate_quality(generated_text, args.reference_text)

    # Print results
    print(f"Model Type: {args.model_type}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Number of Tokens Outputted: {num_tokens_output}")
    print(f"Quality Score: {quality_score}")
    print(f"Generated Text:\n{generated_text}")

if __name__ == '__main__':
    main()