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

# def evaluate_quality(generated_text, reference_text):
#     # Placeholder for quality evaluation
#     # You can implement specific evaluation metrics here
#     # For now, we'll just return 0
#     quality_score = 0.0
#     return quality_score

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

def evaluate_arc_challenge(model, tokenizer):
    import torch
    from datasets import load_dataset
    import re

    # Load the ARC Challenge dataset
    dataset = load_dataset('ai2_arc', 'ARC-Challenge')

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; adjust as needed
    max_samples = 100

    for sample in dataset['test']:
        question = sample['question'].strip()
        choices = sample['choices']['text']
        labels = sample['choices']['label']
        correct_answer = sample['answerKey'].strip()

        # Prepare the prompt
        prompt = f"Question: {question}\nChoices:\n"
        for label, choice in zip(labels, choices):
            prompt += f"{label}: {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the first character after 'Answer:' as the predicted label
        gen_answer_match = re.search(r'Answer:\s*([A-Za-z])', generated_text)
        if gen_answer_match:
            predicted_answer = gen_answer_match.group(1).strip().upper()
        else:
            predicted_answer = None

        if predicted_answer == correct_answer:
            correct += 1

        total += 1
        if total >= max_samples:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_gpqa(model, tokenizer):
    import torch
    from datasets import load_dataset
    import re

    # Load a general QA dataset; using 'nq_open' as an example
    dataset = load_dataset('nq_open')

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; adjust as needed
    max_samples = 100

    for sample in dataset['test']:
        question = sample['question'].strip()
        answers = sample['answers']

        # Prepare the prompt
        prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the generated answer
        generated_answer = generated_text.split('Answer:')[-1].strip()

        # Check if the generated answer matches any of the reference answers
        if any(generated_answer.lower() == ans.lower() for ans in answers):
            correct += 1

        total += 1
        if total >= max_samples:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_hellaswag(model, tokenizer):
    import torch
    from datasets import load_dataset
    import re

    # Load the HellaSwag dataset
    dataset = load_dataset('hellaswag')

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; adjust as needed
    max_samples = 100

    for sample in dataset['validation']:
        context = sample['ctx_a'].strip()
        endings = [sample[f'ending_{i}'].strip() for i in range(4)]
        correct_label = sample['label']

        # Prepare the prompt
        prompt = f"{context}"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            logits = []
            for ending in endings:
                input_ids_option = tokenizer.encode(ending, return_tensors='pt').to(device)
                total_input = torch.cat([inputs['input_ids'], input_ids_option], dim=-1)
                attention_mask = torch.ones_like(total_input).to(device)
                output = model(input_ids=total_input, attention_mask=attention_mask)
                # Get the log probability of the continuation
                logit = output.logits[:, -input_ids_option.size(-1):, :]
                log_probs = torch.nn.functional.log_softmax(logit, dim=-1)
                selected_log_probs = log_probs.gather(2, input_ids_option.unsqueeze(-1)).squeeze(-1)
                total_log_prob = selected_log_probs.sum().item()
                logits.append(total_log_prob)

            predicted_label = logits.index(max(logits))

            if predicted_label == correct_label:
                correct += 1

        total += 1
        if total >= max_samples:
            break

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_infinitebench_mc(model, tokenizer):
    import torch
    import os
    import json

    # Path to InfiniteBench En.MC dataset
    dataset_path = 'path_to_infinitebench/En.MC'  # Update accordingly

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to a few samples for testing due to resource constraints
    max_samples = 10

    # Load the dataset
    with open(os.path.join(dataset_path, 'test.json'), 'r') as f:
        data = json.load(f)

    for sample in data['data'][:max_samples]:
        context = sample['context']
        question = sample['question']
        choices = sample['choices']
        correct_answer = sample['answer']

        # Prepare the prompt
        prompt = f"{context}\n\nQuestion: {question}\nChoices:\n"
        for idx, choice in enumerate(choices):
            prompt += f"{idx}: {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=128000).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the predicted choice index
        predicted_answer = int(generated_text.strip().split('Answer:')[-1].strip())

        if predicted_answer == correct_answer:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

def evaluate_mgsm(model, tokenizer):
    import torch
    from datasets import load_dataset
    import re

    # Load the MGSM dataset; assuming it's available via Hugging Face Datasets
    dataset = load_dataset('mgsm')

    total = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Limit to first 100 samples for testing; adjust as needed
    max_samples = 100

    for sample in dataset['test']:
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
        # The final answer is usually after '####' or the last number
        gen_answer_match = re.search(r'####\s*(.*)', generated_text)
        if gen_answer_match:
            gen_final_answer = gen_answer_match.group(1).strip()
        else:
            # Fallback: extract the last number in the generated text
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

def main():
    import argparse
    import sys

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark different versions of Llama-3.2-1B.')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['original', 'quantized', 'compressed'],
                        help='Type of model to benchmark.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model directory.')
    parser.add_argument('--benchmark', type=str, required=True,
                        choices=['mmlu', 'open_rewrite', 'tldr9', 'ifeval', 'gsm8k', 'math',
                                 'arc_challenge', 'gpqa', 'hellaswag', 'infinitebench_mc', 'mgsm'],
                        help='Name of the benchmark to run.')
    args = parser.parse_args()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load the model based on the type
    if args.model_type == 'original':
        model = run_original_model(args.model_path)
    elif args.model_type == 'quantized':
        model = run_quantized_model(args.model_path)
    elif args.model_type == 'compressed':
        model = run_compressed_model(args.model_path)
    else:
        print("Invalid model type selected.")
        sys.exit(1)

    # Run the selected benchmark
    if args.benchmark == 'mmlu':
        accuracy = evaluate_mmlu(model, tokenizer)
        print(f"MMLU Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'open_rewrite':
        average_rougeL = evaluate_open_rewrite(model, tokenizer)
        print(f"Open-Rewrite Average Rouge-L Score: {average_rougeL:.4f}")
    elif args.benchmark == 'tldr9':
        average_rougeL = evaluate_tldr9(model, tokenizer)
        print(f"TLDR9+ Average Rouge-L Score: {average_rougeL:.4f}")
    elif args.benchmark == 'ifeval':
        average_rougeL = evaluate_ifeval(model, tokenizer)
        print(f"IFEval Average Rouge-L Score: {average_rougeL:.4f}")
    elif args.benchmark == 'gsm8k':
        accuracy = evaluate_gsm8k(model, tokenizer)
        print(f"GSM8K Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'math':
        accuracy = evaluate_math(model, tokenizer)
        print(f"MATH Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'arc_challenge':
        accuracy = evaluate_arc_challenge(model, tokenizer)
        print(f"ARC Challenge Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'gpqa':
        accuracy = evaluate_gpqa(model, tokenizer)
        print(f"GPQA Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'hellaswag':
        accuracy = evaluate_hellaswag(model, tokenizer)
        print(f"HellaSwag Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'infinitebench_mc':
        accuracy = evaluate_infinitebench_mc(model, tokenizer)
        print(f"InfiniteBench En.MC Accuracy: {accuracy:.2%}")
    elif args.benchmark == 'mgsm':
        accuracy = evaluate_mgsm(model, tokenizer)
        print(f"MGSM Accuracy: {accuracy:.2%}")
    else:
        print(f"Benchmark {args.benchmark} not recognized.")
        sys.exit(1)

if __name__ == '__main__':
    main()
    # python benchmark.py --model_type original --model_path path_to_model --benchmark gsm8k