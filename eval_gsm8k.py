import torch
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def extract_answer(text):
    """
    Extract the final numeric answer from generated text.
    The answer is assumed to appear after the token '####'.
    For example, if the generated text ends with: '... #### 42',
    then 42.0 is returned.
    """
    match = re.search(r'####\s*([\d\.\-\+]+)', text)
    if match:
        try:
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None

def enforce_answer_format(text):
    """
    Ensure that the answer ends exactly with the pattern "#### <integer>".
    If extra tokens appear after the final occurrence of the pattern,
    trim them off.
    """
    # Find all occurrences of the target pattern
    matches = list(re.finditer(r'####\s*[\d\.\-\+]+', text))
    if matches:
        # Use the last occurrence and trim any extra text after it
        last_match = matches[-1]
        return text[:last_match.end()].strip()
    return text.strip()

def evaluate_gsm8k(posterior_cot=False):
    
    print("Loading GSM8K test dataset from Hugging Face...")
    gsm8k = load_dataset('openai/gsm8k', 'main')
    test_data = gsm8k['test']
    
    questions = []
    ground_truths = []
    for item in test_data:
        question = item['question']
        answer_text = item['answer']
        # The ground truth answer is expected to follow '####'
        match = re.search(r'####\s*([\d\.\-\+]+)', answer_text)
        if not match:
            continue  # Skip if the answer does not follow the expected format
        ground_truth = float(match.group(1).strip())
        questions.append(question)
        ground_truths.append(ground_truth)
    
    # Create prompts with updated instructions so that the final answer strictly follows the pattern.
    if posterior_cot:
        prompts = [
            "Below is a math problem and its corresponding ground truth answer:\n\nQuestion: " +
            q + "\nAnswer: " + str(int(a)) +
            "\n\nSolve the above problem step-by-step to generate a reasoning trace that leads to the ground truth answer. Conclude with the final answer of the reasoning trace after '####'. "
            "Your response should end exactly with '#### <integer>' (and nothing more)."
            for q, a in zip(questions, ground_truths)
        ]
    else:
        prompts = [
            q + "\nSolve this problem step-by-step and conclude with your final answer after '####'. "
            "Your final answer should end exactly with '#### <integer>' (and nothing more)."
            for q in questions
        ]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    
    print("Generating all answers with vLLM in a single parallel batch...")
    # Define sampling parameters. You can tune these if needed.
    max_tokens = 1024  # Maximum tokens to generate for each prompt
    
    # Initialize the vLLM instance with the Qwen model.
    # Note: vLLM expects a model identifier compatible with Hugging Face.
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-1.5B-Instruct', 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    llm = LLM('Qwen/Qwen2.5-1.5B-Instruct', max_num_seqs=1024, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    requests = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    print(requests[0])
    
    # Generate responses in one parallel batch.
    responses = llm.generate(requests, sampling_params=sampling_params)
    generated_texts = [response.outputs[0].text.strip() for response in responses]
    print(generated_texts[0])
    
    # Ensure each generated text ends exactly with "#### <integer>".
    processed_texts = [enforce_answer_format(text) for text in generated_texts]
    
    # Evaluate numeric accuracy: compare the numeric part extracted from the generated answer with the ground truth.
    correct_numeric = 0
    results = []
    total = len(questions)
    
    for question, gen_text, gt in zip(questions, processed_texts, ground_truths):
        extracted = extract_answer(gen_text)
        is_correct = (extracted is not None and abs(extracted - gt) < 1e-6)
        if is_correct:
            correct_numeric += 1
        results.append({
            "question": question,
            "ground_truth": gt,
            "generated_text": gen_text,
            "extracted_numeric": extracted,
            "numeric_is_correct": is_correct,
        })
    
    numeric_accuracy = correct_numeric / total if total > 0 else 0
    print(f"\nFinal Numeric Accuracy: {numeric_accuracy:.4f} ({correct_numeric}/{total})")
    
    # Save final results to a JSON file.
    with open("posterior_"+ str(posterior_cot) +"_gsm8k_evaluation_results.json", "w") as f:
        json.dump({
            "numeric_accuracy": numeric_accuracy,
            "total": total,
            "results": results,
        }, f, indent=2)

if __name__ == "__main__":
    posterior_cot = True
    evaluate_gsm8k(posterior_cot)
