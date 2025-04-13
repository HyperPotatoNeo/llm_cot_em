import argparse
import torch
import torch.nn.functional as F
import copy
import json
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from vllm import LLM, SamplingParams
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# ----------------------------
# Helper functions
# ----------------------------

def extract_answer(text):
    """
    Extract the final numeric answer from generated text.
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
    Ensure that the answer ends exactly with the pattern "#### <number>".
    Extra tokens beyond the final occurrence of the pattern are trimmed.
    """
    matches = list(re.finditer(r'####\s*[\d\.\-\+]+', text))
    if matches:
        last_match = matches[-1]
        return text[:last_match.end()].strip()
    return text.strip()

def build_prompt(question, posterior_cot=False, ground_truth=None):
    """
    Build the prompt given a question.
    If posterior_cot is True and ground_truth is provided,
    the prompt includes the ground truth answer.
    """
    if posterior_cot and ground_truth is not None:
        return (f"Below is a math problem and its corresponding ground truth answer:\n\n"
                f"Question: {question}\nAnswer: {int(ground_truth)}\n\n"
                "Solve the above problem step-by-step to generate a reasoning trace that leads "
                "to the ground truth answer. Conclude with the final answer of the reasoning trace after '####'. "
                "Your response should end exactly with '#### <integer>' (and nothing more).")
    else:
        return (f"{question}\nSolve this problem step-by-step and conclude with your final answer after '####'. "
                "Your final answer should end exactly with '#### <integer>' (and nothing more).")

def compute_log_prob_batch(model, tokenizer, full_texts, prompt_lengths):
    """
    Compute the summed log probability for the answer tokens for a batch of generated texts.
    Instead of processing one at a time, we tokenize the list 'full_texts' (which are complete prompt+answer sequences)
    with padding and do one forward pass.
    
    Args:
      model: The causal language model.
      tokenizer: Associated tokenizer.
      full_texts: List of strings (each full text is prompt concatenated with generated answer).
      prompt_lengths: List of integer prompt lengths (number of tokens in the prompt) corresponding to each text.
    
    Returns:
      A tensor of shape (B,) of summed log probabilities (one per sample).
      
    The log probability is computed for answer tokens only, i.e. we only sum the log probabilities for tokens
    generated after the prompt. Internally, we use the typical shifting approach:
      - Tokenize all texts with padding.
      - Compute logits.
      - Compute log_softmax over the vocabulary.
      - Gather log probabilities for tokens in the target (i.e. the shifted input_ids).
      - For each sample, use a mask to only include positions from prompt_length up to the last predicted token.
    """
    encodings = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(model.device)          # shape: (B, L_max)
    attention_mask = encodings["attention_mask"].to(model.device)  # shape: (B, L_max)
    outputs = model(input_ids)
    logits = outputs.logits  # shape: (B, L_max, vocab_size)
    log_probs = F.log_softmax(logits/0.7, dim=-1)
    
    # Shift the logits and targets so that each logit corresponds to the probability of
    # the next token.
    # We take predictions for positions [0:L-1] and compare with target tokens from [1:L].
    target_ids = input_ids[:, 1:]
    pred_log_probs = log_probs[:, :-1, :].gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    # pred_log_probs: shape (B, L_max-1)
    
    B, Lm1 = pred_log_probs.shape
    # Convert the list of prompt_lengths into a tensor.
    prompt_tensor = torch.tensor(prompt_lengths, device=pred_log_probs.device, dtype=torch.long).unsqueeze(1)  # (B, 1)
    # Compute actual sequence lengths using the attention mask.
    seq_lengths = attention_mask.sum(dim=1).long().unsqueeze(1)  # (B, 1)
    
    # Create a positions matrix for indices along the token axis.
    positions = torch.arange(Lm1, device=pred_log_probs.device).unsqueeze(0).expand(B, Lm1)  # (B, Lm1)
    # We want to sum the log probabilities for positions corresponding to answer tokens.
    # For each sample i, we mask positions where:
    #    positions >= prompt_lengths[i]   and   positions < (seq_lengths[i] - 1)
    # (We subtract one because our pred_log_probs has length seq_length-1.)
    mask = ((positions >= prompt_tensor) & (positions < (seq_lengths - 1))).float()
    log_prob_sum = (pred_log_probs * mask).sum(dim=1)  # Sum for each sample, shape (B,)
    return log_prob_sum

# ----------------------------
# Custom Dataset for GSM8K
# ----------------------------

class GSM8KDataset(Dataset):
    def __init__(self, split_data):
        self.data = split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        answer_text = item['answer']
        match = re.search(r'####\s*([\d\.\-\+]+)', answer_text)
        if not match:
            return None
        ground_truth = float(match.group(1).strip())
        return {"question": item["question"], "ground_truth": ground_truth}

# ----------------------------
# Evaluation function (using vLLM)
# ----------------------------

def evaluate_test(test_loader, vllm_instance, tokenizer, posterior_cot=False):
    all_questions = []
    all_ground_truths = []
    for batch in test_loader:
        for item in batch:
            if item is not None:
                all_questions.append(item["question"])
                all_ground_truths.append(item["ground_truth"])

    prompts = [build_prompt(q, posterior_cot=posterior_cot, ground_truth=gt)
               for q, gt in zip(all_questions, all_ground_truths)]
    
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    requests = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
    responses = vllm_instance.generate(requests, sampling_params=sampling_params)
    generated_texts = [response.outputs[0].text.strip() for response in responses]

    correct_numeric = 0
    total = len(all_questions)
    for gen_text, gt in zip(generated_texts, all_ground_truths):
        processed_text = enforce_answer_format(gen_text)
        extracted = extract_answer(processed_text)
        if extracted is not None and abs(extracted - gt) < 1e-6:
            correct_numeric += 1

    numeric_accuracy = correct_numeric / total if total > 0 else 0
    return numeric_accuracy

# ----------------------------
# Main training routine with batched log prob computation
# ----------------------------

def train_rl(
    epochs=3,
    batch_size=4,        # number of questions per batch
    num_trajectories=3,  # replicates per question
    wandb_project="gsm8k_rl",
    model_name='Qwen/Qwen2.5-1.5B-Instruct'
):
    wandb.init(project=wandb_project)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and split dataset.
    print("Loading GSM8K dataset...")
    gsm8k = load_dataset('openai/gsm8k', 'main')
    full_data = [item for item in gsm8k['test'] if re.search(r'####\s*[\d\.\-\+]+', item['answer'])]
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    
    train_dataset = GSM8KDataset(train_data)
    test_dataset = GSM8KDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: [i for i in x if i is not None])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: [i for i in x if i is not None])
    
    # Load model and tokenizer.
    print("Loading Qwen model and tokenizer...")
    train_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    train_model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create reference model copy (frozen).
    reference_model = copy.deepcopy(train_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(train_model.parameters(), lr=1e-6)
    
    # Initialize vLLM instance (for generation).
    vllm_instance = LLM(model_name, max_num_seqs=1024, gpu_memory_utilization=0.25)
    
    # Pre-training evaluation.
    print("Evaluating on test set (pre-training)...")
    test_accuracy = evaluate_test(test_loader, vllm_instance, tokenizer, posterior_cot=False)
    wandb.log({"epoch": 0, "test_accuracy": test_accuracy})
    print(f"Initial Test Accuracy: {test_accuracy:.4f}")
    
    global_step = 0

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            # Prepare lists to store prompts, prompt lengths, and ground truths.
            prompt_list = []
            prompt_lengths = []
            ground_truths = []
            
            for item in batch:
                question = item["question"]
                gt = item["ground_truth"]
                ground_truths.append(gt)
                for _ in range(num_trajectories):
                    prompt = build_prompt(question, posterior_cot=False)
                    prompt_list.append(prompt)
                    # Compute prompt token length.
                    tokenized = tokenizer(prompt, return_tensors='pt')
                    prompt_lengths.append(tokenized.input_ids.shape[-1])
            
            # Generate trajectories in batch via vLLM.
            messages = [[{"role": "user", "content": prompt}] for prompt in prompt_list]
            requests = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                        for message in messages]
            sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
            responses = vllm_instance.generate(requests, sampling_params=sampling_params)
            generated_texts = [response.outputs[0].text.strip() for response in responses]

            # Compute rewards (processing each generated sample string).
            rewards = []
            for idx, gen_text in enumerate(generated_texts):
                processed_text = enforce_answer_format(gen_text)
                extracted = extract_answer(processed_text)
                question_idx = idx // num_trajectories
                gt = ground_truths[question_idx]
                reward = 1.0 if (extracted is not None and abs(extracted - gt) < 1e-6) else 0.0
                rewards.append(reward)
            
            rewards_tensor = torch.tensor(rewards, device=train_model.device, dtype=torch.float)
            rewards_tensor = rewards_tensor.view(len(batch), num_trajectories)
            # Compute baseline (average reward per question) and advantage.
            baseline = rewards_tensor.mean(dim=1, keepdim=True)
            advantages = (rewards_tensor - baseline)
            
            # Compute log probabilities in batch (vectorized computation).
            # This returns a tensor of shape (batch_size * num_trajectories,)
            for b in range(len(batch)):
                batch_log_probs = compute_log_prob_batch(train_model, tokenizer, generated_texts[b*num_trajectories:(b+1)*num_trajectories], prompt_lengths[b*num_trajectories:(b+1)*num_trajectories])
                batch_log_probs = batch_log_probs.view(num_trajectories)
                
                # Compute the REINFORCE loss.
                loss = - (advantages[b] * batch_log_probs).mean()
                loss.backward()
            
            # Update parameters immediately after processing the batch.
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            wandb.log({"train_loss": loss.item(), "global_step": global_step})
            print(f"Step {global_step}: Loss {loss.item():.4f}")

            llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
            llmp.load_weights(train_model.named_parameters())

            if global_step % 10 == 0:
                avg_reward = rewards_tensor.mean().item()
                avg_log_prob = batch_log_probs.mean().item()
                wandb.log({
                    "avg_reward": avg_reward,
                    "avg_log_prob": avg_log_prob,
                    "step": global_step,
                })

        # End-of-epoch evaluation.
        test_accuracy = evaluate_test(test_loader, vllm_instance, tokenizer, posterior_cot=False)
        wandb.log({"epoch": epoch, "test_accuracy": test_accuracy})
        print(f"Epoch {epoch} Test Accuracy: {test_accuracy:.4f}")

    #torch.save(train_model.state_dict(), "trained_qwen_policy.pt")
    #wandb.save("trained_qwen_policy.pt")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL on GSM8K using reinforcement learning for LM fine-tuning with vLLM and wandb tracking."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size (number of questions per batch).")
    parser.add_argument("--num_trajectories", type=int, default=5, help="Number of trajectories per question.")
    parser.add_argument("--wandb_project", type=str, default="gsm8k_rl", help="wandb project name.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Pretrained model name.")
    
    args = parser.parse_args()
    
    train_rl(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_trajectories=args.num_trajectories,
        wandb_project=args.wandb_project,
        model_name=args.model_name
    )
