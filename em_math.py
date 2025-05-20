import argparse
import torch
import torch.nn.functional as F
import copy
import json
import numpy as np
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from vllm import LLM, SamplingParams
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math_verify import parse, verify
import pandas as pd
import os

# ----------------------------
# Helper functions
# ----------------------------

import re

def extract_answer(text):
    """
    Extract the final piece of text inside a LaTeX \\boxed{…} tag.
    Returns the contents as a string, or None if no \\boxed{} is found.
    """
    # find all occurrences (non-greedy, across lines)
    matches = re.findall(r'\\boxed\{(.+?)\}', text, re.DOTALL)
    if matches:
        # return the last one, trimmed of surrounding whitespace
        return matches[-1].strip()
    return ""

def expand_histogram(counts, bin_edges):
    """Convert histogram bin counts back into raw samples (bin centers repeated by count)."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expanded = np.repeat(bin_centers, counts)
    return expanded

def build_prompt(question, posterior_cot=False, ground_truth=None):
    """
    Build the prompt given a question.
    If posterior_cot is True and ground_truth is provided,
    the prompt includes the ground truth answer.
    """
    if posterior_cot and ground_truth is not None:
        return (f"Below is a math problem and its corresponding ground truth answer:\n\n"
                f"Question: {question}\nAnswer: {ground_truth}\n\n"
                "Solve the above problem step-by-step to generate a reasoning trace that leads "
                "to the ground truth answer. Conclude with the final answer inside latex boxes, for example \\boxed{A} or \\boxed{\\frac{2}{3}}."
                "Your response should end exactly with this latex boxed expression.")
    else:
        return (f"{question}\nSolve the above problem step-by-step to generate a reasoning trace that leads "
                "to the ground truth answer. Conclude with the final answer inside latex boxes, for example \\boxed{A} or \\boxed{\\frac{2}{3}}."
                "Your response should end exactly with this latex boxed expression.")

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
    mask = ((positions >= prompt_tensor-1) & (positions < (seq_lengths - 1))).bfloat16()
    log_prob_sum = (pred_log_probs * mask).sum(dim=1)  # Sum for each sample, shape (B,)
    return log_prob_sum

# ----------------------------
# Custom Dataset for GSM8K
# ----------------------------

class MathDataset(Dataset):
    def __init__(self, split_data, single_sample=False):
        self.data = split_data
        self.single_sample = single_sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.single_sample:
            idx = 1
        item = self.data[idx]
        ground_truth = item['answer']
        return {"question": item["problem"], "ground_truth": ground_truth}

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
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    responses = vllm_instance.generate(requests, sampling_params=sampling_params)
    generated_texts = [response.outputs[0].text.strip() for response in responses]

    correct_numeric = 0
    avg_len = 0.0
    total = len(all_questions)
    for gen_text, gt in zip(generated_texts, all_ground_truths):
        avg_len += len(gen_text)
        extracted = extract_answer(gen_text)
        gold = parse(gt)
        pred = parse(extracted)
        is_correct = verify(gold, pred)
        if extracted is not None and is_correct:
            correct_numeric += 1

    numeric_accuracy = correct_numeric / total if total > 0 else 0
    return numeric_accuracy, avg_len/len(all_questions)

# ----------------------------
# Main training routine with batched log prob computation
# ----------------------------

def train_rl(
    epochs=3,
    batch_size=4,        # number of questions per batch
    num_trajectories=3,  # replicates per question
    wandb_project="math_rl",
    model_name='Qwen/Qwen2.5-1.5B-Instruct',
    beta=1.0,
    importance_weights=None,
    only_M_step=False,
    E_learning_rate=1e-6,
    M_learning_rate=1e-5,
    share_weights=False,
    dataset_name="agentica-org/DeepScaleR-Preview-Dataset",
    checkpoint='checkpoint.pth',
    load_ckpt=False
):
    wandb.init(project=wandb_project, name=model_name+'_math_em_K_'+str(num_trajectories)+'_share_weights_'+str(share_weights)+'_iw_'+str(importance_weights)+'_batch_'+str(batch_size)+'_beta_'+str(beta)+'_E_lr_'+str(E_learning_rate)+'_M_lr_'+str(M_learning_rate),
            config={                 # everything below gets captured in the run config
            "epochs": epochs,
            "batch_size": batch_size,
            "num_trajectories": num_trajectories,
            "model_name": model_name,
            "beta": beta,
            "importance_weights": importance_weights,
            "only_M_step": only_M_step,
            "E_learning_rate": E_learning_rate,
            "M_learning_rate": M_learning_rate
        })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and split dataset.
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)['train']
    split_idx = int(len(dataset) * 0.85)
    dataset = [item for item in dataset]
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    #train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    
    train_dataset = MathDataset(train_data)
    test_dataset = MathDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: [i for i in x if i is not None])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: [i for i in x if i is not None])
    
    # Load model and tokenizer.
    print("Loading Qwen model and tokenizer...")
    train_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    train_model.train()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create reference model copy (frozen).
    if share_weights:
        posterior_model = train_model
    else:
        posterior_model = copy.deepcopy(train_model)
    #device = train_model.device
    #for param in reference_model.parameters():
    #    param.requires_grad = False

    E_optimizer = AdamW(posterior_model.parameters(), lr=E_learning_rate)
    if share_weights:
        # Use the same optimizer for both E and M steps.
        M_optimizer = E_optimizer
    else:
        # Separate optimizer for M step.
        M_optimizer = AdamW(train_model.parameters(), lr=M_learning_rate)
    
    # Initialize vLLM instance (for generation).
    vllm_instance = LLM(model_name, max_num_seqs=512, gpu_memory_utilization=0.15)
    
    # Load from checkpoint if exist
    if os.path.exists('/usr/workspace/venkatraman1/em_ckpts/'+checkpoint) and load_ckpt:
        ckpt = torch.load('/usr/workspace/venkatraman1/em_ckpts/'+checkpoint, map_location='cpu')#device)
        posterior_model.load_state_dict(ckpt['posterior_model'])
        train_model.load_state_dict(ckpt['train_model'])
        E_optimizer = AdamW(posterior_model.parameters(), lr=E_learning_rate)
        M_optimizer = AdamW(train_model.parameters(), lr=M_learning_rate)
        E_optimizer.load_state_dict(ckpt['E_optimizer'])
        M_optimizer.load_state_dict(ckpt['M_optimizer'])
        for param in E_optimizer.state:
            param_device = param.device
            for k, v in E_optimizer.state[param].items():
                if isinstance(v, torch.Tensor):
                    E_optimizer.state[param][k] = v.to(param_device)
        for param in M_optimizer.state:
            param_device = param.device
            for k, v in M_optimizer.state[param].items():
                if isinstance(v, torch.Tensor):
                    M_optimizer.state[param][k] = v.to(param_device)
        llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
        llmp.load_weights(posterior_model.named_parameters())
        del ckpt
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Pre-training evaluation.
    print("Evaluating on test set (pre-training)...")
    test_accuracy, avg_context_len = evaluate_test(test_loader, vllm_instance, tokenizer, posterior_cot=False)
    wandb.log({"epoch": 0, "test_accuracy": test_accuracy, "Avg context len": avg_context_len})
    print(f"Initial Test Accuracy: {test_accuracy:.4f}")
    
    global_step = 0
    total_reward = 0.0
    if importance_weights is not None:
        accum_min_hist = np.zeros(10, dtype=int)
        accum_max_hist = np.zeros(10, dtype=int)
        bin_edges = np.linspace(0, 1, 11)

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            # Prepare lists to store prompts, prompt lengths, and ground truths.
            prior_prompt_list = []
            prior_prompt_lengths = []
            post_prompt_list = []
            post_prompt_lengths = []
            ground_truths = []
            
            for item in batch:
                question = item["question"]
                gt = item["ground_truth"]
                ground_truths.append(gt)
                for _ in range(num_trajectories):
                    # Prior
                    prior_prompt = build_prompt(question, posterior_cot=False)
                    prior_prompt_list.append(prior_prompt)
                    # Compute prompt token length.
                    tokenized = tokenizer(prior_prompt, return_tensors='pt')
                    prior_prompt_lengths.append(tokenized.input_ids.shape[-1])
                    #Posterior
                    post_prompt = build_prompt(question, posterior_cot=True, ground_truth=gt)
                    post_prompt_list.append(post_prompt)
                    # Compute prompt token length.
                    tokenized = tokenizer(post_prompt, return_tensors='pt')
                    post_prompt_lengths.append(tokenized.input_ids.shape[-1])
            
            # Generate trajectories in batch via vLLM.
            messages = [[{"role": "user", "content": prompt}] for prompt in prior_prompt_list]
            prior_requests = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                        for message in messages]
            messages = [[{"role": "user", "content": prompt}] for prompt in post_prompt_list]
            post_requests = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                        for message in messages]
            for m in range(len(prior_requests)):
                tokenized = tokenizer(prior_requests[m], return_tensors='pt')
                prior_prompt_lengths[m] = tokenized.input_ids.shape[-1]
                tokenized = tokenizer(post_requests[m], return_tensors='pt')
                post_prompt_lengths[m] = tokenized.input_ids.shape[-1]
            sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
            responses = vllm_instance.generate(post_requests, sampling_params=sampling_params)
            generated_texts = [response.outputs[0].text.strip() for response in responses]
            # Generate full prior and post texts
            prior_texts = [base_prompt + gen_text for base_prompt, gen_text in zip(prior_requests, generated_texts)]
            post_texts = [base_prompt + gen_text for base_prompt, gen_text in zip(post_requests, generated_texts)]

            if global_step == 0:
                # Log responses
                llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
                llmp.load_weights(posterior_model.named_parameters())
                post_responses_eval = vllm_instance.generate(post_requests, sampling_params=sampling_params)
                post_samples_eval = [resp.outputs[0].text.strip() for resp in post_responses_eval]

                # 2) Sample with the train (prior) model
                llmp.load_weights(train_model.named_parameters())
                prior_responses_eval = vllm_instance.generate(prior_requests, sampling_params=sampling_params)
                prior_samples_eval = [resp.outputs[0].text.strip() for resp in prior_responses_eval]

                questions_repeated = [
                    item["question"]
                    for item in batch
                    for _ in range(num_trajectories)
                ]
                data = {
                    "question": questions_repeated,
                    "prior sample": prior_samples_eval,
                    "posterior sample": post_samples_eval,
                }
                table = wandb.Table(dataframe=pd.DataFrame(data))

                # 4) Log it
                wandb.log({"0_step_samples": table, "step": global_step})
                del post_responses_eval, prior_responses_eval, post_samples_eval, prior_samples_eval

            # Compute rewards (processing each generated sample string).
            rewards = []
            for idx, gen_text in enumerate(generated_texts):
                extracted = extract_answer(gen_text)
                question_idx = idx // num_trajectories
                gt = ground_truths[question_idx]
                gold = parse(gt)
                pred = parse(extracted)
                reward = 1.0 if verify(gold, pred) else 0.0
                rewards.append(reward)
            
            rewards_tensor = torch.tensor(rewards, device=train_model.device, dtype=torch.bfloat16)
            rewards_tensor = rewards_tensor.view(len(batch), num_trajectories)
            total_reward = total_reward + rewards_tensor.mean()
            if importance_weights is not None:
                iw_tensor = torch.zeros_like(rewards_tensor, device=train_model.device)
            
            # Prior log probs
            #with torch.no_grad():
            #    prior_log_probs = compute_log_prob_batch(train_model, tokenizer, prior_texts, prior_prompt_lengths)
            #    prior_log_probs = prior_log_probs.view(len(batch), num_trajectories)

            # E-step loss
            E_loss_total = 0.0
            kl_total = 0.0
            for b in range(len(batch)):
                with torch.no_grad():
                    prior_log_probs = compute_log_prob_batch(train_model, tokenizer, prior_texts[b*num_trajectories:(b+1)*num_trajectories], prior_prompt_lengths[b*num_trajectories:(b+1)*num_trajectories])
                    prior_log_probs = prior_log_probs.view(num_trajectories)
                #post_log_probs = compute_log_prob_batch(train_model, tokenizer, post_texts[b*num_trajectories:(b+1)*num_trajectories], post_prompt_lengths[b*num_trajectories:(b+1)*num_trajectories])
                post_log_probs = compute_log_prob_batch(posterior_model, tokenizer, post_texts[b*num_trajectories:(b+1)*num_trajectories], post_prompt_lengths[b*num_trajectories:(b+1)*num_trajectories])
                post_log_probs = post_log_probs.view(num_trajectories)
                
                # Compute the REINFORCE loss.
                kl = post_log_probs.detach() - prior_log_probs#[b]
                entropy_reward = rewards_tensor[b] - beta * (kl)
                baseline = entropy_reward.mean()
                advantages = (entropy_reward - baseline)
                E_loss = - (advantages * post_log_probs).mean()
                E_loss.backward()
                E_loss_total = E_loss_total + E_loss.item()
                kl_total = kl_total + kl.mean()
                # Compute importance weights.
                if importance_weights is not None:
                    if importance_weights == "soft_filter":
                        iw_tensor[b] = torch.softmax(entropy_reward, dim=0)
                    elif importance_weights == "RL":
                        advantages = (rewards_tensor[b] - rewards_tensor[b].mean()) / (rewards_tensor[b].std() + 1e-8)
                        iw_tensor[b] = advantages #* torch.exp(-kl)
                        # standardize the importance weights
                        #iw_tensor[b] = iw_tensor[b] / (iw_tensor[b].std() + 1e-8)
            
            # Update parameters with E-step
            torch.nn.utils.clip_grad_norm_(posterior_model.parameters(), max_norm=1.0)
            if not only_M_step:
                E_optimizer.step()
            E_optimizer.zero_grad()
            
            # M-step loss
            M_loss_total = 0.0
            for b in range(len(batch)):
                prior_log_probs = compute_log_prob_batch(train_model, tokenizer, prior_texts[b*num_trajectories:(b+1)*num_trajectories], prior_prompt_lengths[b*num_trajectories:(b+1)*num_trajectories])
                prior_log_probs = prior_log_probs.view(num_trajectories)
                
                # Fiter out wrong answers and maximize log prob for rest
                if importance_weights is not None:
                    M_loss = - (iw_tensor[b] * prior_log_probs).mean()#sum()#mean()
                else:
                    M_loss = - (rewards_tensor[b] * prior_log_probs).mean()
                M_loss.backward()
                M_loss_total = M_loss_total + M_loss.item()
                
            # Update parameters with M-step
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            M_optimizer.step()
            M_optimizer.zero_grad()
            
            # Accumulate min and max importance weights for logging
            if importance_weights is not None:
                iw_tensor = iw_tensor.to(torch.float32).detach().cpu().numpy()
                min_iw = iw_tensor.min(axis=1)
                max_iw = iw_tensor.max(axis=1)
                accum_min_hist += np.histogram(min_iw, bins=bin_edges)[0]
                accum_max_hist += np.histogram(max_iw, bins=bin_edges)[0]
                
            global_step += 1

            wandb.log({"E_loss": E_loss_total/len(batch), "M_loss": M_loss_total/len(batch), "Total_loss": (E_loss_total + M_loss_total)/len(batch), "kl": kl_total/len(batch), "global_step": global_step})
            #print(f"Step {global_step}: Loss {loss.item():.4f}")

            llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
            llmp.load_weights(posterior_model.named_parameters())#(train_model.named_parameters())

            if (global_step) % 10 == 0:
                avg_reward = total_reward/10
                total_reward = 0.0
                wandb.log({
                    "avg_reward": avg_reward,
                    "step": global_step,
                })

            if global_step % 100 == 0:
                # Log importance weights histogram
                if importance_weights is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    width = (bin_edges[1] - bin_edges[0]) * 0.4
                    bins_center = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot side-by-side bar charts for min and max histograms.
                    ax.bar(bins_center - width, accum_min_hist, width=width, alpha=0.7, label='Min Weights')
                    ax.bar(bins_center + width, accum_max_hist, width=width, alpha=0.7, label='Max Weights')
                    
                    ax.set_xlabel("Importance Weight Value (Quantized Bins)")
                    ax.set_ylabel("Accumulated Count")
                    ax.set_title(f"Accumulated Histograms for Min and Max Importance Weights")
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    
                    # Log the figure to wandb as an image
                    wandb.log({"iw_histogram_plot": wandb.Image(fig)})
                    plt.close(fig)
                    accum_min_hist = np.zeros(10, dtype=int)
                    accum_max_hist = np.zeros(10, dtype=int)
                
                # End-of-epoch evaluation.
                # 1) Sample with the posterior model
                if global_step % 500 == 0:
                    llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
                    llmp.load_weights(posterior_model.named_parameters())
                    post_responses = vllm_instance.generate(post_requests, sampling_params=sampling_params)
                    post_samples = [resp.outputs[0].text.strip() for resp in post_responses]

                    # 2) Sample with the train (prior) model
                    llmp.load_weights(train_model.named_parameters())
                    prior_responses = vllm_instance.generate(prior_requests, sampling_params=sampling_params)
                    prior_samples = [resp.outputs[0].text.strip() for resp in prior_responses]

                    questions_repeated = [
                        item["question"]
                        for item in batch
                        for _ in range(num_trajectories)
                    ]
                    data = {
                        "question": questions_repeated,
                        "prior sample": prior_samples,
                        "posterior sample": post_samples,
                    }
                    table = wandb.Table(dataframe=pd.DataFrame(data))

                    # 4) Log it
                    wandb.log({str(global_step)+"_step_samples": table, "step": global_step})
                
                llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
                llmp.load_weights(train_model.named_parameters())
                test_accuracy, avg_context_len = evaluate_test(test_loader, vllm_instance, tokenizer, posterior_cot=False)
                wandb.log({"epoch": epoch, "test_accuracy": test_accuracy, "Avg context len": avg_context_len})
                print(f"Epoch {epoch} Test Accuracy: {test_accuracy:.4f}")
                llmp = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
                llmp.load_weights(posterior_model.named_parameters())

                ckpt = { 
                'epoch': epoch,
                'train_model': train_model.state_dict(),
                'posterior_model': posterior_model.state_dict(),
                'E_optimizer': E_optimizer.state_dict(),
                'M_optimizer': M_optimizer.state_dict()}
                torch.save(ckpt, '/usr/workspace/venkatraman1/em_ckpts/'+checkpoint)
    #torch.save(train_model.state_dict(), "models/trained_qwen_policy.pt")
    #wandb.save("trained_qwen_policy.pt")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL on GSM8K using reinforcement learning for LM fine-tuning with vLLM and wandb tracking."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size (number of questions per batch).")
    parser.add_argument("--num_trajectories", type=int, default=8, help="Number of trajectories per question.")
    parser.add_argument("--wandb_project", type=str, default="math_rl", help="wandb project name.")
    parser.add_argument("--dataset", type=str, default="agentica-org/DeepScaleR-Preview-Dataset", help="Name of math dataset to use.")
    parser.add_argument("--beta", type=float, default=1.0, help="KL term weight for E-step")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Pretrained model name.")
    parser.add_argument("--importance_weights", type=str, choices=["none", "soft_filter", "RL"], default="none", help="Type of importance weighting for M-step. Options: 'none', 'soft_filter', 'RL'.")
    parser.add_argument("--only_M_step", action='store_true', help="Only M-step training.")
    parser.add_argument("--E_learning_rate", type=float, default=5e-6, help="Learning rate for the E optimizer.")
    parser.add_argument("--M_learning_rate", type=float, default=5e-6, help="Learning rate for the E optimizer.")
    parser.add_argument("--share_weights", action='store_true', help="Share weights between E and M steps.")
    parser.add_argument("--checkpoint", type=str, default='checkpoint.pth', help="checkpoint file to load/save.")
    parser.add_argument("--load_ckpt", action='store_true', help="Whether to resume training from checkpoint.")
    
    args = parser.parse_args()
    if args.importance_weights == "none":
        args.importance_weights = None
    
    train_rl(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_trajectories=args.num_trajectories,
        wandb_project=args.wandb_project,
        model_name=args.model_name,
        beta=args.beta,
        importance_weights=args.importance_weights,
        only_M_step=args.only_M_step,
        E_learning_rate=args.E_learning_rate,
        M_learning_rate=args.M_learning_rate,
        share_weights=args.share_weights,
        dataset_name=args.dataset,
        checkpoint=args.checkpoint,
        load_ckpt=args.load_ckpt
    )
