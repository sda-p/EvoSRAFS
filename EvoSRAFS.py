"""
EvoSRAFS ‑ Minimal implementation
================================
Evolutionary Strategies with Reward‑Weighted Averaging & Fitness Shaping
compatible with OpenAI Gymnasium‑style environments and PEFT/LoRA
adaptors on top of a frozen, quantised base language model.

Research direction: Alistair Nexwell
Implementation: ChatGPT (May 2025)
Obligatory debugging: Alistair Nexwell
Licence: CC0

This file focuses on the *algorithmic* core; environment‑specific prompt/
action parsing callbacks must be supplied by the user.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple, Optional

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple, Dict, Optional

import numpy as np
import torch
from torch import nn
import gymnasium as gym
import gym_sokoban
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import wandb

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of tensors into a 1‑D vector (on CPU)."""
    return torch.cat([t.detach().cpu().flatten() for t in tensors], dim=0)


def _unflatten_to_tensors(vector: torch.Tensor, like: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Convert a 1‑D vector back into a list of tensors with the same shapes as *like*."""
    outputs: List[torch.Tensor] = []
    offset = 0
    for t in like:
        n = t.numel()
        outputs.append(vector[offset : offset + n].view_as(t))
        offset += n
    return outputs

class TrainingProgressPrinter:
    def __init__(self, total_steps=None):
        self.total_steps = total_steps
        self.start_time = None
        self.step = 0
        self.last_step_time = None

    def start(self):
        self.start_time = time.time()
        self.step = 0
        self.last_step_time = self.start_time

    def update(self, loss):
        if self.start_time is None:
            raise ValueError("Must call start() before update()")

        self.step += 1
        current_time = time.time()
        step_time = current_time - self.last_step_time
        total_elapsed = current_time - self.start_time
        self.last_step_time = current_time

        print(f"Step: {self.step}", end="")
        print(f", Reward: {loss:.4f}", end="")
        print(f", Step Time: {step_time:.2f}s", end="")
        print(f", Total Elapsed: {self._format_time(total_elapsed)}", end="")

        if self.total_steps is not None and self.step > 0:
            remaining_steps = self.total_steps - self.step
            if remaining_steps > 0:
                eta_seconds = step_time * remaining_steps
                print(f", ETA: {self._format_time(eta_seconds)}", end="")
        print(end="\r")

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

# ---------------------------------------------------------------------------
# Adapter utilities (LoRA via PEFT)
# ---------------------------------------------------------------------------

class LoRAAdapterWrapper:
    """Wraps a PEFT LoRA adapter to expose flat/assignable parameter vectors."""

    def __init__(self, model: PeftModel, adapter_name: str):
        self.model = model
        self.name = adapter_name
        self.params: List[nn.Parameter] = [
            p for p in model.parameters() if p.requires_grad and p.ndimension() > 0
        ]
        self._template = [p.detach().clone() for p in self.params]
        self._numel = sum(p.numel() for p in self.params)

    # -------- vector interface ----------
    def get_flat(self) -> torch.Tensor:
        return _flatten_tensors(self.params)

    def set_flat_(self, vec: torch.Tensor) -> None:
        assert vec.numel() == self._numel, "Vector length mismatch"
        new_tensors = _unflatten_to_tensors(vec, self.params)
        for p, new in zip(self.params, new_tensors):
            p.data.copy_(new.to(p.device))

    # -------- cloning ----------
    def clone_vector(self) -> torch.Tensor:
        """Return a *detached* copy of the current weight vector on CPU."""
        return self.get_flat().clone().cpu()

    # -------- statistics ----------
    @property
    def num_parameters(self) -> int:
        return self._numel


# ---------------------------------------------------------------------------
# EvoSRAFS core
# ---------------------------------------------------------------------------

@dataclass
class EvoConfig:
    sigma: float = 1e-2
    pop_size: int = 16
    lr_initial: float = 0.2      # Initial learning rate
    lr_min: float = 1e-3         # Minimum learning rate
    lr_max: float = 0.5          # Maximum learning rate
    lr_decay_factor: float = 0.9 # Decay factor
    lr_increase_factor: float = 1.05 # Increase factor on improvement
    evaluation_interval: int = 10 # Generations between LR evaluations
    improvement_threshold: float = 1e-3 # Improvement needed to reset/increase LR
    max_steps: int = 128
    generations: int = 100
    device: str = "cuda"

class EvoSRAFS:
    """Minimal Evolutionary Strategy trainer for LoRA adapters."""

    def __init__(
        self,
        base_model: PeftModel,
        tokenizer: AutoTokenizer,
        envs: AsyncVectorEnv,
        env_count: int,
        adapter_name: str,
        config: EvoConfig,
        prompt_fn: Callable[[gym.Env, np.ndarray], str],
        action_fn: Callable[[str], int],
    ):
        self.model = base_model.to(config.device)
        self.tokenizer = tokenizer
        self.envs = envs
        self.env_count = env_count
        self.cfg = config
        self.prompt_fn = prompt_fn
        self.action_fn = action_fn

        # Ensure exactly one active adapter is trainable; freeze everything else
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.set_adapter(adapter_name)
        for p in self.model.parameters():
            if p.ndimension() > 0 and p.requires_grad:
                p.requires_grad = True  # LoRA params

        self.adapter = LoRAAdapterWrapper(self.model, adapter_name)
        self.theta: torch.Tensor = self.adapter.clone_vector()  # master weights (CPU)

        # --- NEW: global-best tracker -------------------------------------------------
        self.best_reward_overall: float = -float("inf")
        self.best_theta: torch.Tensor = self.theta.clone()

        self._eps_buf = torch.zeros((self.cfg.pop_size,
                                     self.adapter.num_parameters),
                                    dtype=torch.float32)

        # Pre‑allocate noise buffer on CPU for efficiency
        self._eps_buf = torch.zeros((self.cfg.pop_size, self.adapter.num_parameters), dtype=torch.float32)

    # ---------------------------------------------------------------------
    # Evolution loop
    # ---------------------------------------------------------------------

    def _sample_noise(self) -> torch.Tensor:
        """Return Gaussian N(0, sigma) noise matrix of shape (pop, dim)."""
        return torch.randn_like(self._eps_buf) * self.cfg.sigma

    def _evaluate_vector(self, vec: torch.Tensor) -> float:
        """Set adapter weights to *vec*, roll out batched envs, return mean reward."""
        self.adapter.set_flat_(vec.to(self.cfg.device))

        # Completely tear out this fucking wrapper code
        obs, infos = envs.reset()

        #batch_size = obs.shape[0]  # number of parallel envs in this vector
        dones = torch.zeros(self.env_count, dtype=torch.bool)
        totals = torch.zeros(self.env_count, dtype=torch.float32)
        steps = torch.zeros(self.env_count, dtype=torch.int)

        while not dones.all() and steps.max().item() < self.cfg.max_steps:
            prompts = [
                prompt_fn(tuple(arr[i] for arr in obs))
                for i in range(self.env_count)
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.cfg.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8)
            act_texts = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            actions = [self.action_fn(text) for text in act_texts]

            # Step only unfinished environments, keep finished ones static
            # Vector envs: pass full action array, get batch results
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            # Only update unfinished envs
            mask = ~dones
            totals[mask] += torch.as_tensor(rewards)[mask]
            steps[mask] += 1
            dones |= torch.as_tensor(terminations) | torch.as_tensor(truncations)
            obs = next_obs

        return totals.mean().item()

    def _fitness_shaping(self, rewards: np.ndarray) -> np.ndarray:
        """Transform raw rewards into centred, rank‑based utility weights."""
        ranks = rewards.argsort().argsort()  # 0..pop‑1 (ascending)
        # Utilities from Salimans et al. 2017 (OpenAI‑ES)
        util = np.maximum(0, math.log(self.cfg.pop_size / 2 + 1) - np.log(self.cfg.pop_size - ranks))
        util -= util.mean()  # centre
        util /= util.std() + 1e-8
        return util.astype(np.float32)

    def train(self, callback: Optional[Callable[[int, float], None]] = None):
        printer = TrainingProgressPrinter(total_steps=self.cfg.generations * self.cfg.pop_size)
        printer.start()
        run = wandb.init()

        lr = self.cfg.lr_initial
        last_evaluation_best = -float("inf")

        for gen in range(self.cfg.generations):
            eps = self._sample_noise()
            rewards = np.zeros(self.cfg.pop_size, np.float32)

            for i in range(self.cfg.pop_size):
                cand_vec = self.theta + eps[i]
                rewards[i] = self._evaluate_vector(cand_vec)
                printer.update(rewards[i])

                if rewards[i] > self.best_reward_overall:
                    self.best_reward_overall = float(rewards[i])
                    self.best_theta = cand_vec.clone()

            keep_mask = rewards >= self.best_reward_overall
            if not keep_mask.any():
                keep_mask[:] = True

            kept_rewards = rewards[keep_mask]
            kept_eps = eps[keep_mask]

            utilities = self._fitness_shaping(kept_rewards)
            grad_est = (torch.from_numpy(utilities).unsqueeze(1) * kept_eps).mean(dim=0)
            self.theta += (lr / self.cfg.sigma) * grad_est

            avg_reward = float(rewards.mean())
            best_reward = float(rewards.max())

            # Adaptive Learning Rate Update
            if gen % self.cfg.evaluation_interval == 0 and gen > 0:
                improvement = self.best_reward_overall - last_evaluation_best
                if improvement < self.cfg.improvement_threshold:
                    lr = max(lr * self.cfg.lr_decay_factor, self.cfg.lr_min)
                else:
                    lr = min(lr * self.cfg.lr_increase_factor, self.cfg.lr_max)
                last_evaluation_best = self.best_reward_overall

            # Logging current learning rate
            run.log({"generation": gen,
                     "best_reward_gen": best_reward,
                     "avg_reward_gen": avg_reward,
                     "best_reward_global": self.best_reward_overall,
                     "current_lr": lr})

            if callback is not None:
                callback(gen, best_reward)
            else:
                print(f"\nGen {gen:04d} | best {best_reward:.3f} | avg {avg_reward:.3f} | "
                      f"global best {self.best_reward_overall:.3f} | lr {lr:.5f}")


        # after training, load final weights
        self.adapter.set_flat_(self.theta.to(self.cfg.device))

# ---------------------------------------------------------------------------
# Convenience: loading a quantised frozen base model
# ---------------------------------------------------------------------------

def load_bnb_lm(model_name: str, device: str = "cuda") -> Tuple[PeftModel, AutoTokenizer]:
    """Load a causal LM with 4‑bit bitsandbytes quantisation & prepare for LoRA."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    # Add a *single* empty LoRA adapter that we will evolve
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["k_proj","o_proj","q_proj","v_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    model.add_adapter("evo", lora_cfg)
    model.set_adapter("evo")
    # Explicitly set PAD to suppress automatic HuggingFace printouts
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok

# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ENV_NAME = "Sokoban-v2"
    BASE_MODEL = "/home/cris/Documents/text-generation-webui/models/huihui-ai_Llama-3.2-1B-Instruct-abliterated"  # Tiny LM for demonstration


    pool_size = 32
    def make_env():
        return gym.make('Blackjack-v1', natural=False, sab=False)


    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(pool_size)])


    #envs = gym.vector.AsyncVectorEnv([make_env for _ in range(pool_size)])
    #env_pool = AsyncEnvPool(make_env, pool_size=4)

    model, tok = load_bnb_lm(BASE_MODEL)

    def prompt_fn(obs):
        return f"""You are playing blackjack. Your current hand has a sum of {obs[0]}, the dealer's showing card is a {obs[1]}, and you {"don't" if not obs[2] else ''} have an ace. What action (stick or hit) do you take?"""

    def action_fn(txt):
        target_txt = txt.lower()
        mapping = {"stick": 0, "hit": 1}
        for key, value in mapping.items():
            if key in target_txt:
                return value
        return 0


    cfg = EvoConfig(pop_size=32, generations=100, sigma=0.05, lr_initial=0.1, lr_min=1e-6, evaluation_interval=1, improvement_threshold=1, max_steps=200)
    trainer = EvoSRAFS(model, tok, envs, pool_size, "evo", cfg, prompt_fn, action_fn)
    trainer.train()

    save_directory = "./LoRA"
    model.save_pretrained(save_directory)
    # The trained LoRA weights are now stored in `model` (adapter "evo") and `trainer.theta`.
    # Save if desired:
    # model.save_pretrained("./evo_cartpole_lora")

