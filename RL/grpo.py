import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    num_epochs: int = 100
    num_steps_per_epoch: int = 10
    num_responses: int = 10
    learning_rate: float = 1e-3
    epsilon: float = 0.01  # For clipping ratio
    kl_penalty: float = 0.0
    compute_ref_model_period: int = 10
    temperature: float = 1.0  # Temperature for softmax
    entropy_bonus: float = 0.01  # Coefficient for entropy regularization
    init_std: float = 0.1  # Standard deviation for weight initialization


class MLPPolicy(nn.Module):
    """MLP policy with standard Linear layers and GELU activation."""
    def __init__(self, vocab_size: int, embedding_dim: int, prompt_length: int, response_length: int, init_std: float = 0.1):
        super().__init__()
        # Store initialization arguments
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.prompt_length = prompt_length
        self.response_length = response_length

        # Initialize embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=init_std)

        # MLP layers for encoding
        self.encode_fc = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.decode_fc = nn.Linear(4 * embedding_dim, embedding_dim)

        # Initialize weights
        nn.init.normal_(self.encode_fc.weight, std=init_std / math.sqrt(embedding_dim))
        nn.init.normal_(self.decode_fc.weight, std=init_std / math.sqrt(4 * embedding_dim))
        nn.init.zeros_(self.encode_fc.bias)
        nn.init.zeros_(self.decode_fc.bias)

    def get_init_args(self):
        """Return the arguments needed to create a copy of this policy."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'prompt_length': self.prompt_length,
            'response_length': self.response_length
        }

    def forward(self, prompts: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            prompts: int[batch pos]
            temperature: float, temperature for softmax
        Returns:
            logits: float[batch pos vocab]
        """
        batch_size = prompts.shape[0]

        # Embed the prompts: [batch pos dim]
        embeddings = self.embedding(prompts)


        # Apply MLP
        hidden = self.encode_fc(embeddings)  # [batch pos (4*dim)]
        hidden = self.gelu(hidden)
        decoded = self.decode_fc(hidden)  # [batch pos dim)]

        # Convert to logits with temperature [batch pos vocab]
        logits = (decoded @ self.embedding.weight.T) / temperature
        return logits


class GRPO:
    """Group Relative Policy Optimization implementation."""
    def __init__(
        self,
        policy: nn.Module,
        config: GRPOConfig,
        reward_fn: Callable[[List[int], List[int]], float]
    ):
        self.policy = policy
        self.config = config
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    def generate_responses(self, prompts: torch.Tensor) -> torch.Tensor:
        """Generate multiple responses for each prompt."""
        logits = self.policy(prompts, temperature=self.config.temperature)  # [batch pos vocab]
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape for sampling
        flattened_logits = logits.view(-1, vocab_size)  # [(batch*pos) vocab]
        flattened_responses = torch.multinomial(
            F.softmax(flattened_logits, dim=-1),
            num_samples=self.config.num_responses,
            replacement=True
        )  # [(batch*pos) trial]

        # Reshape back to original dimensions
        responses = flattened_responses.view(batch_size, seq_len, self.config.num_responses)
        responses = responses.permute(0, 2, 1)  # [batch trial pos]
        return responses

    def compute_log_probs(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        policy: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Compute log probabilities of responses under the policy."""
        if policy is None:
            policy = self.policy

        logits = policy(prompts, temperature=self.config.temperature)  # [batch pos vocab]
        log_probs = F.log_softmax(logits, dim=-1)  # [batch pos vocab]

        # Expand log_probs for each response
        batch_size, seq_len, vocab_size = log_probs.shape
        num_responses = responses.shape[1] # [batch trial pos]
        log_probs = log_probs.unsqueeze(1).expand(-1, num_responses, -1, -1) # [batch trial pos vocab]

        # Gather log probs using responses
        responses_expanded = responses.unsqueeze(-1) # [batch trial pos 1]
        log_probs = log_probs.gather(dim=-1, index=responses_expanded).squeeze(-1) # [batch trial pos]

        # Add small epsilon to avoid log(0)
        log_probs = torch.clamp(log_probs, min=-100, max=100)
        return log_probs

    def compute_rewards(self, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Compute rewards for each response."""
        batch_size, num_responses, _ = responses.shape
        rewards = torch.empty(batch_size, num_responses, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(num_responses):
                rewards[i, j] = self.reward_fn(prompts[i].tolist(), responses[i, j].tolist())
        return rewards

    def compute_centered_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute centered advantages (rewards - baseline).

        Args:
            rewards: [batch trial]
        Returns:
            advantages: [batch trial]
        """
        # Compute baseline per batch
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # [batch 1]
        advantages = rewards - mean_rewards

        # Skip normalization if std is too small
        std = advantages.std()
        if std > 1e-6:
            advantages = advantages / (std + 1e-8)

        return advantages

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy."""
        probs = F.softmax(logits, dim=-1) # [batch pos vocab]
        log_probs = F.log_softmax(logits, dim=-1) # [batch pos vocab]
        entropy = -(probs * log_probs).sum(dim=-1).mean() # [batch]
        return entropy

    def compute_loss(
        self,
        prompts: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GRPO loss with optional clipping and entropy bonus.

        Args:
            prompts: [batch pos] input prompts
            log_probs: [batch trial pos] log probabilities
            advantages: [batch trial] advantage values
            old_log_probs: Optional[batch trial pos] old policy log probs
        Returns:
            total_loss: combined policy and entropy loss
            entropy: policy entropy
        """
        # Sum log_probs over positions to get sequence-level log probs
        seq_log_probs = log_probs.sum(dim=-1)  # [batch trial]

        if old_log_probs is None:
            # Standard policy gradient loss
            weighted_probs = seq_log_probs * advantages
            loss = -weighted_probs.mean()
        else:
            # GRPO loss with clipping
            old_seq_log_probs = old_log_probs.sum(dim=-1)  # [batch trial]
            ratios = torch.exp(seq_log_probs - old_seq_log_probs)  # [batch trial]

            # Clip ratios to prevent numerical instability
            ratios = torch.clamp(ratios, 0.1, 10.0) # [batch trial]

            surr1 = ratios * advantages # [batch trial]
            surr2 = torch.clamp(
                ratios,
                1 - self.config.epsilon,
                1 + self.config.epsilon
            ) * advantages # [batch trial]

            loss = -torch.min(surr1, surr2).mean() # [batch]

        # Add entropy bonus
        logits = self.policy(prompts, temperature=self.config.temperature) # [batch pos vocab]
        entropy = self.compute_entropy(logits) # [batch]
        entropy_loss = -self.config.entropy_bonus * entropy

        total_loss = loss + entropy_loss
        return total_loss, entropy

    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,    # [batch trial pos]
        ref_log_probs: torch.Tensor # [batch trial pos]
    ) -> torch.Tensor:
        """
        Compute an estimate of KL(model | ref_model), where the models are given by:
            log_probs [batch trial pos]
            ref_log_probs [batch trial pos]
        Use the estimate:
            KL(p || q) = E_p[q/p - log(q/p) - 1]
        """
        return (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum(dim=-1).mean()

    def train_step(
        self,
        prompts: torch.Tensor,
        ref_policy: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one training step."""
        # Generate responses and compute rewards
        responses = self.generate_responses(prompts) # [batch trial pos]
        rewards = self.compute_rewards(prompts, responses)
        advantages = self.compute_centered_advantages(rewards)

        # Compute current log probabilities
        log_probs = self.compute_log_probs(prompts, responses) # [batch trial pos]

        # Compute old log probabilities for GRPO
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(prompts, responses, self.policy)

        # Compute loss
        loss, entropy = self.compute_loss(prompts, log_probs, advantages, old_log_probs)

        # Add KL penalty if reference policy is provided
        if ref_policy is not None and self.config.kl_penalty > 0:
            with torch.no_grad():
                # Get log probabilities from both policies for the sampled sequences
                ref_log_probs = self.compute_log_probs(prompts, responses, ref_policy)  # [batch trial pos]

                # Compute KL penalty between sampled sequences
                kl_penalty = self.compute_kl_penalty(log_probs, ref_log_probs)
                loss += self.config.kl_penalty * kl_penalty

        # Update policy with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss, rewards.mean()

    def train(self, prompts: torch.Tensor) -> List[dict]:
        """Train the policy using GRPO."""
        records = []
        ref_policy = None

        for epoch in range(self.config.num_epochs):
            # Update reference policy for KL penalty
            if (self.config.kl_penalty > 0 and
                epoch % self.config.compute_ref_model_period == 0):
                # Create new reference policy using stored init args
                ref_policy = MLPPolicy(**self.policy.get_init_args())
                ref_policy.load_state_dict(self.policy.state_dict())

            # Perform training steps
            for step in range(self.config.num_steps_per_epoch):
                loss, mean_reward = self.train_step(prompts, ref_policy)

                # Record metrics
                records.append({
                    "epoch": epoch,
                    "step": epoch * self.config.num_steps_per_epoch + step,
                    "loss": loss.item(),
                    "mean_reward": mean_reward.item()
                })

                if step % 10 == 0:  # Print every 10 steps
                    print(f"Epoch {epoch}, Step {step}: Loss = {loss:.6f}, Mean Reward = {mean_reward:.4f}")

        return records



def sort_inclusion_ordering_reward(prompt: List[int], response: List[int]) -> float:
    """Reward function that considers both inclusion and ordering with stronger penalties."""
    # Count matching numbers (inclusion)
    prompt_set = set(prompt)
    response_set = set(response)
    matching_numbers = len(prompt_set.intersection(response_set))
    ordering_reward = sum(1 for x, y in zip(response, response[1:]) if x <= y)

    return matching_numbers + ordering_reward



def run_grpo_example():
    """Run a simple GRPO example with longer sequences."""
    # Create more diverse sample data with longer sequences
    prompts = torch.tensor([
        [5, 2, 8, 1, 15, 7, 11, 3, 9, 4],     # Mixed sequence 1
        [12, 6, 1, 14, 3, 9, 4, 10, 7, 2],    # Mixed sequence 2
        [8, 15, 3, 5, 11, 1, 13, 6, 9, 4],    # Mixed sequence 3
        [7, 2, 10, 14, 4, 8, 1, 12, 5, 3],    # Mixed sequence 4
        [20, 16, 25, 18, 22, 17, 23, 19, 21, 24],  # Higher numbers
        [30, 28, 26, 29, 27, 31, 25, 32, 24, 33],  # Even higher numbers
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],      # Already sorted ascending
        [35, 34, 33, 32, 31, 30, 29, 28, 27, 26],  # Already sorted descending
        [15, 15, 12, 12, 18, 18, 20, 20, 25, 25],  # Repeated numbers
        [1, 5, 2, 6, 3, 7, 4, 8, 5, 9],       # Alternating pattern
        [40, 2, 38, 4, 36, 6, 34, 8, 32, 10],  # Large gaps
        [21, 21, 21, 22, 22, 22, 23, 23, 23, 24],  # Blocks of repeats
    ])

    vocab_size = 50  # Increased vocabulary size
    prompt_length = response_length = prompts.shape[1]  # Still 10

    # Create policy and GRPO trainer with adjusted hyperparameters
    policy = MLPPolicy(
        vocab_size=vocab_size,
        embedding_dim=128,  # Increased for larger vocabulary
        prompt_length=prompt_length,
        response_length=response_length,
        init_std=0.1
    )

    config = GRPOConfig(
        num_epochs=30,  # More epochs for more data
        num_steps_per_epoch=20,  # More steps per epoch
        num_responses=10,  # More responses for better exploration
        learning_rate=3e-4,
        epsilon=0.2,
        kl_penalty=0.01,
        temperature=1.0,  # Higher temperature for more exploration
        entropy_bonus=0.1,  # Increased entropy bonus
        init_std=0.1
    )

    grpo = GRPO(
        policy=policy,
        config=config,
        reward_fn=sort_inclusion_ordering_reward
    )

    # Train the policy
    records = grpo.train(prompts)

    # Print final results
    print("\nTraining completed!")
    print(f"Final mean reward: {records[-1]['mean_reward']:.4f}")

    # Print example outputs for different types of inputs
    with torch.no_grad():
        test_prompts = torch.tensor([
            [5, 2, 8, 1, 15, 7, 11, 3, 9, 4],     # Original sequence
            [30, 28, 26, 29, 27, 31, 25, 32, 24, 33],  # Higher numbers
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],      # Already sorted
            [40, 2, 38, 4, 36, 6, 34, 8, 32, 10],  # Large gaps
        ])
        responses = grpo.generate_responses(test_prompts)
        rewards = grpo.compute_rewards(test_prompts, responses)

        print("\nExample outputs:")
        for i in range(len(test_prompts)):
            print(f"\nTest case {i+1}:")
            print(f"Input:     {test_prompts[i].tolist()}")
            print(f"Sorted:    {sorted(test_prompts[i].tolist())}")
            best_response_idx = rewards[i].argmax()
            print(f"Response:  {responses[i, best_response_idx].tolist()}")
            print(f"Reward:    {rewards[i, best_response_idx]:.2f}")


if __name__ == "__main__":
    run_grpo_example()
