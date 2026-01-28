"""
Advantage Actor-Critic (A2C) agent for EVRP.

Implements synchronous actor-critic with:
- Shared encoder for policy and value networks
- Parallel environment rollouts
- Advantage estimation: A = R + gamma * V(s') - V(s)
- Action masking for invalid actions
"""

from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from .base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared encoder.
    
    Architecture:
    - Shared encoder (GAT/MLP) produces node embeddings and graph embedding
    - Actor head: graph_embed + current_node_embed -> action logits
    - Critic head: graph_embed -> state value
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Initialize actor-critic network.
        
        Args:
            encoder: Shared encoder module
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        embed_dim = encoder.get_embed_dim()
        
        # Actor head: produces action logits with proper initialization
        actor_layer1 = nn.Linear(embed_dim * 2 + 2, hidden_dim)
        nn.init.orthogonal_(actor_layer1.weight, gain=np.sqrt(2))
        nn.init.zeros_(actor_layer1.bias)
        
        actor_layer2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(actor_layer2.weight, gain=np.sqrt(2))
        nn.init.zeros_(actor_layer2.bias)
        
        actor_layer3 = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(actor_layer3.weight, gain=0.01)  # Small gain for output
        nn.init.zeros_(actor_layer3.bias)
        
        self.actor = nn.Sequential(
            actor_layer1,
            nn.Tanh(),
            actor_layer2,
            nn.Tanh(),
            actor_layer3,
        )
        
        # Critic head: produces state value with very small initialization
        critic_layer1 = nn.Linear(embed_dim + 2, hidden_dim)
        nn.init.orthogonal_(critic_layer1.weight, gain=np.sqrt(2))
        nn.init.zeros_(critic_layer1.bias)
        
        critic_layer2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(critic_layer2.weight, gain=np.sqrt(2))
        nn.init.zeros_(critic_layer2.bias)
        
        critic_layer3 = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(critic_layer3.weight, gain=0.01)  # Very small gain for value output
        nn.init.zeros_(critic_layer3.bias)
        
        self.critic = nn.Sequential(
            critic_layer1,
            nn.Tanh(),
            critic_layer2,
            nn.Tanh(),
            critic_layer3,
        )
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        current_node: torch.Tensor,
        current_battery: torch.Tensor,
        current_cargo: torch.Tensor,
        valid_actions_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.
        
        Args:
            graph_data: Graph information dictionary
            current_node: Current node indices (batch,)
            current_battery: Current battery levels (batch,)
            current_cargo: Current cargo levels (batch,)
            valid_actions_mask: Mask of valid actions (batch, action_dim)
        
        Returns:
            Tuple of (action_logits, state_values)
        """
        # Encode graph
        node_embeddings, graph_embedding = self.encoder(graph_data)
        
        batch_size = graph_embedding.shape[0]
        
        # Get current node embeddings
        current_node_embeds = node_embeddings[
            torch.arange(batch_size, device=node_embeddings.device),
            current_node
        ]
        
        # Ensure scalar tensors have correct shape
        if current_battery.dim() == 0:
            current_battery = current_battery.unsqueeze(0)
        if current_cargo.dim() == 0:
            current_cargo = current_cargo.unsqueeze(0)
        
        # Actor: concatenate graph, current node, battery, cargo
        actor_input = torch.cat([
            graph_embedding,
            current_node_embeds,
            current_battery.unsqueeze(-1) if current_battery.dim() == 1 else current_battery,
            current_cargo.unsqueeze(-1) if current_cargo.dim() == 1 else current_cargo,
        ], dim=-1)
        
        action_logits = self.actor(actor_input)
        
        # Apply action masking: set invalid actions to -inf
        if valid_actions_mask is not None:
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        # Critic: concatenate graph, battery, cargo
        critic_input = torch.cat([
            graph_embedding,
            current_battery.unsqueeze(-1) if current_battery.dim() == 1 else current_battery,
            current_cargo.unsqueeze(-1) if current_cargo.dim() == 1 else current_cargo,
        ], dim=-1)
        
        state_values = self.critic(critic_input).squeeze(-1)
        
        return action_logits, state_values


class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) agent.
    
    Features:
    - Synchronous updates with advantage estimation
    - Shared encoder between actor and critic
    - Action masking for invalid actions
    - Multiple parallel environments support
    - Entropy regularization for exploration
    
    Algorithm:
    1. Collect rollouts from parallel environments
    2. Compute returns and advantages: A = R + gamma * V(s') - V(s)
    3. Update actor: maximize E[log pi(a|s) * A]
    4. Update critic: minimize MSE(V(s), returns)
    5. Add entropy bonus for exploration
    
    Configuration:
        lr: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        entropy_coef: Entropy coefficient (default: 0.01)
        value_loss_coef: Value loss coefficient (default: 0.5)
        max_grad_norm: Max gradient norm for clipping (default: 0.5)
        n_steps: Number of steps per rollout (default: 5)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        config: Dict[str, Any],
    ):
        """Initialize A2C agent."""
        super().__init__(encoder, action_dim, config)
        
        # Extract hyperparameters
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.n_steps = config.get('n_steps', 5)
        
        # Create actor-critic network
        hidden_dim = config.get('hidden_dim', 256)
        self.ac_network = ActorCriticNetwork(encoder, action_dim, hidden_dim)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=self.lr)
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
    
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using current policy.
        
        Args:
            observation: Environment observation
            deterministic: If True, select argmax action
        
        Returns:
            Tuple of (action, info_dict)
        """
        self.ac_network.eval()
        
        with torch.no_grad():
            # Prepare observation
            obs_tensor = self._prepare_observation(observation)
            
            # Get action logits and value
            action_logits, state_value = self.ac_network(
                {k: v for k, v in obs_tensor.items() if k in ['node_coords', 'node_demands', 'node_types', 'distance_matrix']},
                obs_tensor['current_node'],
                obs_tensor['current_battery'],
                obs_tensor['current_cargo'],
                obs_tensor['valid_actions_mask'],
            )
            
            # Sample or select max action
            if deterministic:
                action = action_logits.argmax(dim=-1).item()
                log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
            else:
                # Add numerical stability to softmax
                probs = F.softmax(action_logits, dim=-1)
                # Ensure probs are valid (no NaN or Inf)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Fallback to uniform distribution over valid actions
                    probs = obs_tensor['valid_actions_mask'].float()
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                dist = Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action], device=action_logits.device)).item()
            
            return action, {
                'log_prob': log_prob,
                'value': state_value.item(),
                'entropy': -(probs * probs.log()).sum().item(),
            }
    
    def update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Update agent from rollout batch.
        
        Args:
            batch: Dictionary with:
                - observations: List of observations
                - actions: List of actions
                - rewards: List of rewards
                - next_observations: List of next observations
                - dones: List of done flags
                - log_probs: List of action log probabilities
                - values: List of state values
        
        Returns:
            Dictionary of training metrics
        """
        self.ac_network.train()
        
        # Convert to tensors
        observations = batch['observations']
        actions = torch.tensor(batch['actions'], dtype=torch.long)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
        dones = torch.tensor(batch['dones'], dtype=torch.float32)
        
        device = next(self.parameters()).device
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        # Prepare batched observations
        batch_size = len(observations)
        
        # Debug: check shapes
        if batch_size > 1:
            shapes = [obs['node_coords'].shape for obs in observations]
            if not all(s == shapes[0] for s in shapes):
                print(f"Warning: Inconsistent observation shapes: {shapes}")
        
        graph_data = {
            'node_coords': torch.stack([torch.from_numpy(obs['node_coords']).float() for obs in observations]).to(device),
            'node_demands': torch.stack([torch.from_numpy(obs['node_demands']).float() for obs in observations]).to(device),
            'node_types': torch.stack([torch.from_numpy(obs['node_types']).float() for obs in observations]).to(device),
            'distance_matrix': torch.stack([torch.from_numpy(obs['distance_matrix']).float() for obs in observations]).to(device),
        }
        
        # Check for NaN/Inf in inputs
        for key, val in graph_data.items():
            if torch.isnan(val).any() or torch.isinf(val).any():
                print(f"Warning: NaN/Inf in input {key}")
                print(f"  Shape: {val.shape}, NaN count: {torch.isnan(val).sum()}, Inf count: {torch.isinf(val).sum()}")
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'entropy': 0.0,
                    'total_loss': 0.0,
                    'mean_value': 0.0,
                    'mean_advantage': 0.0,
                }
        
        current_nodes = torch.tensor([obs['current_node'] if isinstance(obs['current_node'], (int, np.integer)) else obs['current_node'].item() for obs in observations], dtype=torch.long, device=device)
        current_batteries = torch.tensor([obs['current_battery'] if isinstance(obs['current_battery'], (float, int, np.floating, np.integer)) else obs['current_battery'].item() for obs in observations], dtype=torch.float, device=device)
        current_cargos = torch.tensor([obs['current_cargo'] if isinstance(obs['current_cargo'], (float, int, np.floating, np.integer)) else obs['current_cargo'].item() for obs in observations], dtype=torch.float, device=device)
        valid_masks = torch.stack([torch.from_numpy(obs['valid_actions_mask']).bool() for obs in observations]).to(device)
        
        # Forward pass
        action_logits, state_values = self.ac_network(
            graph_data,
            current_nodes,
            current_batteries,
            current_cargos,
            valid_masks,
        )
        
        # Clamp state values to prevent extreme gradients
        state_values = torch.clamp(state_values, min=-1000, max=1000)
        
        # Compute log probabilities and entropy
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Check for -inf in action log probs (happens when action was masked)
        if torch.isinf(action_log_probs).any():
            print(f"Warning: -inf in action_log_probs (invalid action was taken)")
            print(f"  actions: {actions}")
            print(f"  action_logits: {action_logits[torch.isinf(action_log_probs)]}")
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'mean_value': 0.0,
                'mean_advantage': 0.0,
            }
        
        # Compute entropy safely: avoid 0 * -inf = NaN
        # Clamp log_probs to avoid -inf before multiplication
        log_probs_safe = torch.clamp(log_probs, min=-20.0)  # e^(-20) ≈ 2e-9, effectively zero
        entropy = -(probs * log_probs_safe).sum(dim=-1).mean()
        
        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute n-step returns
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else state_values[t].detach()
                returns[t] = rewards[t] + self.gamma * next_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
            
            advantages[t] = returns[t] - state_values[t].detach()
        
        # Normalize advantages (only if we have multiple samples and variance is sufficient)
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-4:  # Only normalize if there's sufficient variance
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-6)
            else:
                # If variance is too small, just center
                advantages = advantages - advantages.mean()
        else:
            # For single sample, just center at 0
            advantages = advantages - advantages.mean()
        
        # Actor loss: -E[log pi(a|s) * A]
        actor_loss = -(action_log_probs * advantages).mean()
        
        # Critic loss: Use Huber loss for robustness to outliers
        critic_loss = F.smooth_l1_loss(state_values, returns)
        
        # Total loss
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Check for NaN in loss components before backward
        if torch.isnan(loss) or torch.isnan(actor_loss) or torch.isnan(critic_loss):
            print(f"Warning: NaN in loss computation")
            print(f"  actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy: {entropy}")
            print(f"  action_log_probs: {action_log_probs}")
            print(f"  advantages: {advantages}")
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'mean_value': 0.0,
                'mean_advantage': 0.0,
            }
        
        # Optimize
        self.optimizer.zero_grad()
        
        # Check for NaN in parameters before backward
        for name, param in self.ac_network.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: NaN/Inf in parameter {name} BEFORE backward")
                return {
                    'actor_loss': 0.0,
                    'critic_loss': 0.0,
                    'entropy': 0.0,
                    'total_loss': 0.0,
                    'mean_value': 0.0,
                    'mean_advantage': 0.0,
                }
        
        loss.backward()
        
        # Check for NaN in gradients
        has_nan_grad = False
        for name, param in self.ac_network.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                nan_count = torch.isnan(param.grad).sum().item()
                inf_count = torch.isinf(param.grad).sum().item()
                print(f"Warning: Bad gradient in {name}: {nan_count} NaN, {inf_count} Inf out of {param.grad.numel()}")
                has_nan_grad = True
                
        if has_nan_grad:
            print("Skipping update due to NaN/Inf gradients")
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'mean_value': 0.0,
                'mean_advantage': 0.0,
            }
        
        nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Track metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'mean_value': state_values.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy.item())
        
        return metrics
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
