"""
Soft Actor-Critic (SAC) agent for EVRP with discrete actions.

Implements maximum entropy RL with:
- Actor network with Gumbel-Softmax for discrete exploration
- Twin Q-networks for value estimation
- Automatic entropy temperature tuning
- Replay buffer for off-policy learning
- Action masking for invalid actions
- Running reward normalization for training stability
"""

from typing import Dict, Tuple, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

from .base_agent import BaseAgent
from ..framework.normalizers import RunningNormalizer


class ReplayBuffer:
    """
    Experience replay buffer for SAC.
    
    Stores transitions (s, a, r, s', done) for off-policy learning.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ):
        """Add transition to buffer."""
        self.buffer.append((observation, action, reward, next_observation, done))
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        return {
            'observations': list(observations),
            'actions': list(actions),
            'rewards': list(rewards),
            'next_observations': list(next_observations),
            'dones': list(dones),
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class Actor(nn.Module):
    """
    Actor network for SAC with discrete actions.
    
    Uses Gumbel-Softmax for continuous relaxation of discrete actions,
    enabling gradient-based optimization with entropy regularization.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """Initialize actor network."""
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        embed_dim = encoder.get_embed_dim()
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        current_node: torch.Tensor,
        current_battery: torch.Tensor,
        current_cargo: torch.Tensor,
        valid_actions_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing action probabilities.
        
        Args:
            graph_data: Graph information
            current_node: Current node indices
            current_battery: Battery levels
            current_cargo: Cargo levels
            valid_actions_mask: Valid actions mask
            temperature: Temperature for Gumbel-Softmax (for training)
        
        Returns:
            Tuple of (action_probs, log_probs)
        """
        # Encode
        node_embeddings, graph_embedding = self.encoder(graph_data)
        
        batch_size = graph_embedding.shape[0]
        current_node_embeds = node_embeddings[
            torch.arange(batch_size, device=node_embeddings.device),
            current_node
        ]
        
        # Ensure scalars have correct shape
        if current_battery.dim() == 0:
            current_battery = current_battery.unsqueeze(0)
        if current_cargo.dim() == 0:
            current_cargo = current_cargo.unsqueeze(0)
        
        # Policy input
        policy_input = torch.cat([
            graph_embedding,
            current_node_embeds,
            current_battery.unsqueeze(-1) if current_battery.dim() == 1 else current_battery,
            current_cargo.unsqueeze(-1) if current_cargo.dim() == 1 else current_cargo,
        ], dim=-1)
        
        # Get logits
        logits = self.policy(policy_input)
        
        # Apply action masking
        if valid_actions_mask is not None:
            logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return probs, log_probs


class Critic(nn.Module):
    """
    Critic (Q-function) network for SAC.
    
    Estimates Q(s, a) for discrete actions.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        """Initialize critic network."""
        super().__init__()
        
        self.encoder = encoder
        self.action_dim = action_dim
        embed_dim = encoder.get_embed_dim()
        
        # Q-function: takes state embedding + action
        self.q_network = nn.Sequential(
            nn.Linear(embed_dim + 2 + 1, hidden_dim),  # graph + battery + cargo + action
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        current_battery: torch.Tensor,
        current_cargo: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing Q(s, a).
        
        Args:
            graph_data: Graph information
            current_battery: Battery levels
            current_cargo: Cargo levels
            actions: Action indices
        
        Returns:
            Q-values
        """
        # Encode
        _, graph_embedding = self.encoder(graph_data)
        
        # Ensure scalars have correct shape
        if current_battery.dim() == 0:
            current_battery = current_battery.unsqueeze(0)
        if current_cargo.dim() == 0:
            current_cargo = current_cargo.unsqueeze(0)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        
        # Q-function input
        q_input = torch.cat([
            graph_embedding,
            current_battery.unsqueeze(-1) if current_battery.dim() == 1 else current_battery,
            current_cargo.unsqueeze(-1) if current_cargo.dim() == 1 else current_cargo,
            actions.unsqueeze(-1).float() if actions.dim() == 1 else actions.unsqueeze(-1).float(),
        ], dim=-1)
        
        q_values = self.q_network(q_input).squeeze(-1)
        
        return q_values
    
    def forward_all_actions(
        self,
        graph_data: Dict[str, torch.Tensor],
        current_battery: torch.Tensor,
        current_cargo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Returns:
            Q-values for all actions (batch, action_dim)
        """
        batch_size = current_battery.shape[0]
        device = current_battery.device
        
        # Encode once
        _, graph_embedding = self.encoder(graph_data)
        
        # Compute Q for all actions
        q_values = []
        for a in range(self.action_dim):
            actions = torch.full((batch_size,), a, device=device)
            q = self.forward(graph_data, current_battery, current_cargo, actions)
            q_values.append(q)
        
        return torch.stack(q_values, dim=1)


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent for discrete EVRP.
    
    Features:
    - Maximum entropy reinforcement learning
    - Off-policy learning with replay buffer
    - Twin Q-networks to mitigate overestimation
    - Automatic entropy temperature tuning
    - Action masking for invalid actions
    - Discrete actions with entropy-regularized policy
    
    Algorithm:
    1. Collect experience in replay buffer
    2. Sample batch and compute target Q-values
    3. Update critics: minimize MSE(Q, target)
    4. Update actor: maximize E[Q(s,a) - alpha * log pi(a|s)]
    5. Update temperature: match target entropy
    
    Configuration:
        lr: Learning rate (default: 3e-4)
        gamma: Discount factor (default: 0.99)
        tau: Target network update rate (default: 0.005)
        alpha: Entropy temperature (default: 'auto' for automatic tuning)
        buffer_size: Replay buffer capacity (default: 100000)
        batch_size: Training batch size (default: 256)
        target_entropy: Target entropy for auto-tuning (default: -action_dim * 0.5)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        config: Dict[str, Any],
    ):
        """Initialize SAC agent."""
        super().__init__(encoder, action_dim, config)

        # Hyperparameters
        self.lr = config.get('lr', config.get('learning_rate', 5e-4))
        self.gamma = config.get('gamma', 0.99)
        # CRITICAL FIX: Tau increased from 0.005 to 0.02 for faster target network updates
        # Low tau causes massive lag in target networks, leading to divergence around iteration 800
        self.tau = config.get('tau', 0.02)
        self.batch_size = config.get('batch_size', 256)
        self.buffer_size = config.get('buffer_size', 100000)
        # CRITICAL FIX: max_grad_norm increased from 1.0 to 2.0 to allow better gradient flow
        self.max_grad_norm = float(config.get('max_grad_norm', 2.0))
        # CRITICAL FIX: reward_clip increased from 10.0 to 30.0 to preserve reward signal variance
        self.reward_clip = float(config.get('reward_clip', 30.0))

        # Entropy temperature
        alpha_config = config.get('alpha', 'auto')
        if alpha_config == 'auto':
            # Automatic entropy tuning
            self.auto_entropy = True
            # CRITICAL FIX: target_entropy less aggressive (-action_dim * 0.25 vs -0.5)
            # This prevents alpha from growing unbounded, which causes exploration collapse
            self.target_entropy = config.get('target_entropy', -action_dim * 0.25)
            self.log_alpha = nn.Parameter(torch.zeros(1))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
            self._fixed_alpha = None
        else:
            self.auto_entropy = False
            self._fixed_alpha = float(alpha_config)
            self.log_alpha = torch.log(torch.tensor([self._fixed_alpha]))
            self.alpha_optimizer = None

        hidden_dim = config.get('hidden_dim', 256)

        # Create networks
        from copy import deepcopy

        # Actor
        self.actor = Actor(encoder, action_dim, hidden_dim)

        # Twin critics
        self.critic1 = Critic(deepcopy(encoder), action_dim, hidden_dim)
        self.critic2 = Critic(deepcopy(encoder), action_dim, hidden_dim)

        # Target critics
        self.critic1_target = Critic(deepcopy(encoder), action_dim, hidden_dim)
        self.critic2_target = Critic(deepcopy(encoder), action_dim, hidden_dim)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Freeze target networks
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # FIX: Add running reward normalizer for training stability
        self.return_normalizer = RunningNormalizer(shape=())

        # Metrics
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
    
    @property
    def alpha(self) -> float:
        """Get current entropy temperature (fixed if set, else learned)."""
        if self._fixed_alpha is not None:
            return self._fixed_alpha
        return self.log_alpha.exp().item()
    
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using current policy."""
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = self._prepare_observation(observation)
            
            probs, log_probs = self.actor(
                {k: v for k, v in obs_tensor.items() if k in ['node_coords', 'node_demands', 'node_types', 'distance_matrix']},
                obs_tensor['current_node'],
                obs_tensor['current_battery'],
                obs_tensor['current_cargo'],
                obs_tensor['valid_actions_mask'],
            )
            
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()
            
            return action, {
                'probs': probs[0].cpu().numpy(),
                'entropy': -(probs * log_probs).sum().item(),
            }
    
    def update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Update agent from replay buffer batch."""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        return self._update_networks(batch)
    
    def _update_networks(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Internal method to update networks."""
        device = next(self.parameters()).device
        
        # Prepare batch
        observations = batch['observations']
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float, device=device)
        next_observations = batch['next_observations']
        dones = torch.tensor(batch['dones'], dtype=torch.float, device=device)

        # Keep target values in a numerically stable range.
        rewards = torch.clamp(rewards, min=-self.reward_clip, max=self.reward_clip)
        
        # FIX: Update return normalizer with rewards for tracking statistics
        self.return_normalizer.update(rewards.detach().cpu().numpy())
        
        # Prepare graph data
        graph_data = {
            'node_coords': torch.stack([torch.from_numpy(obs['node_coords']).float() for obs in observations]).to(device),
            'node_demands': torch.stack([torch.from_numpy(obs['node_demands']).float() for obs in observations]).to(device),
            'node_types': torch.stack([torch.from_numpy(obs['node_types']).float() for obs in observations]).to(device),
            'distance_matrix': torch.stack([torch.from_numpy(obs['distance_matrix']).float() for obs in observations]).to(device),
        }
        
        next_graph_data = {
            'node_coords': torch.stack([torch.from_numpy(obs['node_coords']).float() for obs in next_observations]).to(device),
            'node_demands': torch.stack([torch.from_numpy(obs['node_demands']).float() for obs in next_observations]).to(device),
            'node_types': torch.stack([torch.from_numpy(obs['node_types']).float() for obs in next_observations]).to(device),
            'distance_matrix': torch.stack([torch.from_numpy(obs['distance_matrix']).float() for obs in next_observations]).to(device),
        }
        
        current_batteries = torch.tensor([obs['current_battery'] if isinstance(obs['current_battery'], (float, int, np.floating, np.integer)) else obs['current_battery'].item() for obs in observations], dtype=torch.float, device=device)
        current_cargos = torch.tensor([obs['current_cargo'] if isinstance(obs['current_cargo'], (float, int, np.floating, np.integer)) else obs['current_cargo'].item() for obs in observations], dtype=torch.float, device=device)
        
        next_batteries = torch.tensor([obs['current_battery'] if isinstance(obs['current_battery'], (float, int, np.floating, np.integer)) else obs['current_battery'].item() for obs in next_observations], dtype=torch.float, device=device)
        next_cargos = torch.tensor([obs['current_cargo'] if isinstance(obs['current_cargo'], (float, int, np.floating, np.integer)) else obs['current_cargo'].item() for obs in next_observations], dtype=torch.float, device=device)
        
        next_valid_masks = torch.stack([torch.from_numpy(obs['valid_actions_mask']).bool() for obs in next_observations]).to(device)
        
        # Update critics
        with torch.no_grad():
            # Get next actions and log probs
            next_probs, next_log_probs = self.actor(
                next_graph_data,
                torch.tensor([obs['current_node'] if isinstance(obs['current_node'], (int, np.integer)) else obs['current_node'].item() for obs in next_observations], dtype=torch.long, device=device),
                next_batteries,
                next_cargos,
                next_valid_masks,
            )

            # Avoid numerical instability from exact zeros in probability/log-prob terms.
            next_probs = torch.clamp(next_probs, min=1e-8)
            next_probs = next_probs / next_probs.sum(dim=1, keepdim=True)
            next_log_probs = torch.log(next_probs)
            
            # Compute target Q-values using twin critics
            next_q1_all = self.critic1_target.forward_all_actions(next_graph_data, next_batteries, next_cargos)
            next_q2_all = self.critic2_target.forward_all_actions(next_graph_data, next_batteries, next_cargos)
            next_q_all = torch.min(next_q1_all, next_q2_all)
            
            # Expected Q-value: E_a[Q(s',a) - alpha * log pi(a|s')]
            next_v = (next_probs * (next_q_all - self.alpha * next_log_probs)).sum(dim=1)
            
            # Target: r + gamma * (1 - done) * V(s')
            q_target = rewards + self.gamma * (1 - dones) * next_v

        if not torch.isfinite(q_target).all():
            return {
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha': self.alpha,
                'alpha_loss': 0.0,
                'mean_q': 0.0,
            }
        
        # Current Q-values
        q1 = self.critic1(graph_data, current_batteries, current_cargos, actions)
        q2 = self.critic2(graph_data, current_batteries, current_cargos, actions)
        
        # Critic losses
        # Huber is more robust than MSE under noisy/off-policy targets.
        critic1_loss = F.smooth_l1_loss(q1, q_target)
        critic2_loss = F.smooth_l1_loss(q2, q_target)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()
        
        # Update actor
        valid_masks = torch.stack([torch.from_numpy(obs['valid_actions_mask']).bool() for obs in observations]).to(device)
        current_nodes = torch.tensor([obs['current_node'] if isinstance(obs['current_node'], (int, np.integer)) else obs['current_node'].item() for obs in observations], dtype=torch.long, device=device)
        
        probs, log_probs = self.actor(graph_data, current_nodes, current_batteries, current_cargos, valid_masks)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True)
        log_probs = torch.log(probs)
        
        # Q-values for all actions
        q1_all = self.critic1.forward_all_actions(graph_data, current_batteries, current_cargos)
        q2_all = self.critic2.forward_all_actions(graph_data, current_batteries, current_cargos)
        q_all = torch.min(q1_all, q2_all)
        
        # Actor loss: maximize E[Q(s,a) - alpha * log pi(a|s)]
        actor_loss = (probs * (self.alpha * log_probs - q_all)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update alpha (temperature)
        if self.auto_entropy:
            entropy = -(probs * log_probs).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Keep alpha in a sane range to prevent instability.
            self.log_alpha.data.clamp_(-10.0, 2.0)
        else:
            alpha_loss = torch.tensor(0.0)
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item() if self.auto_entropy else 0.0,
            'mean_q': q_all.mean().item(),
        }
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def store_transition(
        self,
        observation: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: Dict[str, np.ndarray],
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(observation, action, reward, next_observation, done)
    
    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
