"""
Unit tests for the modular RL framework.
"""

import pytest
import yaml
import tempfile
from pathlib import Path
import numpy as np

from evrp_rl.framework import (
    EnvFactory,
    EncoderFactory,
    AgentFactory,
    RewardModule,
    MaskModule,
    ConfigLoader,
    create_experiment_config,
    MetricsLogger,
    ExperimentRunner,
)
from evrp_rl.env import EVRPEnvironment
from evrp_rl.encoders import MLPEncoder, GATEncoder
from evrp_rl.agents import A2CAgent, SACAgent


class TestEnvFactory:
    """Test environment factory."""
    
    def test_create_basic(self):
        """Test basic environment creation."""
        config = {
            'num_customers': 5,
            'num_chargers': 2,
        }
        
        env = EnvFactory.create(config)
        
        assert isinstance(env, EVRPEnvironment)
        assert env.num_customers == 5
        assert env.num_chargers == 2
    
    def test_create_with_all_params(self):
        """Test environment creation with all parameters."""
        config = {
            'num_customers': 10,
            'num_chargers': 3,
            'battery_capacity': 80.0,
            'cargo_capacity': 40.0,
            'max_demand': 15.0,
            'grid_size': 50,
            'seed': 123,
        }
        
        env = EnvFactory.create(config)
        
        assert env.max_battery == 80.0
        assert env.max_cargo == 40.0
    
    def test_create_with_defaults(self):
        """Test environment creation uses defaults."""
        config = {}
        
        env = EnvFactory.create(config)
        
        # Should use default values
        assert env.num_customers == 10
        assert env.num_chargers == 3


class TestEncoderFactory:
    """Test encoder factory."""
    
    def test_create_mlp(self):
        """Test MLP encoder creation."""
        config = {
            'type': 'mlp',
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_layers': 2,
        }
        
        encoder = EncoderFactory.create(config)
        
        assert isinstance(encoder, MLPEncoder)
        assert encoder.embed_dim == 64
    
    def test_create_gat(self):
        """Test GAT encoder creation."""
        config = {
            'type': 'gat',
            'embed_dim': 128,
            'num_layers': 3,
            'num_heads': 4,
        }
        
        encoder = EncoderFactory.create(config)
        
        assert isinstance(encoder, GATEncoder)
        assert encoder.embed_dim == 128
    
    def test_invalid_encoder_type(self):
        """Test error for invalid encoder type."""
        config = {'type': 'invalid'}
        
        with pytest.raises(ValueError, match="Unknown encoder type"):
            EncoderFactory.create(config)
    
    def test_create_with_defaults(self):
        """Test encoder creation with defaults."""
        config = {}
        
        encoder = EncoderFactory.create(config)
        
        # Should create MLP by default
        assert isinstance(encoder, MLPEncoder)


class TestAgentFactory:
    """Test agent factory."""
    
    def test_create_a2c(self):
        """Test A2C agent creation."""
        config = {
            'type': 'a2c',
            'encoder': {
                'type': 'mlp',
                'embed_dim': 64,
            },
            'hyperparameters': {
                'lr': 1e-3,
            }
        }
        
        agent = AgentFactory.create(config, action_dim=5)
        
        assert isinstance(agent, A2CAgent)
    
    def test_create_sac(self):
        """Test SAC agent creation."""
        config = {
            'type': 'sac',
            'encoder': {
                'type': 'mlp',
                'embed_dim': 64,
            },
            'hyperparameters': {
                'lr': 1e-3,
            }
        }
        
        agent = AgentFactory.create(config, action_dim=5)
        
        assert isinstance(agent, SACAgent)
    
    def test_create_with_provided_encoder(self):
        """Test agent creation with pre-created encoder."""
        encoder = MLPEncoder(embed_dim=64)
        config = {
            'type': 'a2c',
            'hyperparameters': {},
        }
        
        agent = AgentFactory.create(config, action_dim=5, encoder=encoder)
        
        assert agent.encoder is encoder
    
    def test_invalid_agent_type(self):
        """Test error for invalid agent type."""
        config = {'type': 'invalid'}
        
        with pytest.raises(ValueError, match="Unknown agent type"):
            AgentFactory.create(config, action_dim=5)


class TestRewardModule:
    """Test reward shaping module."""
    
    def test_initialization(self):
        """Test reward module initialization."""
        config = {
            'invalid_action_penalty': -5.0,
            'battery_penalty_coef': 0.2,
        }
        
        reward_module = RewardModule(config)
        
        assert reward_module.invalid_action_penalty == -5.0
        assert reward_module.battery_penalty_coef == 0.2
    
    def test_default_config(self):
        """Test reward module with default config."""
        reward_module = RewardModule()
        
        assert reward_module.invalid_action_penalty == -10.0
    
    def test_compute_reward_battery_penalty(self):
        """Test battery penalty computation."""
        reward_module = RewardModule({'battery_penalty_coef': 0.1})
        
        state = {}
        next_state = {'current_battery': 15.0, 'current_cargo': 0.0}
        info = {}
        
        shaped_reward = reward_module.compute_reward(
            base_reward=-1.0,
            action=1,
            state=state,
            next_state=next_state,
            done=False,
            info=info,
        )
        
        # Should apply battery penalty: -1.0 - 0.1 * (20 - 15) = -1.5
        assert shaped_reward == pytest.approx(-1.5)
    
    def test_compute_reward_completion_bonus(self):
        """Test completion bonus."""
        reward_module = RewardModule({'completion_bonus': 20.0})
        
        state = {}
        next_state = {'current_battery': 50.0, 'current_cargo': 0.0}
        info = {'all_customers_visited': True}
        
        shaped_reward = reward_module.compute_reward(
            base_reward=0.0,
            action=0,
            state=state,
            next_state=next_state,
            done=True,
            info=info,
        )
        
        # Should get completion bonus
        assert shaped_reward == pytest.approx(20.0)
    
    def test_callable(self):
        """Test reward module is callable."""
        reward_module = RewardModule()
        
        result = reward_module(
            base_reward=-1.0,
            action=0,
            state={},
            next_state={'current_battery': 50.0, 'current_cargo': 0.0},
            done=False,
            info={},
        )
        
        assert isinstance(result, float)


class TestMaskModule:
    """Test action masking module."""
    
    def test_initialization(self):
        """Test mask module initialization."""
        config = {
            'use_battery_constraint': False,
            'battery_safety_margin': 0.2,
        }
        
        mask_module = MaskModule(config)
        
        assert mask_module.use_battery_constraint is False
        assert mask_module.battery_safety_margin == 0.2
    
    def test_default_config(self):
        """Test mask module with defaults."""
        mask_module = MaskModule()
        
        assert mask_module.use_battery_constraint is True


class TestConfigLoader:
    """Test configuration loading and validation."""
    
    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = {
            'env': {'num_customers': 10},
            'agent': {'type': 'a2c'},
        }
        
        validated = ConfigLoader.validate(config)
        
        assert 'run' in validated
        assert validated['run']['epochs'] == 100
    
    def test_validate_missing_keys(self):
        """Test validation catches missing keys."""
        config = {'env': {}}
        
        with pytest.raises(ValueError, match="Missing required key"):
            ConfigLoader.validate(config)
    
    def test_load_and_save(self):
        """Test loading and saving config files."""
        config = {
            'env': {'num_customers': 10},
            'agent': {'type': 'a2c'},
            'run': {'epochs': 50},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'
            
            # Save
            ConfigLoader.save(config, str(config_path))
            
            # Load
            loaded = ConfigLoader.load(str(config_path))
            
            assert loaded['env']['num_customers'] == 10
            assert loaded['agent']['type'] == 'a2c'
            assert loaded['run']['epochs'] == 50


class TestCreateExperimentConfig:
    """Test experiment config creation helper."""
    
    def test_create_basic(self):
        """Test basic config creation."""
        config = create_experiment_config(
            env_config={'num_customers': 20},
            agent_config={'type': 'sac'},
        )
        
        assert config['env']['num_customers'] == 20
        assert config['agent']['type'] == 'sac'
        assert 'run' in config
    
    def test_create_with_run_config(self):
        """Test config creation with run config."""
        config = create_experiment_config(
            env_config={'num_customers': 20},
            agent_config={'type': 'sac'},
            run_config={'epochs': 200},
        )
        
        assert config['run']['epochs'] == 200


class TestMetricsLogger:
    """Test metrics logging."""
    
    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            assert logger.log_dir.exists()
            assert 'train' in logger.metrics
            assert 'eval' in logger.metrics
    
    def test_log_train_episode(self):
        """Test logging training episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            logger.log_train_episode(
                episode=1,
                reward=-10.0,
                length=50,
                metrics={'actor_loss': 0.5, 'critic_loss': 1.0},
            )
            
            assert len(logger.metrics['train']['episodes']) == 1
            assert logger.metrics['train']['rewards'][0] == -10.0
    
    def test_log_eval_episode(self):
        """Test logging evaluation episode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            logger.log_eval_episode(
                episode=1,
                reward=-5.0,
                length=30,
                route_length=150.0,
                charge_visits=2,
                success=True,
            )
            
            assert len(logger.metrics['eval']['episodes']) == 1
            assert logger.metrics['eval']['success_rate'][0] == 1.0
    
    def test_get_recent_stats(self):
        """Test getting recent statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            # Log several episodes
            for i in range(10):
                logger.log_train_episode(
                    episode=i,
                    reward=-10.0 + i,
                    length=50,
                    metrics={},
                )
            
            stats = logger.get_recent_stats('train', window=5)
            
            assert 'mean_reward' in stats
            assert 'std_reward' in stats
            # Last 5 rewards: -5, -4, -3, -2, -1, mean = -3.0
            assert stats['mean_reward'] == pytest.approx(-3.0)
    
    def test_save_metrics(self):
        """Test saving metrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir)
            
            logger.log_train_episode(1, -10.0, 50, {})
            logger.save('test_metrics.json')
            
            metrics_file = Path(tmpdir) / 'test_metrics.json'
            assert metrics_file.exists()


class TestExperimentRunner:
    """Test experiment runner (integration tests)."""
    
    @pytest.fixture
    def simple_config(self):
        """Simple config for testing."""
        return {
            'env': {
                'num_customers': 3,
                'num_chargers': 1,
            },
            'agent': {
                'type': 'a2c',
                'encoder': {
                    'type': 'mlp',
                    'embed_dim': 32,
                    'hidden_dim': 64,
                    'num_layers': 2,
                },
                'hyperparameters': {
                    'lr': 1e-3,
                    'hidden_dim': 64,
                },
            },
            'run': {
                'name': 'test_run',
                'epochs': 5,
                'eval_frequency': 3,
                'save_frequency': 5,
                'max_steps_per_episode': 20,
                'num_eval_episodes': 2,
            },
        }
    
    def test_runner_initialization(self, simple_config):
        """Test runner initialization."""
        env = EnvFactory.create(simple_config['env'])
        agent = AgentFactory.create(simple_config['agent'], env.action_space.n)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                env, agent, simple_config,
                log_dir=tmpdir,
                checkpoint_dir=tmpdir,
            )
            
            assert runner.num_epochs == 5
            assert runner.eval_frequency == 3
    
    def test_train_episode(self, simple_config):
        """Test single training episode."""
        env = EnvFactory.create(simple_config['env'])
        agent = AgentFactory.create(simple_config['agent'], env.action_space.n)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                env, agent, simple_config,
                log_dir=tmpdir,
                checkpoint_dir=tmpdir,
            )
            
            reward, length, metrics = runner.train_episode()
            
            assert isinstance(reward, float)
            assert isinstance(length, int)
            assert 'actor_loss' in metrics
            assert length > 0
    
    def test_eval_episode(self, simple_config):
        """Test single evaluation episode."""
        env = EnvFactory.create(simple_config['env'])
        agent = AgentFactory.create(simple_config['agent'], env.action_space.n)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                env, agent, simple_config,
                log_dir=tmpdir,
                checkpoint_dir=tmpdir,
            )
            
            reward, length, info = runner.eval_episode()
            
            assert isinstance(reward, float)
            assert isinstance(length, int)
            assert 'route_length' in info
            assert 'charge_visits' in info
            assert 'success' in info
    
    def test_train_short(self, simple_config):
        """Test short training run."""
        env = EnvFactory.create(simple_config['env'])
        agent = AgentFactory.create(simple_config['agent'], env.action_space.n)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = ExperimentRunner(
                env, agent, simple_config,
                log_dir=tmpdir,
                checkpoint_dir=tmpdir,
            )
            
            runner.train()
            
            # Check that metrics were logged
            assert len(runner.logger.metrics['train']['episodes']) == 5
            
            # Check that checkpoints were saved
            final_checkpoint = runner.checkpoint_dir / 'final_model.pt'
            assert final_checkpoint.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
