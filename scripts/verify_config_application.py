#!/usr/bin/env python3
"""
Verify that hyperparameter configs are properly applied to agents.
Run this BEFORE training to catch config application issues.
"""

import yaml
from pathlib import Path
import torch
import sys

sys.path.insert(0, str(Path.cwd()))

from src.framework import AgentFactory, EnvFactory
from src.encoders.gat_encoder import GATEncoder

def verify_configs():
    print("\n" + "="*70)
    print("🔍 HYPERPARAMETER CONFIG VERIFICATION")
    print("="*70)
    
    config_dir = Path('examples/configs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Read config files
    print("\n1️⃣  READING CONFIG FILES")
    print("-" * 70)
    
    configs_from_file = {}
    for agent_name in ['a2c', 'sac']:
        cfg_path = config_dir / f'benchmark_{agent_name}.yaml'
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        configs_from_file[agent_name] = cfg
        
        hyperparams = cfg['hyperparameters']
        print(f"\n{agent_name.upper()} Config:")
        print(f"  Learning Rate:  {hyperparams.get('learning_rate', 'NOT FOUND')}")
        print(f"  Max Grad Norm:  {hyperparams.get('max_grad_norm', 'NOT FOUND')}")
        if agent_name == 'a2c':
            print(f"  Entropy Coef:   {hyperparams.get('entropy_coef', 'NOT FOUND')}")
        elif agent_name == 'sac':
            print(f"  Alpha:          {hyperparams.get('alpha', 'NOT FOUND')}")
    
    # Step 2: Create encoder and agents using AgentFactory
    print("\n2️⃣  CREATING AGENTS WITH AgentFactory")
    print("-" * 70)
    
    env_config = {'num_customers': 20, 'num_chargers': 5, 'max_battery': 500.0, 'seed': 42}
    env = EnvFactory.create(env_config)
    action_dim = env.action_space.n
    
    encoder = GATEncoder(embed_dim=64, num_layers=2, num_heads=4)
    
    agents_created = {}
    import copy
    for agent_name in ['a2c', 'sac']:
        cfg = copy.deepcopy(configs_from_file[agent_name])
        # Ensure correct agent type in config
        if 'agent' in cfg:
            if isinstance(cfg['agent'], dict):
                cfg['agent']['type'] = agent_name
            else:
                cfg['agent'] = agent_name
        else:
            cfg['agent'] = agent_name
        try:
            agent = AgentFactory.create(cfg, action_dim=action_dim)
            agent.to(device)
            agents_created[agent_name] = agent
            print(f"\n✅ {agent_name.upper()} agent created successfully")
        except Exception as e:
            print(f"\n❌ {agent_name.upper()} agent creation FAILED: {e}")
            agents_created[agent_name] = None
    
    # Step 3: Verify hyperparameters in agents
    print("\n3️⃣  VERIFYING HYPERPARAMETERS IN AGENTS")
    print("-" * 70)
    
    results = {}
    for agent_name, agent in agents_created.items():
        if agent is None:
            continue
            
        print(f"\n{agent_name.upper()}:")
        config_values = configs_from_file[agent_name]['hyperparameters']
        
        # Check learning rate
        expected_lr = config_values.get('learning_rate')
        if hasattr(agent, 'optimizer'):
            actual_lr = agent.optimizer.param_groups[0]['lr']
        else:
            actual_lr = 'NO OPTIMIZER'
        
        lr_match = (abs(actual_lr - expected_lr) < 1e-7) if isinstance(actual_lr, float) else False
        lr_status = "✅ MATCH" if lr_match else "❌ MISMATCH"
        
        print(f"  Learning Rate:")
        print(f"    Expected: {expected_lr}")
        print(f"    Actual:   {actual_lr}")
        print(f"    {lr_status}")
        
        # Check max_grad_norm
        expected_grad_norm = config_values.get('max_grad_norm', 0.5)
        actual_grad_norm = getattr(agent, 'max_grad_norm', 'NOT SET')
        grad_norm_match = (actual_grad_norm == expected_grad_norm) if isinstance(actual_grad_norm, (int, float)) else False
        grad_status = "✅ MATCH" if grad_norm_match else "❌ MISMATCH"
        
        print(f"  Max Grad Norm:")
        print(f"    Expected: {expected_grad_norm}")
        print(f"    Actual:   {actual_grad_norm}")
        print(f"    {grad_status}")
        
        # Check agent-specific params
        if agent_name == 'a2c':
            expected_entropy = config_values.get('entropy_coef')
            actual_entropy = getattr(agent, 'entropy_coef', 'NOT SET')
            entropy_match = (abs(actual_entropy - expected_entropy) < 1e-6) if isinstance(actual_entropy, float) else False
            entropy_status = "✅ MATCH" if entropy_match else "❌ MISMATCH"
            
            print(f"  Entropy Coef:")
            print(f"    Expected: {expected_entropy}")
            print(f"    Actual:   {actual_entropy}")
            print(f"    {entropy_status}")
            
            results[agent_name] = lr_match and grad_norm_match and entropy_match
        
        elif agent_name == 'sac':
            expected_alpha = config_values.get('alpha')
            # Print all attributes for debugging
            print("  [DEBUG] SAC agent __dict__:")
            for k, v in agent.__dict__.items():
                print(f"    {k}: {v}")
            fixed_alpha = getattr(agent, '_fixed_alpha', None)
            learned_alpha = torch.exp(agent.log_alpha).item() if hasattr(agent, 'log_alpha') else None
            if fixed_alpha is not None:
                alpha_value = fixed_alpha
                alpha_source = 'fixed (from config)'
            elif learned_alpha is not None:
                alpha_value = learned_alpha
                alpha_source = 'learned (auto entropy)'
            else:
                alpha_value = 'NOT SET'
                alpha_source = 'unknown'

            alpha_match = (abs(alpha_value - expected_alpha) < 1e-6) if isinstance(alpha_value, float) else False
            alpha_status = "✅ MATCH" if alpha_match else "❌ MISMATCH"

            print(f"  Alpha:")
            print(f"    Expected: {expected_alpha}")
            print(f"    Actual:   {alpha_value} [{alpha_source}]")
            print(f"    {alpha_status}")

            results[agent_name] = lr_match and grad_norm_match and alpha_match
    
    # Final summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    all_ok = all(results.values())
    if all_ok:
        print("\n✅ ALL HYPERPARAMETERS CORRECTLY APPLIED")
        print("\n🚀 Safe to proceed with training!")
        return 0
    else:
        print("\n❌ HYPERPARAMETER MISMATCH DETECTED")
        for agent_name, is_ok in results.items():
            status = "✅" if is_ok else "❌"
            print(f"  {status} {agent_name.upper()}")
        print("\n⚠️  DO NOT TRAIN YET - Config application issue detected!")
        print("\nSee CRITICAL_CONFIG_DEBUG.md for debugging steps.")
        return 1

if __name__ == '__main__':
    exit_code = verify_configs()
    sys.exit(exit_code)
