# EVRP-RL Architecture Diagram

This file provides a simple, standalone architecture diagram (Mermaid + ASCII fallback) that explains how the project components interact at runtime.

## Mermaid component diagram

```mermaid
flowchart LR
  Configs[Configs\nconfigs/*.yaml]
  Runner[Runner / Trainer\nsrc/framework/runner.py]
  EV[EVRP Environment\nenv/evrp_env.py]
  subgraph Encoders[Encoders]
    GAT[GAT Encoder\nsrc/encoders/gat_encoder.py]
    MLP[MLP Encoder\nsrc/encoders/mlp_encoder.py]
  end
  subgraph Agents[Agents]
    A2C[A2C Agent\nsrc/agents/a2c_agent.py]
    SAC[SAC Agent\nsrc/agents/sac_agent.py]
  end
  Check[Checkpoints\ncheckpoints/]
  Results[Results & Logs\nresults/]
  Examples[Examples\nexamples/]
  Tests[Tests\ntests/]

  Configs --> Runner
  Runner --> EV
  EV -->|observation| Encoders
  Encoders -->|embedding| Agents
  Agents -->|action| EV
  Runner -->|save/load| Check
  Runner -->|log| Results
  Examples --> Runner
  Tests --> Runner
```

## Sequence diagram (one training step)

```mermaid
sequenceDiagram
  participant R as Runner
  participant E as Env
  participant Enc as Encoder
  participant A as Agent
  participant C as Checkpoints

  R->>E: reset / step -> obs
  E-->>R: observation
  R->>Enc: encode(observation)
  Enc-->>R: embedding
  R->>A: act(embedding, mask)
  A-->>R: action
  R->>E: apply(action)
  E-->>R: next_obs, reward, done, info
  R->>R: store transition
  alt update step
    R->>A: update(parameters)
    R->>C: save(checkpoint)
  end
```

## ASCII fallback

```
  [Configs YAML] -> [Runner/Trainer]
         |
         v
  [EVRP Environment] -> [Encoders] -> [Agents] -> [EVRP Environment]
         |                                   |
         +-> logs -> [Results & Metrics]     +-> saves -> [Checkpoints]

  Examples/Tests -> invoke Runner
```

## Notes (concise)

- Runner orchestration: reset/step env, encode observations, select actions with masking, apply actions, store transitions, and run updates according to algorithm (A2C/SAC).
- Encoders produce tensors shaped (batch, node_count, D); agents output actions or distributions over nodes.
- Checkpoints should contain model + optimizer states, trainer step/epoch, RNG states, and a config snapshot for reproducibility.
- Extension points: add new encoders under `src/encoders/`, new agents under `src/agents/`, and new environment variants under `env/`.

---

If you'd like, I can also:

- render this diagram as an SVG and save it to `docs/` for viewers that don't support Mermaid, or
- add a short `examples/resume_from_checkpoint.py` demonstrating loading a checkpoint and running evaluation.
