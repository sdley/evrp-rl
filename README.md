# Electric Vehicle Routing Problem – EVRP

## Project Overview

The Electric Vehicle Routing Problem (EVRP) is a complex combinatorial optimization challenge central to logistics and urban transportation. It involves routing a fleet of battery-constrained electric vehicles (EVs) to serve customers within specified time windows while minimizing travel distance and respecting charging constraints. This project tackles EVRP using Deep Reinforcement Learning (DRL), inspired by the pioneering LIN202 framework, to develop adaptive and scalable routing solutions. Conducted as part of an academic internship focused on reinforcement learning approaches, this research aims to push the boundaries of intelligent routing under realistic EV operational constraints.

## Objectives

- Model the EVRP as a reinforcement learning environment reflecting battery limitations, time windows, and charging station availability.
- Develop, implement, and train RL agents (e.g., DQN, PPO, policy gradient methods) to generate feasible and energy-efficient routing policies.
- Benchmark RL models against classical optimization approaches such as heuristics and mixed-integer linear programming.
- Evaluate model scalability, route efficiency, and charging optimization across diverse problem instances.

## Methodology Summary

The approach represents the routing problem as a graph, encompassing customers, charging stations, and depot nodes. Each node encodes local data (coordinates, customer demand, time windows) and global system states (current time, battery level, available EVs). The core RL model uses a graph embedding technique (Structure2Vec) combined with an attention mechanism and an LSTM decoder to estimate action probabilities for route construction. The training process employs policy gradient optimization with a rollout baseline, guided by a reward function balancing route distance minimization, constraint satisfaction, and penalty terms for infeasibilities.

## Repository Structure

```text
📦 evrp-rl
┣ 📂 src/ # RL models, environment, and utility code
┣ 📂 data/ # Benchmark datasets and synthetic instance generators
┣ 📂 experiments/ # Experiment scripts and Jupyter notebooks
┣ 📂 results/ # Model outputs, route visualizations, and metrics
┣ 📂 docs/ # Literature reviews, reports, and academic papers
┣ README.md
┣ LICENSE
┗ CODE_OF_CONDUCT.md
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- TensorFlow or PyTorch
- NumPy
- Pandas
- Matplotlib

### Installation

```shell
git clone https://github.com/sdley/evrp-rl.git
cd evrp-rl
pip install -r requirements.txt
```

### Running Experiments

To train the RL agent:

```python
python src/train_agent.py
```

Other scripts and notebooks for evaluation and visualization are available in the `experiments/` directory.

## Usage

- Execute experiments with configurable environment parameters, such as the number of EVs, charging stations, and customer time windows.
- Visualize optimized routes and performance metrics with provided scripts and interactive Jupyter notebooks.
- Modify environment constraints or model architecture for tailored EVRP variants.

## Results & Benchmarks

- RL models demonstrate robust performance on large-scale EVRP instances where classical methods falter.
- Stochastic decoding strategies yield high-quality routing solutions with significant efficiency gains.
- Visual route examples highlight the model’s ability to balance time windows, battery constraints, and charging station visits.
- Benchmarks indicate superior scalability and adaptability suitable for real-time EV fleet operations.

## Contributing

Contributions, suggestions, and improvements are warmly welcomed. Please ensure adherence to the repository’s Code of Conduct. Feel free to open issues or submit pull requests for collaboration.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments

- Authors of the paper "Deep Reinforcement Learning for the Electric Vehicle Routing Problem With Time Windows" (LIN202).
- Supporting academic institution and research lab.
