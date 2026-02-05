# Multi-Agent DQN with Manual Forward & Backpropagation

A reinforcement learning implementation featuring multiple agents navigating a grid world to pick up and deliver items. The DQN (Deep Q-Network) is implemented from scratch using NumPy with manual forward propagation and backpropagation.

## Features

- **Hardcoded Neural Network**: Pure NumPy implementation without deep learning frameworks
- **Manual Forward Propagation**: Matrix multiplication with ReLU activation
- **Manual Backpropagation**: Gradient calculation using chain rule
- **Online Learning**: Agents learn while performing tasks (no separate training phase)
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Multi-Agent Coordination**: 4 agents navigate simultaneously with collision avoidance

## Project Structure

```
pacman-main/
├── NAgentDQN.py          # Main implementation file
├── config.json           # Configuration parameters
├── train_log.txt         # Training output log (auto-generated)
├── requirements-dev.txt  # Development dependencies
├── tests/                # Test files
│   ├── conftest.py
│   ├── test_integration_smoke.py
│   └── test_multiagent.py
└── README.md             # This file
```

## Neural Network Architecture

```
Input Layer (12 neurons)
    ↓
Hidden Layer (64 neurons, ReLU activation)
    ↓
Output Layer (4 neurons - Q-values for each action)
```

**Total Parameters**: 1,092 (12×64 + 64 + 64×4 + 4)

### State Vector (12 dimensions)
- Agent position (x, y) - normalized
- Target position (x, y) - normalized
- Direction to target (dx, dy)
- Distance to target
- Has item flag
- Nearby agent flags (3 directions)

## Configuration

All parameters can be configured in `config.json`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| environment | grid_size | 5 | Grid dimensions (5×5) |
| environment | num_agents | 4 | Number of agents |
| network | input_dim | 12 | State vector size |
| network | hidden_dim | 64 | Hidden layer neurons |
| network | output_dim | 4 | Number of actions |
| training | learning_rate | 0.001 | Learning rate |
| training | gamma | 0.95 | Discount factor |
| training | batch_size | 64 | Mini-batch size |
| training | memory_size | 20000 | Replay buffer size |
| rewards | pickup | 5.0 | Reward for picking up item |
| rewards | delivery | 20.0 | Reward for delivery |
| rewards | move_toward | 0.5 | Reward for moving toward target |
| rewards | move_away | -0.3 | Penalty for moving away |
| rewards | step | -0.1 | Step penalty |
| rewards | wall | -1.0 | Wall collision penalty |
| rewards | collision | -2.0 | Agent collision penalty |

## Usage

### Running Training

```bash
python NAgentDQN.py
```

### Output

Training progress is displayed in the console and saved to `train_log.txt`:

```
============================================================
Training Session Started: 2026-02-03 14:30:22
============================================================

Episode 1/100 - Steps: 45, Deliveries: 3, Collisions: 0, Success Rate: 100.0%
Episode 2/100 - Steps: 38, Deliveries: 4, Collisions: 1, Success Rate: 75.0%
...

=== FINAL RESULTS ===
Average Steps: 42.5
Average Success Rate: 87.3%
Total Deliveries: 350
Total Collisions: 23
Collision-Free Deliveries: 312
```

### Metrics Explained

- **Steps**: Number of moves taken in an episode
- **Deliveries**: Items successfully delivered to destination
- **Collisions**: Number of agent-to-agent collisions
- **Success Rate**: Percentage of collision-free deliveries

## Actions

| Action | Value | Direction |
|--------|-------|-----------|
| UP | 0 | (0, -1) |
| DOWN | 1 | (0, 1) |
| LEFT | 2 | (-1, 0) |
| RIGHT | 3 | (1, 0) |

## Algorithm Flow

```
┌────────────────────────────────────────────────────────-─────┐
│                         run()                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Initialize: DQN, ReplayBuffer, Environment         │     │
│  └─────────────────────────────────────────────────────┘     │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────-─┐    │
│  │  FOR each episode:                                   │    │
│  │    ┌────────────────────────────────────────-───┐    │    │
│  │    │  get_smart_actions() - Select actions      │    │    │
│  │    │  (heuristic-based with collision avoid)    │    │    │
│  │    └─────────────────────────────────────────-──┘    │    │
│  │                        ↓                             │    │
│  │    ┌───────────────────────────────────────-────┐    │    │
│  │    │  env.step() - Execute actions              │    │    │
│  │    │  Get rewards, new states                   │    │    │
│  │    └────────────────────────────────────────-───┘    │    │
│  │                        ↓                             │    │
│  │    ┌────────────────────────────────────────-───┐    │    │
│  │    │  replay_buffer.push() - Store experience   │    │    │
│  │    └────────────────────────────────────────-───┘    │    │
│  │                        ↓                             │    │
│  │    ┌───────────────────────────────────────-────┐    │    │
│  │    │  train_step() - Online learning            │    │    │
│  │    │  • Sample batch from buffer                │    │    │
│  │    │  • Forward propagation                     │    │    │
│  │    │  • Calculate TD error                      │    │    │
│  │    │  • Backpropagation                         │    │    │
│  │    │  • Update weights                          │    │    │
│  │    └──────────────────────────────────────-─────┘    │    │
│  └────────────────────────────────────────────-─────────┘    │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────-─────┐    │
│  │  Log results to console and train_log.txt            │    │
│  └────────────────────────────────────────────────-─────┘    │
└───────────────────────────────────────────────────────────-──┘
```

## Key Components

### DQN Class
- `forward()`: Manual forward propagation with ReLU
- `backward()`: Manual backpropagation with gradient descent
- `train_step()`: Batch training with experience replay

### ReplayBuffer Class
- Stores (state, action, reward, next_state, done) tuples
- Random sampling for training batches

### MultiAgentGridWorld Class
- Grid environment with pickup (A) and delivery (B) locations
- Handles agent movements and collisions
- Calculates rewards based on agent actions

### get_smart_actions()
- Heuristic-based action selection
- Collision avoidance with other agents
- Wall boundary checking

## Requirements

- Python 3.7+
- NumPy

## License

This project is for educational purposes as part of a Multi-Agent Reinforcement Learning assignment.
