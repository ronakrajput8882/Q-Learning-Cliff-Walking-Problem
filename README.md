<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🧠%20Q-Learning%20Cliff%20Walking&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Reinforcement%20Learning%20|%20Model-Free%20|%20Gymnasium&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-4B8BBE?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
</div>

---

## 📌 Project Overview

The **Cliff Walking Problem** is a classic Reinforcement Learning benchmark where an agent must navigate from a start state to a goal state on a 4×12 grid — without falling off a cliff that carries a heavy penalty of **-100**. This project implements **Q-Learning** (an off-policy, model-free TD control algorithm) with an **ε-greedy exploration policy** to learn an optimal path.

The agent learns purely through trial and error — no environment model, no supervised labels, just reward signals.

| Parameter | Value |
|:---|:---:|
| Algorithm | Q-Learning (off-policy TD) |
| Environment | `CliffWalking-v1` (Gymnasium) |
| State Space | 48 states (4×12 grid) |
| Action Space | 4 (Up, Down, Left, Right) |
| Discount Factor (γ) | 0.99 |
| Learning Rate (α) | 0.5 |
| Epsilon (ε) | 0.1 |
| Episodes | 500 |

---

## 📂 Environment

**CliffWalking-v1** from [Gymnasium (Farama Foundation)](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

- Grid: 4 rows × 12 columns = **48 states**
- Start: Bottom-left (state 36)
- Goal: Bottom-right (state 47)
- Cliff: Bottom row states 37–46 → reward **-100** and reset
- Step reward: **-1** per step (encourages shortest path)
- Agent observes state as a single integer (0–47)

---

## 🔄 Pipeline Workflow

```
Environment Init → Q-Table Init (48×4 zeros) → ε-Greedy Action Selection → Step & Observe → Q-Update → Convergence → Greedy Deployment
```

### 1️⃣ Environment Initialization
`gymnasium.make("CliffWalking-v1")` initializes the 4×12 grid. State is a flat integer in `[0, 47]`.

### 2️⃣ Q-Table Initialization
`Q = np.zeros((48, 4))` — zero-initialized table mapping every (state, action) pair to expected return.

### 3️⃣ ε-Greedy Policy
```python
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()   # explore
    else:
        return np.argmax(Q[state])         # exploit
```
With ε=0.1, the agent explores 10% of the time and exploits the best known action 90% of the time.

### 4️⃣ Q-Learning Update (Bellman Equation)
```
Q[s, a] ← Q[s, a] + α × (r + γ × max_a' Q[s', a'] − Q[s, a])
```
Off-policy: the target always uses `max` over next-state actions regardless of what policy was followed.

### 5️⃣ Greedy Deployment
After 500 training episodes, the agent runs one episode using a fully greedy policy (`np.argmax(Q[state])`) with `render_mode="human"` to visualize the learned path.

---

## 🤖 Algorithm ⭐ Best Model

### Q-Learning (Off-Policy TD Control)

```python
Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
```

- **Off-policy**: learns the optimal policy while following a different (exploratory) behavior policy
- **Temporal Difference**: updates Q-values after every step, not at end of episode
- **Convergence**: guaranteed to converge to optimal Q* under standard conditions (decaying α, sufficient exploration)
- **vs SARSA**: Q-Learning is more aggressive — it learns the optimal path even if the agent occasionally falls off the cliff during training; SARSA learns a safer path because it accounts for exploration in its update target

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| Training Episodes | 500 |
| Final Episode Reward | ~-13 |
| Optimal Path Length | 13 steps |
| Cliff Penalty | -100 per fall |
| Convergence | ~200–300 episodes |
| Deployment Reward | -13 (optimal) |

> **Note:** Q-Learning converges to the **optimal (shortest) path** hugging the cliff edge. SARSA, by contrast, learns a safer route further from the cliff due to its on-policy update — a well-documented behavioral difference in RL literature.

---

## 🔍 Key Insights

- 📈 **Discount factor γ=0.99** is critical — lower values cause the agent to prioritize immediate rewards and avoid the cliff too conservatively
- ⚡ **α=0.5** provides fast learning; lower α (0.1) leads to slower but smoother convergence
- 🎯 **Q-Learning finds the cliff-edge optimal path** (total reward ≈ -13) — 13 steps along the bottom row, accepting the risk of falling during training
- 🔄 **ε-greedy with ε=0.1** strikes the right explore-exploit balance; higher ε causes frequent cliff falls even in later episodes
- 🧠 **Off-policy nature** is the core advantage: the agent can learn from sub-optimal exploratory behavior without that exploration degrading the target policy

---

## 🗂️ Repository Structure

```
Q-Learning-Cliff-Walking-Problem/
│
├── Q_Learning_Cliff_Walking_Problem.py   # Main implementation
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ronakrajput8882/Q-Learning-Cliff-Walking-Problem.git
cd Q-Learning-Cliff-Walking-Problem

# Install dependencies
pip install numpy gymnasium

# Run training + visualization
python Q_Learning_Cliff_Walking_Problem.py
```

The script will:
1. Train the Q-table over 500 episodes (prints reward + episode length each episode)
2. Render the learned optimal policy in a visual window

---

## 🧠 Key Learnings

- Q-Learning is guaranteed to find the **globally optimal policy** in tabular environments with discrete state/action spaces
- The **Bellman optimality equation** is the mathematical backbone of Q-Learning — every update brings Q-values closer to their true optimal values
- **Off-policy vs on-policy** is not just a technicality — it fundamentally changes the behavior of the learned policy in risky environments like Cliff Walking
- Even with just 63 lines of Python, you can implement a complete RL training loop from scratch
- Gymnasium's `CliffWalking-v1` is an excellent testbed for understanding TD methods before scaling to deep RL

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| NumPy | Q-table storage and argmax operations |
| Gymnasium | RL environment (`CliffWalking-v1`) |
| random | ε-greedy exploration sampling |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>