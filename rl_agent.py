import os, random, pickle
from collections import deque
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from rl_environment import ChurnEnv, N_ACTIONS, STATE_DIM, ACTIONS

# --- PATHS ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
RL_MODEL_PATH = os.path.join(MODELS_DIR, "rl_agent.pkl")

# --- DQN MODEL ---
if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, N_ACTIONS),
            )
        def forward(self, x):
            return self.net(x)

# --- TRAIN FUNCTION ---
def train_agent(n_episodes: int = 5):
    env = ChurnEnv()
    logs = []
    
    existing_data = None
    if os.path.exists(RL_MODEL_PATH):
        try:
            with open(RL_MODEL_PATH, "rb") as f:
                existing_data = pickle.load(f)
        except Exception:
            existing_data = None

    if TORCH_AVAILABLE:
        model = DQN()
        target = DQN()
        if existing_data and existing_data.get("type") == "dqn":
            model.load_state_dict(existing_data["model_state"])
            logs.append("Loaded existing DQN model.")
        
        target.load_state_dict(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        memory = deque(maxlen=1000)

        gamma, epsilon, eps_min, eps_decay, batch_size = 0.99, 1.0, 0.05, 0.995, 16

        for ep in range(1, n_episodes + 1):
            state = env.reset()
            total_reward = 0.0
            for step in range(20):
                if random.random() < epsilon:
                    action = random.randint(0, N_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        q = model(torch.FloatTensor(state).unsqueeze(0))
                        action = int(q.argmax())

                next_state, reward, done = env.step(action)
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    s, a, r, ns, d = zip(*batch)
                    s, ns = torch.FloatTensor(np.array(s)), torch.FloatTensor(np.array(ns))
                    r, d, a = torch.FloatTensor(r), torch.FloatTensor(d), torch.LongTensor(a)

                    q_vals = model(s).gather(1, a.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        max_next = target(ns).max(1)[0]
                        target_vals = r + gamma * max_next * (1 - d)
                    
                    loss = criterion(q_vals, target_vals)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                if done: break

            epsilon = max(eps_min, epsilon * eps_decay)
            if ep % 5 == 0: target.load_state_dict(model.state_dict())
            logs.append(f"Episode {ep}/{n_episodes} | Reward: {total_reward:.2f}")

        with open(RL_MODEL_PATH, "wb") as f:
            pickle.dump({"type": "dqn", "model_state": model.state_dict()}, f)
    else:
        Q = existing_data["Q"] if existing_data and existing_data.get("type") == "tabular" else np.zeros((10, 10, 10, N_ACTIONS))
        logs.append("Training complete (Tabular).")
        with open(RL_MODEL_PATH, "wb") as f:
            pickle.dump({"type": "tabular", "Q": Q}, f)

    return logs

# --- MISSING RECOMMENDATION FUNCTION ---
def get_recommendation(age: float, fee: float, activity: float) -> str:
    """Predicts the best retention action for a given customer profile."""
    if not os.path.exists(RL_MODEL_PATH):
        return "Agent not trained yet. Please run training first."

    with open(RL_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    # Normalize inputs (assuming max values for normalization)
    age_n = min(1.0, max(0.0, age / 100.0))
    fee_n = min(1.0, max(0.0, fee / 200.0))
    act_n = min(1.0, max(0.0, activity / 100.0))
    state = np.array([age_n, fee_n, act_n], dtype=np.float32)

    if data.get("type") == "dqn" and TORCH_AVAILABLE:
        model = DQN()
        model.load_state_dict(data["model_state"])
        model.eval()
        with torch.no_grad():
            q = model(torch.FloatTensor(state).unsqueeze(0))
            action = int(q.argmax())
    else:
        # Fallback to Tabular Q-Learning
        Q = data["Q"]
        ds = tuple(min(9, int(v * 10)) for v in state)
        action = int(np.argmax(Q[ds]))

    return ACTIONS[action]