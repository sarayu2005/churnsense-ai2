import numpy as np

ACTIONS = ["No Action", "Offer Promo", "Send Email", "Upgrade Plan", "Call Customer"]
N_ACTIONS = len(ACTIONS)
STATE_DIM = 3  # age_norm, fee_norm, activity_norm


class ChurnEnv:
    """Simple churn-reduction RL environment."""

    def __init__(self):
        self.state = None
        self.step_count = 0
        self.max_steps = 20

    def reset(self):
        # random customer state: [age_norm, fee_norm, activity_norm]
        self.state = np.random.rand(STATE_DIM).astype(np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: int):
        self.step_count += 1

        # reward heuristic
        age, fee, activity = self.state
        reward = 0.0

        if action == 1:   # Offer Promo  – works best for high-fee customers
            reward = 1.0 if fee > 0.6 else 0.1
        elif action == 2:  # Send Email   – works best for low-activity
            reward = 1.0 if activity < 0.4 else 0.2
        elif action == 3:  # Upgrade Plan – works for medium activity
            reward = 0.8 if 0.3 < activity < 0.7 else 0.1
        elif action == 4:  # Call Customer – high-value rescue
            reward = 1.2 if fee > 0.5 and activity < 0.5 else 0.3
        else:              # No Action
            reward = -0.2

        # transition: improve activity slightly on positive reward
        if reward > 0.5:
            self.state[2] = min(1.0, self.state[2] + 0.05)

        done = self.step_count >= self.max_steps
        return self.state.copy(), reward, done
