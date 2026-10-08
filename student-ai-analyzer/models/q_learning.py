"""
models/q_learning.py
---------------------
Simple Q-Learning demonstration.

Environment:
  A student must move through a sequence of learning milestones to reach the Goal.

States (6):
  0 – Start
  1 – Learn Python
  2 – Practice Coding
  3 – Learn ML
  4 – Complete Project
  5 – Goal  (terminal)

Actions:
  0 – Stay   (remain in current state, small negative reward)
  1 – Advance (move to next state)

Rewards:
  Advancing through states gives small positive rewards (+1).
  Reaching the Goal gives a big reward (+10).
  Staying in place gives a small penalty (-0.5).

The Q-table has shape (n_states, n_actions).
After training, the agent should always choose "Advance".
"""

import numpy as np

# ── Environment definition ────────────────────────────────────────────────────

STATES = [
    "Start",
    "Learn Python",
    "Practice Coding",
    "Learn ML",
    "Complete Project",
    "Goal",
]

ACTIONS = ["Stay", "Advance"]

N_STATES  = len(STATES)
N_ACTIONS = len(ACTIONS)
GOAL_STATE = N_STATES - 1


def get_reward(state: int, action: int) -> tuple:
    """
    Given current state and action, return (next_state, reward).

    action 0 = Stay, action 1 = Advance
    """
    if state == GOAL_STATE:
        # Already at goal – no transition
        return GOAL_STATE, 0.0

    if action == 1:  # Advance
        next_state = state + 1
        reward = 10.0 if next_state == GOAL_STATE else 1.0
    else:  # Stay
        next_state = state
        reward = -0.5

    return next_state, reward


# ── Q-Learning algorithm ──────────────────────────────────────────────────────

def train_q_learning(
    episodes:      int   = 200,
    alpha:         float = 0.8,   # learning rate
    gamma:         float = 0.9,   # discount factor
    epsilon:       float = 1.0,   # initial exploration rate
    epsilon_decay: float = 0.02,  # decay per episode
    epsilon_min:   float = 0.05,
):
    """
    Train a Q-table using the Q-Learning update rule:
      Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') − Q(s, a)]

    Returns
    -------
    Q_table         : ndarray shape (N_STATES, N_ACTIONS)
    episode_rewards : list of total reward per episode
    learned_path    : list of state names the greedy policy follows
    """
    # Initialise Q-table to zeros
    Q = np.zeros((N_STATES, N_ACTIONS))

    episode_rewards = []

    for ep in range(episodes):
        state = 0  # always start at "Start"
        total_reward = 0.0

        for _ in range(20):  # max steps per episode
            if state == GOAL_STATE:
                break

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(N_ACTIONS)   # explore
            else:
                action = int(np.argmax(Q[state]))        # exploit

            next_state, reward = get_reward(state, action)

            # Q-Learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        episode_rewards.append(total_reward)

    # Extract the greedy path the trained agent follows from Start to Goal
    learned_path = []
    state = 0
    for _ in range(N_STATES + 2):
        learned_path.append(STATES[state])
        if state == GOAL_STATE:
            break
        action = int(np.argmax(Q[state]))
        next_state, _ = get_reward(state, action)
        if next_state == state:
            # Stuck – shouldn't happen after training, but guard against infinite loop
            break
        state = next_state

    return Q, episode_rewards, learned_path
