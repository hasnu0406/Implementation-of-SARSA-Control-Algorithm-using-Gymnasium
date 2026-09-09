# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement


Implement the SARSA (State-Action-Reward-State-Action) control algorithm in the Gymnasium FrozenLake-v1 environment. The objective is to train an agent to learn an optimal action-value function using an epsilon-greedy policy and reach the goal while avoiding holes.


## Software Requirements

- Python 3.x
- Jupyter Notebook / Google Colab
- NumPy
- Matplotlib
- Gymnasium

## Environment Description



## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Initialize the FrozenLake environment and the Q-table with zeros.
2. Set the learning rate `α`, discount factor `γ`, exploration rate `ε`, and number of episodes.
3. Reset the environment and select the initial action using the ε-greedy policy.
4. Execute the action and observe the next state and reward.
5. Select the next action using the ε-greedy policy.
6. Update the Q-value using the SARSA update rule:

   `Q(S,A) ← Q(S,A) + α[R + γQ(S',A') − Q(S,A)]`
7. Set the next state and action as the current state and action.
8. Repeat steps 4–7 until the episode reaches a terminal state.
9. Decay `ε` after each episode to gradually reduce exploration.
10. After training, extract the state-value function and learned policy from the Q-table.
11. Display the final Q-table, estimated state-value function, and learned policy.

## Python Program

```python

!pip install gymnasium

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create Custom FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "SFFF",
    "FHFH",
    "FFHF",
    "HFFG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=True
)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Custom FrozenLake Map:")

for row in custom_map:
    print(row)

print("\nNumber of States:", n_states)
print("Number of Actions:", n_actions)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

initial_value = 0.5

Q = np.full(
    (n_states, n_actions),
    initial_value,
    dtype=float
)

print("\nInitial Q-table:")
print(Q)

print("\nInitial State-Value Function:")
print(np.max(Q, axis=1))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    max_q = np.max(Q[state])

    # Select randomly among equally good actions
    best_actions = np.flatnonzero(
        Q[state] == max_q
    )

    return np.random.choice(best_actions)


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Select initial action
    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # -------------------------------------------------
        # Terminal State
        # -------------------------------------------------

        if terminated or truncated:

            # For terminal states:
            # Q(s,a) <- Q(s,a) + alpha[R - Q(s,a)]

            td_target = reward

            td_error = (
                td_target
                - Q[state, action]
            )

            Q[state, action] += (
                alpha * td_error
            )

            break

        # -------------------------------------------------
        # Select Next Action
        # -------------------------------------------------

        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # -------------------------------------------------
        # SARSA Update
        # -------------------------------------------------

        td_target = (
            reward
            + gamma * Q[next_state, next_action]
        )

        td_error = (
            td_target
            - Q[state, action]
        )

        Q[state, action] += (
            alpha * td_error
        )

        # Move to next state and action
        state = next_state
        action = next_action

    # Store episode reward
    episode_rewards.append(total_reward)

    # -------------------------------------------------
    # Variable Epsilon Decay
    # -------------------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Extract State Values and Learned Policy
# -------------------------------------------------

state_values = np.max(
    Q,
    axis=1
)

learned_policy = np.argmax(
    Q,
    axis=1
)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")

    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(
        Q,
        3
    )
)

print_value_function(
    state_values
)

print_policy(
    learned_policy
)


# -------------------------------------------------
# Average Reward
# -------------------------------------------------

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 4)
)

print(
    "Final Epsilon:",
    round(epsilon, 4)
)


# -------------------------------------------------
# Plot SARSA Learning Curve
# -------------------------------------------------

window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    moving_average,
    linewidth=2
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Average Reward"
)

plt.title(
    "SARSA Learning Curve - Custom FrozenLake"
)

plt.grid(
    True
)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()



```
---

## Output

Final Q-table:

<img width="631" height="387" alt="image" src="https://github.com/user-attachments/assets/98cc2ee4-ff4d-4e55-aeee-e74b24c5eb6e" />




Estimated State-Value Function:

<img width="385" height="135" alt="image" src="https://github.com/user-attachments/assets/371bf482-a7a5-46c0-9d1b-8ce08ac968c5" />




Learned Policy:

<img width="266" height="131" alt="image" src="https://github.com/user-attachments/assets/50347523-9162-4558-83a5-cef746c464f1" />



Average reward over last 1000 episodes: 

<img width="1031" height="615" alt="image" src="https://github.com/user-attachments/assets/e909aef7-5054-418f-a5ff-86ed3ca11556" />



---

## Result


The SARSA algorithm successfully learned an action-value function for the FrozenLake environment. The Q-table was updated through interaction with the environment using the SARSA update rule and an epsilon-greedy policy. The learned policy shows the preferred action for each state, while the state-value function represents the maximum learned value for each state.

The learning curve shows the improvement in the agent's performance over training episodes, with the average reward generally increasing as the agent learns a better policy.


---

## Inference
The experiment demonstrates that SARSA can learn an effective policy through on-policy reinforcement learning. By balancing exploration and exploitation using the epsilon-greedy strategy, the agent gradually learns actions that lead toward the goal while avoiding undesirable states. The use of a slippery FrozenLake environment also shows that SARSA can learn under stochastic state transitions.




