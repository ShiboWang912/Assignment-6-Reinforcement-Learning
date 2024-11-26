import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Create the LunarLander environment
env = gym.make('LunarLander-v2')

# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.epsilon = 1.0   # Initial epsilon value
        self.epsilon_min = 0.02  # Final epsilon value
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output layer with 4 actions
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Hyperparameters
num_episodes = 1000
max_steps = 200
batch_size = 64
state_size = env.observation_space.shape[0]  # 8-dimensional state space
action_size = env.action_space.n  # 4 possible actions

# Initialize the agent
agent = DQNAgent(state_size, action_size)

# List to store metrics for plotting
episode_rewards = []

# Training loop
for e in range(num_episodes):
    state, _ = env.reset()  # Reset the environment and ignore the 'info'
    state = np.reshape(state, [1, state_size])  # Reshape the state to match the input shape
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)  # Unpack 5 values
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    agent.replay(batch_size)
    episode_rewards.append(total_reward)

    # Print the progress
    if (e + 1) % 100 == 0:
        print(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

agent.model.save("dqn_lunar_lander_model.h5")


import torch
import torch.nn as nn
import torch.optim as optim

# Define the PyTorch model
class DQNAgentPyTorch(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgentPyTorch, self).__init__()
        # Ensure that the architecture matches the Keras model's architecture
        self.fc1 = nn.Linear(state_size, 24)  # First layer with 64 units
        self.fc2 = nn.Linear(24, 24)          # Second layer with 64 units
        self.out = nn.Linear(24, action_size)  # Output layer for actions

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Initialize the PyTorch model
state_size = 8  # 8-dimensional state space
action_size = 4  # 4 possible actions
pytorch_model = DQNAgentPyTorch(state_size, action_size)
import tensorflow as tf
# Load the Keras model
keras_model = tf.keras.models.load_model("dqn_lunar_lander_model.h5")

# Load the PyTorch model
pytorch_model = DQNAgentPyTorch(state_size, action_size)

# Transfer weights from Keras to PyTorch (manually)
with torch.no_grad():
    # Get Keras weights
    keras_weights = keras_model.get_weights()

    # Set PyTorch model weights
    pytorch_model.fc1.weight.data = torch.Tensor(keras_weights[0].T)  # Keras weights are in (input, output), PyTorch expects (output, input)
    pytorch_model.fc1.bias.data = torch.Tensor(keras_weights[1])
    pytorch_model.fc2.weight.data = torch.Tensor(keras_weights[2].T)
    pytorch_model.fc2.bias.data = torch.Tensor(keras_weights[3])
    pytorch_model.out.weight.data = torch.Tensor(keras_weights[4].T)
    pytorch_model.out.bias.data = torch.Tensor(keras_weights[5])

# Save the PyTorch model as a .pt file
torch.save(pytorch_model.state_dict(), "dqn_model.pt")


# Plot training performance
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Performance')
plt.show()
