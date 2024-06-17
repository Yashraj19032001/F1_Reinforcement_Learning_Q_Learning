# F1_Reinforcement_Learning_Q_Learning
This is a Custom Gym Environment named: YashrajEnv

Overview
--------
YashrajEnv is a custom Gym environment designed for an F1-Racing game simulation. The environment provides a grid-based track where an agent navigates to reach a goal while managing fuel and encountering various obstacles. Currently controlled manually, will add Q-Learning(Reinforcement Learning method in the upcoming days).

Features
--------
- Grid Size: Default grid size is 10x10.
- Initial Fuel: The agent starts with a specified amount of fuel.
- Actions: Four discrete actions available (up, down, right, left).
- Observation Space: 2D position on the grid.
- Rewards: The agent receives rewards based on its actions and proximity to the goal or obstacles.
- Agent: It includes Lewis Hamilton driving the Mercedes W11 F-1 car.
- Obstacles: Includes red flags, yellow flags, and other vehicle, each affecting the agent's reward or behavior.
- Visualization: Renders the environment using Pygame for visual feedback.

Installation
------------
1. Clone the repository:
git clone https://github.com/Yashraj19032001/F1_Reinforcement_Learning_Q_Learning.git
cd your_repository

2. Install dependencies:
pip install -r requirements.txt


Usage
-----
Creating the Environment:

import gymnasium as gym

# Create an instance of the environment
env = gym.make('YashrajEnv', grid_size=10, initial_fuel=17)

Environment Configuration:

- Adding Obstacles: Red flags, yellow flags, and red bull states can be added using specific methods (add_red_flag_states, add_yellow_flag_states, add_red_bull_states).

Example:
env.add_red_flag_states(red_flag_state_coordinates=(4, 1))

- Setting Track States: Define track coordinates to customize the environment layout.

Example:
track_state_coordinates_list = [(0, 5), (0, 6), ...]
env.add_track_states(track_state_coordinates_list)

Running the Environment:

observation, info = env.reset()
done = False

while not done:
 action = int(input("Choose action (0=Up, 1=Down, 2=Right, 3=Left): "))
 observation, reward, done, info = env.step(action)
 env.render()

Rendering:

The environment renders using Pygame, providing visual feedback on the agent's position, obstacles, and goal.

Contributing
------------
Contributions are welcome! Feel free to fork the repository and submit pull requests.

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.


