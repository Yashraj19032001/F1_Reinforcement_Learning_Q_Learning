import sys
import pygame
import time
import numpy as np
import gymnasium as gym
import os

class YashrajEnv(gym.Env):
    """
    Custom Gym environment for an F1-Racing game.
    """

    def __init__(self, grid_size=10, initial_fuel=17) -> None:
        """
        Initialize the environment.
        
        Parameters:
        - grid_size: 5x5 grid size.
        - initial_fuel: The initial amount of fuel available.
        """
        super(YashrajEnv, self).__init__()

        # Environment configuration
        self.grid_size = grid_size
        self.cell_size = 80
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array([9, 9])
        self.done = False
        self.red_flag_states = []
        self.yellow_flag_states = []
        self.red_bull_states = []
        self.track_states = []
        self.game_started = False
        self.goal_reached = False
        self.goal_reached_time = None
        self.fuel = initial_fuel
        self.out_of_fuel = False

        # Action-space: 4 discrete actions (up, down, right, left)
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: 2D position on the grid
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        # Initialize the window
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size * self.grid_size, self.cell_size * self.grid_size))

        # Determine the base directory
        self.base_dir = os.path.dirname(__file__)

        # Define the asset paths
        self.font_path = os.path.join(self.base_dir, 'assets', 'fonts', 'Formula1-Regular.ttf')
        self.agent_image_path = os.path.join(self.base_dir, 'assets', 'images', 'Agent.jpg')
        self.goal_image_path = os.path.join(self.base_dir, 'assets', 'images', 'Goal.jpg')
        self.red_flag_image_path = os.path.join(self.base_dir, 'assets', 'images', 'Red_flag.jpeg')
        self.yellow_flag_image_path = os.path.join(self.base_dir, 'assets', 'images', 'Yellow_flag.jpg')
        self.red_bull_image_path = os.path.join(self.base_dir, 'assets', 'images', 'Redbull.webp')
        self.start_sound_path = os.path.join(self.base_dir, 'assets', 'sounds', 'Start_audio.mp3')
        self.red_bull_sound_path = os.path.join(self.base_dir, 'assets', 'sounds', 'Redbull.mp3')
        self.finish_sound_path = os.path.join(self.base_dir, 'assets', 'sounds', 'Goal.mp3')
        self.red_bull_1_sound_path = os.path.join(self.base_dir, 'assets', 'sounds', 'Redbull_1.mp3')

        # Load a stylistic font
        self.font = pygame.font.Font(self.font_path, 24)
        self.large_font = pygame.font.Font(self.font_path, 120)

        # Load images
        self.agent_image = pygame.image.load(self.agent_image_path)
        self.goal_image = pygame.image.load(self.goal_image_path)
        self.red_flag_states_image = pygame.image.load(self.red_flag_image_path)
        self.yellow_flag_states_image = pygame.image.load(self.yellow_flag_image_path)
        self.red_bull_states_image = pygame.image.load(self.red_bull_image_path)

        # Scale images to fit grid cells
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))
        self.red_flag_states_image = pygame.transform.scale(self.red_flag_states_image, (self.cell_size, self.cell_size))
        self.yellow_flag_states_image = pygame.transform.scale(self.yellow_flag_states_image, (self.cell_size, self.cell_size))
        self.red_bull_states_image = pygame.transform.scale(self.red_bull_states_image, (self.cell_size, self.cell_size))

        # Initialize sounds
        pygame.mixer.init()
        self.start_sound = pygame.mixer.Sound(self.start_sound_path)
        self.red_bull_sound = pygame.mixer.Sound(self.red_bull_sound_path)
        self.finish_sound = pygame.mixer.Sound(self.finish_sound_path)
        self.red_bull_1_sound = pygame.mixer.Sound(self.red_bull_1_sound_path)

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.state = np.array([1, 1])  # Initial position
        self.done = False
        self.reward = 0
        self.goal_reached = False
        self.goal_reached_time = None
        self.out_of_fuel = False
        self.fuel = 17  # Max fuel to reach the goal via the shortest path

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )  # Distance to goal for the initial state

        return self.state, self.info

    def add_red_bull_states(self, red_bull_state_coordinates):
        """
        Add coordinates of red bull states to the environment.
        """
        self.red_bull_states.append(np.array(red_bull_state_coordinates))

    def add_red_flag_states(self, red_flag_state_coordinates):
        """
        Add coordinates of red flag states to the environment.
        """
        self.red_flag_states.append(np.array(red_flag_state_coordinates))

    def add_yellow_flag_states(self, yellow_flag_state_coordinates):
        """
        Add coordinates of yellow flag states to the environment.
        """
        self.yellow_flag_states.append(np.array(yellow_flag_state_coordinates))

    def add_track_states(self, track_state_coordinates_list):
        """
        Add coordinates of track states to the environment.
        """
        for track_state_coordinates in track_state_coordinates_list:
            self.track_states.append(np.array(track_state_coordinates))

    def step(self, action):
        """
        Execute the given action in the environment.
        
        Parameters:
        - action: The action to be executed (0=up, 1=down, 2=right, 3=left).
        
        Returns:
        - state: The new state of the environment.
        - reward: The reward obtained by the action.
        - done: Whether the episode is finished.
        - info: Additional information.
        """
        valid_actions = [0, 1, 2, 3]
        if action not in valid_actions:
            print("Invalid input")
            return self.state, self.reward, self.done, self.info

        # Up
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1

        # Down
        if action == 1 and self.state[0] < self.grid_size - 1:
            self.state[0] += 1

        # Right
        if action == 2 and self.state[1] < self.grid_size - 1:
            self.state[1] += 1

        # Left
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # Deduct fuel with each action
        self.fuel -= 0.5

        if self.fuel <= 0:
            self.done = True
            self.out_of_fuel = True
            print("Fuel exhausted! Game Over.")

        # Calculate reward
        reward = -0.05  # Default penalty for each step
        if np.array_equal(self.state, self.goal):  # Check goal condition
            reward += 10 + self.fuel  # Reward for reaching goal plus remaining fuel
            self.done = True
            self.goal_reached = True
            self.goal_reached_time = time.time()
            self.finish_sound.play()
        elif any(np.array_equal(self.state, each_red_flag) for each_red_flag in self.red_flag_states):  # Check red-flag-states
            reward += -3
            self.done = False
        elif any(np.array_equal(self.state, each_yellow_flag) for each_yellow_flag in self.yellow_flag_states):  # Check yellow-flag states
            reward += -2
            self.done = False
        elif any(np.array_equal(self.state, each_red_bull) for each_red_bull in self.red_bull_states):  # Check red_bull states
            reward += -1
            self.done = False
        elif any(np.array_equal(self.state, each_track) for each_track in self.track_states):  # Check track states
            reward += 0
            self.done = False

        # Play sounds based on specific states
        if np.array_equal(self.state, [1, 2]):
            self.start_sound.play()
        if np.array_equal(self.state, [4, 6]):
            self.red_bull_sound.play()
        if np.array_equal(self.state, [7, 7]):
            self.red_bull_1_sound.play()

        self.reward += reward  # Accumulate the reward

        # Update info
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )

        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        Render the current state of the environment.
        """
        # Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Default background color
        background_color = (78, 138, 108)  # Green

        # Check if the agent is in a red_flag state
        if any(np.array_equal(self.state, each_red_flag) for each_red_flag in self.red_flag_states):
            background_color = (255, 0, 0)  # Red
        # Check if the agent is in a yellow_flag state
        elif any(np.array_equal(self.state, each_yellow_flag) for each_yellow_flag in self.yellow_flag_states):
            background_color = (255, 255, 0)  # Yellow

        # Fill the background with the selected color
        self.screen.fill(background_color)

        # Draw Grid lines
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), grid, 1)

        # Draw the Goal-state
        goal_pos = (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size)
        self.screen.blit(self.goal_image, goal_pos)

        # Draw the track states
        for each_track in self.track_states:
            track_pos = pygame.Rect(each_track[1] * self.cell_size, each_track[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (169, 169, 169), track_pos)

        # Draw the red flag states
        for each_red_flag in self.red_flag_states:
            red_flag_pos = (each_red_flag[1] * self.cell_size, each_red_flag[0] * self.cell_size)
            self.screen.blit(self.red_flag_states_image, red_flag_pos)

        # Draw the yellow flag states
        for each_yellow_flag in self.yellow_flag_states:
            yellow_flag_pos = (each_yellow_flag[1] * self.cell_size, each_yellow_flag[0] * self.cell_size)
            self.screen.blit(self.yellow_flag_states_image, yellow_flag_pos)

        # Draw the red-bull states
        for each_red_bull in self.red_bull_states:
            red_bull_pos = (each_red_bull[1] * self.cell_size, each_red_bull[0] * self.cell_size)
            self.screen.blit(self.red_bull_states_image, red_bull_pos)

        # Draw the agent
        agent_pos = (self.state[1] * self.cell_size, self.state[0] * self.cell_size)
        self.screen.blit(self.agent_image, agent_pos)

        # Display remaining fuel
        fuel_text = self.font.render(f"Fuel: {self.fuel}", True, (139, 0, 0))
        self.screen.blit(fuel_text, (10, 10))

        # Check if the agent has reached the goal and display "You Won" message
        if self.goal_reached:
            text_surface = self.large_font.render("You Won!", True, (255, 0, 0))
            self.screen.blit(text_surface, (self.cell_size * self.grid_size // 2 - text_surface.get_width() // 2, self.cell_size * self.grid_size // 2 - text_surface.get_height() // 2))
            pygame.display.flip()
            self.wait_for_close()
        elif self.out_of_fuel:
            text_surface = self.font.render("Ran Out of Fuel!", True, (255, 0, 0))
            self.screen.blit(text_surface, (self.cell_size * self.grid_size // 2 - text_surface.get_width() // 2, self.cell_size * self.grid_size // 2 - text_surface.get_height() // 2))
            pygame.display.flip()
            self.wait_for_close()

        # Update contents on the window
        pygame.display.flip()

    def wait_for_close(self):
        """
        Wait for the user to close the window.
        """
        # Delay before waiting for user to close the window
        time.sleep(8)

        waiting = False  # Keep the window open until the user closes it
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
        self.close()

    def close(self):
        """
        Close the pygame window.
        """
        pygame.quit()


# Step 2: Create an instance of the environment and test the implementation
my_env = YashrajEnv(grid_size=10)

# Red flag states
my_env.add_red_flag_states(red_flag_state_coordinates=(4, 1))
my_env.add_yellow_flag_states(yellow_flag_state_coordinates=(6, 1))
my_env.add_red_bull_states(red_bull_state_coordinates=(3, 7))
my_env.add_red_bull_states(red_bull_state_coordinates=(6, 8))

# Track-states coordinates
track_state_coordinates_list = [
    (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 9), (2, 9),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5),
    (3, 6), (3, 7), (3, 8), (3, 9), (4, 1), (4, 2), (4, 3),
    (4, 4), (4, 5), (4, 6), (4, 9), (5, 1), (5, 6), (5, 9),
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
    (6, 8), (6, 9), (7, 7), (7, 9), (8, 7), (8, 8), (8, 9)
]

my_env.add_track_states(track_state_coordinates_list)

# Step 3: Manually control the environment and check the implementation
# --------
observation, info = my_env.reset()
print(f"Initial state: {observation}, Info: {info}")

for _ in range(20):
    # Choose a random action
    action = int(input("Choose action (0=Up, 1=Down, 2=Right, 3=Left): "))  # manual action
    # action = my_env.action_space.sample()  # random action

    # Take the action in your environment
    new_state, reward, done, info = my_env.step(action)
    print(f"New state: {new_state}, Reward: {reward}, Done: {done}, Info: {info}")

    # Render the environment
    my_env.render()

    # Check for termination condition
    if done:
        break
