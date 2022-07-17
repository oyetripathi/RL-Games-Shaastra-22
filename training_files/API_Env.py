from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from nqueens import Queen
from stable_baselines import DQN
import numpy as np
# ----------------------------------------------------- #


class APIVasuki(Env):
    @staticmethod
    def _food_position_(n):
        # Using the N-Queens problem to uniformly distribute the food spawning location
        qq = Queen(n)
        food_pos = np.empty(shape=[0, 2])
        chess = qq.queen_data[0]
        for x in range(n):
            for y in range(n):
                if chess[y][x] == 1:
                    arr = np.array([[x, y]])
                    food_pos = np.append(food_pos, arr, axis=0)
        # Returning the n food locations which are spatially distributed uniformly
        return food_pos

    def _init_agent_(self, score=0):
        # Creating a dictionary to store the information related to the agent
        agent = dict()
        # Set initial direction of head of the Snake :  North = 0, East = 1, South = 2, West = 3
        agent['head'] = np.random.randint(low=0, high=4, size=(1)).item()
        # The score for each agent
        agent['score'] = score
        # Set initial position
        agent['state'] = np.random.randint(low=0, high=self.n, size=(2))
        # Velocity of the snake
        agent['velocity'] = 1
        # Returning the Agent Properties
        return agent

    def __init__(self, n, rewards, game_length=100):
        # Parameters
        self.n = n
        self.rewards = rewards
        # Actions we can take : left = 0, forward = 1, right = 2
        self.action_space = Discrete(3)
        # The nxn grid
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3, self.n, self.n), dtype=np.float32) # MultiDiscrete([self.n, self.n])
        # Set Total Game length
        self.game_length = game_length
        self.game_length_ = self.game_length
        # Set Food Spawning locations. Totally there are only n locations
        self.foodspawn_space = self._food_position_(self.n)
        # Out of the n food locations, at any time only n/2 random locations have food
        self.live_index = np.random.choice(len(self.foodspawn_space), size=(self.n // 2), replace=False)
        self.live_foodspawn_space = self.foodspawn_space[self.live_index]
        # Initializing the Agents
        self.agentA = self._init_agent_()
        self.agentB = self._init_agent_()
        encoded, _ = self.encode()
        self.history = []
        self.model_b = DQN.load('benchmarks/double_dqn_sb_1.zip')

    def _movement_(self, action, agent):
        # Loading the states
        illegal = 0  # If the snake hits the walls
        n = self.n
        head = agent['head']
        state = agent['state'].copy()
        velocity = agent['velocity']
        score = agent['score']
        # Applying the Action
        if action == 0:  # Go Left
            if head == 0:
                if state[1] == velocity - 1:  # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
            elif head == 1:
                if state[0] == velocity - 1:  # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
            elif head == 2:
                if state[1] == n - velocity:  # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head == 3:
                if state[0] == n - velocity:  # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2
        elif action == 1:  # Move Forward
            if head == 0:
                if state[0] == velocity - 1:  # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
            elif head == 1:
                if state[1] == n - velocity:  # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head == 2:
                if state[0] == n - velocity:  # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2
            elif head == 3:
                if state[1] == velocity - 1:  # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
        elif action == 2:  # Go Right
            if head == 0:
                if state[1] == n - velocity:  # Right Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, velocity])
                head = 1
            elif head == 1:
                if state[0] == n - velocity:  # Bottom Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([velocity, 0])
                head = 2
            elif head == 2:
                if state[1] == velocity - 1:  # Left Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([0, -velocity])
                head = 3
            elif head == 3:
                if state[0] == velocity - 1:  # Top Wall
                    illegal = 1
                    change = np.array([0, 0])
                else:
                    change = np.array([-velocity, 0])
                head = 0
        # Updating the agent properties
        modified = {'head': head, 'state': state + change, 'score': score, 'velocity': velocity}
        return modified, illegal

    def _reward_(self, agent, illegal):
        # Loading the states
        head = agent['head']
        state = agent['state'].copy()
        velocity = agent['velocity']
        score = agent['score']
        # Calculating the reward
        if illegal == 1:  # If the snake hits the wall
            reward = self.rewards['Illegal']
        else:
            if True in np.all((state == self.live_foodspawn_space), axis=1):
                # Finding the index of the state
                index = np.where(np.all((state == self.live_foodspawn_space), axis=1) == True)[0].item()
                # Computing the empty foodspawn spaces
                empty_foodspawn_space = [space for space in self.foodspawn_space if
                                         space not in self.live_foodspawn_space]
                # Removing the state from live foodspawn space
                self.live_foodspawn_space = np.delete(self.live_foodspawn_space, index, 0)
                # Updating the live foodspawn space
                addition = np.random.choice(len(empty_foodspawn_space), size=1, replace=False)
                self.live_foodspawn_space = np.append(self.live_foodspawn_space,
                                                      np.expand_dims(empty_foodspawn_space[addition.item(0)], axis=0),
                                                      axis=0)
                assert len(set([(x, y) for (x, y) in self.live_foodspawn_space])) == 4
                # If the snake lands on the food
                reward = self.rewards['Food']
            else:
                # If the snake just moves
                reward = self.rewards['Movement']
        return reward

    def step(self, action):
        state_b = self.get_state('b')
        actionA = action
        actionB, __ = self.model_b.predict(state_b, deterministic=True)
        # Applying the actions
        self.agentA, illegalA = self._movement_(actionA, self.agentA)
        self.agentB, illegalB = self._movement_(actionB, self.agentB)
        # Calculating the reward
        if (self.agentA['state'] == self.agentB['state']).all():
            if self.agentA['score'] > self.agentB['score']:
                rewardA = 5 * abs(self.agentB['score'] // (self.agentA['score'] - self.agentB['score']))
                rewardB = - 3 * abs(self.agentB['score'] // (self.agentA['score'] - self.agentB['score']))
                _ = self._reward_(self.agentA, illegalA)
                score = self.agentB['score']
                while True:
                    self.agentB = self._init_agent_(score)
                    if (self.agentB['state'] != self.agentA['state']).all():
                        _ = self._reward_(self.agentB, illegalB)
                        break
            elif self.agentA['score'] < self.agentB['score']:
                rewardA = - 3 * abs(self.agentA['score'] // (self.agentA['score'] - self.agentB['score']))
                rewardB = 5 * abs(self.agentA['score'] // (self.agentA['score'] - self.agentB['score']))
                _ = self._reward_(self.agentB, illegalB)
                score = self.agentA['score']
                while True:
                    self.agentA = self._init_agent_(score)
                    if (self.agentA['state'] != self.agentB['state']).all():
                        _ = self._reward_(self.agentA, illegalA)
                        break
            elif self.agentA['score'] == self.agentB['score']:
                rewardA = - abs(self.agentA['score'] // 2)
                rewardB = - abs(self.agentB['score'] // 2)
                while True:
                    self.agentA = self._init_agent_(score=self.agentA['score'])
                    if (self.agentA['state'] != self.agentB['state']).all():
                        _ = self._reward_(self.agentA, illegalA)
                        break
                while True:
                    self.agentB = self._init_agent_(score=self.agentB['score'])
                    if (self.agentB['state'] != self.agentA['state']).all():
                        _ = self._reward_(self.agentB, illegalB)
                        break
        else:
            rewardA = self._reward_(self.agentA, illegalA)
            rewardB = self._reward_(self.agentB, illegalB)
        # Adding the reward to the score
        self.agentA['score'] = self.agentA['score'] + rewardA
        self.agentB['score'] = self.agentB['score'] + rewardB
        # Updating history
        encoded, _ = self.encode()
        self.history.append(
            {"agentA": self.agentA, "agentB": self.agentB, "live_foodspawn_space": self.live_foodspawn_space,
             "encoded": encoded,
             "rewardA": rewardA, "actionA": actionA, "rewardB": rewardB, "actionB": actionB})
        # Check if game is done
        self.game_length -= 1
        if self.game_length <= 0:
            done = True
        else:
            done = False
        # Set placeholder for info
        info = {'agentA': 0, 'agentB': 0}
        # state_a = self.get_state('a')
        state_a = self.get_cnn_state()
        return state_a, rewardA, done, info

    def encode(self):
        # Loading the states
        encoder = {'blank': 0, 'foodspawn_space': 1, 'agentA': 2, 'agentB': 3}
        state = np.zeros((self.n, self.n))
        live_foodspawn_space = self.live_foodspawn_space.astype(int)
        snakeA = self.agentA['state']
        snakeB = self.agentB['state']
        # Adding the agents and snakes
        state[live_foodspawn_space[:, 0], live_foodspawn_space[:, 1]] = encoder['foodspawn_space']
        state[snakeA[0], snakeA[1]] = encoder['agentA']
        state[snakeB[0], snakeB[1]] = encoder['agentB']
        # One-Hot encoding the state
        encoded = np.eye(len(encoder.keys()))[state.astype(int)]
        encoded = np.moveaxis(encoded, -1, 0)
        # Returning the encoded and state
        return encoded, state

    def get_cnn_state(self):
        cnn_state = np.zeros((self.n, self.n, 3))
        live_foodspawn_space = self.live_foodspawn_space.astype(int)
        for i in range(live_foodspawn_space.shape[0]):
            cnn_state[live_foodspawn_space[i, 0], live_foodspawn_space[i, 1], 0] = 1
        A = self.agentA['state']
        B = self.agentB['state']
        cnn_state[A[0], A[1], 1] = self.agentA['head'] + 1
        cnn_state[B[0], B[1], 2] = self.agentB['head'] + 1
        return np.swapaxes(cnn_state, 0, 2)

    def reset(self):
        # Reset Total Game length
        self.game_length = self.game_length_
        # Reset Food Spawning locations
        self.foodspawn_space = self._food_position_(self.n)
        # Reset Live Food Spawning locations
        self.live_index = np.random.choice(len(self.foodspawn_space), size=(self.n // 2), replace=False)
        self.live_foodspawn_space = self.foodspawn_space[self.live_index]
        # Reset Agents
        self.agentA = self._init_agent_()
        self.agentB = self._init_agent_()
        # Clear History
        self.history = []
        encoded, _ = self.encode()
        # state_a = self.get_state('a')
        state_a = self.get_cnn_state()
        return state_a  # np.ravel(encoded[1:])

    def get_state(self, c):
        if c == 'a':
            agent = self.agentA
            enemy = self.agentB
        else:
            agent = self.agentB
            enemy = self.agentA
        state = list()
        # -------------------------------------------------
        for coord in self.live_foodspawn_space:
            for i in range(coord.shape[0]):
                x = np.zeros(shape=7)
                idx = int(coord[i] - agent['state'][i])
                if idx > 0:
                    x[idx-1] = 1
                else:
                    x[-idx-1] = -1
                for j in x:
                    state.append(j)
        # -----------------------------------------------------
        for i in range(enemy['state'].shape[0]):
            x = np.zeros(shape=7)
            idx = int(enemy['state'][i] - agent['state'][i])
            if idx > 0:
                x[idx-1] = 1
            else:
                x[-idx-1] = -1
            for j in x:
                state.append(j)
        # -------------------------------------------------------
        if c != 'b':
            row, col = agent['state'][0], agent['state'][1]
            x = np.zeros(shape=3)
            if agent['head'] == 0:
                if row == 0:
                    x[0] = -1
                if col == 0:
                    x[1] = -1
                if col == 7:
                    x[2] = -1
            elif agent['head'] == 1:
                if row == 0:
                    x[1] = -1
                if row == 7:
                    x[2] = -1
                if col == 7:
                    x[0] = -1
            elif agent['head'] == 2:
                if row == 7:
                    x[0] = -1
                if col == 0:
                    x[2] = -1
                if col == 7:
                    x[1] = -1
            else:
                if row == 0:
                    x[2] = -1
                if col == 0:
                    x[0] = -1
                if row == 7:
                    x[1] = -1
            for j in x:
                state.append(j)
            # -----------------------------------------
            state.append(agent['score'] - enemy['score'])
        # -----------------------------------------
        x = np.zeros(shape=4)
        x[agent['head']] = 1
        for j in x:
            state.append(j)
        return np.array(state)


'''config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100}
env = APIVasuki(**config)
env.reset()
for i in range(100):
    action_A = env.action_space.sample()
    obs, reward, done, info = env.step(action_A)
    print(obs.shape)
    if done:
        print("episode finished")
        break'''


