from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import csv

""""learning notes: 
    7/31/2019
    *Can check all available actions with python -m pysc2.bin.valid_actions in console 
    * 
    
    
    Implementation notes:
    7/31/2019
    *In this version we use 'feature_units' observation data. this includes all the observable units (not sure if screen or minimap) in an array.
     see more about what information each array contain in pysc2/lib/features.py (class FeatureUnit).
     Most important for our use is the feature.units.unit_type (MineralShard and Marine) and feature_units.x and y for screen coordinates.
     
        * One question arise from the previous note; whats the size of the state? We know that we will have max 20 mineralshards at given time.
          with the above description, we have for each shard, [x,y] coordinate (total 40 scalars) + 2 marine coordinate. 
          But for each shard we collect, the amount of shards decrease. In the literature we have checked so far, the size of the state is
          always held constant. So should we set "collected shards" to some value (example: negative), or is there a model where the size of
          the state can change.
     
        * Another question is if we need to take several frames of data for markov property. Do we, for example need the velocity of marines
          included in our state (for this, we can simply collect the actions and deduce velocity by taking the difference and divide with frame).
    
    * One simplification: In RL, when Q table is approximated by a Neural network, we can either choose the action along the input and the 
      Q value as output, or omit the action as input, and get the Q value for each action in the output.
      For our case, with the screen resolution of 84*84, if our action is Move_screen(x,y), the output layer has to be 84*84 = 7056!
      We could potentially reduce this by the Move_screen() ONLY representing the direction, meaning the x,y values can only be the edges
      of the screen. This will reduce the output layer to 84*4 = 336! 
      
      
      TODO: 
    """


class Agent(base_agent.BaseAgent):

    def __init__(self, learning_rate, state_size=44,
                 action_size=84*4, hidden_size=24, epsilon=0.5,
                 gamma=0.7, epsilon_min=0.01,
                 epsilon_decay=0.8, batch_size=32,
                 decay=0.01):
        super(Agent, self).__init__()
        # Pysc2 related variables
        self.coordinates = None # This is the coordinate (x,y) on the minimap the action is supposed to learn
        self.coordinates = tuple(np.random.randint(84, size=2))
        self.MineralShardID = 1680  # the feature_unit (or pysc2 general unit) ID
        self.MarineID = 48  # check more in pysc2/lib/features.py
        self.screen_dimension = 84
        # for Memorizing
        self.state = []
        self.taken_action = 0
        self.reward = 0
        # Keras / RL related variables
        # RL parameters
        self.memory = deque(maxlen=100000)
        self.state_size = state_size # 2 * 20 + 2 * 2 (shard + marines) = 44
        self.gamma = gamma
        self.action_size = action_size
        # (x,y, no_op or not) = 84*84 + 1) => dimensionality reduction (84*4 + 1, or remove noop) => (84*4)
        self.batch_size = batch_size
        # epsilon learning param
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Dense(hidden_size, input_dim=state_size, activation='tanh'))
        self.model.add(layers.Dense(hidden_size, activation='tanh'))
        self.model.add(layers.Dense(action_size, activation='linear'))
        self.model.compile(loss='mse',
                           optimizer=Adam())  # potential improvement in AdamOpt

    def action(self, state, no_random=False):

        if (np.random.random() <= self.epsilon) and not no_random:
            return np.random.randint(self.action_size)
        return np.argmax(self.model.predict(state))

    def memorize(self, obs):
        next_state = self.get_state(obs)
        reward = obs.reward
        done = obs.last()
        self.memory.append((self.state, self.taken_action, reward, next_state, done))
        self.state = next_state

    def forget(self):
        self.memory.clear()

    def learn(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        x_batch, y_batch = [], []
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in batch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(
                self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    # Should be used between sending action to env, but not when memorizing

    def action_dimension_reduction(self, action: int):  # action [0,84*4-1]
        row = (action // self.screen_dimension)
        col = (action % self.screen_dimension)
        x, y = 0, 0
        if row <= 1:  # x = 0,1
            x = row * (self.screen_dimension-1)
            y = col
        else: # y = 0,1
            x = col
            y = (self.screen_dimension-1) * (row - 2)
        return tuple([x, y])

    def get_unit_coordinates(self, obs, unitID):
         return [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == unitID]

    def get_state(self,obs):
        mineral_shard_position = self.get_unit_coordinates(obs, self.MineralShardID)
        if len(mineral_shard_position) < 20:
            mineral_shard_position.extend([[-1, -1] for _ in range(20-len(mineral_shard_position))])
        marine_position = self.get_unit_coordinates(obs, self.MarineID)
        state_combined = np.array(marine_position + mineral_shard_position)
        return state_combined.reshape(1, -1)

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def step(self, obs, no_random = False):
        super(Agent, self).step(obs)
        # init select marines
        if obs.first():
            self.state = self.get_state(obs)
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                if not self.unit_type_is_selected(obs, units.Terran.Marine):
                    return actions.FUNCTIONS.select_army("select")
        self.state = self.get_state(obs)
        self.taken_action = self.action(self.state, no_random)
        self.coordinates = self.action_dimension_reduction(self.taken_action)
        # Move with marines
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs,actions.FUNCTIONS.Move_screen.id):
                return actions.FUNCTIONS.Move_screen("now", self.coordinates)
        return actions.FUNCTIONS.no_op()

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model.load_weights(model_name)

def main(unused_argv):
    agent = Agent(learning_rate=1e-5)
    amount = 100
    last_run = True # set to true if last run removes random step choice
    i = 0
    j = 0
    model_name ='CollectMineralShard_model.h5'
    reward_total = []
    try:
        import os.path
        try:
            if os.path.isfile(model_name):
                agent.load_model(model_name)
                print(f'Loading and using file: {model_name} ')
        except:
            pass
        with sc2_env.SC2Env(
                map_name="CollectMineralShards",
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=True) as env:
            agent.setup(env.observation_spec(), env.action_spec())
            no_random = True
            while i < amount:
                if last_run and (amount - i <= 2):  # due to some weird error, every 2 game is shut down.
                    no_random = True
                    print("last run. The predicted step is fully used")
                reward = 0
                timesteps = env.reset()
                agent.reset()
                agent.forget()  # clear memory
                while True:
                    step_actions = [agent.step(timesteps[0], no_random)]
                    reward += timesteps[0].reward
                    if timesteps[0].last():
                        agent.memorize(timesteps[0])
                        if reward:
                            j += 1
                            agent.learn()
                            print(f'learning done: {j+1}({i+1})/{amount} \n')
                        reward_total.append(reward)
                        break
                    timesteps = env.step(step_actions)
                    if not timesteps[0].first():
                        agent.memorize(timesteps[0])
                i += 1
        agent.save_model(model_name)
        print(f'The game was run {amount} times and obtained the following rewards: {reward_total}')
    except KeyboardInterrupt:
        pass
    with open("reward_list.csv", "w") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(reward_total)


if __name__ == "__main__":
    app.run(main)
