from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import tensorflow as tf
import random
import pdb

class Agent(base_agent.BaseAgent):
    
    def __init__(self):
        super(Agent,self).__init__()

        self.attack_coordinates = None


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
 
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def step(self, obs):
        super(Agent, self).step(obs)
        ##init select marines
        if obs.first():
            marines = self.get_units_by_type(obs,units.Terran.Marine)
            if self.can_do(obs,actions.FUNCTIONS.select_army.id):
                if self.unit_type_is_selected(obs,units.Terran.Marine):
                    return actions.FUNCTIONS.select_army("select")
        
#Attack with Zergling
        if self.unit_type_is_selected(obs,units.Terran.Marine):
            return actions.FUNCTIONS.select_army("select")
        if(len(zerglings) > 20):
            
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs,actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",self.attack_coordinates)
                if self.can_do(obs,actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")
            
#Create Spawning_pool        
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if (self.can_do(obs,actions.FUNCTIONS.Build_SpawningPool_screen.id)):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)
          
                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)

                return actions.FUNCTIONS.select_point("select_all_type", (drone.x,drone.y))  
#Create Overlord if no supply            
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
              free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
              if free_supply == 0:
                if (self.can_do(obs,actions.FUNCTIONS.Train_Overlord_quick.id)):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")
#Train zergling
              if (self.can_do(obs,actions.FUNCTIONS.Train_Zergling_quick.id)):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")

        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        
        if len(larvae) > 0:
            larva = random.choice(larvae)
      
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x,larva.y))
        return actions.FUNCTIONS.no_op()

def main(unused_argv):
    agent = Agent()
    try:
        while True:
            with sc2_env.SC2Env(
            map_name="CollectMineralShards",
            agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),use_feature_units=True),
            step_mul=16,
            game_steps_per_episode=0,           
            visualize=True) as env:        
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
