import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, "/home/dan/myCode/spinningup")


import spinup
"""RUN agent code runs code I made and optionally also the SpinningUpRL code. 
This code expect the spinning up RL git repo to be set by the sys.path.insert above. 
(Note that this might fuck up some IDE) """



if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser()

   parser.add_argument("-env", help="the gym environment it runs on", default="MountainCar-v0")
   parser.add_argument("-ma", "--my-agent", help="Choose my agent (currently only support REINFORCE)",
                       default="REINFORCE")
   parser.add_argument("-sa", "--spin-agent", help="Agent from SpinningUPRL", default="REINFORCE")
   from REINFORCE import REINFORCE
   my_agent = REINFORCE()