# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional TLC algorithms and reinforcement learning based methods.


## About this fork

 * Adds flow and roadmap files from [Colight's repo](https://github.com/wingsweihua/colight/tree/master/data)
 * Random agent for lower bound performance
 * (WIP) Better and unified logging without conflicting files. (copying what was done with run_dqn.py) 
 * (WIP) Insta plotting

## Requirements

 * CityFlow
 * Tensorflow 1.1x
 * OpenAI gym
 * Numpy

# Run

 Just run any of _the run\__* scripts
 
  `python run_dqn.py`
