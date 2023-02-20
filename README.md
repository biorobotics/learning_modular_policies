**Overview**
Code used in the publication "Learning modular robot control policies." (see https://arxiv.org/abs/2105.10049) Trains and runs modular robot policies with model based reinforcement learning, from the Biorobotics Laboratory at Carnegie Mellon University. Written and maintained by Julian Whitman.

**System requirements**
- Training: NVIDIA GPU with minimum 8 Gb VRAM, ideally multiple GPUs with >12 Gb. 
- Running policies: most CPUs can run the policy, but we have only verified computers with at least four Intel i7 cores running it in real-time. 

**Dependencies**
- python3
- pybullet for simulation: pip3 install pybullet
- pytorch for deep neural networks: see https://pytorch.org/, install the version corresponding to your OS and GPU.
- scipy for interpolation utility: pip3 install scipy
- If you want to compile the modular robot urdfs from xacros, this requires a ROS verison of at least kinetic and with at least the xacro command installed.
- If you are using a joystick to control the trained policy, get pygame for joystick reading: pip install pygame
- The physical robot control (run_robot_policy.py) uses the hebi python API: pip install hebi-py, but is not needed for training or simulation, so most users will not need to install this package.
- Some analysis scripts (with file extension .ipynb) use jupyter notebook, but it is not necessary to run the training or simulation tests.

**Running**
- The first step after installing dependencies is to simulate a pre-trained policy with modular_policy/simulate_policy.py.
- If you would like to train the modular policy from scratch, the main modular policy training script is modular_policy/mbrl.py. See "Learning modular robot control policies" for more information about the training process and compute time.

**Repository contents**
- modular_policy contains scripts and utilities for training and executing modular policies.
- mpl_policy contains scripts and utilities for training and executing multi-layer perceptron policies, which serve as a basis of comparison.
- urdf contains the robot models used in simulations.



