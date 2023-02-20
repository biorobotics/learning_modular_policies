# modular_mbrl
Model based reinforcement learning for modular robots

dependencies:


pybullet for simulation
(pip install pybullet)

pytorch for learning

mpc package for trajectory optimization/planning
(pip install mpc)
Note: go to /home/<USER>/.local/lib/python3.6/site-packages/mpc/pnqp.py and comment out print("[WARNING] pnqp warning: Did not converge") 


pygame for joystick reading 
(pip install pygame)

If you want to compile the modular robot urdfs from xacros, requires a ROS verison of at least kinetic and with at least the xacro command

The physical robot control uses the hebi python API. 