'''
Source code for paper "Learning modular robot control policies" in Transactions on Robotics
Julian Whitman, Dec. 2022. 
'''
# Reads joystick values using the pygame module and updates lists to read
# commands from joystick

import pygame
import numpy as np
import time

def init_joystick():
    pygame.init()
    pygame.joystick.init()
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(joy.get_name())
    return joy

def read(joy):
    pygame.event.get()
    axes = []
    buttons = []
    povs = []
    for ax in range(joy.get_numaxes()):
        axes.append(joy.get_axis(ax))
    for button in range(joy.get_numbuttons()):
        buttons.append( joy.get_button(button))
    for hat in range(joy.get_numhats()):
        povs.append( joy.get_hat(hat) )
    return axes, buttons, povs

# Test for readJoy functionality
def test():
    pygame.init()
    pygame.joystick.init()
    pygame.joystick.Joystick(0).init()

    vL.axes = np.array(pygame.joystick.Joystick(0).get_numaxes()*[0.0])
    vL.buttons = np.array(pygame.joystick.Joystick(0).get_numbuttons()*[0])
    vL.povs = np.array(pygame.joystick.Joystick(0).get_numhats()*[[0,0]])
    while (1):
        pygame.event.get()
        read()
        time.sleep(1)

# test out
if __name__ == '__main__':
    joy = init_joystick()
    time_start = time.time()
    while time.time()<time_start+10:
        axes, buttons, povs = read(joy)
        print(np.round(axes,2), buttons, povs)
        print(np.arctan2(-axes[0],-axes[1]))

        time.sleep(0.2)