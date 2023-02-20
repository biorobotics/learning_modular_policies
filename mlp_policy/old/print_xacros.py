'''
Systematically print out all permutations of parts on the chassis.
The xacros here adapt those from Hebi Robotics, www.hebirobotics.com

'''
from itertools import product 

import os
folder = 'urdf'
if not os.path.exists(folder):
    os.mkdir(folder)
cwd = os.path.dirname(os.path.realpath(__file__))

def moment_of_inertia_cuboid(dx, dy, dz, mass):
    Ixx = 1/12 * mass * (dy**2 + dz**2)
    Iyy = 1/12 * mass * (dx**2 + dy**2)
    Izz = 1/12 * mass * (dx**2 + dy**2)
    return Ixx, Iyy, Izz

# take a letter list and print a urdf
def print_xacros(letter_list):
    # fname = folder + letter_list + '.xacro'
    fname = os.path.join(folder,  letter_list + '.xacro')
    with open(fname, 'w') as handle:
        handle.write('<?xml version="1.0"?>\n')
        # handle.write('<robot name="' + letter_list + '">\n\n')
        handle.write('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="'+ letter_list + '"> \n\n')

        handle.write('<xacro:include filename="hebi.xacro"/>\n')
        handle.write('<xacro:include filename="leg.xacro"/>\n')
        handle.write('<xacro:include filename="wheel.xacro"/>\n')
        handle.write('<xacro:include filename="chassis.xacro"/>\n')
        handle.write('<xacro:chassis/>\n')

        handle.write('\n')

        # leg params
        
        mount_leg = [
            'mount_xyz="0.1524 0.0880 0.0095" mount_rpy="0 0 ${pi/6 - pi/2}"',
            'mount_xyz="0 0.1760    0.0095" mount_rpy="0 0 ${3*pi/6- pi/2}"',
            'mount_xyz="-0.1524 0.0880 0.0095" mount_rpy="0 0 ${5*pi/6- pi/2}"',
            'mount_xyz="-0.1524 -0.0880  0.0095" mount_rpy="0 0 ${7*pi/6- pi/2}"',
            'mount_xyz="0 -0.1760  0.0095" mount_rpy="0 0 ${9*pi/6- pi/2}"',
            'mount_xyz="0.1524 -0.0880 0.0095" mount_rpy="0 0 ${11*pi/6- pi/2}"',
            ]

        # wheel params
        mount_wheel=[
            'mount_xyz="0.1524 0.0880 0" mount_rpy="${pi} 0 ${pi/6 - pi/2}"',
            'mount_xyz="0 0.1760    0" mount_rpy="${pi} 0 ${3*pi/6- pi/2}"',
            'mount_xyz="-0.1524 0.0880 0" mount_rpy="${pi} 0 ${5*pi/6- pi/2}"',
            'mount_xyz="-0.1524 -0.0880  0" mount_rpy="${pi} 0 ${7*pi/6- pi/2}"',
            'mount_xyz="0 -0.1760  0" mount_rpy="${pi} 0 ${9*pi/6- pi/2}"',
            'mount_xyz="0.1524 -0.0880 0" mount_rpy="${pi} 0 ${11*pi/6- pi/2}"',
            ]

        current_module = 1
        for i in range(6): # prints limb where each body on the limb has the index of this module
            if letter_list[i] == 'l':
                handle.write('<xacro:leg num="' + str(i+1) + '" ' + mount_leg[i] + '/> \n')
                current_module+=1

            elif letter_list[i] == 'w':
                handle.write('<xacro:wheel num="' + str(i+1) + '" ' + mount_wheel[i] + '/> \n')

                current_module+=1

            elif letter_list[i] == 'n':
                handle.write('<!-- No limb at port ' + str(i+1) + '-->\n')

            handle.write('\n')
        handle.write('</robot>')
        # print('Wrote file ' + fname)

# for test

# name = 'llwlww'
# print_xacros(name)
# in_name = os.path.join(cwd,   name + '.xacro')
# out_name = os.path.join(cwd,  name + '.urdf')
# os.system('rosrun xacro xacro --inorder --xacro-ns ' + in_name + ' > ' + out_name)
# print('compiled ' + in_name + ' to ' + out_name)

def get_symmetric_names():
    # print xacros for 6 legs where it is left-right symmetric
    # limbs are leg, wheel, or none
    # module_letters = 'lwn'
    module_letters = 'lwn' # legs of wheels
    l1 = list(product(module_letters, repeat=3)) 
    # print(l1)

    # allow at most one "none" slot,
    # allow only "none" in the middle
    l2 = []
    for l in l1:
        # if (l.count('n') <= 1):
        if (l[0] is not 'n') and (l[2] is not 'n'):
            l2.append(''.join(l))

    # allow anything
    # l2 = []
    # for l in l1:
    #     l2.append(''.join(l))

    # since the limbs are numbered in a circle, and I want left-right symmetry,
    # add letters forwards then backwards
    # leg numbering:
    #   1 - [ ] - 6 
    #  2 - [   ] - 5
    #   3 - [ ] - 4



    name_list = []
    for l in l2:
        l_doubled = ''
        for letter in l:
            l_doubled += letter
        for letter in l[::-1]:
            l_doubled += letter
        name_list.append(l_doubled)
    return name_list

def get_names():
    # print xacros for 6 legs 
    # limbs are leg, wheel, or none
    module_letters = 'lwn' # legs of wheels
    l1 = list(product(module_letters, repeat=6)) 

    # # allow at most two "none" slot
    # l2 = []
    # for l in l1:
    #     if l.count('n') <= 1:
    #         l2.append(''.join(l))

    # allow only "none" in the middle slots
    l2 = []
    for l in l1:
        # if (l.count('n') <= 1):
        if ((l[0] is not 'n') and (l[2] is not 'n') and
            (l[3] is not 'n') and (l[5] is not 'n')): 
            l2.append(''.join(l))

    return l2

def compile_to_urdf(name):
        print_xacros(name)
        in_name = os.path.join(cwd,  folder, name + '.xacro')
        out_name = os.path.join(cwd,  folder, name + '.urdf')
        os.system('rosrun xacro xacro --inorder --xacro-ns ' + in_name + ' > ' + out_name)
        print('compiled ' + in_name + ' to ' + out_name)


        

if __name__ == "__main__":

    sym_name_list = get_symmetric_names()
    all_name_list = get_names()
    # name_list = ['lllwnw', 'wlllnw', 'lnwlnl', 'llwlll']
    # compile list to urdf: NOTE this requires ROS and that the hebi_description package be compiled.

    print('Compiling symmetric names')
    for name in sym_name_list:
        compile_to_urdf(name)
    print('Compiled')
    print(sym_name_list)

    # print('Compiling asymmetric names')
    # # look for asymetric names not yet compiled and do those
    # asym_name_list = []
    # for name in all_name_list:
    #     if (name not in sym_name_list):
    #         asym_name_list.append(name)
    #         compile_to_urdf(name)
    # print('Compiled')
    # print(asym_name_list)

    # print('Compiling out-of-distribution designs')
    # out_distribution_names = ['nwnnwn', 'nwllwn', 'lnwnln', 'wnwnwn']  
    # out_distribution_names = ['llllll','llwwll']
    # for name in out_distribution_names:
    #     compile_to_urdf(name)

    print('Done.')

