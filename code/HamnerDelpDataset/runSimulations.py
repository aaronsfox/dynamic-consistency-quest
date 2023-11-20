# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through the process of reducing residuals using different
    methods, namely the standard residual reduction algorithm packaged with
    OpenSim and a new method - using OpenSim Moco tracking simulations. 
    
    Please refer to the main README that comes with this repository for details
    around how to use this script.
    
    This uses the Hamner & Delp 2013 dataset available at:
        https://simtk.org/projects/nmbl_running
        
    There are four simulation processes included with this script:
        > Standard RRA: a single iteration of RRA replicating the original approach
          of Hamner & Delp
        > Iterative RRA: three repeat iterations of RRA. The approach replicates
          what is done in the Rajagopal et al. model example, where the desired
          kinematics are kept from IK each iteration while using an updated model
        > Moco Track: torque-actuated simulation using Moco functionality
        > AddBiomechanics: setting up data for input to the AddBiomechanics server
        
    Currently this script focuses on the 5 metres per second running but could
    be adapted to the other speeds.

"""

# %% Import packages

import opensim as osim
import osimFunctions as helper
import os
import pickle
import numpy as np
import time
import re
import shutil
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# %% Set-up

"""

The options in this code block can be modified to re-run

"""

#Add OpenSim geometry path
#Can be helpful with running into any issues around geometry path
#Set this to OpenSim install directory
#### NOTE: change geometry path to OpenSim install directory
geomDir = os.path.join('C:', os.sep, 'OpenSim 4.3', 'Geometry')
osim.ModelVisualizer.addDirToGeometrySearchPaths(geomDir)
print(f'***** OpenSim Geometry installation directory set at {geomDir} *****')
print('***** Please change the geomDir variable in runSimulations.py if incorrect *****')

##### SETTINGS FOR RUNNING THE DESIRED ANALYSES #####

#Settings for processes to run across subjects
#These are currently set to False but if changed to True the analyses for the 
#desired process will be run in subsequent parts of the script. Keep in mind that
#some of these can be time consuming to re-run (particulary the Moco step).
runRRA = False
runRRA3 = False
runMoco = False
runAddBiomech = False

#Print out some info/warnings for certain things
if runMoco:
    print('***** You have selected to re-run the Moco analyses. *****')
    print('***** These analyses take some time, so prepare to be here for a while... *****')
if runAddBiomech:
    print('***** Note that the AddBiomechanics processing code only does not run these analyses. *****')
    print('***** Instead, the data is set-up in the subjects AddBiomechanics folder. *****')
    print('***** To re-run these analyses, you can re-upload the data files to the AddBiomechanics server. *****')

##### SETTINGS FOR COMPILING THE DATA TO FILE #####

#When set to True, the script will collate the RRA, RRA3, Moco Tracking and processed
#AddBiomechanics data to file so that it is easier to load in subsequent aspects
#of this script or when creating new code. Note that this should only really be 
#done again if the simulation results are re-run or changed.
compileData = False

#Within the compiling data step, there are options on which data to read and check
#by creating figures for each of the subjects. These are the joint kinematics, 
#kinetics and residuals. Note that setting compileData to True is necessary for
#these flags to have any effect.
readAndCheckKinematics = True
readAndCheckKinetics = True
readAndCheckResiduals = True

##### TODO: analyse data flag...

# %% Settings and global variables

#Set matplotlib parameters
from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

#Get home path
homeDir = os.getcwd()

#Set subject list
subList = ['subject01',
           'subject02',
           'subject03',
           'subject04', #some noisy kinematics in Moco (arm kinematics)
           'subject08',
           'subject10', #some noisy kinematics in Moco (arm kinematics)
           'subject11',
           'subject17', #some noisy kinematics in Moco (arm kinematics)
           'subject19',
           'subject20'] #some noisy kinematics in Moco (arm kinematics)
    
#Set run names list
#This is useful if you want to assess further running speeds
runList  = ['run2',
            'run3',
            'run4',
            'run5']
    
#Set the run label to examine
#Currently just one trial but could be adapted to use the list above
runLabel = 'run5'
runName = 'Run_5'
    
#Set run cycle list
cycleList = ['cycle1',
             'cycle2',
             'cycle3']
    
#Create a dictionary of the coordinate tasks originally used in Hamner & Delp
rraTasks = {'pelvis_tx': 2.5e1, 'pelvis_ty': 1.0e2, 'pelvis_tz': 2.5e1,
            'pelvis_tilt': 7.5e2, 'pelvis_list': 2.5e2, 'pelvis_rotation': 5.0e1,
            'hip_flexion_r': 7.5e1, 'hip_adduction_r': 5.0e1, 'hip_rotation_r': 1.0e1,
            'knee_angle_r': 1.0e1, 'ankle_angle_r': 1.0e1,
            'hip_flexion_l': 7.5e1, 'hip_adduction_l': 5.0e1, 'hip_rotation_l': 1.0e1,
            'knee_angle_l': 1.0e1, 'ankle_angle_l': 1.0e1,
            'lumbar_extension': 7.5e1, 'lumbar_bending': 5.0e1, 'lumbar_rotation': 2.5e1,
            'arm_flex_r': 1.0e0, 'arm_add_r': 1.0e0, 'arm_rot_r': 1.0e0,
            'elbow_flex_r': 1.0e0, 'pro_sup_r': 1.0e0,
            'arm_flex_l': 1.0e0, 'arm_add_l': 1.0e0, 'arm_rot_l': 1.0e0,
            'elbow_flex_l': 1.0e0, 'pro_sup_l': 1.0e0
            }

#Create a dictionary of the optimal forces for actuators used in Hamner & Delp
rraActuators = {'pelvis_tx': 1, 'pelvis_ty': 1, 'pelvis_tz': 1,
                'pelvis_tilt': 1, 'pelvis_list': 1, 'pelvis_rotation': 1,
                'hip_flexion_r': 1000, 'hip_adduction_r': 1000, 'hip_rotation_r': 1000,
                'knee_angle_r': 1000, 'ankle_angle_r': 1000,
                'hip_flexion_l': 1000, 'hip_adduction_l': 1000, 'hip_rotation_l': 1000,
                'knee_angle_l': 1000, 'ankle_angle_l': 1000,
                'lumbar_extension': 1000, 'lumbar_bending': 1000, 'lumbar_rotation': 1000,
                'arm_flex_r': 500, 'arm_add_r': 500, 'arm_rot_r': 500,
                'elbow_flex_r': 500, 'pro_sup_r': 500,
                'arm_flex_l': 500, 'arm_add_l': 500, 'arm_rot_l': 500,
                'elbow_flex_l': 500, 'pro_sup_l': 500
                }

#Create a dictionary for actuator limits used in Hamner & Delp
rraLimits = {'pelvis_tx': 10000, 'pelvis_ty': 10000, 'pelvis_tz': 10000,
             'pelvis_tilt': 10000, 'pelvis_list': 10000, 'pelvis_rotation': 10000,
             'hip_flexion_r': 1, 'hip_adduction_r': 1, 'hip_rotation_r': 1,
             'knee_angle_r': 1, 'ankle_angle_r': 1,
             'hip_flexion_l': 1, 'hip_adduction_l': 1, 'hip_rotation_l': 1,
             'knee_angle_l': 1, 'ankle_angle_l': 1,
             'lumbar_extension': 1, 'lumbar_bending': 1, 'lumbar_rotation': 1,
             'arm_flex_r': 1, 'arm_add_r': 1, 'arm_rot_r': 1,
             'elbow_flex_r': 1, 'pro_sup_r': 1,
             'arm_flex_l': 1, 'arm_add_l': 1, 'arm_rot_l': 1,
             'elbow_flex_l': 1, 'pro_sup_l': 1
             }

#Create a dictionary for kinematic boundary limits (+/- to max and min)
kinematicLimits = {'pelvis_tx': 0.2, 'pelvis_ty': 0.1, 'pelvis_tz': 0.2,
                   'pelvis_tilt': np.deg2rad(10), 'pelvis_list': np.deg2rad(10), 'pelvis_rotation': np.deg2rad(10),
                   'hip_flexion_r': np.deg2rad(10), 'hip_adduction_r': np.deg2rad(5), 'hip_rotation_r': np.deg2rad(5),
                   'knee_angle_r': np.deg2rad(15), 'ankle_angle_r': np.deg2rad(10),
                   'hip_flexion_l': np.deg2rad(10), 'hip_adduction_l': np.deg2rad(5), 'hip_rotation_l': np.deg2rad(5),
                   'knee_angle_l': np.deg2rad(15), 'ankle_angle_l': np.deg2rad(10),
                   'lumbar_extension': np.deg2rad(10), 'lumbar_bending': np.deg2rad(5), 'lumbar_rotation': np.deg2rad(5),
                   'arm_flex_r': np.deg2rad(5), 'arm_add_r': np.deg2rad(5), 'arm_rot_r': np.deg2rad(5),
                   'elbow_flex_r': np.deg2rad(10), 'pro_sup_r': np.deg2rad(5),
                   'arm_flex_l': np.deg2rad(5), 'arm_add_l': np.deg2rad(5), 'arm_rot_l': np.deg2rad(5),
                   'elbow_flex_l': np.deg2rad(10), 'pro_sup_l': np.deg2rad(5)
                   }

#Create a list of markers to set as fixed in the generic model
fixedMarkers = ['RACR', 'LACR', 'C7', 'CLAV', 'RSJC', 'RLEL', 'RMEL',
                'RFAradius', 'RFAulna', 'LSJC', 'LLEL', 'LMEL', 
                'LFAradius', 'LFAulna', 'RASI', 'LASI', 'RPSI', 'LPSI',
                'LHJC', 'RHJC', 'RLFC', 'RMFC', 'RKJC', 'RLMAL', 'RMMAL',
                'RAJC', 'RCAL', 'LLFC', 'LMFC', 'LKJC', 'LLMAL', 'LMMAL',
                'LAJC', 'LCAL', 'REJC', 'LEJC']

#Set a list for kinematic vars
kinematicVars = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                 'knee_angle_r', 'ankle_angle_r',
                 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                 'knee_angle_l', 'ankle_angle_l',
                 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
                 'elbow_flex_r', 'pro_sup_r',
                 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
                 'elbow_flex_l', 'pro_sup_l'
                 ]

#Set a list for kinetic vars
kineticVars = ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
               'knee_angle_r', 'ankle_angle_r',
               'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
               'knee_angle_l', 'ankle_angle_l',
               'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
               'arm_flex_r', 'arm_add_r', 'arm_rot_r',
               'elbow_flex_r', 'pro_sup_r',
               'arm_flex_l', 'arm_add_l', 'arm_rot_l',
               'elbow_flex_l', 'pro_sup_l'
               ]

#Set a list for residual variables
residualVars = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ', 'F', 'M']

#Set variables for residual data in respective files for rra vs. moco
rraResidualVars = ['ground_pelvis_pelvis_offset_FX', 'ground_pelvis_pelvis_offset_FY', 'ground_pelvis_pelvis_offset_FZ',
                   'ground_pelvis_pelvis_offset_MX', 'ground_pelvis_pelvis_offset_MY', 'ground_pelvis_pelvis_offset_MZ']
mocoResidualVars = ['/forceset/pelvis_tx_actuator', '/forceset/pelvis_ty_actuator', '/forceset/pelvis_tz_actuator',
                    '/forceset/pelvis_list_actuator', '/forceset/pelvis_rotation_actuator', '/forceset/pelvis_tilt_actuator']
addBiomechResidualVars = ['pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force',
                          'pelvis_list_moment', 'pelvis_rotation_moment', 'pelvis_tilt_moment']

#Set dictionary for plotting axes

#Kinematics
kinematicAx = {'pelvis_tx': [0,0], 'pelvis_ty': [0,1], 'pelvis_tz': [0,2],
               'pelvis_tilt': [1,0], 'pelvis_list': [1,1], 'pelvis_rotation': [1,2],
               'hip_flexion_r': [2,0], 'hip_adduction_r': [2,1], 'hip_rotation_r': [2,2],
               'knee_angle_r': [3,0], 'ankle_angle_r': [3,1],
               'hip_flexion_l': [4,0], 'hip_adduction_l': [4,1], 'hip_rotation_l': [4,2],
               'knee_angle_l': [5,0], 'ankle_angle_l': [5,1],
               'lumbar_extension': [6,0], 'lumbar_bending': [6,1], 'lumbar_rotation': [6,2],
               'arm_flex_r': [7,0], 'arm_add_r': [7,1], 'arm_rot_r': [7,2],
               'elbow_flex_r': [8,0], 'pro_sup_r': [8,1],
               'arm_flex_l': [9,0], 'arm_add_l': [9,1], 'arm_rot_l': [9,2],
               'elbow_flex_l': [10,0], 'pro_sup_l': [10,1]
               }

#Kinetics
kineticAx = {'hip_flexion_r': [0,0], 'hip_adduction_r': [0,1], 'hip_rotation_r': [0,2],
             'knee_angle_r': [1,0], 'ankle_angle_r': [1,1],
             'hip_flexion_l': [2,0], 'hip_adduction_l': [2,1], 'hip_rotation_l': [2,2],
             'knee_angle_l': [3,0], 'ankle_angle_l': [3,1],
             'lumbar_extension': [4,0], 'lumbar_bending': [4,1], 'lumbar_rotation': [4,2],
             'arm_flex_r': [5,0], 'arm_add_r': [5,1], 'arm_rot_r': [5,2],
             'elbow_flex_r': [6,0], 'pro_sup_r': [6,1],
             'arm_flex_l': [7,0], 'arm_add_l': [7,1], 'arm_rot_l': [7,2],
             'elbow_flex_l': [8,0], 'pro_sup_l': [8,1]
             }

#Residuals
residualAx = {'FX': [0,0], 'FY': [0,1], 'FZ': [0,2], 'F': [0,3],
              'MX': [1,0], 'MY': [1,1], 'MZ': [1,2], 'M': [1,3]
               }

#Set colours for plots
ikCol = '#000000' #IK = black
rraCol = '#e569ce' #RRA = purple
rra3Col = '#ff6876' #RRA3 = pink
mocoCol = '#4885ed' #Moco = blue
addBiomechCol = '#ffa600' #AddBiomechanics = gold

#### TODO: get the line-styles right - don't look great right now...

#Set line-style for plots
ikLineStyle = 'solid' #IK = solid
rraLineStyle = 'dashed' #RRA = dashed
rra3LineStyle = 'dashdot' #RRA3 = dash-dot
mocoLineStyle = ':' #Moco = dotted
addBiomechLineStyle = (0, (3,5,1,5,1,5)) #AddBiomechanics = dash dot dotted

# %% Loop through subject list

for subject in subList:
    
    # %% Set-up for individual subject
    
    #Create dictionary to store timing data for RRA process
    if runRRA:
        rraRunTimeData = {run: {cyc: {'rraRunTime': []} for cyc in cycleList} for run in runList}
    
    #Create dictionary to store timing data for RRA3 process
    if runRRA3:
        rra3RunTimeData = {run: {cyc: {'rra3RunTime': []} for cyc in cycleList} for run in runList}
    
    #Create dictionary to store timing data
    if runMoco:
        mocoRunTimeData = {run: {cyc: {'mocoRunTime': [], 'nIters': [], 'solved': []} for cyc in cycleList} for run in runList}
    
    #Load in the subjects gait timing data
    with open(os.path.join('..','..','data','HamnerDelp2014',subject,'expData','gaitTimes.pkl'), 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
        
    #Create an RRA directory in the subjects folder
    if runRRA:
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'rra'),
                    exist_ok = True)        
        #Create run trial specific directory as well
        #Note this is currently just run5
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel),
                    exist_ok = True)         
        
    #Create an RRA3 directory in the subjects folder
    if runRRA3:
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3'),
                    exist_ok = True)            
        #Create run trial specific directory as well
        #Note this is currently just run5
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel),
                    exist_ok = True)  
        
    #Create a Moco directory in the subjects folder
    if runMoco:
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'moco'),
                    exist_ok = True)        
        #Create run trial specific directory as well
        #Note this is currently just run5
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel),
                    exist_ok = True) 
            
    #Create an AddBiomechanics directory in the subjects folder
    if runAddBiomech:
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics'),
                    exist_ok = True)        
        #Create run trial specific directory as well
        #Note this is currently just run5
        os.makedirs(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel),
                    exist_ok = True)
        
    # %% Check for running RRA process
    
    if runRRA:
        
        # %% Set-up for RRA
        
        #Change to rra directory for ease of use with tools
        os.chdir(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel))
        
        #Add in opensim logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink('rraLog.log')
        
        #Load the subject model to refer to body parameters
        osimModel = osim.Model(os.path.join('..','..','model',f'{subject}_adjusted_scaled.osim'))
    
        #Create dictionary to store mass adjustments
        bodyList = [osimModel.updBodySet().get(ii).getName() for ii in range(osimModel.updBodySet().getSize())]
        massAdjustmentData = {run: {cyc: {body: {'origMass': [], 'newMass': [], 'massChange': []} for body in bodyList} for cyc in cycleList} for run in runList}
        
        #Create the RRA actuators file
        
        #Create and set name
        rraForceSet = osim.ForceSet()
        rraForceSet.setName(f'{subject}_{runLabel}_RRA_Actuators')
        
        #Loop through coordinates and append to force set
        for actuator in rraActuators.keys():
            
            #Create the actuator. First we must check if point, torque or coordinate
            #actuators are required depending on the coordinate
            
            #Check for point actuator
            if actuator in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:            
                #Create the point actuator
                pointActuator = osim.PointActuator()            
                #Set the name to the residual coordinate
                pointActuator.setName(f'F{actuator[-1].capitalize()}')            
                #Set the max and min controls to those provided
                pointActuator.set_min_control(rraLimits[actuator]*-1)
                pointActuator.set_max_control(rraLimits[actuator])
                #Set the force body as the pelvis
                pointActuator.set_body('pelvis')
                #Set the direction
                pointActuator.set_direction(osim.Vec3(
                    (np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[0],
                     np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[1],
                     np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[2])
                    ))            
                #Set force to be global
                pointActuator.set_point_is_global(True)
                #Set the point from the model
                pointActuator.set_point(osimModel.updBodySet().get('pelvis').get_mass_center())
                #Set optimal force
                pointActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(pointActuator)
                
            #Check for torque actuator
            elif actuator in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']:
                #Create a torque actuator
                torqueActuator = osim.TorqueActuator()
                #Set the name to the residual coordinate            
                torqueActuator.setName(f'M{[x for i, x in enumerate(["X","Y","Z"]) if [actuator == ii for ii in ["pelvis_list", "pelvis_rotation", "pelvis_tilt"]][i]][0]}')
                #Set the max and min controls to those provided
                torqueActuator.set_min_control(rraLimits[actuator]*-1)
                torqueActuator.set_max_control(rraLimits[actuator])
                #Set the torque to act on the pelvis relative to the ground
                torqueActuator.set_bodyA('pelvis')
                torqueActuator.set_bodyB('ground')
                #Set the axis
                torqueActuator.set_axis(osim.Vec3(
                    (np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[0],
                     np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[1],
                     np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[2])
                    ))
                #Set torque to be global
                torqueActuator.set_torque_is_global(True)
                #Set optimal force
                torqueActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(torqueActuator)
                
            #Remaining should be coordinate actuators
            else:
                #Create a coordinate actuator
                coordActuator = osim.CoordinateActuator()
                #Set name to coordinate
                coordActuator.setName(actuator)
                #Set coordinate
                coordActuator.set_coordinate(actuator)
                #Set min and max control to those provided
                coordActuator.set_min_control(rraLimits[actuator]*-1)
                coordActuator.set_max_control(rraLimits[actuator])
                #Set optimal force
                coordActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(coordActuator)
                
        #Print the force set to file
        rraForceSet.printToXML(f'{subject}_{runLabel}_RRA_Actuators.xml')
        
        #Create the RRA tasks file
        
        #Create and set name
        rraTaskSet = osim.CMC_TaskSet()
        rraTaskSet.setName(f'{subject}_{runLabel}_RRA_Tasks')
        
        #Loop through coordinates and append to force set
        for task in rraTasks.keys():
            
            #Create the task
            cmcTask = osim.CMC_Joint()
            
            #Set the name to the coordinate
            cmcTask.setName(task)
            
            #Set task weight
            cmcTask.setWeight(rraTasks[task])
            
            #Set active parameters
            cmcTask.setActive(True, False, False)
            
            #Set kp and kv
            cmcTask.setKP(100)
            cmcTask.setKV(20)
            
            #Set coordinate
            cmcTask.setCoordinateName(task)
            
            #Clone and append to task set
            rraTaskSet.cloneAndAppend(cmcTask)
                
        #Print the force set to file
        rraTaskSet.printToXML(f'{subject}_{runLabel}_RRA_Tasks.xml')
    
        # %% Run the standard RRA
        
        #Create a generic RRA tool to manipulate for the 3 cycles
        rraTool = osim.RRATool()
        
        #Set the generic elements in the tool
        
        #Model file
        rraTool.setModelFilename(os.path.join('..','..','model',f'{subject}_adjusted_scaled.osim'))
        
        #Append the force set files
        forceSetFiles = osim.ArrayStr()
        forceSetFiles.append(f'{subject}_{runLabel}_RRA_Actuators.xml')
        rraTool.setForceSetFiles(forceSetFiles)
        rraTool.setReplaceForceSet(True)
        
        #External loads file
        rraTool.setExternalLoadsFileName(os.path.join('..','..','expData',f'{runName}_grf.xml'))

        #Kinematics file
        rraTool.setDesiredKinematicsFileName(os.path.join('..','..','ik',f'{runName}.mot'))
        
        #Cutoff frequency for kinematics
        rraTool.setLowpassCutoffFrequency(15.0)
        
        #Task set file
        rraTool.setTaskSetFileName(f'{subject}_{runLabel}_RRA_Tasks.xml')
        
        #Output precision
        rraTool.setOutputPrecision(20)
        
        #Loop through gait cycles
        for cycle in cycleList:
            
            #Create directory for cycle
            os.makedirs(cycle, exist_ok = True)
            
            #Add in opensim logger for cycle
            osim.Logger.removeFileSink()
            osim.Logger.addFileSink(os.path.join(cycle,f'{runLabel}_{cycle}_rraLog.log'))
            
            #Add in cycle specific details
            
            #Tool name
            rraTool.setName(f'{subject}_{runLabel}_{cycle}')
            
            #Start and end time
            rraTool.setInitialTime(gaitTimings[runLabel][cycle]['initialTime'])
            rraTool.setFinalTime(gaitTimings[runLabel][cycle]['finalTime'])
            
            #Results directory
            rraTool.setResultsDir(f'{cycle}/')
            
            #Output model file
            rraTool.setOutputModelFileName(f'{cycle}/{subject}_{runLabel}_{cycle}_rraAdjusted.osim')
            
            #Adjusted COM body
            rraTool.setAdjustCOMToReduceResiduals(True)
            rraTool.setAdjustedCOMBody('torso')
            
            #Print to file
            rraTool.printToXML(f'{subject}_{runLabel}_{cycle}_setupRRA.xml')
            
            #Load and run rra tool
            #For some reason rra works better when the tool is reloaded
            rraToolRun = osim.RRATool(f'{subject}_{runLabel}_{cycle}_setupRRA.xml')
            
            #Set-up start timer
            startRunTime = time.time()
            
            #Run tool        
            rraToolRun.run()
            
            #End timer and record
            rraRunTime = round(time.time() - startRunTime, 2)
            
            #Record run-time to dictionary
            rraRunTimeData[runLabel][cycle]['rraRunTime'] = rraRunTime
            
            #Mass adjustments
            #Stop the logger
            osim.Logger.removeFileSink()
            #Read in the log file
            fid = open(os.path.join(cycle,f'{runLabel}_{cycle}_rraLog.log'), 'r')
            fileText = fid.readlines()
            fid.close()
            #Loop through the bodies
            for body in bodyList:
                #Search through log file lines for current body adjustment
                for li in fileText:
                    if body in li and 'orig mass' in li and 'new mass' in li:
                        #Extract out the original mass and new mass
                        stringToGetOrig = re.search('orig mass = (.*),', li)
                        stringToGetNew = re.search('new mass = (.*)\n', li)
                        #Get the values and append to dictionary
                        massAdjustmentData[runLabel][cycle][body]['origMass'] = float(stringToGetOrig.group(1))
                        massAdjustmentData[runLabel][cycle][body]['newMass'] = float(stringToGetNew.group(1))
                        massAdjustmentData[runLabel][cycle][body]['massChange'] = float(stringToGetNew.group(1)) - float(stringToGetOrig.group(1))
            
            #Adjust mass in the newly created model
            #Load the model
            rraAdjustedModel = osim.Model(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted.osim'))
            #Loop through the bodies and set the mass from the dictionary
            for body in bodyList:
                #Get the new mass
                newMass = massAdjustmentData[runLabel][cycle][body]['newMass']
                #Update in the model
                rraAdjustedModel.updBodySet().get(body).setMass(newMass)
            #Finalise the model connections
            rraAdjustedModel.finalizeConnections()
            #Re-save the model
            rraAdjustedModel.printToXML(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted.osim'))
            
            #Calculate the final residuals and joint torques with new kinematics and
            #model using inverse dynamics
            
            #Create the tool
            #Generate this from the blank set-up file as we can't edit the body forces part
            idTool = osim.InverseDynamicsTool(os.path.join('..','..','..','..','..','tools','blank_id_setup.xml'))
            
            #Set the results directory        
            idTool.setResultsDir(f'{cycle}/')
            
            #Set model file
            idTool.setModelFileName(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted.osim'))
            
            #Set the external loads file
            idTool.setExternalLoadsFileName(os.path.join('..','..','expData',f'{runName}_grf.xml'))
            
            #Set the kinematics file from RRA
            idTool.setCoordinatesFileName(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_Kinematics_q.sto'))
            
            #Set the time range
            idTool.setStartTime(osim.Storage(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_Kinematics_q.sto')).getFirstTime())
            idTool.setEndTime(osim.Storage(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_Kinematics_q.sto')).getLastTime())
            
            #Set the output forces file
            idTool.setOutputGenForceFileName(f'{subject}_{runLabel}_{cycle}_id.sto')
                    
            #Print to file
            idTool.printToXML(f'{subject}_{runLabel}_{cycle}_setupID.xml')
            
            #Run tool
            idTool.run()
            
            #Rename the body forces file
            os.replace(os.path.join(cycle,'body_forces_at_joints.sto'),
                       os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_bodyForces.sto'))
            
            #Print confirmation
            print(f'RRA completed for {subject} {runLabel} {cycle}...')
            
        #Save run time and mass adjustment data dictionaries
        with open(f'{subject}_rraRunTimeData.pkl', 'wb') as writeFile:
            pickle.dump(rraRunTimeData, writeFile)
        with open(f'{subject}_massAdjustmentData.pkl', 'wb') as writeFile:
            pickle.dump(massAdjustmentData, writeFile)
    
        #Navigate back to home directory for next subject
        os.chdir(homeDir)
    
    # %% Check for running RRA3 process
    
    if runRRA3:
        
        # %% Set-up for RRA3
        
        #Change to rra directory for ease of use with tools
        os.chdir(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel))
        
        #Perform the generic processes relevant to all steps
        
        #Add in opensim logger for generic processes
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink('rra3Log.log')
        
        #Load the subject model to refer to body parameters
        osimModel = osim.Model(os.path.join('..','..','model',f'{subject}_adjusted_scaled.osim'))
        
        #Create dictionary to store mass adjustments
        #Slightly different to earlier version where 3 iterations are the upper dict level
        bodyList = [osimModel.updBodySet().get(ii).getName() for ii in range(osimModel.updBodySet().getSize())]
        massAdjustmentData3 = {}
        for rraIter in range(1,4):
            massAdjustmentData3[f'rra{rraIter}'] = {run: {cyc: {body: {'origMass': [], 'newMass': [], 'massChange': []} for body in bodyList} for cyc in cycleList} for run in runList}
        
        #Create the RRA actuators file
        
        #Create and set name
        rraForceSet = osim.ForceSet()
        rraForceSet.setName(f'{subject}_{runLabel}_RRA_Actuators')
        
        #Loop through coordinates and append to force set
        for actuator in rraActuators.keys():
            
            #Create the actuator. First we must check if point, torque or coordinate
            #actuators are required depending on the coordinate
            
            #Check for point actuator
            if actuator in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:            
                #Create the point actuator
                pointActuator = osim.PointActuator()            
                #Set the name to the residual coordinate
                pointActuator.setName(f'F{actuator[-1].capitalize()}')            
                #Set the max and min controls to those provided
                pointActuator.set_min_control(rraLimits[actuator]*-1)
                pointActuator.set_max_control(rraLimits[actuator])
                #Set the force body as the pelvis
                pointActuator.set_body('pelvis')
                #Set the direction
                pointActuator.set_direction(osim.Vec3(
                    (np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[0],
                     np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[1],
                     np.array([actuator[-1] == ii for ii in ['x','y','z']], dtype = int)[2])
                    ))            
                #Set force to be global
                pointActuator.set_point_is_global(True)
                #Set the point from the model
                pointActuator.set_point(osimModel.updBodySet().get('pelvis').get_mass_center())
                #Set optimal force
                pointActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(pointActuator)
                
            #Check for torque actuator
            elif actuator in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']:
                #Create a torque actuator
                torqueActuator = osim.TorqueActuator()
                #Set the name to the residual coordinate            
                torqueActuator.setName(f'M{[x for i, x in enumerate(["X","Y","Z"]) if [actuator == ii for ii in ["pelvis_list", "pelvis_rotation", "pelvis_tilt"]][i]][0]}')
                #Set the max and min controls to those provided
                torqueActuator.set_min_control(rraLimits[actuator]*-1)
                torqueActuator.set_max_control(rraLimits[actuator])
                #Set the torque to act on the pelvis relative to the ground
                torqueActuator.set_bodyA('pelvis')
                torqueActuator.set_bodyB('ground')
                #Set the axis
                torqueActuator.set_axis(osim.Vec3(
                    (np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[0],
                     np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[1],
                     np.array([actuator == ii for ii in ['pelvis_list', 'pelvis_rotation', 'pelvis_tilt']], dtype = int)[2])
                    ))
                #Set torque to be global
                torqueActuator.set_torque_is_global(True)
                #Set optimal force
                torqueActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(torqueActuator)
                
            #Remaining should be coordinate actuators
            else:
                #Create a coordinate actuator
                coordActuator = osim.CoordinateActuator()
                #Set name to coordinate
                coordActuator.setName(actuator)
                #Set coordinate
                coordActuator.set_coordinate(actuator)
                #Set min and max control to those provided
                coordActuator.set_min_control(rraLimits[actuator]*-1)
                coordActuator.set_max_control(rraLimits[actuator])
                #Set optimal force
                coordActuator.set_optimal_force(rraActuators[actuator])
                #Clone and append to force set
                rraForceSet.cloneAndAppend(coordActuator)
                
        #Print the force set to file
        rraForceSet.printToXML(f'{subject}_{runLabel}_RRA_Actuators.xml')
        
        #Create the RRA tasks file
        
        #Create and set name
        rraTaskSet = osim.CMC_TaskSet()
        rraTaskSet.setName(f'{subject}_{runLabel}_RRA_Tasks')
        
        #Loop through coordinates and append to force set
        for task in rraTasks.keys():
            
            #Create the task
            cmcTask = osim.CMC_Joint()
            
            #Set the name to the coordinate
            cmcTask.setName(task)
            
            #Set task weight
            cmcTask.setWeight(rraTasks[task])
            
            #Set active parameters
            cmcTask.setActive(True, False, False)
            
            #Set kp and kv
            cmcTask.setKP(100)
            cmcTask.setKV(20)
            
            #Set coordinate
            cmcTask.setCoordinateName(task)
            
            #Clone and append to task set
            rraTaskSet.cloneAndAppend(cmcTask)
                
        #Print the force set to file
        rraTaskSet.printToXML(f'{subject}_{runLabel}_RRA_Tasks.xml')
        
        # %% Loop through 3 iterations of RRA
        
        for rraIter in range(1,4):
            
            #Create the directory to store the current iteration in
            os.makedirs(f'rra{rraIter}', exist_ok = True)
            
            #Shift to iteration directory
            os.chdir(f'rra{rraIter}')
            
            #Add in opensim logger for current iteration
            osim.Logger.removeFileSink()
            osim.Logger.addFileSink(f'rra3Log_{rraIter}.log')
            
            #Create a generic RRA tool to manipulate for the 3 cycles
            rraTool = osim.RRATool()
            
            #Set the generic elements in the tool
            
            #Append the force set files
            forceSetFiles = osim.ArrayStr()
            forceSetFiles.append(os.path.join('..',f'{subject}_{runLabel}_RRA_Actuators.xml'))
            rraTool.setForceSetFiles(forceSetFiles)
            rraTool.setReplaceForceSet(True)
            
            #External loads file
            rraTool.setExternalLoadsFileName(os.path.join('..','..','..','expData',f'{runName}_grf.xml'))
            
            # #Kinematics file
            # #This remains consistent across all iterations
            # rraTool.setDesiredKinematicsFileName(f'..\\..\\..\\ik\\{runName}.mot')
            
            # #Cutoff frequency for kinematics
            # rraTool.setLowpassCutoffFrequency(15.0)
            
            #Task set file
            rraTool.setTaskSetFileName(os.path.join('..',f'{subject}_{runLabel}_RRA_Tasks.xml'))
            
            #Output precision
            rraTool.setOutputPrecision(20)
            
            #Loop through gait cycles
            for cycle in cycleList:
                
                #Create directory for cycle
                os.makedirs(cycle, exist_ok = True)
                
                #Add in opensim logger for cycle
                osim.Logger.removeFileSink()
                osim.Logger.addFileSink(os.path.join(cycle,f'{runLabel}_{cycle}_rra3Log_{rraIter}.log'))
                
                #Add in cycle and iteration specific details
                if rraIter == 1:
                
                    #Use the originally scaled model
                    rraTool.setModelFilename(os.path.join('..','..','..','model',f'{subject}_adjusted_scaled.osim'))
                    
                    #Use the original IK file and filter
                    rraTool.setDesiredKinematicsFileName(os.path.join('..','..','..','ik',f'{runName}.mot'))
                    rraTool.setLowpassCutoffFrequency(15.0)
                    
                    #Set the timings using the gait timings data
                    rraTool.setInitialTime(gaitTimings[runLabel][cycle]['initialTime'])
                    rraTool.setFinalTime(gaitTimings[runLabel][cycle]['finalTime'])
                    
                    
                else:
                    
                    #Use the adjusted RRA model from previous iteration
                    rraTool.setModelFilename(os.path.join('..',f'rra{rraIter-1}',cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted_iter{rraIter-1}.osim'))
                    
                    #Use the adjusted RRA kinematics and don't filter
                    rraTool.setDesiredKinematicsFileName(os.path.join('..',f'rra{rraIter-1}',cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter-1}_Kinematics_q.sto'))
                    rraTool.setLowpassCutoffFrequency(-1)
                    
                    #Set the timings using the previous iteration kinematic data
                    rraTool.setInitialTime(osim.Storage(os.path.join('..',f'rra{rraIter-1}',cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter-1}_Kinematics_q.sto')).getFirstTime())
                    rraTool.setFinalTime(osim.Storage(os.path.join('..',f'rra{rraIter-1}',cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter-1}_Kinematics_q.sto')).getLastTime())
                
                #Tool name
                rraTool.setName(f'{subject}_{runLabel}_{cycle}_iter{rraIter}')
                
                #Results directory
                rraTool.setResultsDir(f'{cycle}/')
                
                #Output model file
                rraTool.setOutputModelFileName(f'{cycle}/{subject}_{runLabel}_{cycle}_rraAdjusted_iter{rraIter}.osim')
                
                #Adjusted COM body
                rraTool.setAdjustCOMToReduceResiduals(True)
                rraTool.setAdjustedCOMBody('torso')
                
                #Print to file
                rraTool.printToXML(f'{subject}_{runLabel}_{cycle}_setupRRA_iter{rraIter}.xml')
                
                #Load and run rra tool
                #For some reason rra works better when the tool is reloaded
                rraToolRun = osim.RRATool(f'{subject}_{runLabel}_{cycle}_setupRRA_iter{rraIter}.xml')
                
                #Set-up start timer
                startRunTime = time.time()
                
                #Run tool        
                rraToolRun.run()
                
                #End timer and record
                rraRunTime = round(time.time() - startRunTime, 2)
                
                #Record run-time to dictionary
                #Append to list as we're going to get 3 times for iterations here
                rra3RunTimeData[runLabel][cycle]['rra3RunTime'].append(rraRunTime)
                
                #Mass adjustments
                #Stop the logger
                osim.Logger.removeFileSink()
                #Read in the log file
                fid = open(os.path.join(cycle,f'{runLabel}_{cycle}_rra3Log_{rraIter}.log'), 'r')
                fileText = fid.readlines()
                fid.close()
                #Loop through the bodies
                for body in bodyList:
                    #Search through log file lines for current body adjustment
                    for li in fileText:
                        if body in li and 'orig mass' in li and 'new mass' in li:
                            #Extract out the original mass and new mass
                            stringToGetOrig = re.search('orig mass = (.*),', li)
                            stringToGetNew = re.search('new mass = (.*)\n', li)
                            #Get the values and append to dictionary
                            massAdjustmentData3[f'rra{rraIter}'][runLabel][cycle][body]['origMass'] = float(stringToGetOrig.group(1))
                            massAdjustmentData3[f'rra{rraIter}'][runLabel][cycle][body]['newMass'] = float(stringToGetNew.group(1))
                            massAdjustmentData3[f'rra{rraIter}'][runLabel][cycle][body]['massChange'] = float(stringToGetNew.group(1)) - float(stringToGetOrig.group(1))
                
                #Adjust mass in the newly created model
                #Load the model
                rraAdjustedModel = osim.Model(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted_iter{rraIter}.osim'))
                #Loop through the bodies and set the mass from the dictionary
                for body in bodyList:
                    #Get the new mass
                    newMass = massAdjustmentData3[f'rra{rraIter}'][runLabel][cycle][body]['newMass']
                    #Update in the model
                    rraAdjustedModel.updBodySet().get(body).setMass(newMass)
                #Finalise the model connections
                rraAdjustedModel.finalizeConnections()
                #Re-save the model
                rraAdjustedModel.printToXML(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted_iter{rraIter}.osim'))
                
                #Calculate the final residuals and joint torques with new kinematics and
                #model using inverse dynamics
                
                #Create the tool
                #Generate this from the blank set-up file as we can't edit the body forces part
                idTool = osim.InverseDynamicsTool(os.path.join('..','..','..','..','..','..','tools','blank_id_setup.xml'))
                
                #Set the results directory        
                idTool.setResultsDir(f'{cycle}/')
                
                #Set model file
                idTool.setModelFileName(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_rraAdjusted_iter{rraIter}.osim'))
                
                #Set the external loads file
                idTool.setExternalLoadsFileName(os.path.join('..','..','..','expData',f'{runName}_grf.xml'))
                
                #Set the kinematics file from RRA
                idTool.setCoordinatesFileName(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter}_Kinematics_q.sto'))
                
                #Set the time range
                idTool.setStartTime(osim.Storage(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter}_Kinematics_q.sto')).getFirstTime())
                idTool.setEndTime(osim.Storage(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter}_Kinematics_q.sto')).getLastTime())
                
                #Set the output forces file
                idTool.setOutputGenForceFileName(f'{subject}_{runLabel}_{cycle}_iter{rraIter}_id.sto')
                        
                #Print to file
                idTool.printToXML(f'{subject}_{runLabel}_{cycle}_setupID_iter{rraIter}.xml')
                
                #Run tool
                idTool.run()
                
                #Rename the body forces file
                os.replace(os.path.join(cycle,'body_forces_at_joints.sto'),
                           os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_iter{rraIter}_bodyForces.sto'))
                
                #Print confirmation
                print(f'RRA completed for {subject} {runLabel} {cycle} iteration {rraIter}...')
                
            #Return to main trial level directory
            os.chdir('..')
                        
        #Save run time and mass adjustment data dictionaries
        with open(f'{subject}_rra3RunTimeData.pkl', 'wb') as writeFile:
            pickle.dump(rra3RunTimeData, writeFile)
        with open(f'{subject}_massAdjustmentData3.pkl', 'wb') as writeFile:
            pickle.dump(massAdjustmentData3, writeFile)
        
        #Navigate back to home directory for next subject
        os.chdir(homeDir)
        
    # %% Check for running Moco process
    
    if runMoco:
    
        # %% Set-up for Moco approach
            
        #Change to rra directory for ease of use with tools
        os.chdir(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel))

        #Add in opensim logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink('mocoLog.log')
    
        #Copy external load files across as there are issues with using these out of
        #directory with Moco tools
        shutil.copyfile(os.path.join('..','..','expData',f'{runName}_grf.xml'),
                        f'{runName}_grf.xml')
        shutil.copyfile(os.path.join('..','..','expData',f'{runName}_grf.mot'),
                        f'{runName}_grf.mot')
        
        #Convert kinematics to states version for use with Moco
        helper.kinematicsToStates(kinematicsFileName = os.path.join('..','..','ik',f'{runName}.mot'),
                           osimModelFileName = os.path.join('..','..','model',f'{subject}_adjusted_scaled.osim'),
                           outputFileName = f'{runName}_coordinates.sto',
                           inDegrees = True, outDegrees = False,
                           filtFreq = 15.0)
    
        # %% Run the standard Moco driven residual reduction approach
    
        #Create a generic tracking tool to manipulate for the 3 cycles
        mocoTrack = osim.MocoTrack()
        mocoTrack.setName('mocoResidualReduction')
        
        # Construct a ModelProcessor and set it on the tool.
        modelProcessor = osim.ModelProcessor(os.path.join('..','..','model',f'{subject}_adjusted_scaled.osim'))
        modelProcessor.append(osim.ModOpAddExternalLoads(f'{runName}_grf.xml'))
        modelProcessor.append(osim.ModOpRemoveMuscles())
        
        #Process model to edit
        mocoModel = modelProcessor.process()
        
        #Add in torque actuators that replicate the RRA actuators
        mocoModel = helper.addTorqueActuators(osimModel = mocoModel,
                                              optForces = rraActuators,
                                              controlLimits = rraLimits)
        
        #Set model in tracking tool
        mocoTrack.setModel(osim.ModelProcessor(mocoModel))
        
        #Construct a table processor to append to the tracking tool for kinematics
        #The kinematics can't be filtered here with the operator as it messes with
        #time stamps in a funky way. This however has already been done in the 
        #conversion to state coordinates
        tableProcessor = osim.TableProcessor(f'{runName}_coordinates.sto')
        mocoTrack.setStatesReference(tableProcessor)
        
        #Create a dictionary to set kinematic bounds
        #Create this based on maximum and minimum values in the kinematic data
        #plus/minus some generic values
        
        #Load the kinematics file as a table
        ikTable = osim.TimeSeriesTable(f'{runName}_coordinates.sto')
        
        #Create the bounds dictionary
        kinematicBounds = {}
        #Loop through the coordinates
        for coord in kinematicLimits.keys():
            #Get the coordinate path
            coordPath = mocoModel.updCoordinateSet().get(coord).getAbsolutePathString()+'/value'
            #Set bounds in dictionary
            kinematicBounds[coord] = [ikTable.getDependentColumn(coordPath).to_numpy().min() - kinematicLimits[coord],
                                      ikTable.getDependentColumn(coordPath).to_numpy().max() + kinematicLimits[coord]]
    
        #Set the global states tracking weight in the tracking problem
        mocoTrack.set_states_global_tracking_weight(1)
        
        #Set tracking tool to apply states to guess
        mocoTrack.set_apply_tracked_states_to_guess(True)
        
        #Provide the setting to ignore unused columns in kinematic data
        mocoTrack.set_allow_unused_references(True)
        
        #Set Moco to not track the speed derivatives from kinematic data
        #This isn't done in RRA so we don't do it here
        mocoTrack.set_track_reference_position_derivatives(False)
        
        #Set tracking mesh interval time
        mocoTrack.set_mesh_interval(0.01) #### note that this is likely a different time step to RRA
        
        #Set the coordinate reference task weights to match RRA
    
        #Create weight set for state tracking
        stateWeights = osim.MocoWeightSet()
        
        #Loop through coordinates to apply weights
        for coordInd in range(mocoModel.updCoordinateSet().getSize()):
            
            #Get name and absolute path to coordinate
            coordName = mocoModel.updCoordinateSet().get(coordInd).getName()
            coordPath = mocoModel.updCoordinateSet().get(coordInd).getAbsolutePathString()
        
            #If a task weight is provided, add it in
            if coordName in list(rraTasks.keys()):
                #Append state into weight set
                #Track the coordinate value
                stateWeights.cloneAndAppend(osim.MocoWeight(f'{coordPath}/value',
                                                            rraTasks[coordName]))
                #Don't track the Coordinate speed
                stateWeights.cloneAndAppend(osim.MocoWeight(f'{coordPath}/speed',
                                                            0))
                
        #Add state weights to the tracking tool
        mocoTrack.set_states_weight_set(stateWeights)
    
        #Loop through gait cycles
        for cycle in cycleList:
        
            #Create directory for cycle
            os.makedirs(cycle, exist_ok = True)
            
            #Add in opensim logger for cycle
            osim.Logger.removeFileSink()
            osim.Logger.addFileSink(os.path.join(cycle,f'{runLabel}_{cycle}_mocoLog.log'))
            
            #Add in cycle specific details
        
            #Set the gait timings in tracking tool
            mocoTrack.set_initial_time(gaitTimings[runLabel][cycle]['initialTime'])
            mocoTrack.set_final_time(gaitTimings[runLabel][cycle]['finalTime'])
            
            #Initialise the Moco study
            study = mocoTrack.initialize()
            problem = study.updProblem()
            
            #Set the parameters for the regularization term on MocoTrack problem
            #(minimize squared excitations)
            effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
            effort.setWeight(0.001)
                    
            #Lock time bounds to the IK data
            problem.setTimeBounds(gaitTimings[runLabel][cycle]['initialTime'],
                                  gaitTimings[runLabel][cycle]['finalTime'])
            
            #Set kinematic bounds using the dictionary values and experimental data
            for coordInd in range(mocoModel.updCoordinateSet().getSize()):
                #First check if coordinate is in kinematic bounds dictionary
                if mocoModel.updCoordinateSet().get(coordInd).getName() in list(kinematicBounds.keys()):                
                    #Get coordinate name and path
                    coordName = mocoModel.updCoordinateSet().get(coordInd).getName()
                    coordPath = mocoModel.updCoordinateSet().get(coordInd).getAbsolutePathString()+'/value'
                    #Set bounds in problem
                    problem.setStateInfo(coordPath,
                                         #Bounds set to model ranges
                                         [kinematicBounds[coordName][0], kinematicBounds[coordName][1]]
                                         )
            
            #Get the solver
            solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
                    
            #Set solver parameters
            solver.set_optim_max_iterations(1000)
            solver.set_optim_constraint_tolerance(1e-2)
            solver.set_optim_convergence_tolerance(1e-2)
            
            #Reset problem (required if changing to implicit mode)
            solver.resetProblem(problem)
            
            #Print to file
            study.printToXML(f'{subject}_{runLabel}_{cycle}_setupMoco.omoco')
            
            #Set-up start timer
            startRunTime = time.time()
            
            #Solve!       
            solution = study.solve()
            
            #End timer and record
            mocoRunTime = round(time.time() - startRunTime, 2)
            
            #Record run-time to dictionary
            mocoRunTimeData[runLabel][cycle]['mocoRunTime'] = mocoRunTime
            
            #Check need to unseal and store outcome
            if solution.isSealed():
                solution.unseal()
                mocoRunTimeData[runLabel][cycle]['solved'] = False
            else:
                mocoRunTimeData[runLabel][cycle]['solved'] = True
                
            #Store number of iterations
            mocoRunTimeData[runLabel][cycle]['nIters'] = solution.getNumIterations()
    
            #Write solution to file
            solution.write(os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_mocoSolution.sto'))
            
            #Remove tracked states file
            os.remove('mocoResidualReduction_tracked_states.sto')
            
            #Calculate the final residuals and joint torques using inverse dynamics
            
            #First convert the solution to a states table and back to standard kinematic coordinates
            
            #Write states table to file
            osim.STOFileAdapter().write(solution.exportToStatesTable(),
                                        os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_mocoStates.sto'))
            
            #Convert states back to kinematic coordinates with helper function
            helper.statesToKinematics(statesFileName = os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_mocoStates.sto'),
                                      outputFileName = os.path.join(cycle,f'{subject}_{runLabel}_{cycle}_mocoKinematics.sto'))
            
            #Stop the logger
            osim.Logger.removeFileSink()
            
        #Save run time and mass adjustment data dictionaries
        with open(f'{subject}_mocoRunTimeData.pkl', 'wb') as writeFile:
            pickle.dump(mocoRunTimeData, writeFile)
    
    # %% Check for running AddBiomechanics process
    
    """
    
    This section of the script does not run the entire AddBiomechanics process.
    Rather, it sets the data up in a folder ready for upload to the server where
    it is subsequently processed. Later code will then read these processed data
    back in to collate and review.
    
    """
    
    if runAddBiomech:
    
        # %% Set-up for AddBiomechanics approach
        
        #Change to rra directory for ease of use with tools
        os.chdir(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel))
        
        #Add in opensim logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink('addBiomechanicsLog.log')
        
        #Read in the generic model to save and feed into AddBiomechanics
        genModel = osim.Model(os.path.join('..','..','model','genericModel.osim'))
        
        #Set the appropriate markers to fixed in the model
        for markerInd in range(genModel.updMarkerSet().getSize()):
            if genModel.updMarkerSet().get(markerInd).getName() in fixedMarkers:
                genModel.updMarkerSet().get(markerInd).set_fixed(True)
            else:
                genModel.updMarkerSet().get(markerInd).set_fixed(False)
                
        #Pronation-supination coordinate limits need to expand to work properly
        genModel.updCoordinateSet().get('pro_sup_r').setRangeMax(np.deg2rad(180))
        genModel.updCoordinateSet().get('pro_sup_l').setRangeMax(np.deg2rad(180))
                
        #Finalize model connections
        genModel.finalizeConnections()
        
        #Print model to file
        genModel.printToXML('genericModel.osim')
        
        #Copy overall TRC and MOT files across to directory
        shutil.copyfile(os.path.join('..','..','expData',f'{runName}.trc'),
                        f'{runName}.trc')
        shutil.copyfile(os.path.join('..','..','expData',f'{runName}_grf.mot'),
                        f'{runName}_grf.mot')
            
        #Print confirmation
        print(f'Data extracted for {subject} {runLabel} for AddBiomechanics processing...')

    
    # %% Finalise before next subject
        
    #Navigate back to home directory for next subject
    os.chdir(homeDir)

# %% Compile data from simulations

#Check for whether or to compile data
if compileData:
    
    # %% Loop through subject list

    for subject in subList:
        
        #Load in the subjects gait timing data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'expData','gaitTimes.pkl'), 'rb') as openFile:
            gaitTimings = pickle.load(openFile)
            
        #Calculate residual force and moment recommendations based on original experimental data
        #Force residual recommendations are 5% of maximum external force
        #Moment residual recommendations are 1% of COM height * maximum external force
        
        #Read in external GRF and get peak force residual recommendation
        expGRF = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'expData',f'{runName}_grf.mot'))
        peakVGRF = np.array((expGRF.getDependentColumn('R_ground_force_vy').to_numpy().max(),
                             expGRF.getDependentColumn('L_ground_force_vy').to_numpy().max())).max()
        forceResidualRec = peakVGRF * 0.05
        
        #Extract centre of mass from static output
        #Load in scaled model
        scaledModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        modelState = scaledModel.initSystem()
        #Read in static motion output
        staticMotion = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_static_output.mot'))
        #Set model to joint coordinates from static output
        for coord in kinematicAx.keys():
            #Get absolute path to joint coordinate value in static output
            jointPath = scaledModel.updCoordinateSet().get(coord).getAbsolutePathString()+'/value'
            #Get value from static output
            staticCoordVal = staticMotion.getDependentColumn(jointPath).to_numpy()[0]
            #Set value in model
            scaledModel.updCoordinateSet().get(coord).setValue(modelState, staticCoordVal)
        #Realise model to position
        scaledModel.realizePosition(modelState)
        #Get model centre of mass
        modelCOM = float(scaledModel.getOutput('com_position').getValueAsString(modelState).split(',')[1])
        #Calculate moment residual recommendation
        momentResidualRec = peakVGRF * modelCOM * 0.01
        
        # %% Read in and compare kinematics
        
        #Check whether to evaluate kinematics
        if readAndCheckKinematics:
        
            #Create dictionaries to store data from the various tools
            
            #Individual cycle data
            ikKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
            rraKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
            rra3Kinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
            mocoKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
            addBiomechKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
            
            #Mean data
            ikMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
            rraMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
            rra3MeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
            mocoMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
            addBiomechMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
            
            #Load in original IK kinematics
            ikData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'ik',f'{runName}.mot'))
            ikTime = np.array(ikData.getIndependentColumn())
            
            #Loop through cycles, load and normalise gait cycle to 101 points
            for cycle in cycleList:
                
                #Load RRA kinematics
                rraData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_Kinematics_q.sto'))
                rraTime = np.array(rraData.getIndependentColumn())
                
                #Load RRA3 kinematics
                rra3Data = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel,'rra3',cycle,f'{subject}_{runLabel}_{cycle}_iter3_Kinematics_q.sto'))
                rra3Time = np.array(rra3Data.getIndependentColumn())
                
                #Load Moco kinematics
                mocoData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_mocoKinematics.sto'))
                mocoTime = np.array(mocoData.getIndependentColumn())
                
                #Load AddBiomechanics kinematics
                #Slightly different as able to load these from .csv file
                addBiomechData = pd.read_csv(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel,'ID',f'{runName}_full.csv'))
                addBiomechTime = addBiomechData['time'].to_numpy()
                
                #Associate start and stop indices to IK data for this cycle
                
                #Get times
                initialTime = rraTime[0]
                finalTime = rraTime[-1]
                
                #Get IK indices
                initialInd = np.argmax(ikTime > initialTime)
                finalInd = np.argmax(ikTime > finalTime) - 1
                
                #Get AddBiomechanics indices
                addBiomechStart = np.argmax(addBiomechTime > initialTime)
                addBiomechStop = np.argmax(addBiomechTime > finalTime) - 1
                
                #Loop through kinematic variables to extract
                for var in kinematicVars:
                    
                    #Extract kinematic variable data
                    #RRA
                    rraKinematicVar = rraData.getDependentColumn(var).to_numpy()
                    #RRA3
                    rra3KinematicVar = rra3Data.getDependentColumn(var).to_numpy()
                    #Moco
                    if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                        mocoKinematicVar = mocoData.getDependentColumn(var).to_numpy()
                    else:
                        mocoKinematicVar = np.rad2deg(mocoData.getDependentColumn(var).to_numpy()) #still in radians for joint angles
                    #AddBiomechanics
                    if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                        addBiomechKinematicVar = addBiomechData[f'pos_{var}'].to_numpy()[addBiomechStart:addBiomechStop]
                    else:
                        addBiomechKinematicVar = np.rad2deg(addBiomechData[f'pos_{var}'].to_numpy())[addBiomechStart:addBiomechStop] #still in radians for joint angles
                    
                    #Get the time cycle for AddBiomechanics data
                    addBiomechTimeCycle = addBiomechTime[addBiomechStart:addBiomechStop]
                    
                    #Extract inverse kinematics over time period
                    ikKinematicVar = ikData.getDependentColumn(var).to_numpy()[initialInd:finalInd]
                    ikTimeCycle = ikTime[initialInd:finalInd]
                    
                    #Interpolate to 101 points
                    
                    #Create interpolation function
                    rraInterpFunc = interp1d(rraTime, rraKinematicVar)
                    rra3InterpFunc = interp1d(rra3Time, rra3KinematicVar)
                    mocoInterpFunc = interp1d(mocoTime, mocoKinematicVar)
                    addBiomechInterpFunc = interp1d(addBiomechTimeCycle, addBiomechKinematicVar)
                    ikInterpFunc = interp1d(ikTimeCycle, ikKinematicVar)
                    
                    #Interpolate data and store in relevant dictionary
                    rraKinematics[runLabel][cycle][var] = rraInterpFunc(np.linspace(rraTime[0], rraTime[-1], 101))
                    rra3Kinematics[runLabel][cycle][var] = rra3InterpFunc(np.linspace(rra3Time[0], rra3Time[-1], 101))
                    mocoKinematics[runLabel][cycle][var] = mocoInterpFunc(np.linspace(mocoTime[0], mocoTime[-1], 101))
                    addBiomechKinematics[runLabel][cycle][var] = addBiomechInterpFunc(np.linspace(addBiomechTimeCycle[0], addBiomechTimeCycle[-1], 101))
                    ikKinematics[runLabel][cycle][var] = ikInterpFunc(np.linspace(ikTimeCycle[0], ikTimeCycle[-1], 101))
            
            #Create a plot of the kinematics
    
            #Create the figure
            fig, ax = plt.subplots(nrows = 11, ncols = 3, figsize = (8,16))
            
            #Adjust subplots
            plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.05, top = 0.95,
                                hspace = 0.4, wspace = 0.5)
            
            #Loop through variables and plot data
            for var in kinematicVars:
                
                #Set the appropriate axis
                plt.sca(ax[kinematicAx[var][0],kinematicAx[var][1]])
                        
                #Loop through cycles to plot individual curves
                for cycle in cycleList:
                    
                    #Plot RRA data
                    plt.plot(np.linspace(0,100,101), rraKinematics[runLabel][cycle][var],
                             linestyle = rraLineStyle, lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Kinematics[runLabel][cycle][var],
                             ls = rra3LineStyle, lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoKinematics[runLabel][cycle][var],
                             ls = mocoLineStyle, lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechKinematics[runLabel][cycle][var],
                             ls = (0, (1,10)), lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
                    #Plot IK data
                    plt.plot(np.linspace(0,100,101), ikKinematics[runLabel][cycle][var],
                             ls = ikLineStyle, lw = 0.5, c = ikCol, alpha = 0.4, zorder = 2)
                    
                #Plot mean curves
                
                #Calculate mean for current kinematic variable
                
                #RRA data
                rraMeanKinematics[runLabel][var] = np.mean(np.vstack((rraKinematics[runLabel]['cycle1'][var],
                                                                      rraKinematics[runLabel]['cycle2'][var],
                                                                      rraKinematics[runLabel]['cycle3'][var])),
                                                           axis = 0)
                
                #RRA3 data
                rra3MeanKinematics[runLabel][var] = np.mean(np.vstack((rra3Kinematics[runLabel]['cycle1'][var],
                                                                       rra3Kinematics[runLabel]['cycle2'][var],
                                                                       rra3Kinematics[runLabel]['cycle3'][var])),
                                                            axis = 0)
                
                #Moco data
                mocoMeanKinematics[runLabel][var] = np.mean(np.vstack((mocoKinematics[runLabel]['cycle1'][var],
                                                                       mocoKinematics[runLabel]['cycle2'][var],
                                                                       mocoKinematics[runLabel]['cycle3'][var])),
                                                            axis = 0)
                
                #AddBiomechanics data
                addBiomechMeanKinematics[runLabel][var] = np.mean(np.vstack((addBiomechKinematics[runLabel]['cycle1'][var],
                                                                             addBiomechKinematics[runLabel]['cycle2'][var],
                                                                             addBiomechKinematics[runLabel]['cycle3'][var])),
                                                                  axis = 0)
                
                #IK data
                ikMeanKinematics[runLabel][var] = np.mean(np.vstack((ikKinematics[runLabel]['cycle1'][var],
                                                                     ikKinematics[runLabel]['cycle2'][var],
                                                                     ikKinematics[runLabel]['cycle3'][var])),
                                                          axis = 0)
                
                #Plot means
                
                #Plot RRA mean
                plt.plot(np.linspace(0,100,101), rraMeanKinematics[runLabel][var],
                         ls = rraLineStyle, lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanKinematics[runLabel][var],
                         ls = rra3LineStyle, lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanKinematics[runLabel][var],
                         ls = mocoLineStyle, lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechMeanKinematics[runLabel][var],
                         ls = addBiomechLineStyle, lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
                
                #Plot Ik mean
                plt.plot(np.linspace(0,100,101), ikMeanKinematics[runLabel][var],
                         ls = ikLineStyle, lw = 1.5, c = ikCol, alpha = 1.0, zorder = 3)
    
                #Clean up axis properties
                
                #Set x-limits
                plt.gca().set_xlim([0,100])
                
                #Add labels
                
                #X-axis (if bottom row)
                if kinematicAx[var][0] == 10:
                    plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 8, fontweight = 'bold')
                    
                #Y-axis (dependent on kinematic variable)
                if var in ['pelvis_tx', 'pevis_ty', 'pelvis_tz']:
                    plt.gca().set_ylabel('Position (m)', fontsize = 8, fontweight = 'bold')
                else:
                    plt.gca().set_ylabel('Joint Angle (\u00b0)', fontsize = 8, fontweight = 'bold')
        
                #Set title
                plt.gca().set_title(var.replace('_',' ').title(), pad = 3, fontsize = 10, fontweight = 'bold')
                    
                #Add zero-dash line if necessary
                if plt.gca().get_ylim()[0] < 0 < plt.gca().get_ylim()[-1]:
                    plt.gca().axhline(y = 0, color = 'dimgrey', linewidth = 0.5, ls = ':', zorder = 1)
                        
                #Turn off top-right spines
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                
                #Set axis ticks in
                plt.gca().tick_params('both', direction = 'in', length = 3)
                
                #Set x-ticks at 0, 50 and 100
                plt.gca().set_xticks([0,50,100])
                #Remove labels if not on bottom row
                if kinematicAx[var][1] != 10:
                    plt.gca().set_xticklabels([])
                    
            #Turn off un-used axes
            ax[3,2].axis('off')
            ax[5,2].axis('off')
            ax[8,2].axis('off')
            ax[10,2].axis('off')
            
            #Add figure title
            fig.suptitle(f'{subject} Kinematics Comparison (IK = Black, RRA = Purple, RRA3 = Pink, Moco = Blue, AddBiomechanics = Gold)',
                         fontsize = 10, fontweight = 'bold', y = 0.99)
    
            #Save figure
            fig.savefig(os.path.join('..','..','data','HamnerDelp2013',subject,'results','figures',f'{subject}_{runLabel}_kinematicsComparison.png'),
                        format = 'png', dpi = 300)
            
            #Close figure
            plt.close('all')
            
            #Save kinematic data dictionaries
            #IK data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(ikKinematics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikMeanKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(ikMeanKinematics, writeFile)
            #RRA data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(rraKinematics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(rraMeanKinematics, writeFile)
            #RRA3 data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3Kinematics.pkl'), 'wb') as writeFile:
                pickle.dump(rra3Kinematics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(rra3MeanKinematics, writeFile)
            #Moco data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(mocoKinematics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(mocoMeanKinematics, writeFile)
            #AddBiomechanics data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechKinematics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinematics.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechMeanKinematics, writeFile)
            
            #Calculate RMSD of all tools vs. one another
            toolList = ['IK', 'RRA', 'RRA3', 'Moco', 'AddBiomechanics']
            
            #Create dictionaries for RMSE data (inc. spot for mean data)
            ikKinematicsRMSE = {tool: {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList} for tool in toolList}
            rraKinematicsRMSE = {tool: {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList} for tool in toolList}
            rra3KinematicsRMSE = {tool: {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList} for tool in toolList}
            mocoKinematicsRMSE = {tool: {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList} for tool in toolList}
            addBiomechKinematicsRMSE = {tool: {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList} for tool in toolList}
            
            #Loop through variables
            for var in kinematicVars:    
                
                #Loop through cycles
                for cycle in cycleList:                    
                    
                    #IK vs. all other tools
                    ikKinematicsRMSE['IK'][runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - ikKinematics[runLabel]['cycle1'][var])**2))
                    ikKinematicsRMSE['RRA'][runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))
                    ikKinematicsRMSE['RRA3'][runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - rra3Kinematics[runLabel]['cycle1'][var])**2))
                    ikKinematicsRMSE['Moco'][runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
                    ikKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - addBiomechKinematics[runLabel]['cycle1'][var])**2))
                    
                    #RRA vs. all other tools
                    rraKinematicsRMSE['IK'][runLabel][cycle][var] = np.sqrt(np.mean((rraKinematics[runLabel]['cycle1'][var] - ikKinematics[runLabel]['cycle1'][var])**2))
                    rraKinematicsRMSE['RRA'][runLabel][cycle][var] = np.sqrt(np.mean((rraKinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))
                    rraKinematicsRMSE['RRA3'][runLabel][cycle][var] = np.sqrt(np.mean((rraKinematics[runLabel]['cycle1'][var] - rra3Kinematics[runLabel]['cycle1'][var])**2))
                    rraKinematicsRMSE['Moco'][runLabel][cycle][var] = np.sqrt(np.mean((rraKinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
                    rraKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] = np.sqrt(np.mean((rraKinematics[runLabel]['cycle1'][var] - addBiomechKinematics[runLabel]['cycle1'][var])**2))
                    
                    #RRA3 vs. all other tools
                    rra3KinematicsRMSE['IK'][runLabel][cycle][var] = np.sqrt(np.mean((rra3Kinematics[runLabel]['cycle1'][var] - ikKinematics[runLabel]['cycle1'][var])**2))
                    rra3KinematicsRMSE['RRA'][runLabel][cycle][var] = np.sqrt(np.mean((rra3Kinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))
                    rra3KinematicsRMSE['RRA3'][runLabel][cycle][var] = np.sqrt(np.mean((rra3Kinematics[runLabel]['cycle1'][var] - rra3Kinematics[runLabel]['cycle1'][var])**2))
                    rra3KinematicsRMSE['Moco'][runLabel][cycle][var] = np.sqrt(np.mean((rra3Kinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
                    rra3KinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] = np.sqrt(np.mean((rra3Kinematics[runLabel]['cycle1'][var] - addBiomechKinematics[runLabel]['cycle1'][var])**2))
                    
                    #Moco vs. all other tools
                    mocoKinematicsRMSE['IK'][runLabel][cycle][var] = np.sqrt(np.mean((mocoKinematics[runLabel]['cycle1'][var] - ikKinematics[runLabel]['cycle1'][var])**2))
                    mocoKinematicsRMSE['RRA'][runLabel][cycle][var] = np.sqrt(np.mean((mocoKinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))
                    mocoKinematicsRMSE['RRA3'][runLabel][cycle][var] = np.sqrt(np.mean((mocoKinematics[runLabel]['cycle1'][var] - rra3Kinematics[runLabel]['cycle1'][var])**2))
                    mocoKinematicsRMSE['Moco'][runLabel][cycle][var] = np.sqrt(np.mean((mocoKinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
                    mocoKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] = np.sqrt(np.mean((mocoKinematics[runLabel]['cycle1'][var] - addBiomechKinematics[runLabel]['cycle1'][var])**2))
                    
                    #AddBiomechanics vs. all other tools
                    addBiomechKinematicsRMSE['IK'][runLabel][cycle][var] = np.sqrt(np.mean((addBiomechKinematics[runLabel]['cycle1'][var] - ikKinematics[runLabel]['cycle1'][var])**2))
                    addBiomechKinematicsRMSE['RRA'][runLabel][cycle][var] = np.sqrt(np.mean((addBiomechKinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))
                    addBiomechKinematicsRMSE['RRA3'][runLabel][cycle][var] = np.sqrt(np.mean((addBiomechKinematics[runLabel]['cycle1'][var] - rra3Kinematics[runLabel]['cycle1'][var])**2))
                    addBiomechKinematicsRMSE['Moco'][runLabel][cycle][var] = np.sqrt(np.mean((addBiomechKinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
                    addBiomechKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] = np.sqrt(np.mean((addBiomechKinematics[runLabel]['cycle1'][var] - addBiomechKinematics[runLabel]['cycle1'][var])**2))
                
                #Calculate mean RMSE across all cycles
                
                #IK vs. all other tools
                ikKinematicsRMSE['IK'][runLabel]['mean'][var] = np.mean([ikKinematicsRMSE['IK'][runLabel][cycle][var] for cycle in cycleList])
                ikKinematicsRMSE['RRA'][runLabel]['mean'][var] = np.mean([ikKinematicsRMSE['RRA'][runLabel][cycle][var] for cycle in cycleList])
                ikKinematicsRMSE['RRA3'][runLabel]['mean'][var] = np.mean([ikKinematicsRMSE['RRA3'][runLabel][cycle][var] for cycle in cycleList])
                ikKinematicsRMSE['Moco'][runLabel]['mean'][var] = np.mean([ikKinematicsRMSE['Moco'][runLabel][cycle][var] for cycle in cycleList])
                ikKinematicsRMSE['AddBiomechanics'][runLabel]['mean'][var] = np.mean([ikKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] for cycle in cycleList])
                
                #RRA vs. all other tools
                rraKinematicsRMSE['IK'][runLabel]['mean'][var] = np.mean([rraKinematicsRMSE['IK'][runLabel][cycle][var] for cycle in cycleList])
                rraKinematicsRMSE['RRA'][runLabel]['mean'][var] = np.mean([rraKinematicsRMSE['RRA'][runLabel][cycle][var] for cycle in cycleList])
                rraKinematicsRMSE['RRA3'][runLabel]['mean'][var] = np.mean([rraKinematicsRMSE['RRA3'][runLabel][cycle][var] for cycle in cycleList])
                rraKinematicsRMSE['Moco'][runLabel]['mean'][var] = np.mean([rraKinematicsRMSE['Moco'][runLabel][cycle][var] for cycle in cycleList])
                rraKinematicsRMSE['AddBiomechanics'][runLabel]['mean'][var] = np.mean([rraKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] for cycle in cycleList])
                
                #RRA3 vs. all other tools
                rra3KinematicsRMSE['IK'][runLabel]['mean'][var] = np.mean([rra3KinematicsRMSE['IK'][runLabel][cycle][var] for cycle in cycleList])
                rra3KinematicsRMSE['RRA'][runLabel]['mean'][var] = np.mean([rra3KinematicsRMSE['RRA'][runLabel][cycle][var] for cycle in cycleList])
                rra3KinematicsRMSE['RRA3'][runLabel]['mean'][var] = np.mean([rra3KinematicsRMSE['RRA3'][runLabel][cycle][var] for cycle in cycleList])
                rra3KinematicsRMSE['Moco'][runLabel]['mean'][var] = np.mean([rra3KinematicsRMSE['Moco'][runLabel][cycle][var] for cycle in cycleList])
                rra3KinematicsRMSE['AddBiomechanics'][runLabel]['mean'][var] = np.mean([rra3KinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] for cycle in cycleList])
                
                #Moco vs. all other tools
                mocoKinematicsRMSE['IK'][runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE['IK'][runLabel][cycle][var] for cycle in cycleList])
                mocoKinematicsRMSE['RRA'][runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE['RRA'][runLabel][cycle][var] for cycle in cycleList])
                mocoKinematicsRMSE['RRA3'][runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE['RRA3'][runLabel][cycle][var] for cycle in cycleList])
                mocoKinematicsRMSE['Moco'][runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE['Moco'][runLabel][cycle][var] for cycle in cycleList])
                mocoKinematicsRMSE['AddBiomechanics'][runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] for cycle in cycleList])
                
                #AddBiomechanics vs. all other tools
                addBiomechKinematicsRMSE['IK'][runLabel]['mean'][var] = np.mean([addBiomechKinematicsRMSE['IK'][runLabel][cycle][var] for cycle in cycleList])
                addBiomechKinematicsRMSE['RRA'][runLabel]['mean'][var] = np.mean([addBiomechKinematicsRMSE['RRA'][runLabel][cycle][var] for cycle in cycleList])
                addBiomechKinematicsRMSE['RRA3'][runLabel]['mean'][var] = np.mean([addBiomechKinematicsRMSE['RRA3'][runLabel][cycle][var] for cycle in cycleList])
                addBiomechKinematicsRMSE['Moco'][runLabel]['mean'][var] = np.mean([addBiomechKinematicsRMSE['Moco'][runLabel][cycle][var] for cycle in cycleList])
                addBiomechKinematicsRMSE['AddBiomechanics'][runLabel]['mean'][var] = np.mean([addBiomechKinematicsRMSE['AddBiomechanics'][runLabel][cycle][var] for cycle in cycleList])
    
            #Save kinematic RMSE data dictionaries
            #IK
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikKinematicsRMSE.pkl'), 'wb') as writeFile:
                pickle.dump(ikKinematicsRMSE, writeFile)
            #RRA
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraKinematicsRMSE.pkl'), 'wb') as writeFile:
                pickle.dump(rraKinematicsRMSE, writeFile)
            #RRA3
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3KinematicsRMSE.pkl'), 'wb') as writeFile:
                pickle.dump(rra3KinematicsRMSE, writeFile)
            #Moco data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoKinematicsRMSE.pkl'), 'wb') as writeFile:
                pickle.dump(mocoKinematicsRMSE, writeFile)
            #AddBiomechanics data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechKinematicsRMSE.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechKinematicsRMSE, writeFile)
        
        # %% Read in and compare kinetics
        
        #Check whether to evaluate kinetics
        if readAndCheckKinetics:
            
            #Create dictionaries to store data from the various tools
            
            #Individual cycle data
            ikKinetics = {run: {cyc: {var: np.zeros(101) for var in kineticVars} for cyc in cycleList} for run in runList}
            rraKinetics = {run: {cyc: {var: np.zeros(101) for var in kineticVars} for cyc in cycleList} for run in runList}
            rra3Kinetics = {run: {cyc: {var: np.zeros(101) for var in kineticVars} for cyc in cycleList} for run in runList}
            mocoKinetics = {run: {cyc: {var: np.zeros(101) for var in kineticVars} for cyc in cycleList} for run in runList}
            addBiomechKinetics = {run: {cyc: {var: np.zeros(101) for var in kineticVars} for cyc in cycleList} for run in runList}
            
            #Mean data
            ikMeanKinetics = {run: {var: np.zeros(101) for var in kineticVars} for run in runList}
            rraMeanKinetics = {run: {var: np.zeros(101) for var in kineticVars} for run in runList}
            rra3MeanKinetics = {run: {var: np.zeros(101) for var in kineticVars} for run in runList}
            mocoMeanKinetics = {run: {var: np.zeros(101) for var in kineticVars} for run in runList}
            addBiomechMeanKinetics = {run: {var: np.zeros(101) for var in kineticVars} for run in runList}
    
            #Loop through cycles, load and normalise gait cycle to 101 points
            for cycle in cycleList:
                
                #Load RRA kinetics
                rraData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_Actuation_force.sto'))
                rraTime = np.array(rraData.getIndependentColumn())
                
                #Load RRA3 kinetics
                rra3Data = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel,'rra3',cycle,f'{subject}_{runLabel}_{cycle}_iter3_Actuation_force.sto'))
                rra3Time = np.array(rra3Data.getIndependentColumn())
                
                #Load Moco kinetics
                mocoData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_mocoSolution.sto'))
                mocoTime = np.array(mocoData.getIndependentColumn())
                
                #Load AddBiomechanics kinetics
                #Slightly different as able to load these from .csv file
                addBiomechData = pd.read_csv(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel,'ID',f'{runName}_full.csv'))
                addBiomechTime = addBiomechData['time'].to_numpy()
                
                #Associate start and stop indices to IK data for this cycle
                
                #Get times
                initialTime = rraTime[0]
                finalTime = rraTime[-1]
                
                #Get AddBiomechanics indices
                addBiomechStart = np.argmax(addBiomechTime > initialTime)
                addBiomechStop = np.argmax(addBiomechTime > finalTime) - 1
                
                #Loop through kinetic variables to extract
                for var in kineticVars:
                    
                    #Extract kinetic variable data
                    #RRA
                    rraKineticVar = rraData.getDependentColumn(var).to_numpy()
                    #RRA3
                    rra3KineticVar = rra3Data.getDependentColumn(var).to_numpy()
                    #Moco
                    #Requires full path to forceset and multiply by optimal force
                    mocoKineticVar = mocoData.getDependentColumn(f'/forceset/{var}_actuator').to_numpy() * rraActuators[var]
                    #AddBiomechanics
                    addBiomechKineticVar = addBiomechData[f'tau_{var}'].to_numpy()[addBiomechStart:addBiomechStop]
                    
                    #Get the time cycle for AddBiomechanics data
                    addBiomechTimeCycle = addBiomechTime[addBiomechStart:addBiomechStop]
    
                    #Interpolate to 101 points
                    
                    #Create interpolation function
                    rraInterpFunc = interp1d(rraTime, rraKineticVar)
                    rra3InterpFunc = interp1d(rra3Time, rra3KineticVar)
                    mocoInterpFunc = interp1d(mocoTime, mocoKineticVar)
                    addBiomechInterpFunc = interp1d(addBiomechTimeCycle, addBiomechKineticVar)
                    
                    #Interpolate data and store in relevant dictionary
                    rraKinetics[runLabel][cycle][var] = rraInterpFunc(np.linspace(rraTime[0], rraTime[-1], 101))
                    rra3Kinetics[runLabel][cycle][var] = rra3InterpFunc(np.linspace(rra3Time[0], rra3Time[-1], 101))
                    mocoKinetics[runLabel][cycle][var] = mocoInterpFunc(np.linspace(mocoTime[0], mocoTime[-1], 101))
                    addBiomechKinetics[runLabel][cycle][var] = addBiomechInterpFunc(np.linspace(addBiomechTimeCycle[0], addBiomechTimeCycle[-1], 101))
            
            #Create a plot of the kinetics
            
            #Create the figure
            fig, ax = plt.subplots(nrows = 9, ncols = 3, figsize = (8,12))
            
            #Adjust subplots
            plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.05, top = 0.95,
                                hspace = 0.4, wspace = 0.5)
            
            #Loop through variables and plot data
            for var in kineticVars:
                
                #Set the appropriate axis
                plt.sca(ax[kineticAx[var][0],kineticAx[var][1]])
                        
                #Loop through cycles to plot individual curves
                for cycle in cycleList:
                    
                    #Plot RRA data
                    plt.plot(np.linspace(0,100,101), rraKinetics[runLabel][cycle][var],
                             ls = rraLineStyle, lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Kinetics[runLabel][cycle][var],
                             ls = rra3LineStyle, lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoKinetics[runLabel][cycle][var],
                             ls = mocoLineStyle, lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechKinetics[runLabel][cycle][var],
                             ls = addBiomechLineStyle, lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
                #Plot mean curves
                
                #Calculate mean for current kinetic variable
                
                #RRA data
                rraMeanKinetics[runLabel][var] = np.mean(np.vstack((rraKinetics[runLabel]['cycle1'][var],
                                                                    rraKinetics[runLabel]['cycle2'][var],
                                                                    rraKinetics[runLabel]['cycle3'][var])),
                                                         axis = 0)
                
                #RRA3 data
                rra3MeanKinetics[runLabel][var] = np.mean(np.vstack((rra3Kinetics[runLabel]['cycle1'][var],
                                                                     rra3Kinetics[runLabel]['cycle2'][var],
                                                                     rra3Kinetics[runLabel]['cycle3'][var])),
                                                          axis = 0)
                
                #Moco data
                mocoMeanKinetics[runLabel][var] = np.mean(np.vstack((mocoKinetics[runLabel]['cycle1'][var],
                                                                     mocoKinetics[runLabel]['cycle2'][var],
                                                                     mocoKinetics[runLabel]['cycle3'][var])),
                                                          axis = 0)
                
                #AddBiomechanics data
                addBiomechMeanKinetics[runLabel][var] = np.mean(np.vstack((addBiomechKinetics[runLabel]['cycle1'][var],
                                                                           addBiomechKinetics[runLabel]['cycle2'][var],
                                                                           addBiomechKinetics[runLabel]['cycle3'][var])),
                                                                axis = 0)
                
                #Plot means
                
                #Plot RRA mean
                plt.plot(np.linspace(0,100,101), rraMeanKinetics[runLabel][var],
                         ls = rraLineStyle, lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanKinetics[runLabel][var],
                         ls = rra3LineStyle, lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanKinetics[runLabel][var],
                         ls = mocoLineStyle, lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechMeanKinetics[runLabel][var],
                         ls = addBiomechLineStyle, lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
    
                #Clean up axis properties
                
                #Set x-limits
                plt.gca().set_xlim([0,100])
                
                #Add labels
                
                #X-axis (if bottom row)
                if kineticAx[var][0] == 8:
                    plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 8, fontweight = 'bold')
                    
                #Y-axis
                plt.gca().set_ylabel('Joint Torque (Nm)', fontsize = 8, fontweight = 'bold')
        
                #Set title
                plt.gca().set_title(var.replace('_',' ').title()+' Torque', pad = 3, fontsize = 10, fontweight = 'bold')
                    
                #Add zero-dash line if necessary
                if plt.gca().get_ylim()[0] < 0 < plt.gca().get_ylim()[-1]:
                    plt.gca().axhline(y = 0, color = 'dimgrey', linewidth = 0.5, ls = ':', zorder = 1)
                        
                #Turn off top-right spines
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                
                #Set axis ticks in
                plt.gca().tick_params('both', direction = 'in', length = 3)
                
                #Set x-ticks at 0, 50 and 100
                plt.gca().set_xticks([0,50,100])
                #Remove labels if not on bottom row
                if kinematicAx[var][1] != 8:
                    plt.gca().set_xticklabels([])
                    
            #Turn off un-used axes
            ax[1,2].axis('off')
            ax[3,2].axis('off')
            ax[6,2].axis('off')
            ax[8,2].axis('off')
            
            #Add figure title
            fig.suptitle(f'{subject} Kinetics Comparison (RRA = Purple, RRA3 = Pink, Moco = Blue, AddBiomechanics = Gold)',
                         fontsize = 10, fontweight = 'bold', y = 0.99)
    
            #Save figure
            fig.savefig(os.path.join('..','..','data','HamnerDelp2013',subject,'results','figures',f'{subject}_{runLabel}_kineticsComparison.png'),
                        format = 'png', dpi = 300)
            
            #Close figure
            plt.close('all')
            
            #Save kinetic data dictionaries
            #RRA data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(rraKinetics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(rraMeanKinetics, writeFile)
            #RRA3 data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3Kinetics.pkl'), 'wb') as writeFile:
                pickle.dump(rra3Kinetics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(rra3MeanKinetics, writeFile)
            #Moco data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(mocoKinetics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(mocoMeanKinetics, writeFile)
            #AddBiomechanics data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechKinetics, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinetics.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechMeanKinetics, writeFile)
        
        # %% Read in and compare residuals
        
        #Check whether to evaluate residuals
        if readAndCheckResiduals:
            
            #Create dictionaries to store data from the various tools
            
            #Individual cycle data
            rraResiduals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
            rra3Residuals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
            mocoResiduals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
            addBiomechResiduals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
            
            #Mean data
            rraMeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
            rra3MeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
            mocoMeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
            addBiomechMeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
            
            #Loop through cycles, load and normalise gait cycle to 101 points
            for cycle in cycleList:
                
                #Load RRA body forces
                rraData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_bodyForces.sto'))
                rraTime = np.array(rraData.getIndependentColumn())
                
                #Load RRA3 body forces
                rra3Data = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel,'rra3',cycle,f'{subject}_{runLabel}_{cycle}_iter3_bodyForces.sto'))
                rra3Time = np.array(rra3Data.getIndependentColumn())
                
                #Load Moco solution
                mocoData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel,cycle,f'{subject}_{runLabel}_{cycle}_mocoSolution.sto'))
                mocoTime = np.array(mocoData.getIndependentColumn())
                
                #Load AddBiomechanics solution
                addBiomechData = osim.TimeSeriesTable(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel,'ID',f'{runName}_id.sto'))
                addBiomechTime = np.array(addBiomechData.getIndependentColumn())
                
                #Get AddBiomechanics start and stop indices for this cycle
                
                #Get times
                initialTime = rraTime[0]
                finalTime = rraTime[-1]
                
                #Get AddBiomechanics indices
                addBiomechStart = np.argmax(addBiomechTime > initialTime)
                addBiomechStop = np.argmax(addBiomechTime > finalTime) - 1
                addBiomechTimeCycle = addBiomechTime[addBiomechStart:addBiomechStop]
                
                #Loop through residual variables to extract
                for var in residualVars:
                    
                    #Check for individual or summative variable
                    if var.endswith('X') or var.endswith('Y') or var.endswith('Z'):
                    
                        #Map residual variable to appropriate column label in respective data
                        rraVar = rraResidualVars[residualVars.index(var)]
                        rra3Var = rraResidualVars[residualVars.index(var)]
                        mocoVar = mocoResidualVars[residualVars.index(var)]
                        addBiomechVar = addBiomechResidualVars[residualVars.index(var)]
                        
                        #Extract residual data
                        rraResidualVar = rraData.getDependentColumn(rraVar).to_numpy()
                        rra3ResidualVar = rra3Data.getDependentColumn(rraVar).to_numpy()
                        mocoResidualVar = mocoData.getDependentColumn(mocoVar).to_numpy() #no need to multiply by optForce as it was 1
                        addBiomechResidualVar = addBiomechData.getDependentColumn(addBiomechVar).to_numpy()[addBiomechStart:addBiomechStop]
        
                        # #Normalise data to model mass
                        
                        # #Load models
                        # rraModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_rraAdjusted.osim')
                        # mocoModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
                        
                        # #Get body mass
                        # rraModelMass = np.sum([rraModel.updBodySet().get(ii).getMass() for ii in range(rraModel.updBodySet().getSize())])
                        # mocoModelMass = np.sum([mocoModel.updBodySet().get(ii).getMass() for ii in range(mocoModel.updBodySet().getSize())])
                        
                        # #Normalise data
                        # rraResidualVarNorm = rraResidualVar / rraModelMass
                        # mocoResidualVarNorm = mocoResidualVar / mocoModelMass
                        
                        #Interpolate to 101 points
                        
                        #Create interpolation function
                        rraInterpFunc = interp1d(rraTime, rraResidualVar)
                        rra3InterpFunc = interp1d(rra3Time, rra3ResidualVar)
                        mocoInterpFunc = interp1d(mocoTime, mocoResidualVar)
                        addBiomechInterpFunc = interp1d(addBiomechTimeCycle, addBiomechResidualVar)
                        
                        #Interpolate data and store in relevant dictionary
                        rraResiduals[runLabel][cycle][var] = rraInterpFunc(np.linspace(rraTime[0], rraTime[-1], 101))
                        rra3Residuals[runLabel][cycle][var] = rra3InterpFunc(np.linspace(rra3Time[0], rra3Time[-1], 101))
                        mocoResiduals[runLabel][cycle][var] = mocoInterpFunc(np.linspace(mocoTime[0], mocoTime[-1], 101))
                        addBiomechResiduals[runLabel][cycle][var] = addBiomechInterpFunc(np.linspace(addBiomechTimeCycle[0], addBiomechTimeCycle[-1], 101))
                        
                    #Else create summative data for force or moment data
                    else:
                        
                        #Find variables related to the current parameter
                        if var == 'F':
                            sumVars = ['FX', 'FY', 'FZ']
                        elif var == 'M':
                            sumVars = ['MX', 'MY', 'MZ']
                            
                        #Sum the relevant data to the dictionary
                        rraResiduals[runLabel][cycle][var] = np.sum(np.vstack([np.abs(rraResiduals[runLabel][cycle][getVar]) for getVar in sumVars]), axis = 0)
                        rra3Residuals[runLabel][cycle][var] = np.sum(np.vstack([np.abs(rra3Residuals[runLabel][cycle][getVar]) for getVar in sumVars]), axis = 0)
                        mocoResiduals[runLabel][cycle][var] = np.sum(np.vstack([np.abs(mocoResiduals[runLabel][cycle][getVar]) for getVar in sumVars]), axis = 0)
                        addBiomechResiduals[runLabel][cycle][var] = np.sum(np.vstack([np.abs(addBiomechResiduals[runLabel][cycle][getVar]) for getVar in sumVars]), axis = 0)
            
            #Create the figure
            fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (12, 4))
            
            #Adjust subplots
            plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.085, top = 0.875,
                                hspace = 0.4, wspace = 0.35)
            
            #Loop through variables and plot data
            for var in residualVars:
                
                #Set the appropriate axis
                plt.sca(ax[residualAx[var][0],residualAx[var][1]])
                        
                #Loop through cycles to plot individual curves
                for cycle in cycleList:
                    
                    #Plot RRA data
                    plt.plot(np.linspace(0,100,101), rraResiduals[runLabel][cycle][var],
                             ls = rraLineStyle, lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Residuals[runLabel][cycle][var],
                             ls = rra3LineStyle, lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoResiduals[runLabel][cycle][var],
                             ls = mocoLineStyle, lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechResiduals[runLabel][cycle][var],
                             ls = addBiomechLineStyle, lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
                #Plot mean curves
                
                #Calculate mean for current residual variable
                
                #RRA data
                rraMeanResiduals[runLabel][var] = np.mean(np.vstack((rraResiduals[runLabel]['cycle1'][var],
                                                                     rraResiduals[runLabel]['cycle2'][var],
                                                                     rraResiduals[runLabel]['cycle3'][var])),
                                                          axis = 0)
                
                #RRA3 data
                rra3MeanResiduals[runLabel][var] = np.mean(np.vstack((rra3Residuals[runLabel]['cycle1'][var],
                                                                      rra3Residuals[runLabel]['cycle2'][var],
                                                                      rra3Residuals[runLabel]['cycle3'][var])),
                                                           axis = 0)
                
                #Moco data
                mocoMeanResiduals[runLabel][var] = np.mean(np.vstack((mocoResiduals[runLabel]['cycle1'][var],
                                                                      mocoResiduals[runLabel]['cycle2'][var],
                                                                      mocoResiduals[runLabel]['cycle3'][var])),
                                                           axis = 0)
                
                #AddBiomechanics data
                addBiomechMeanResiduals[runLabel][var] = np.mean(np.vstack((addBiomechResiduals[runLabel]['cycle1'][var],
                                                                            addBiomechResiduals[runLabel]['cycle2'][var],
                                                                            addBiomechResiduals[runLabel]['cycle3'][var])),
                                                                 axis = 0)
                
                #Plot means
                
                #Plot RRA mean
                plt.plot(np.linspace(0,100,101), rraMeanResiduals[runLabel][var],
                         ls = rraLineStyle, lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanResiduals[runLabel][var],
                         ls = rra3LineStyle, lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanResiduals[runLabel][var],
                         ls = mocoLineStyle, lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechMeanResiduals[runLabel][var],
                         ls = addBiomechLineStyle, lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
    
                #Clean up axis properties
                
                #Set x-limits
                plt.gca().set_xlim([0,100])
                
                #Set y-limits to 10% either side of residuals recommendation 
                #Expand if not there already
                if var.startswith('F'):
                    #Check if axis limits are inside residual limits
                    if plt.gca().get_ylim()[1] < (forceResidualRec * 1.10):
                        plt.gca().set_ylim(plt.gca().get_ylim()[0], forceResidualRec * 1.10)
                    if plt.gca().get_ylim()[0] > (forceResidualRec * 1.10 * -1) and var != 'F':
                        plt.gca().set_ylim(forceResidualRec * 1.10 * -1, plt.gca().get_ylim()[1])
                elif var.startswith('M'):
                    #Check if axis limits are inside residual limits
                    if plt.gca().get_ylim()[1] < (momentResidualRec * 1.10):
                        plt.gca().set_ylim(plt.gca().get_ylim()[0], momentResidualRec * 1.10)
                    if plt.gca().get_ylim()[0] > (momentResidualRec * 1.10 * -1) and var != 'M':
                        plt.gca().set_ylim(momentResidualRec * 1.10 * -1, plt.gca().get_ylim()[1])
                
                #Add dashed line at residual recommendation limits
                if var.endswith('X') or var.endswith('Y') or var.endswith('Z'):
                    if var.startswith('F'):
                        plt.gca().axhline(y = forceResidualRec, color = 'black', linewidth = 1, ls = '--', zorder = 1)
                        plt.gca().axhline(y = forceResidualRec * -1, color = 'black', linewidth = 1, ls = '--', zorder = 1)
                    elif var.startswith('M'):
                        plt.gca().axhline(y = momentResidualRec, color = 'black', linewidth = 1, ls = '--', zorder = 1)
                        plt.gca().axhline(y = momentResidualRec * -1, color = 'black', linewidth = 1, ls = '--', zorder = 1)
                
                #Add labels
                
                #X-axis (if bottom row)
                if var.startswith('M'):
                    plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 8, fontweight = 'bold')
                    
                #Y-axis (dependent on kinematic variable)
                if var.startswith('F'):
                    plt.gca().set_ylabel('Residual Force (N)', fontsize = 8, fontweight = 'bold')
                else:
                    plt.gca().set_ylabel('Residual Moment (Nm)', fontsize = 8, fontweight = 'bold')
        
                #Set title
                if var.endswith('X') or var.endswith('Y') or var.endswith('Z'):
                    plt.gca().set_title(var, pad = 3, fontsize = 12, fontweight = 'bold')
                else:
                    plt.gca().set_title('Total '+var, pad = 3, fontsize = 12, fontweight = 'bold')
                    
                #Add zero-dash line if necessary
                if plt.gca().get_ylim()[0] < 0 < plt.gca().get_ylim()[-1]:
                    plt.gca().axhline(y = 0, color = 'dimgrey', linewidth = 0.5, ls = ':', zorder = 1)
                        
                #Turn off top-right spines
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                
                #Set axis ticks in
                plt.gca().tick_params('both', direction = 'in', length = 3)
                
                #Set x-ticks at 0, 50 and 100
                plt.gca().set_xticks([0,50,100])
                #Remove labels if not on bottom row
                if not var.startswith('M'):
                    plt.gca().set_xticklabels([])
            
            #Add figure title
            fig.suptitle(f'{subject} Residuals Comparison (RRA = Purple, RRA3 = Pink, Moco = Blue, AddBiomechanics = Gold)',
                         fontsize = 10, fontweight = 'bold', y = 0.99)
            
            #Save figure
            fig.savefig(os.path.join('..','..','data','HamnerDelp2013',subject,'results','figures',f'{subject}_{runLabel}_residualsComparison.png'),
                        format = 'png', dpi = 300)
            
            #Close figure
            plt.close('all')
            
            #Save residual data dictionaries
            #RRA data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(rraResiduals, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(rraMeanResiduals, writeFile)
            #RRA3 data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3Residuals.pkl'), 'wb') as writeFile:
                pickle.dump(rra3Residuals, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(rra3MeanResiduals, writeFile)
            #Moco data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(mocoResiduals, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(mocoMeanResiduals, writeFile)
            #AddBiomechanics data
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechResiduals, writeFile)
            with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanResiduals.pkl'), 'wb') as writeFile:
                pickle.dump(addBiomechMeanResiduals, writeFile)
    
    



# %% ----- end of runSimulations.py ----- %% #
