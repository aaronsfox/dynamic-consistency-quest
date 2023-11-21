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
import seaborn as sns
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    print('***** If you do re-run these analyses, you need to update the processingLogs.txt files with the updated AddBiomechanics logs. *****')

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

##### SETTINGS FOR ANALYSING THE SIMULATION DATA #####

#When set to True, the script will take the collated data from each of the simulation
#tools to generate the group descriptive data to save to file, while also generating
#the associated figures for these group data. Note that this should only really
#be done again if the simulation results are re-run or changed.
analyseData = False

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

#Set a list for generic kinematic vars
kinematicVarsGen = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                    'hip_flexion', 'hip_adduction', 'hip_rotation',
                    'knee_angle', 'ankle_angle',
                    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                    'arm_flex', 'arm_add', 'arm_rot',
                    'elbow_flex', 'pro_sup'
                    ]

#Set a dictionary for plotting kinematic vars and their axes
# kinematicVarsPlot = {'pelvis_tilt': [0,0], 'pelvis_list': [0,1], 'pelvis_rotation': [0,2],
#                      'hip_flexion': [1,0], 'hip_adduction': [1,1], 'hip_rotation': [1,2],
#                      'knee_angle': [2,0], 'ankle_angle': [2,1],
#                      'lumbar_extension': [3,0], 'lumbar_bending': [3,1], 'lumbar_rotation': [3,2],
#                      'arm_flex': [4,0], 'arm_add': [4,1], 'arm_rot': [4,2],
#                      'elbow_flex': [5,0], 'pro_sup': [5,1]
#                      }
kinematicVarsPlot = {'pelvis_tx': [0,0], 'pelvis_ty': [0,1], 'pelvis_tz': [0,2], 'pelvis_tilt': [0,3], 'pelvis_list': [0,4], 'pelvis_rotation': [0,5],
                     'hip_flexion': [1,0], 'hip_adduction': [1,1], 'hip_rotation': [1,2], 'knee_angle': [1,3], 'ankle_angle': [1,4],
                     'lumbar_extension': [2,0], 'lumbar_bending': [2,1], 'lumbar_rotation': [2,2],
                     'arm_flex': [3,0], 'arm_add': [3,1], 'arm_rot': [3,2], 'elbow_flex': [3,3], 'pro_sup': [3,4]
                     }

#Set a dictionary for plotting kinetic vars and their axes
kineticVarsPlot = {'hip_flexion': [0,0], 'hip_adduction': [0,1], 'hip_rotation': [0,2],
                   'knee_angle': [1,0], 'ankle_angle': [1,1],
                   'lumbar_extension': [2,0], 'lumbar_bending': [2,1], 'lumbar_rotation': [2,2],
                   'arm_flex': [3,0], 'arm_add': [3,1], 'arm_rot': [3,2],
                   'elbow_flex': [4,0], 'pro_sup': [4,1]
                   }

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

#Set a list for kinematic plot titles
kinematicVarsTitle = ['Pelvis Forwards Translation', 'Pelvis Vertical Translation', 'Pelvis Side Translation',
                      'Pelvis Post. (+) / Ant. (-) Tilt', 'Pelvis Right (+) / Left (-) List', 'Pelvis Left (+) / Right (-) Rot.',
                      'Hip Flex. (+) / Ext. (-)', 'Hip Add. (+) / Abd. (-)', 'Hip Int. Rot (+) / Ext. Rot (-)',
                      'Knee Flex. (-)', 'Ankle DF (+) / PF (-)',
                      'Lumbar Ext. (+) / Flex. (-)', 'Lumbar Right (+) / Left (-) Bend', 'Lumbar Left (+) / Right (-) Rot.',
                      'Shoulder Flex. (+) / Ext. (-)', 'Shoulder Add. (+) / Abd. (-)', 'Shoulder Int. (+) / Ext. (-) Rot',
                      'Elbow Flex. (+)', 'Forearm Pro. (+)'
                      ]

#Set a list for kinematic plot titles
kineticVarsTitle = ['Hip Ext. (+) / Flex. (-) Mom.', 'Hip Abd. (+) / Add. (-) Mom.', 'Hip Ext. Rot (+) / Int. Rot (-) Mom.',
                    'Knee Flex. (+) / Ext. (-) Mom.', 'Ankle PF (+) / DF (-) Mom.',
                    'Lumbar Flex. (+) / Ext.(-) Mom.', 'Lumbar Left (+) / Right (-) Bend Mom.', 'Lumbar Right (+) / Left (-) Rot. Mom.',
                    'Shoulder Ext. (+) / Flex. (-) Mom.', 'Shoulder Abd. (+) / Add. (-) Mom.', 'Shoulder Ext. (+) / Int. (-) Rot Mom.',
                    'Elbow Ext. (+) / Flex. (-) Mom.', 'Forearm Sup. (+) / Pro. Mom.'
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

#Set colours for plot in dictionary format for certain approaches
colDict = {'ik': '#000000', 'rra': '#e569ce', 'rra3': '#ff6876', 'moco': '#4885ed', 'addBiomech': '#ffa600'}

#Set markers for plot in dictionary format for certain approaches
markerDict = {'rra': 'o', 'rra3': 'h', 'moco': 's', 'addBiomech': 'd'}

#Set HEX as RGB colours (https://www.rapidtables.com/convert/color/hex-to-rgb.html)
#These are only used as osim Vec3 objects so they can be set that way here
ikColRGB = osim.Vec3(0,0,0) #IK = black
rraColRGB = osim.Vec3(0.8980392156862745,0.4117647058823529,0.807843137254902) #RRA = purple
rra3ColRGB = osim.Vec3(1,0.40784313725490196,0.4627450980392157) #RRA3 = pink
mocoColRGB = osim.Vec3(0.2823529411764706,0.5215686274509804,0.9294117647058824) #Moco = blue
addBiomechColRGB = osim.Vec3(1,0.6509803921568628,0) #AddBiomechanics = gold

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

#Check for whether to compile data
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
                             linestyle = '-', lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Kinematics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoKinematics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechKinematics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
                    #Plot IK data
                    plt.plot(np.linspace(0,100,101), ikKinematics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = ikCol, alpha = 0.4, zorder = 2)
                    
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
                         ls = '-', lw = 1, c = rraCol,
                         marker = markerDict['rra'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanKinematics[runLabel][var],
                         ls = ':', lw = 1, c = rra3Col,
                         marker = markerDict['rra3'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanKinematics[runLabel][var],
                         ls = '--', lw = 1, c = mocoCol,
                         marker = markerDict['moco'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechMeanKinematics[runLabel][var],
                         ls = '--', lw = 1, c = addBiomechCol,
                         marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot Ik mean
                plt.plot(np.linspace(0,100,101), ikMeanKinematics[runLabel][var],
                         ls = '-', lw = 1, c = ikCol, alpha = 1.0, zorder = 3)
    
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
                             ls = '-', lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Kinetics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoKinetics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechKinetics[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
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
                         ls = '-', lw = 1, c = rraCol,
                         marker = markerDict['rra'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanKinetics[runLabel][var],
                         ls = ':', lw = 1, c = rra3Col,
                         marker = markerDict['rra3'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanKinetics[runLabel][var],
                         ls = '--', lw = 1, c = mocoCol,
                         marker = markerDict['moco'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechMeanKinetics[runLabel][var],
                         ls = '--', lw = 1, c = addBiomechCol,
                         marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
    
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
                             ls = '-', lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                    
                    #Plot RRA3 data
                    plt.plot(np.linspace(0,100,101), rra3Residuals[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = rra3Col, alpha = 0.4, zorder = 2)
                    
                    #Plot Moco data
                    plt.plot(np.linspace(0,100,101), mocoResiduals[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = mocoCol, alpha = 0.4, zorder = 2)
                    
                    #Plot AddBiomechanics data
                    plt.plot(np.linspace(0,100,101), addBiomechResiduals[runLabel][cycle][var],
                             ls = '-', lw = 0.5, c = addBiomechCol, alpha = 0.4, zorder = 2)
                    
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
                         ls = '-', lw = 1, c = rraCol,
                         marker = markerDict['rra'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot RRA3 mean
                plt.plot(np.linspace(0,100,101), rra3MeanResiduals[runLabel][var],
                         ls = ':', lw = 1, c = rra3Col,
                         marker = markerDict['rra3'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot Moco mean
                plt.plot(np.linspace(0,100,101), mocoMeanResiduals[runLabel][var],
                         ls = '--', lw = 1, c = mocoCol,
                         marker = markerDict['moco'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
                
                #Plot AddBiomechanics mean
                plt.plot(np.linspace(0,100,101), addBiomechResiduals[runLabel][var],
                         ls = '--', lw = 1, c = addBiomechCol,
                         marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
                         alpha = 1.0, zorder = 3)
    
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
    
# %% Analyse data from simulations

#Check for whether to analyse data
if analyseData:
    
    # %% Extract solution times

    #Set a place to store average solution times
    solutionTimes = {'rra': np.zeros(len(subList)), 'rra3': np.zeros(len(subList)),
                     'moco': np.zeros(len(subList)), 'addBiomech': np.zeros(len(subList))}
    
    #Loop through subject list
    for subject in subList:
        
        #Load in the subjects gait timing data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'expData','gaitTimes.pkl'), 'rb') as openFile:
            gaitTimings = pickle.load(openFile)
        
        #Load RRA solution time data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'rra',runLabel,f'{subject}_rraRunTimeData.pkl'), 'rb') as openFile:
            rraRunTime = pickle.load(openFile)
            
        #Load RRA3 solution time data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'rra3',runLabel,f'{subject}_rra3RunTimeData.pkl'), 'rb') as openFile:
            rra3RunTime = pickle.load(openFile)
            
        #Load Moco solution time data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'moco',runLabel,f'{subject}_mocoRunTimeData.pkl'), 'rb') as openFile:
            mocoRunTime = pickle.load(openFile)
            
        #Extract AddBiomechanics processing time from logs
        
        #Read in the log file
        fid = open(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel,'processingLogs.txt'), 'r')
        logText = fid.readlines()
        fid.close()
        
        #Identify rows of text with IPOPT timing outputs and add to initialised time
        addBiomechanicsTime = 0
        for textLine in logText:
            if 'Total seconds in IPOPT' in textLine:
                addBiomechanicsTime += float(re.findall('\d+\.\d+', textLine)[0])
                
        #Normalise AddBiomechanics time to a factor of whole trial length vs. average cycle length
        #Get the average duration across cycles
        avgCycleDuration = np.array([gaitTimings[runLabel][cycle]['finalTime'] - gaitTimings[runLabel][cycle]['initialTime'] for cycle in cycleList]).mean()
        #Get duration of AddBiomechanics entire trial
        addBiomechTime = osim.TimeSeriesTableVec3(os.path.join('..','..','data','HamnerDelp2013',subject,'addBiomechanics',runLabel,f'{runName}.trc')).getIndependentColumn()
        addBiomechDuration = addBiomechTime[-1] - addBiomechTime[0]
        #Determine the proportion of the entire AddBiomechanics trial that the avergae cycle would cover
        #Multiply the total AddBiomechanics timeby this to scale
        addBiomechanicsTimeScaled = addBiomechanicsTime * (avgCycleDuration / addBiomechDuration)
            
        #Append summary timing data to dictionary
        #RRA
        solutionTimes['rra'][subList.index(subject)] = np.array([rraRunTime[runLabel][cycle]['rraRunTime'] for cycle in cycleList]).mean()
        #RRA3 (slightly different as need to sum the three iterations)
        solutionTimes['rra3'][subList.index(subject)] = np.array([np.sum(rra3RunTime[runLabel][cycle]['rra3RunTime']) for cycle in cycleList]).mean()
        #Moco
        solutionTimes['moco'][subList.index(subject)] = np.array([mocoRunTime[runLabel][cycle]['mocoRunTime'] for cycle in cycleList]).mean()
        #AddBiomechanics
        solutionTimes['addBiomech'][subList.index(subject)] = addBiomechanicsTimeScaled
        
    #Average and display these results
    # print(f'Average RRA run time (s): {np.round(solutionTimes["rra"].mean(),2)} +/- {np.round(solutionTimes["rra"].std(),2)}')
    # print(f'Average RRA3 run time (s): {np.round(solutionTimes["rra3"].mean(),2)} +/- {np.round(solutionTimes["rra3"].std(),2)}')
    # print(f'Average Moco run time (s): {np.round(solutionTimes["moco"].mean(),2)} +/- {np.round(solutionTimes["moco"].std(),2)}')
    # print(f'Average AddBiomechanics run time (s): {np.round(solutionTimes["addBiomech"].mean(),2)} +/- {np.round(solutionTimes["addBiomech"].std(),2)}')
    print(f'Average RRA run time (mins): {np.round((solutionTimes["rra"]/60).mean(),2)} +/- {np.round((solutionTimes["rra"]/60).std(),2)}')
    print(f'Average RRA3 run time (mins): {np.round((solutionTimes["rra3"]/60).mean(),2)} +/- {np.round((solutionTimes["rra3"]/60).std(),2)}')
    print(f'Average Moco run time (mins): {np.round((solutionTimes["moco"]/60).mean(),2)} +/- {np.round((solutionTimes["moco"]/60).std(),2)}')
    print(f'Average AddBiomechanics run time (mins): {np.round((solutionTimes["addBiomech"]/60).mean(),2)} +/- {np.round((solutionTimes["addBiomech"]/60).std(),2)}')
    
    #Convert to dataframe for plotting
    #Convert to minutes here too
    solutionTimes_df = pd.DataFrame(list(zip((list(solutionTimes['rra'] / 60) + list(solutionTimes['rra3'] / 60) + list(solutionTimes['moco'] / 60) + list(solutionTimes['addBiomech'] / 60)),
                                             (['RRA']*len(solutionTimes['rra']) + ['RRA3']*len(solutionTimes['rra3']) + ['Moco']*len(solutionTimes['moco']) + ['AddBiomechanics']*len(solutionTimes['addBiomech'])),
                                             subList * 4)),
                                    columns = ['Time', 'Solver', 'subjectId'])
    
    #Create figure
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,6))
    
    #Create boxplot using matplotlib
    bp = ax.boxplot(
            np.vstack((solutionTimes['rra'] / 60,
                       solutionTimes['rra3'] / 60,
                       solutionTimes['moco'] / 60,
                       solutionTimes['addBiomech'] / 60)).T,
            whis = (0,100),
            patch_artist = True,
            labels = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
            positions = range(0,4),
            widths = 0.3,
        )
    
    #Set colouring order for boxplots
    bpColOrder = [rraCol, rra3Col, mocoCol, addBiomechCol]
    
    #Loop through and adjust everything
    
    #Caps & whiskets (2 per colour)
    boxInd = 0
    for bpCol in bpColOrder:
        for _ in range(2):
            bp['caps'][boxInd].set_color(bpCol)
            bp['caps'][boxInd].set_linewidth(1.5)
            bp['whiskers'][boxInd].set_color(bpCol)
            bp['whiskers'][boxInd].set_linewidth(1.5)
            boxInd += 1
        
    #Boxes and medians (1 per colour)
    boxInd = 0
    for bpCol in bpColOrder:
        bp['medians'][boxInd].set_color(bpCol)
        bp['medians'][boxInd].set_linewidth(1.5)
        bp['boxes'][boxInd].set_facecolor('none')
        bp['boxes'][boxInd].set_edgecolor(bpCol)
        bp['boxes'][boxInd].set_linewidth(1.5)
        boxInd += 1
        
    #Add the strip plot for points
    #Do this in a loop hacky-ish way to alter marker shape
    solverList = ['rra', 'rra3', 'moco', 'addBiomech']
    for solver in solverList:
        sp = sns.stripplot(x = solverList.index(solver), y = solutionTimes[solver] / 60,
                           color = colDict[solver],
                           marker = markerDict[solver],
                           size = 6, alpha = 0.5,
                           jitter = True, dodge = False, 
                           native_scale = True, zorder = 5,
                           ax = ax)
    
    #Set y-axes limits
    ax.set_ylim([0,ax.get_ylim()[1]])
    
    #Remove x-label
    ax.set_xlabel('')
    
    #Set y-label
    ax.set_ylabel('Solution Time (mins)', fontsize = 14, labelpad = 10)
    
    #Set x-labels
    ax.set_xticklabels(['RRA', 'RRA3', 'Moco', 'AddBiomechanics'], fontsize = 12,
                       rotation = 45, ha = 'right')
    ax.set_xlabel('')
    
    #Despine top and right axes
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(os.path.join('..','..','results','HamnerDelpDataset','figures','averageSolutionTimes.png'),
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close()
    
    #Export solution times dictionary to file
    with open(os.path.join('..','..','results','HamnerDelpDataset','outputs','solutionTimes.pkl'), 'wb') as writeFile:
        pickle.dump(solutionTimes, writeFile)
        
    #Export summary data to csv file
    solutionTimes_df.to_csv(os.path.join('..','..','results','HamnerDelpDataset','outputs','solutionTimes_summary.csv'),
                            index = False)
    
    # %% Extract average and peak residual forces/moments
    
    #Set a place to store residual data
    avgResiduals = {'rra': {var: np.zeros(len(subList)) for var in residualVars},
                    'rra3': {var: np.zeros(len(subList)) for var in residualVars},
                    'moco':{var: np.zeros(len(subList)) for var in residualVars},
                    'addBiomech':{var: np.zeros(len(subList)) for var in residualVars}}
    peakResiduals = {'rra': {var: np.zeros(len(subList)) for var in residualVars},
                     'rra3': {var: np.zeros(len(subList)) for var in residualVars},
                     'moco':{var: np.zeros(len(subList)) for var in residualVars},
                     'addBiomech':{var: np.zeros(len(subList)) for var in residualVars}}
    
    #Set a place to store the proposed thresholds for residuals of each subject
    residualThresholds = {'F': np.zeros(len(subList)), 'M': np.zeros(len(subList))}
    
    #Loop through subject list
    for subject in subList:
        
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
        for coord in kinematicVars:
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
        
        #Add residual thresholds to dictionary
        residualThresholds['F'][subList.index(subject)] = forceResidualRec
        residualThresholds['M'][subList.index(subject)] = momentResidualRec
        
        #Load RRA residuals data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraResiduals.pkl'), 'rb') as openFile:
            rraResiduals = pickle.load(openFile)
            
        #Load RRA3 residuals data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3Residuals.pkl'), 'rb') as openFile:
            rra3Residuals = pickle.load(openFile)
            
        #Load Moco residuals data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoResiduals.pkl'), 'rb') as openFile:
            mocoResiduals = pickle.load(openFile)
            
        #Load AddBiomechanics residuals data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechResiduals.pkl'), 'rb') as openFile:
            addBiomechResiduals = pickle.load(openFile)
    
        #Loop through and extract peak residuals and average
        for var in residualVars:
            
            #Extract the average from each cycle and average into the dictionaries
            avgResiduals['rra'][var][subList.index(subject)] = np.array([np.abs(rraResiduals[runLabel][cycle][var]).mean() for cycle in cycleList]).mean()
            avgResiduals['rra3'][var][subList.index(subject)] = np.array([np.abs(rra3Residuals[runLabel][cycle][var]).mean() for cycle in cycleList]).mean()
            avgResiduals['moco'][var][subList.index(subject)] = np.array([np.abs(mocoResiduals[runLabel][cycle][var]).mean() for cycle in cycleList]).mean()
            avgResiduals['addBiomech'][var][subList.index(subject)] = np.array([np.abs(addBiomechResiduals[runLabel][cycle][var]).mean() for cycle in cycleList]).mean()
            
            #Extract the peak from each cycle and average into the dictionaries
            peakResiduals['rra'][var][subList.index(subject)] = np.array([np.abs(rraResiduals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()
            peakResiduals['rra3'][var][subList.index(subject)] = np.array([np.abs(rra3Residuals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()
            peakResiduals['moco'][var][subList.index(subject)] = np.array([np.abs(mocoResiduals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()
            peakResiduals['addBiomech'][var][subList.index(subject)] = np.array([np.abs(addBiomechResiduals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()
    
    #Average and display results for average residual variables
    for var in residualVars:
        print(f'******** {var} ********')
        print(f'Average RRA average residuals for {var}: {np.round(avgResiduals["rra"][var].mean(),3)} +/- {np.round(avgResiduals["rra"][var].std(),3)}')
        print(f'Average RRA3 average residuals for {var}: {np.round(avgResiduals["rra3"][var].mean(),3)} +/- {np.round(avgResiduals["rra3"][var].std(),3)}')
        print(f'Average Moco average residuals for {var}: {np.round(avgResiduals["moco"][var].mean(),3)} +/- {np.round(avgResiduals["moco"][var].std(),3)}')
        print(f'Average AddBiomechanics average residuals for {var}: {np.round(avgResiduals["addBiomech"][var].mean(),3)} +/- {np.round(avgResiduals["addBiomech"][var].std(),3)}')
        print('********************')
    
    #Average and display results for peak residual variables
    for var in residualVars:
        print(f'******** {var} ********')
        print(f'Average RRA peak residuals for {var}: {np.round(peakResiduals["rra"][var].mean(),3)} +/- {np.round(peakResiduals["rra"][var].std(),3)}')
        print(f'Average RRA3 peak residuals for {var}: {np.round(peakResiduals["rra3"][var].mean(),3)} +/- {np.round(peakResiduals["rra3"][var].std(),3)}')
        print(f'Average Moco peak residuals for {var}: {np.round(peakResiduals["moco"][var].mean(),3)} +/- {np.round(peakResiduals["moco"][var].std(),3)}')
        print(f'Average AddBiomechanics peak residuals for {var}: {np.round(peakResiduals["addBiomech"][var].mean(),3)} +/- {np.round(peakResiduals["addBiomech"][var].std(),3)}')
        print('********************')
    
    #Convert to dataframe for plotting
    
    #Average Forces
    residuals = (list(avgResiduals['rra']['FX']) + list(avgResiduals['rra']['FY']) + list(avgResiduals['rra']['FZ']) + \
                 list(avgResiduals['rra3']['FX']) + list(avgResiduals['rra3']['FY']) + list(avgResiduals['rra3']['FZ']) + \
                 list(avgResiduals['moco']['FX']) + list(avgResiduals['moco']['FY']) + list(avgResiduals['moco']['FZ']) + \
                 list(avgResiduals['addBiomech']['FX']) + list(avgResiduals['addBiomech']['FY']) + list(avgResiduals['addBiomech']['FZ'])
                 )
    solver = (['RRA']*len(avgResiduals['rra']['FX'])*3 + ['RRA3']*len(avgResiduals['rra3']['FX'])*3 + ['Moco']*len(avgResiduals['moco']['FX'])*3 + ['AddBiomechanics']*len(avgResiduals['addBiomech']['FX'])*3)
    axis = (['FX']*len(avgResiduals['rra']['FX']) + ['FY']*len(avgResiduals['rra']['FY']) + ['FZ']*len(avgResiduals['rra']['FZ']) + \
            ['FX']*len(avgResiduals['rra3']['FX']) + ['FY']*len(avgResiduals['rra3']['FY']) + ['FZ']*len(avgResiduals['rra3']['FZ']) + \
            ['FX']*len(avgResiduals['moco']['FX']) + ['FY']*len(avgResiduals['moco']['FY']) + ['FZ']*len(avgResiduals['moco']['FZ']) + \
            ['FX']*len(avgResiduals['addBiomech']['FX']) + ['FY']*len(avgResiduals['addBiomech']['FY']) + ['FZ']*len(avgResiduals['addBiomech']['FZ']))
    subjectId = subList * 3 * 4
    avgResidualForces_df = pd.DataFrame(list(zip(residuals, solver, axis, subjectId)),
                                        columns = ['Average Residual Force', 'Solver', 'Axis', 'Subject'])
    
    #Average Moments
    residuals = (list(avgResiduals['rra']['MX']) + list(avgResiduals['rra']['MY']) + list(avgResiduals['rra']['MZ']) + \
                 list(avgResiduals['rra3']['MX']) + list(avgResiduals['rra3']['MY']) + list(avgResiduals['rra3']['MZ']) + \
                 list(avgResiduals['moco']['MX']) + list(avgResiduals['moco']['MY']) + list(avgResiduals['moco']['MZ']) + \
                 list(avgResiduals['addBiomech']['MX']) + list(avgResiduals['addBiomech']['MY']) + list(avgResiduals['addBiomech']['MZ'])
                 )
    solver = (['RRA']*len(avgResiduals['rra']['MX'])*3 + ['RRA3']*len(avgResiduals['rra3']['MX'])*3 + ['Moco']*len(avgResiduals['moco']['MX'])*3 + ['AddBiomechanics']*len(avgResiduals['addBiomech']['MX'])*3)
    axis = (['MX']*len(avgResiduals['rra']['MX']) + ['MY']*len(avgResiduals['rra']['MY']) + ['MZ']*len(avgResiduals['rra']['MZ']) + \
            ['MX']*len(avgResiduals['rra3']['MX']) + ['MY']*len(avgResiduals['rra3']['MY']) + ['MZ']*len(avgResiduals['rra3']['MZ']) + \
            ['MX']*len(avgResiduals['moco']['MX']) + ['MY']*len(avgResiduals['moco']['MY']) + ['MZ']*len(avgResiduals['moco']['MZ']) + \
            ['MX']*len(avgResiduals['addBiomech']['MX']) + ['MY']*len(avgResiduals['addBiomech']['MY']) + ['MZ']*len(avgResiduals['addBiomech']['MZ']))
    subjectId = subList * 3 * 4
    avgResidualMoments_df = pd.DataFrame(list(zip(residuals, solver, axis, subjectId)),
                                         columns = ['Average Residual Moment', 'Solver', 'Axis', 'Subject'])
    
    #Peak Forces
    residuals = (list(peakResiduals['rra']['FX']) + list(peakResiduals['rra']['FY']) + list(peakResiduals['rra']['FZ']) + \
                 list(peakResiduals['rra3']['FX']) + list(peakResiduals['rra3']['FY']) + list(peakResiduals['rra3']['FZ']) + \
                 list(peakResiduals['moco']['FX']) + list(peakResiduals['moco']['FY']) + list(peakResiduals['moco']['FZ']) + \
                 list(peakResiduals['addBiomech']['FX']) + list(peakResiduals['addBiomech']['FY']) + list(peakResiduals['addBiomech']['FZ'])
                 )
    solver = (['RRA']*len(peakResiduals['rra']['FX'])*3 + ['RRA3']*len(peakResiduals['rra3']['FX'])*3 + ['Moco']*len(peakResiduals['moco']['FX'])*3 + ['AddBiomechanics']*len(peakResiduals['addBiomech']['FX'])*3)
    axis = (['FX']*len(peakResiduals['rra']['FX']) + ['FY']*len(peakResiduals['rra']['FY']) + ['FZ']*len(peakResiduals['rra']['FZ']) + \
            ['FX']*len(peakResiduals['rra3']['FX']) + ['FY']*len(peakResiduals['rra3']['FY']) + ['FZ']*len(peakResiduals['rra3']['FZ']) + \
            ['FX']*len(peakResiduals['moco']['FX']) + ['FY']*len(peakResiduals['moco']['FY']) + ['FZ']*len(peakResiduals['moco']['FZ']) + \
            ['FX']*len(peakResiduals['addBiomech']['FX']) + ['FY']*len(peakResiduals['addBiomech']['FY']) + ['FZ']*len(peakResiduals['addBiomech']['FZ']))
    subjectId = subList * 3 * 4
    peakResidualForces_df = pd.DataFrame(list(zip(residuals, solver, axis, subjectId)),
                                         columns = ['Peak Residual Force', 'Solver', 'Axis', 'Subject'])
    
    #Peak Moments
    residuals = (list(peakResiduals['rra']['MX']) + list(peakResiduals['rra']['MY']) + list(peakResiduals['rra']['MZ']) + \
                 list(peakResiduals['rra3']['MX']) + list(peakResiduals['rra3']['MY']) + list(peakResiduals['rra3']['MZ']) + \
                 list(peakResiduals['moco']['MX']) + list(peakResiduals['moco']['MY']) + list(peakResiduals['moco']['MZ']) + \
                 list(peakResiduals['addBiomech']['MX']) + list(peakResiduals['addBiomech']['MY']) + list(peakResiduals['addBiomech']['MZ'])
                 )
    solver = (['RRA']*len(peakResiduals['rra']['MX'])*3 + ['RRA3']*len(peakResiduals['rra3']['MX'])*3 + ['Moco']*len(peakResiduals['moco']['MX'])*3 + ['AddBiomechanics']*len(peakResiduals['addBiomech']['MX'])*3)
    axis = (['MX']*len(peakResiduals['rra']['MX']) + ['MY']*len(peakResiduals['rra']['MY']) + ['MZ']*len(peakResiduals['rra']['MZ']) + \
            ['MX']*len(peakResiduals['rra3']['MX']) + ['MY']*len(peakResiduals['rra3']['MY']) + ['MZ']*len(peakResiduals['rra3']['MZ']) + \
            ['MX']*len(peakResiduals['moco']['MX']) + ['MY']*len(peakResiduals['moco']['MY']) + ['MZ']*len(peakResiduals['moco']['MZ']) + \
            ['MX']*len(peakResiduals['addBiomech']['MX']) + ['MY']*len(peakResiduals['addBiomech']['MY']) + ['MZ']*len(peakResiduals['addBiomech']['MZ']))
    subjectId = subList * 3 * 4
    peakResidualMoments_df = pd.DataFrame(list(zip(residuals, solver, axis, subjectId)),
                                          columns = ['Peak Residual Moment', 'Solver', 'Axis', 'Subject'])
        
    #Create figure for residual forces
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5), sharey = True)
    
    #Set colouring order for boxplots
    bpColOrder = [rraCol, rra3Col, mocoCol, addBiomechCol]
    
    #Create position dictionary for variables
    varPosX = {resVar: np.linspace(0.4 + ['FX','FY','FZ'].index(resVar),
                                   1.4 + ['FX','FY','FZ'].index(resVar),
                                   4) + ['FX','FY','FZ'].index(resVar) for resVar in ['FX', 'FY', 'FZ']}
    
    #Create boxplot of average residual forces using matplotlib
    
    #Loop through average force residual variables
    for resVar in ['FX','FY','FZ']:
    
        #Set positions for boxplot based
        
        #Create boxplot for current variable
        bp = ax[0].boxplot(
            np.vstack((avgResiduals['rra'][resVar],
                       avgResiduals['rra3'][resVar],
                       avgResiduals['moco'][resVar],
                       avgResiduals['addBiomech'][resVar])).T,
            whis = (0,100),
            patch_artist = True,
            positions = varPosX[resVar],
            widths = 0.3,
            )
        
        #Loop through and adjust everything
        
        #Caps & whiskets (2 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            for _ in range(2):
                bp['caps'][boxInd].set_color(bpCol)
                bp['caps'][boxInd].set_linewidth(1.5)
                bp['whiskers'][boxInd].set_color(bpCol)
                bp['whiskers'][boxInd].set_linewidth(1.5)
                boxInd += 1
            
        #Boxes and medians (1 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            bp['medians'][boxInd].set_color(bpCol)
            bp['medians'][boxInd].set_linewidth(1.5)
            bp['boxes'][boxInd].set_facecolor('none')
            bp['boxes'][boxInd].set_edgecolor(bpCol)
            bp['boxes'][boxInd].set_linewidth(1.5)
            boxInd += 1
    
        #Add the strip plot for points
        #Do this in a loop hacky-ish way to alter marker shape
        solverList = ['rra', 'rra3', 'moco', 'addBiomech']
        solverLabel = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics']
        for solver in solverList:
            #Criteria check for whether to have legend label (only on first input)
            if resVar == 'FX':
                legLabel = solverLabel[solverList.index(solver)]
            else:
                legLabel = '_'+solverLabel[solverList.index(solver)]
            sp = sns.stripplot(x = varPosX[resVar][solverList.index(solver)],
                               y = avgResiduals[solver][resVar],
                               color = colDict[solver],
                               marker = markerDict[solver],
                               label = legLabel,
                               size = 6, alpha = 0.5,
                               jitter = True, dodge = False, 
                               native_scale = True, zorder = 5,
                               ax = ax[0])
        
    #Add the average recommended threshold for residuals
    ax[0].axhline(y = residualThresholds['F'].mean(), color = 'black',
                  linewidth = 1, ls = '--', zorder = 1)
    
    #Set y-axes limits
    ax[0].set_ylim([0.0,ax[0].get_ylim()[1]])
    
    #Remove x-label
    ax[0].set_xlabel('')
    
    #Set y-label
    ax[0].set_ylabel('Average Residual Force (N)', fontsize = 14, labelpad = 10)
    
    #Set x-ticks
    ax[0].set_xticks([1,3,5])
    
    #Set x-labels
    ax[0].set_xticklabels(['FX', 'FY', 'FZ'], fontsize = 12, rotation = 45, ha = 'right')
    
    #Despine top and right axes
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    
    #Loop through peak force residual variables
    for resVar in ['FX','FY','FZ']:
    
        #Set positions for boxplot based
        
        #Create boxplot for current variable
        bp = ax[1].boxplot(
            np.vstack((peakResiduals['rra'][resVar],
                       peakResiduals['rra3'][resVar],
                       peakResiduals['moco'][resVar],
                       peakResiduals['addBiomech'][resVar])).T,
            whis = (0,100),
            patch_artist = True,
            positions = varPosX[resVar],
            widths = 0.3,
            )
        
        #Loop through and adjust everything
        
        #Caps & whiskets (2 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            for _ in range(2):
                bp['caps'][boxInd].set_color(bpCol)
                bp['caps'][boxInd].set_linewidth(1.5)
                bp['whiskers'][boxInd].set_color(bpCol)
                bp['whiskers'][boxInd].set_linewidth(1.5)
                boxInd += 1
            
        #Boxes and medians (1 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            bp['medians'][boxInd].set_color(bpCol)
            bp['medians'][boxInd].set_linewidth(1.5)
            bp['boxes'][boxInd].set_facecolor('none')
            bp['boxes'][boxInd].set_edgecolor(bpCol)
            bp['boxes'][boxInd].set_linewidth(1.5)
            boxInd += 1
    
        #Add the strip plot for points
        #Do this in a loop hacky-ish way to alter marker shape
        solverList = ['rra', 'rra3', 'moco', 'addBiomech']
        solverLabel = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics']
        for solver in solverList:
            #Criteria check for whether to have legend label (only on first input)
            if resVar == 'FX':
                legLabel = solverLabel[solverList.index(solver)]
            else:
                legLabel = '_'+solverLabel[solverList.index(solver)]
            sp = sns.stripplot(x = varPosX[resVar][solverList.index(solver)],
                               y = peakResiduals[solver][resVar],
                               color = colDict[solver],
                               marker = markerDict[solver],
                               label = legLabel,
                               size = 6, alpha = 0.5,
                               jitter = True, dodge = False, 
                               native_scale = True, zorder = 5,
                               ax = ax[1])
    
    #Add the average recommended threshold for residuals
    ax[1].axhline(y = residualThresholds['F'].mean(), color = 'black',
                  linewidth = 1, ls = '--', zorder = 1)

    #Set y-axes limits so that peak values get included
    ax[1].set_ylim([0.0,
                    peakResidualForces_df['Peak Residual Force'].max() * 1.05])
    
    #Remove x-label
    ax[1].set_xlabel('')
    
    #Set y-label
    ax[1].set_ylabel('Peak Residual Force (N)', fontsize = 14, labelpad = 10)
    
    #Set x-ticks
    ax[1].set_xticks([1,3,5])
    
    #Set x-labels
    ax[1].set_xticklabels(['FX', 'FY', 'FZ'], fontsize = 12, rotation = 45, ha = 'right')
    
    #Despine top and right axes
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(os.path.join('..','..','results','HamnerDelpDataset','figures','residualForces.png'),
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close()
    
    #Create figure for residual moments
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5), sharey = True)
    
    #Create position dictionary for variables
    varPosX = {resVar: np.linspace(0.4 + ['MX','MY','MZ'].index(resVar),
                                   1.4 + ['MX','MY','MZ'].index(resVar),
                                   4) + ['MX','MY','MZ'].index(resVar) for resVar in ['MX', 'MY', 'MZ']}
    
    #Create boxplot of average residual forces using matplotlib
    
    #Loop through average moment residual variables
    for resVar in ['MX','MY','MZ']:
    
        #Set positions for boxplot based
        
        #Create boxplot for current variable
        bp = ax[0].boxplot(
            np.vstack((avgResiduals['rra'][resVar],
                       avgResiduals['rra3'][resVar],
                       avgResiduals['moco'][resVar],
                       avgResiduals['addBiomech'][resVar])).T,
            whis = (0,100),
            patch_artist = True,
            positions = varPosX[resVar],
            widths = 0.3,
            )
        
        #Loop through and adjust everything
        
        #Caps & whiskets (2 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            for _ in range(2):
                bp['caps'][boxInd].set_color(bpCol)
                bp['caps'][boxInd].set_linewidth(1.5)
                bp['whiskers'][boxInd].set_color(bpCol)
                bp['whiskers'][boxInd].set_linewidth(1.5)
                boxInd += 1
            
        #Boxes and medians (1 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            bp['medians'][boxInd].set_color(bpCol)
            bp['medians'][boxInd].set_linewidth(1.5)
            bp['boxes'][boxInd].set_facecolor('none')
            bp['boxes'][boxInd].set_edgecolor(bpCol)
            bp['boxes'][boxInd].set_linewidth(1.5)
            boxInd += 1
    
        #Add the strip plot for points
        #Do this in a loop hacky-ish way to alter marker shape
        solverList = ['rra', 'rra3', 'moco', 'addBiomech']
        solverLabel = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics']
        for solver in solverList:
            #Criteria check for whether to have legend label (only on first input)
            if resVar == 'MX':
                legLabel = solverLabel[solverList.index(solver)]
            else:
                legLabel = '_'+solverLabel[solverList.index(solver)]
            sp = sns.stripplot(x = varPosX[resVar][solverList.index(solver)],
                               y = avgResiduals[solver][resVar],
                               color = colDict[solver],
                               marker = markerDict[solver],
                               label = legLabel,
                               size = 6, alpha = 0.5,
                               jitter = True, dodge = False, 
                               native_scale = True, zorder = 5,
                               ax = ax[0])
        
    #Add the average recommended threshold for residuals
    ax[0].axhline(y = residualThresholds['M'].mean(), color = 'black',
                  linewidth = 1, ls = '--', zorder = 1)
    
    #Set y-axes limits
    ax[0].set_ylim([0.0,ax[0].get_ylim()[1]])
    
    #Remove x-label
    ax[0].set_xlabel('')
    
    #Set y-label
    ax[0].set_ylabel('Average Residual Moment (Nm)', fontsize = 14, labelpad = 10)
    
    #Set x-ticks
    ax[0].set_xticks([1,3,5])
    
    #Set x-labels
    ax[0].set_xticklabels(['MX', 'MY', 'MZ'], fontsize = 12, rotation = 45, ha = 'right')
    
    #Despine top and right axes
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    
    #Loop through peak moment residual variables
    for resVar in ['MX','MY','MZ']:
    
        #Set positions for boxplot based
        
        #Create boxplot for current variable
        bp = ax[1].boxplot(
            np.vstack((peakResiduals['rra'][resVar],
                       peakResiduals['rra3'][resVar],
                       peakResiduals['moco'][resVar],
                       peakResiduals['addBiomech'][resVar])).T,
            whis = (0,100),
            patch_artist = True,
            positions = varPosX[resVar],
            widths = 0.3,
            )
        
        #Loop through and adjust everything
        
        #Caps & whiskets (2 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            for _ in range(2):
                bp['caps'][boxInd].set_color(bpCol)
                bp['caps'][boxInd].set_linewidth(1.5)
                bp['whiskers'][boxInd].set_color(bpCol)
                bp['whiskers'][boxInd].set_linewidth(1.5)
                boxInd += 1
            
        #Boxes and medians (1 per colour)
        boxInd = 0
        for bpCol in bpColOrder:
            bp['medians'][boxInd].set_color(bpCol)
            bp['medians'][boxInd].set_linewidth(1.5)
            bp['boxes'][boxInd].set_facecolor('none')
            bp['boxes'][boxInd].set_edgecolor(bpCol)
            bp['boxes'][boxInd].set_linewidth(1.5)
            boxInd += 1
    
        #Add the strip plot for points
        #Do this in a loop hacky-ish way to alter marker shape
        solverList = ['rra', 'rra3', 'moco', 'addBiomech']
        solverLabel = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics']
        for solver in solverList:
            #Criteria check for whether to have legend label (only on first input)
            if resVar == 'MX':
                legLabel = solverLabel[solverList.index(solver)]
            else:
                legLabel = '_'+solverLabel[solverList.index(solver)]
            sp = sns.stripplot(x = varPosX[resVar][solverList.index(solver)],
                               y = peakResiduals[solver][resVar],
                               color = colDict[solver],
                               marker = markerDict[solver],
                               label = legLabel,
                               size = 6, alpha = 0.5,
                               jitter = True, dodge = False, 
                               native_scale = True, zorder = 5,
                               ax = ax[1])
    
    #Add the average recommended threshold for residuals
    ax[1].axhline(y = residualThresholds['M'].mean(), color = 'black',
                  linewidth = 1, ls = '--', zorder = 1)

    #Set y-axes limits so that peak values get included
    ax[1].set_ylim([0.0,
                    peakResidualMoments_df['Peak Residual Moment'].max() * 1.05])
    
    #Remove x-label
    ax[1].set_xlabel('')
    
    #Set y-label
    ax[1].set_ylabel('Peak Residual Moment (Nm)', fontsize = 14, labelpad = 10)
    
    #Set x-ticks
    ax[1].set_xticks([1,3,5])
    
    #Set x-labels
    ax[1].set_xticklabels(['MX', 'MY', 'MZ'], fontsize = 12, rotation = 45, ha = 'right')
    
    #Despine top and right axes
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    
    #Tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(os.path.join('..','..','results','HamnerDelpDataset','figures','residualMoments.png'),
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close()
    
    #Export residual summary dataframes to file
    avgResidualForces_df.to_csv(os.path.join('..','..','results','HamnerDelpDataset','outputs','avgResidualForces.csv'), index = False)
    avgResidualMoments_df.to_csv(os.path.join('..','..','results','HamnerDelpDataset','outputs','avgResidualMoments.csv'), index = False)
    peakResidualForces_df.to_csv(os.path.join('..','..','results','HamnerDelpDataset','outputs','peakResidualForces.csv'), index = False)
    peakResidualMoments_df.to_csv(os.path.join('..','..','results','HamnerDelpDataset','outputs','peakResidualMoments.csv'), index = False)
       
    # %% Extract root mean square deviations of kinematic data
         
    #Set list of approaches
    approachList = ['IK', 'RRA', 'RRA3', 'Moco', 'AddBiomechanics']
        
    #Set a place to store average solution times
    kinematicsRMSD = {outerApproach: {innerApproach: {var: np.zeros(len(subList)) for var in kinematicVarsGen} for innerApproach in approachList} for outerApproach in approachList}
    
    #Loop through subject list
    for subject in subList:
        
        #Define dictionary to store subject data
        rmseData = {}
        
        #Load IK RMSD data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikKinematicsRMSE.pkl'), 'rb') as openFile:
            rmseData['IK'] = pickle.load(openFile)
        
        #Load RRA RMSD data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraKinematicsRMSE.pkl'), 'rb') as openFile:
            rmseData['RRA'] = pickle.load(openFile)
            
        #Load RRA3 RMSD data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3KinematicsRMSE.pkl'), 'rb') as openFile:
            rmseData['RRA3'] = pickle.load(openFile)
        
        #Load Moco RMSD data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoKinematicsRMSE.pkl'), 'rb') as openFile:
            rmseData['Moco'] = pickle.load(openFile)
            
        #Load AddBiomechanics RMSD data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechKinematicsRMSE.pkl'), 'rb') as openFile:
            rmseData['AddBiomechanics'] = pickle.load(openFile)
    
        #Loop through and extract mean for generic kinematic variables
        for var in kinematicVarsGen:
            
            #Check for pelvis/lumbar variable
            if 'pelvis' in var or 'lumbar' in var:
                #Loop through approaches
                for outerApproach in approachList:
                    for innerApproach in approachList:
                        #Extract the mean and place in dictionary 
                        kinematicsRMSD[outerApproach][innerApproach][var][subList.index(subject)] = \
                            rmseData[outerApproach][innerApproach][runLabel]['mean'][var]
            else:
                #Loop through approaches
                for outerApproach in approachList:
                    for innerApproach in approachList:
                        #Extract the mean for combined left and right sidess and place in dictionary 
                        kinematicsRMSD[outerApproach][innerApproach][var][subList.index(subject)] = \
                            np.array((rmseData[outerApproach][innerApproach][runLabel]['mean'][f'{var}_r'], rmseData[outerApproach][innerApproach][runLabel]['mean'][f'{var}_l'])).mean()
    
    #Average and display results for variables
    #Loop through approaches
    for outerApproach in approachList:
        for innerApproach in approachList:
            #Avoid printing duplicate approaches
            if outerApproach != innerApproach:
                #Loop through kinematic variables
                for var in kinematicVarsGen:
                    if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                        #Convert to cm for translations
                        print(f'Average {var} RMSD for {innerApproach} vs. {outerApproach}: {np.round(kinematicsRMSD[outerApproach][innerApproach][var].mean()*100,2)} +/- {np.round(kinematicsRMSD[outerApproach][innerApproach][var].std()*100,2)}')
                    else:
                        #Present in degrees
                        print(f'Average {var} RMSD for {innerApproach} vs. {outerApproach}: {np.round(kinematicsRMSD[outerApproach][innerApproach][var].mean(),2)} +/- {np.round(kinematicsRMSD[outerApproach][innerApproach][var].std(),2)}')
    
    #Export RMSD dictionary to file
    with open(os.path.join('..','..','results','HamnerDelpDataset','outputs','kinematicsRMSD.pkl'), 'wb') as writeFile:
        pickle.dump(kinematicsRMSD, writeFile)
        
    # %% Compare average kinematics across approaches
    
    #Set a place to store subject kinematic data
    meanKinematics = {'ik': {var: np.zeros((len(subList),101)) for var in kinematicVars},
                      'rra': {var: np.zeros((len(subList),101)) for var in kinematicVars},
                      'rra3': {var: np.zeros((len(subList),101)) for var in kinematicVars},
                      'moco':{var: np.zeros((len(subList),101)) for var in kinematicVars},
                      'addBiomech':{var: np.zeros((len(subList),101)) for var in kinematicVars}}
    
    #Loop through subject list
    for subject in subList:
    
        #Read in IK kinematic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikMeanKinematics.pkl'), 'rb') as openFile:
            ikMeanKinematics = pickle.load(openFile)
            
        #Read in RRA kinematic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinematics.pkl'), 'rb') as openFile:
            rraMeanKinematics = pickle.load(openFile)
        
        #Read in RRA3 kinematic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinematics.pkl'), 'rb') as openFile:
            rra3MeanKinematics = pickle.load(openFile)
            
        #Read in Moco kinematic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinematics.pkl'), 'rb') as openFile:
            mocoMeanKinematics = pickle.load(openFile)
            
        #Read in AddBiomechanics kinematic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinematics.pkl'), 'rb') as openFile:
            addBiomechMeanKinematics = pickle.load(openFile)
            
        #Loop through and extract kinematic data
        for var in kinematicVars:
            meanKinematics['ik'][var][subList.index(subject),:] = ikMeanKinematics[runLabel][var]
            meanKinematics['rra'][var][subList.index(subject),:] = rraMeanKinematics[runLabel][var]
            meanKinematics['rra3'][var][subList.index(subject),:] = rra3MeanKinematics[runLabel][var]
            meanKinematics['moco'][var][subList.index(subject),:] = mocoMeanKinematics[runLabel][var]
            meanKinematics['addBiomech'][var][subList.index(subject),:] = addBiomechMeanKinematics[runLabel][var]
            
    #Create figure of group kinematics across the different approaches
    #Note that generic kinematic variables are used here and right side values are presented
    
    #Create the figure
    fig, ax = plt.subplots(nrows = 4, ncols = 6, figsize = (14,8))
    
    #Adjust subplots
    plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.05, top = 0.95,
                        hspace = 0.4, wspace = 0.5)
    
    #Loop through variables and plot data
    for var in kinematicVarsPlot.keys():
        
        #Set the appropriate axis
        plt.sca(ax[kinematicVarsPlot[var][0],kinematicVarsPlot[var][1]])
        
        #Set the plotting variable based on whether it is a general or side variable
        if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                   'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']:
            plotVar = str(var)
        else:
            plotVar = var+'_r'
                
        #Plot mean and SD curves
        
        #IK mean
        plt.plot(np.linspace(0,100,101), meanKinematics['ik'][plotVar].mean(axis = 0),
                 ls = '-', lw = 1, c = ikCol, alpha = 1.0, zorder = 3)
        # #IK sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinematics['ik'][plotVar].mean(axis = 0) + meanKinematics['ik'][plotVar].std(axis = 0),
        #                  meanKinematics['ik'][plotVar].mean(axis = 0) - meanKinematics['ik'][plotVar].std(axis = 0),
        #                  color = ikCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #RRA mean
        plt.plot(np.linspace(0,100,101), meanKinematics['rra'][plotVar].mean(axis = 0),
                 ls = '-', lw = 1, c = rraCol,
                 marker = markerDict['rra'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #RRA sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinematics['rra'][plotVar].mean(axis = 0) + meanKinematics['rra'][plotVar].std(axis = 0),
        #                  meanKinematics['rra'][plotVar].mean(axis = 0) - meanKinematics['rra'][plotVar].std(axis = 0),
        #                  color = rraCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #RRA3 mean
        plt.plot(np.linspace(0,100,101), meanKinematics['rra3'][plotVar].mean(axis = 0),
                 ls = ':', lw = 1, c = rra3Col, 
                 marker = markerDict['rra3'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #RRA3 sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinematics['rra3'][plotVar].mean(axis = 0) + meanKinematics['rra3'][plotVar].std(axis = 0),
        #                  meanKinematics['rra3'][plotVar].mean(axis = 0) - meanKinematics['rra3'][plotVar].std(axis = 0),
        #                  color = rra3Col, alpha = 0.1, zorder = 2, lw = 0)
        
        #Moco mean
        plt.plot(np.linspace(0,100,101), meanKinematics['moco'][plotVar].mean(axis = 0),
                 ls = '--', lw = 1, c = mocoCol,
                 marker = markerDict['moco'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #Moco sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinematics['moco'][plotVar].mean(axis = 0) + meanKinematics['moco'][plotVar].std(axis = 0),
        #                  meanKinematics['moco'][plotVar].mean(axis = 0) - meanKinematics['moco'][plotVar].std(axis = 0),
        #                  color = mocoCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #AddBiomechanics mean
        plt.plot(np.linspace(0,100,101), meanKinematics['addBiomech'][plotVar].mean(axis = 0),
                 ls = '--', lw = 1, c = addBiomechCol,
                 marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #AddBiomechanics sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinematics['addBiomech'][plotVar].mean(axis = 0) + meanKinematics['addBiomech'][plotVar].std(axis = 0),
        #                  meanKinematics['addBiomech'][plotVar].mean(axis = 0) - meanKinematics['addBiomech'][plotVar].std(axis = 0),
        #                  color = addBiomechCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #Clean up axis properties
        
        #Set x-limits
        plt.gca().set_xlim([0,100])
        
        #Add labels
        
        #X-axis (if bottom row)
        if kinematicVarsPlot[var][0] == 3:
            plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 10, fontweight = 'bold')
            
        #Y-axis
        if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
            plt.gca().set_ylabel('Position (m)', fontsize = 10, fontweight = 'bold')
        else:
            plt.gca().set_ylabel('Joint Angle (\u00b0)', fontsize = 10, fontweight = 'bold')
    
        #Set title
        plt.gca().set_title(kinematicVarsTitle[list(kinematicVarsPlot.keys()).index(var)],
                            pad = 5, fontsize = 10, fontweight = 'bold')
            
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
        
        #Remove x-tick labels if not bottom row
        if kinematicVarsPlot[var][0] != 3:
            plt.gca().set_xticklabels([])
        
    #Turn off un-used axes
    ax[1,5].axis('off')
    ax[2,3].axis('off')
    ax[2,4].axis('off')
    ax[2,5].axis('off')
    
    #Create legend on dummy axis in bottom right
    plt.sca(ax[3,5])
    
    #Plot dummy data
    #IK
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'IK',
             ls = '-', lw = 1, c = ikCol, alpha = 1.0, zorder = 3)
    #RRA
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'RRA',
             ls = '-', lw = 1, c = rraCol,
             marker = markerDict['rra'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #RRA3
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'RRA3',
             ls = ':', lw = 1, c = rra3Col, 
             marker = markerDict['rra3'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #Moco
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'Moco',
             ls = '--', lw = 1, c = mocoCol,
             marker = markerDict['moco'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #AddBiomechanics
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'AddBiomechanics',
             ls = '--', lw = 1, c = addBiomechCol,
             marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    
    #Add legend
    plt.legend()
    
    #Remove all axis properties
    #Spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #Ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    #Axis limits to avoid data
    plt.gca().set_ylim([50,100])
    
    #Set tight layout
    plt.tight_layout()
    
    #Save figure
    fig.savefig(os.path.join('..','..','results','HamnerDelpDataset','figures','meanKinematics.png'),
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close('all')
    
    #Export mean kinematics dictionary to file
    with open(os.path.join('..','..','results','HamnerDelpDataset','outputs','meanKinematics.pkl'), 'wb') as writeFile:
        pickle.dump(meanKinematics, writeFile)
        
    # %% Compare average kinetics across approaches
    
    #Set a place to store subject kinematic data
    meanKinetics = {'rra': {var: np.zeros((len(subList),101)) for var in kineticVars},
                    'rra3': {var: np.zeros((len(subList),101)) for var in kineticVars},
                    'moco':{var: np.zeros((len(subList),101)) for var in kineticVars},
                    'addBiomech':{var: np.zeros((len(subList),101)) for var in kineticVars}}
    
    #Loop through subject list
    for subject in subList:
            
        #Read in RRA kinetic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinetics.pkl'), 'rb') as openFile:
            rraMeanKinetics = pickle.load(openFile)
        
        #Read in RRA3 kinetic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinetics.pkl'), 'rb') as openFile:
            rra3MeanKinetics = pickle.load(openFile)
            
        #Read in Moco kinetic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinetics.pkl'), 'rb') as openFile:
            mocoMeanKinetics = pickle.load(openFile)
            
        #Read in AddBiomechanics kinetic data
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinetics.pkl'), 'rb') as openFile:
            addBiomechMeanKinetics = pickle.load(openFile)
            
        #Loop through and extract kinematic data
        for var in kineticVars:
            meanKinetics['rra'][var][subList.index(subject),:] = rraMeanKinetics[runLabel][var]
            meanKinetics['rra3'][var][subList.index(subject),:] = rra3MeanKinetics[runLabel][var]
            meanKinetics['moco'][var][subList.index(subject),:] = mocoMeanKinetics[runLabel][var]
            meanKinetics['addBiomech'][var][subList.index(subject),:] = addBiomechMeanKinetics[runLabel][var]
            
    #Create figure of group kinetics across the different approaches
    #Note that generic kinetic variablea are used here and right side values are presented
    
    #Create the figure
    fig, ax = plt.subplots(nrows = 5, ncols = 3, figsize = (8,10))
    
    #Adjust subplots
    plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.05, top = 0.95,
                        hspace = 0.4, wspace = 0.5)
    
    #Loop through variables and plot data
    for var in kineticVarsPlot.keys():
        
        #Set the appropriate axis
        plt.sca(ax[kineticVarsPlot[var][0],kineticVarsPlot[var][1]])
        
        #Set the plotting variable based on whether it is a general or side variable
        if 'pelvis' in var or 'lumbar' in var:
            plotVar = str(var)
        else:
            plotVar = var+'_r'
                
        #Plot mean and SD curves
        
        #RRA mean
        plt.plot(np.linspace(0,100,101), meanKinetics['rra'][plotVar].mean(axis = 0),
                 ls = '-', lw = 1, c = rraCol,
                 marker = markerDict['rra'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #RRA sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinetics['rra'][plotVar].mean(axis = 0) + meanKinetics['rra'][plotVar].std(axis = 0),
        #                  meanKinetics['rra'][plotVar].mean(axis = 0) - meanKinetics['rra'][plotVar].std(axis = 0),
        #                  color = rraCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #RRA3 mean
        plt.plot(np.linspace(0,100,101), meanKinetics['rra3'][plotVar].mean(axis = 0),
                 ls = ':', lw = 1, c = rra3Col,
                 marker = markerDict['rra3'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #RRA3 sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinetics['rra3'][plotVar].mean(axis = 0) + meanKinetics['rra3'][plotVar].std(axis = 0),
        #                  meanKinetics['rra3'][plotVar].mean(axis = 0) - meanKinetics['rra3'][plotVar].std(axis = 0),
        #                  color = rra3Col, alpha = 0.1, zorder = 2, lw = 0)
        
        #Moco mean
        plt.plot(np.linspace(0,100,101), meanKinetics['moco'][plotVar].mean(axis = 0),
                 ls = '--', lw = 1, c = mocoCol,
                 marker = markerDict['moco'], markevery = 2, markersize = 3, ### different mark every used due to noisyness
                 alpha = 1.0, zorder = 3)
        # #Moco sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinetics['moco'][plotVar].mean(axis = 0) + meanKinetics['moco'][plotVar].std(axis = 0),
        #                  meanKinetics['moco'][plotVar].mean(axis = 0) - meanKinetics['moco'][plotVar].std(axis = 0),
        #                  color = mocoCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #AddBiomechanics mean
        plt.plot(np.linspace(0,100,101), meanKinetics['addBiomech'][plotVar].mean(axis = 0),
                 ls = '--', lw = 1.5, c = addBiomechCol,
                 marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
                 alpha = 1.0, zorder = 3)
        # #AddBiomechanics sd
        # plt.fill_between(np.linspace(0,100,101),
        #                  meanKinetics['addBiomech'][plotVar].mean(axis = 0) + meanKinetics['addBiomech'][plotVar].std(axis = 0),
        #                  meanKinetics['addBiomech'][plotVar].mean(axis = 0) - meanKinetics['addBiomech'][plotVar].std(axis = 0),
        #                  color = addBiomechCol, alpha = 0.1, zorder = 2, lw = 0)
        
        #Clean up axis properties
        
        #Set x-limits
        plt.gca().set_xlim([0,100])
        
        #Add labels
        
        #X-axis (if bottom row)
        if kineticVarsPlot[var][0] == 4:
            plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 10, fontweight = 'bold')
            
        #Y-axis
        plt.gca().set_ylabel('Joint Moment (Nm)', fontsize = 10, fontweight = 'bold')
    
        #Set title
        plt.gca().set_title(kineticVarsTitle[list(kineticVarsPlot.keys()).index(var)],
                            pad = 5, fontsize = 10, fontweight = 'bold')
            
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
        
        #Remove x-tick labels if not bottom row
        if kineticVarsPlot[var][0] != 4:
            plt.gca().set_xticklabels([])
        
    #Turn off un-used axes
    ax[1,2].axis('off')
    
    #Create legend on dummy axis in bottom right
    plt.sca(ax[4,2])
    
    #Plot dummy data
    #RRA
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'RRA',
             ls = '-', lw = 1, c = rraCol,
             marker = markerDict['rra'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #RRA3
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'RRA3',
             ls = ':', lw = 1, c = rra3Col, 
             marker = markerDict['rra3'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #Moco
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'Moco',
             ls = '--', lw = 1, c = mocoCol,
             marker = markerDict['moco'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    #AddBiomechanics
    plt.plot(np.linspace(0,100,101), np.arange(0,1,1/101), label = 'AddBiomechanics',
             ls = '--', lw = 1, c = addBiomechCol,
             marker = markerDict['addBiomech'], markevery = 5, markersize = 3,
             alpha = 1.0, zorder = 3)
    
    #Add legend
    plt.legend()
    
    #Remove all axis properties
    #Spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #Ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    #Axis limits to avoid data
    plt.gca().set_ylim([50,100])
        
    #Save figure
    fig.savefig(os.path.join('..','..','results','HamnerDelpDataset','figures','meanKinetics.png'),
                format = 'png', dpi = 300)
    
    #Close figure
    plt.close('all')
    
    #Export mean kinematics dictionary to file
    with open(os.path.join('..','..','results','HamnerDelpDataset','outputs','meanKinetics.pkl'), 'wb') as writeFile:
        pickle.dump(meanKinetics, writeFile)
    
    # %% Create coloured models and average kinematic datafiles for each participant
    
    #Create dictionaries to store group kinematic data
    ikGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
    rraGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
    rra3GroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
    mocoGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
    addBiomechGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
    avgGroupTimes = {run: {'time': np.zeros((len(subList),101))} for run in runList}
    
    #Convert kinematic variables to table labels
    tableLabels = osim.StdVectorString()
    for currLabel in kinematicVars:
        tableLabels.append(currLabel)
    
    #Loop through subjects
    for subject in subList:
        
        #Read in gait timings
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'expData','gaitTimes.pkl'), 'rb') as openFile:
            gaitTimings = pickle.load(openFile)
        
        #Read in the kinematic data
        
        #IK
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikMeanKinematics.pkl'), 'rb') as openFile:
            ikKinematics = pickle.load(openFile)
        #RRA
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinematics.pkl'), 'rb') as openFile:
            rraKinematics = pickle.load(openFile)
        #RRA3
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinematics.pkl'), 'rb') as openFile:
            rra3Kinematics = pickle.load(openFile)
        #Moco
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinematics.pkl'), 'rb') as openFile:
            mocoKinematics = pickle.load(openFile)
        #AddBiomechanics
        with open(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinematics.pkl'), 'rb') as openFile:
            addBiomechKinematics = pickle.load(openFile)
            
        #Build a time series table with the mean kinematics from each category
        
        #Create the three tables
        ikTable = osim.TimeSeriesTable()
        rraTable = osim.TimeSeriesTable()
        rra3Table = osim.TimeSeriesTable()
        mocoTable = osim.TimeSeriesTable()
        addBiomechTable = osim.TimeSeriesTable()
        
        #Set the column labels
        ikTable.setColumnLabels(tableLabels)
        rraTable.setColumnLabels(tableLabels)
        rra3Table.setColumnLabels(tableLabels)
        mocoTable.setColumnLabels(tableLabels)
        addBiomechTable.setColumnLabels(tableLabels)
        
        #Create a time variable for the participant based on the average gait timings
        avgDur = np.array([gaitTimings[runLabel][cycle]['finalTime'] - gaitTimings[runLabel][cycle]['initialTime'] for cycle in cycleList]).mean()
        avgTime = np.linspace(0, avgDur, 101)
        
        #Fill the table with rows by looping through the rows and columns
        #By default OpenSim assumes radians, so convert those here (if not a translation)
        for iRow in range(len(avgTime)):
            #Create an array that gets each kinematic variable at the current point in time
            ikRowData = np.array([ikKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(ikKinematics[runLabel][var][iRow]) for var in kinematicVars])
            rraRowData = np.array([rraKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(rraKinematics[runLabel][var][iRow]) for var in kinematicVars])
            rra3RowData = np.array([rra3Kinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(rra3Kinematics[runLabel][var][iRow]) for var in kinematicVars])
            mocoRowData = np.array([mocoKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(mocoKinematics[runLabel][var][iRow]) for var in kinematicVars])
            addBiomechRowData = np.array([addBiomechKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(addBiomechKinematics[runLabel][var][iRow]) for var in kinematicVars])
            #Convert each of these to an osim row variable
            ikRow = osim.RowVector.createFromMat(ikRowData)
            rraRow = osim.RowVector.createFromMat(rraRowData)
            rra3Row = osim.RowVector.createFromMat(rra3RowData)
            mocoRow = osim.RowVector.createFromMat(mocoRowData)
            addBiomechRow = osim.RowVector.createFromMat(addBiomechRowData)
            #Append to the relevant table
            ikTable.appendRow(iRow, ikRow)
            rraTable.appendRow(iRow, rraRow)
            rra3Table.appendRow(iRow, rra3Row)
            mocoTable.appendRow(iRow, mocoRow)
            addBiomechTable.appendRow(iRow, addBiomechRow)
            #Set the time for current row
            ikTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
            rraTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
            rra3Table.setIndependentValueAtIndex(iRow, avgTime[iRow])
            mocoTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
            addBiomechTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
            
        #Write to mot file format
        osim.STOFileAdapter().write(ikTable, os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikMeanKinematics.sto'))
        osim.STOFileAdapter().write(rraTable, os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraMeanKinematics.sto'))
        osim.STOFileAdapter().write(rra3Table, os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3MeanKinematics.sto'))
        osim.STOFileAdapter().write(mocoTable, os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoMeanKinematics.sto'))
        osim.STOFileAdapter().write(addBiomechTable, os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechMeanKinematics.sto'))
        
        #Append participants kinematics to the broader group dictionary
        for var in kinematicVars:
            ikGroupKinematics[runLabel][var][subList.index(subject)] = ikKinematics[runLabel][var]
            rraGroupKinematics[runLabel][var][subList.index(subject)] = rraKinematics[runLabel][var]
            rra3GroupKinematics[runLabel][var][subList.index(subject)] = rra3Kinematics[runLabel][var]
            mocoGroupKinematics[runLabel][var][subList.index(subject)] = mocoKinematics[runLabel][var]
            addBiomechGroupKinematics[runLabel][var][subList.index(subject)] = addBiomechKinematics[runLabel][var]
            
        #Store average time in dictionary
        avgGroupTimes[runLabel]['time'][subList.index(subject)] = avgTime
            
        #Create colured versions of models for the categories
        
        #Read in multiple versions of the subject model
        ikModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        rraModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        rra3Model = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        mocoModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        addBiomechModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'))
        
        #Delete the forceset in each model to get rid of muscles
        ikModel.updForceSet().clearAndDestroy()
        rraModel.updForceSet().clearAndDestroy()
        rra3Model.updForceSet().clearAndDestroy()
        mocoModel.updForceSet().clearAndDestroy()
        addBiomechModel.updForceSet().clearAndDestroy()
        
        #Delete marker set in model
        ikModel.updMarkerSet().clearAndDestroy()
        rraModel.updMarkerSet().clearAndDestroy()
        rra3Model.updMarkerSet().clearAndDestroy()
        mocoModel.updMarkerSet().clearAndDestroy()
        addBiomechModel.updMarkerSet().clearAndDestroy()
    
        #Loop through the bodies and set the colouring
        #Also adjust the opacity here
        for bodyInd in range(ikModel.updBodySet().getSize()):
            
            #Loop through the attached geomtries on body
            for gInd in range(ikModel.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
                
                    #Set the colours for the current body and geometry
                    ikModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(ikColRGB) 
                    rraModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(rraColRGB)
                    rra3Model.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(rra3ColRGB)
                    mocoModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(mocoColRGB)
                    addBiomechModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(addBiomechColRGB)
                    
                    #Opacity
                    ikModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4) 
                    rraModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                    rra3Model.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                    mocoModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                    addBiomechModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                    
        #Set names
        ikModel.setName(f'{subject}_IK')
        rraModel.setName(f'{subject}_RRA')
        rra3Model.setName(f'{subject}_RRA3')
        mocoModel.setName(f'{subject}_Moco')
        addBiomechModel.setName(f'{subject}_AddBiomechanics')
        
        #Finalise model connections
        ikModel.finalizeConnections()
        rraModel.finalizeConnections()
        rra3Model.finalizeConnections()
        mocoModel.finalizeConnections()
        addBiomechModel.finalizeConnections()
        
        #Print to file
        ikModel.printToXML(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_ikModel.osim'))
        rraModel.printToXML(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rraModel.osim'))
        rra3Model.printToXML(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_rra3Model.osim'))
        mocoModel.printToXML(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_mocoModel.osim'))
        addBiomechModel.printToXML(os.path.join('..','..','data','HamnerDelp2013',subject,'results','outputs',f'{subject}_addBiomechModel.osim'))
    
    # %% Create mean models and kinematic files
    
    #Create coloured mean models based on generic model
    
    #Read in multiple versions of generic model
    ikMeanModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013','subject01','model','genericModel.osim'))
    rraMeanModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013','subject01','model','genericModel.osim'))
    rra3MeanModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013','subject01','model','genericModel.osim'))
    mocoMeanModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013','subject01','model','genericModel.osim'))
    addBiomechMeanModel = osim.Model(os.path.join('..','..','data','HamnerDelp2013','subject01','model','genericModel.osim'))
    
    #Delete the forceset in each model to get rid of muscles
    ikMeanModel.updForceSet().clearAndDestroy()
    rraMeanModel.updForceSet().clearAndDestroy()
    rra3MeanModel.updForceSet().clearAndDestroy()
    mocoMeanModel.updForceSet().clearAndDestroy()
    addBiomechMeanModel.updForceSet().clearAndDestroy()
    
    #Delete marker set in model
    ikMeanModel.updMarkerSet().clearAndDestroy()
    rraMeanModel.updMarkerSet().clearAndDestroy()
    rra3MeanModel.updMarkerSet().clearAndDestroy()
    mocoMeanModel.updMarkerSet().clearAndDestroy()
    addBiomechMeanModel.updMarkerSet().clearAndDestroy()
    
    #Loop through the bodies and set the colouring
    #Also adjust the opacity here
    for bodyInd in range(ikMeanModel.updBodySet().getSize()):
        
        #Loop through the attached geomtries on body
        for gInd in range(ikMeanModel.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
        
            #Set the colours for the current body and geometry
            ikMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(ikColRGB) 
            rraMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(rraColRGB)
            rra3MeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(rra3ColRGB)
            mocoMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(mocoColRGB)
            addBiomechMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(addBiomechColRGB)
            
            #Opacity
            ikMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4) 
            rraMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
            rra3MeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
            mocoMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
            addBiomechMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
            
    #Set names
    ikMeanModel.setName('generic_IK')
    rraMeanModel.setName('generic_RRA')
    rra3MeanModel.setName('generic_RRA3')
    mocoMeanModel.setName('generic_Moco')
    addBiomechMeanModel.setName('generic_AddBiomechanics')
    
    #Finalise model connections
    ikMeanModel.finalizeConnections()
    rraMeanModel.finalizeConnections()
    rra3MeanModel.finalizeConnections()
    mocoMeanModel.finalizeConnections()
    addBiomechMeanModel.finalizeConnections()
    
    #Print to file
    ikMeanModel.printToXML(os.path.join('..','..','results','HamnerDelpDataset','outputs','generic_ikModel.osim'))
    rraMeanModel.printToXML(os.path.join('..','..','results','HamnerDelpDataset','outputs','generic_rraModel.osim'))
    rra3MeanModel.printToXML(os.path.join('..','..','results','HamnerDelpDataset','outputs','generic_rra3Model.osim'))
    mocoMeanModel.printToXML(os.path.join('..','..','results','HamnerDelpDataset','outputs','generic_mocoModel.osim'))
    addBiomechMeanModel.printToXML(os.path.join('..','..','results','HamnerDelpDataset','outputs','generic_addBiomechModel.osim'))
    
    #Build a mean time series table with the kinematics from each category
    
    #Create the three tables
    ikMeanTable = osim.TimeSeriesTable()
    rraMeanTable = osim.TimeSeriesTable()
    rra3MeanTable = osim.TimeSeriesTable()
    mocoMeanTable = osim.TimeSeriesTable()
    addBiomechMeanTable = osim.TimeSeriesTable()
    
    #Set the column labels
    ikMeanTable.setColumnLabels(tableLabels)
    rraMeanTable.setColumnLabels(tableLabels)
    rra3MeanTable.setColumnLabels(tableLabels)
    mocoMeanTable.setColumnLabels(tableLabels)
    addBiomechMeanTable.setColumnLabels(tableLabels)
    
    #Create an average time variable
    avgMeanTime = np.mean(avgGroupTimes[runLabel]['time'], axis = 0)
    
    #Fill the table with rows by looping through the rows and columns
    #By default OpenSim assumes radians, so convert those here (if not a translation)
    for iRow in range(len(avgMeanTime)):
        #Create an array that gets each kinematic variable at the current point in time
        ikRowData = np.array([np.mean(ikGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(ikGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
        rraRowData = np.array([np.mean(rraGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(rraGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
        rra3RowData = np.array([np.mean(rra3GroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(rra3GroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
        mocoRowData = np.array([np.mean(mocoGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(mocoGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
        addBiomechRowData = np.array([np.mean(addBiomechGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(addBiomechGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
        #Convert each of these to an osim row variable
        ikRow = osim.RowVector.createFromMat(ikRowData)
        rraRow = osim.RowVector.createFromMat(rraRowData)
        rra3Row = osim.RowVector.createFromMat(rra3RowData)
        mocoRow = osim.RowVector.createFromMat(mocoRowData)
        addBiomechRow = osim.RowVector.createFromMat(addBiomechRowData)
        #Append to the relevant table
        ikMeanTable.appendRow(iRow, ikRow)
        rraMeanTable.appendRow(iRow, rraRow)
        rra3MeanTable.appendRow(iRow, rra3Row)
        mocoMeanTable.appendRow(iRow, mocoRow)
        addBiomechMeanTable.appendRow(iRow, addBiomechRow)
        #Set the time for current row
        ikMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
        rraMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
        rra3MeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
        mocoMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
        addBiomechMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
        
    #Write to mot file format
    osim.STOFileAdapter().write(ikMeanTable, os.path.join('..','..','results','HamnerDelpDataset','outputs','group_ikMeanKinematics.sto'))
    osim.STOFileAdapter().write(rraMeanTable, os.path.join('..','..','results','HamnerDelpDataset','outputs','group_rraMeanKinematics.sto'))
    osim.STOFileAdapter().write(rra3MeanTable, os.path.join('..','..','results','HamnerDelpDataset','outputs','group_rra3MeanKinematics.sto'))
    osim.STOFileAdapter().write(mocoMeanTable, os.path.join('..','..','results','HamnerDelpDataset','outputs','group_mocoMeanKinematics.sto'))
    osim.STOFileAdapter().write(addBiomechMeanTable, os.path.join('..','..','results','HamnerDelpDataset','outputs','group_addBiomechMeanKinematics.sto'))

# %% ----- end of runSimulations.py ----- %% #
