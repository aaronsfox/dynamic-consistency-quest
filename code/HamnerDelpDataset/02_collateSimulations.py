# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through the process of collating the RRA, Moco Tracking and
    AddBiomechanics simulations run on the Hamner & Delp 2013 data. Specifically
    we load in results from each subject to compare the various data and store
    these data in easy to load in formats for subsequent analyses.
    
    TODO: 
        > Incorporate AddBiomechanics results
            >> Are pro_sup limits allowed to be exceeded in AddBiomechanics?
                >>> If expanding limits helps pro_sup accuracy, then need to adapt
                    how model is created, and re-run AddBiomechanics processing
                >>> It does --- need to re-run...
        > Incorporate review of joint torques

"""

# %% Import packages

import opensim as osim
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# %% Set-up

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

#Add OpenSim geometry path
#Can be helpful with running into any issues around geometry path
#Set this to OpenSim install directory
osim.ModelVisualizer.addDirToGeometrySearchPaths('C:\\OpenSim 4.3\\Geometry')

#Get home path
homeDir = os.getcwd()

#Set conditionals for analyses to evaluate
readAndCheckKinematics = True
readAndCheckKinetics = True
readAndCheckResiduals = True

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
#Currently just one trial but could be adapted to a list
runLabel = 'run5'
runName = 'Run_5'
    
#Set run cycle list
cycleList = ['cycle1',
             'cycle2',
             'cycle3']

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

#Set colours for plots
ikCol = '#000000' #IK = black
rraCol = '#e569ce' #RRA = purple
rra3Col = '#ff6876' #RRA3 = pink
mocoCol = '#4885ed' #Moco = blue
addBiomechCol = '#ffa600' #AddBiomechanics = gold

# %% Loop through subject list

for subject in subList:
    
    #Load in the subjects gait timing data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\gaitTimes.pkl', 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
        
    #Calculate residual force and moment recommendations based on original experimental data
    #Force residual recommendations are 5% of maximum external force
    #Moment residual recommendations are 1% of COM height * maximum external force
    
    #Read in external GRF and get peak force residual recommendation
    expGRF = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\{runName}_grf.mot')
    peakVGRF = np.array((expGRF.getDependentColumn('R_ground_force_vy').to_numpy().max(),
                         expGRF.getDependentColumn('L_ground_force_vy').to_numpy().max())).max()
    forceResidualRec = peakVGRF * 0.05
    
    #Extract centre of mass from static output
    #Load in scaled model
    scaledModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    modelState = scaledModel.initSystem()
    #Read in static motion output
    staticMotion = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_static_output.mot')
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
        ikData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\ik\\{runName}.mot')
        ikTime = np.array(ikData.getIndependentColumn())
        
        #Loop through cycles, load and normalise gait cycle to 101 points
        for cycle in cycleList:
            
            #Load RRA kinematics
            rraData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_Kinematics_q.sto')
            rraTime = np.array(rraData.getIndependentColumn())
            
            #Load RRA3 kinematics
            rra3Data = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra3\\{runLabel}\\rra3\\{cycle}\\{subject}_{runLabel}_{cycle}_iter3_Kinematics_q.sto')
            rra3Time = np.array(rra3Data.getIndependentColumn())
            
            #Load Moco kinematics
            mocoData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_mocoKinematics.sto')
            mocoTime = np.array(mocoData.getIndependentColumn())
            
            #Load AddBiomechanics kinematics
            #Slightly different as able to load these from .csv file
            addBiomechData = pd.read_csv(f'..\\..\\data\\HamnerDelp2013\\{subject}\\addBiomechanics\\{runLabel}\\ID\\{runName}_full.csv')
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
                         ls = '-', lw = 0.5, c = rraCol, alpha = 0.4, zorder = 2)
                
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
                     ls = '-', lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
            
            #Plot RRA3 mean
            plt.plot(np.linspace(0,100,101), rra3MeanKinematics[runLabel][var],
                     ls = '-', lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
            
            #Plot Moco mean
            plt.plot(np.linspace(0,100,101), mocoMeanKinematics[runLabel][var],
                     ls = '-', lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
            
            #Plot AddBiomechanics mean
            plt.plot(np.linspace(0,100,101), addBiomechMeanKinematics[runLabel][var],
                     ls = '-', lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
            
            #Plot Ik mean
            plt.plot(np.linspace(0,100,101), ikMeanKinematics[runLabel][var],
                     ls = '-', lw = 1.5, c = ikCol, alpha = 1.0, zorder = 3)

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
        fig.savefig(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\figures\\{subject}_{runLabel}_kinematicsComparison.png',
                    format = 'png', dpi = 300)
        
        #Close figure
        plt.close('all')
        
        #Save kinematic data dictionaries
        #IK data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikKinematics.pkl', 'wb') as writeFile:
            pickle.dump(ikKinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.pkl', 'wb') as writeFile:
            pickle.dump(ikMeanKinematics, writeFile)
        #RRA data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinematics.pkl', 'wb') as writeFile:
            pickle.dump(rraKinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.pkl', 'wb') as writeFile:
            pickle.dump(rraMeanKinematics, writeFile)
        #RRA3 data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3Kinematics.pkl', 'wb') as writeFile:
            pickle.dump(rra3Kinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinematics.pkl', 'wb') as writeFile:
            pickle.dump(rra3MeanKinematics, writeFile)
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematics.pkl', 'wb') as writeFile:
            pickle.dump(mocoKinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.pkl', 'wb') as writeFile:
            pickle.dump(mocoMeanKinematics, writeFile)
        #AddBiomechanics data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechKinematics.pkl', 'wb') as writeFile:
            pickle.dump(addBiomechKinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinematics.pkl', 'wb') as writeFile:
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
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikKinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(ikKinematicsRMSE, writeFile)
        #RRA
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(rraKinematicsRMSE, writeFile)
        #RRA3
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3KinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(rra3KinematicsRMSE, writeFile)
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(mocoKinematicsRMSE, writeFile)
        #AddBiomechanics data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechKinematicsRMSE.pkl', 'wb') as writeFile:
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
            rraData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_Actuation_force.sto')
            rraTime = np.array(rraData.getIndependentColumn())
            
            #Load RRA3 kinetics
            rra3Data = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra3\\{runLabel}\\rra3\\{cycle}\\{subject}_{runLabel}_{cycle}_iter3_Actuation_force.sto')
            rra3Time = np.array(rra3Data.getIndependentColumn())
            
            #Load Moco kinetics
            mocoData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_mocoSolution.sto')
            mocoTime = np.array(mocoData.getIndependentColumn())
            
            #Load AddBiomechanics kinetics
            #Slightly different as able to load these from .csv file
            addBiomechData = pd.read_csv(f'..\\..\\data\\HamnerDelp2013\\{subject}\\addBiomechanics\\{runLabel}\\ID\\{runName}_full.csv')
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
                     ls = '-', lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
            
            #Plot RRA3 mean
            plt.plot(np.linspace(0,100,101), rra3MeanKinetics[runLabel][var],
                     ls = '-', lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
            
            #Plot Moco mean
            plt.plot(np.linspace(0,100,101), mocoMeanKinetics[runLabel][var],
                     ls = '-', lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
            
            #Plot AddBiomechanics mean
            plt.plot(np.linspace(0,100,101), addBiomechMeanKinetics[runLabel][var],
                     ls = '-', lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)

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
        fig.savefig(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\figures\\{subject}_{runLabel}_kineticsComparison.png',
                    format = 'png', dpi = 300)
        
        #Close figure
        plt.close('all')
        
        #Save kinetic data dictionaries
        #RRA data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinetics.pkl', 'wb') as writeFile:
            pickle.dump(rraKinetics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinetics.pkl', 'wb') as writeFile:
            pickle.dump(rraMeanKinetics, writeFile)
        #RRA3 data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3Kinetics.pkl', 'wb') as writeFile:
            pickle.dump(rra3Kinetics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinetics.pkl', 'wb') as writeFile:
            pickle.dump(rra3MeanKinetics, writeFile)
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinetics.pkl', 'wb') as writeFile:
            pickle.dump(mocoKinetics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinetics.pkl', 'wb') as writeFile:
            pickle.dump(mocoMeanKinetics, writeFile)
        #AddBiomechanics data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechKinetics.pkl', 'wb') as writeFile:
            pickle.dump(addBiomechKinetics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinetics.pkl', 'wb') as writeFile:
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
            rraData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_bodyForces.sto')
            rraTime = np.array(rraData.getIndependentColumn())
            
            #Load RRA3 body forces
            rra3Data = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra3\\{runLabel}\\rra3\\{cycle}\\{subject}_{runLabel}_{cycle}_iter3_bodyForces.sto')
            rra3Time = np.array(rra3Data.getIndependentColumn())
            
            #Load Moco solution
            mocoData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_mocoSolution.sto')
            mocoTime = np.array(mocoData.getIndependentColumn())
            
            #Load AddBiomechanics solution
            addBiomechData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\addBiomechanics\\{runLabel}\\ID\\{runName}_id.sto')
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
                     ls = '-', lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
            
            #Plot RRA3 mean
            plt.plot(np.linspace(0,100,101), rra3MeanResiduals[runLabel][var],
                     ls = '-', lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
            
            #Plot Moco mean
            plt.plot(np.linspace(0,100,101), mocoMeanResiduals[runLabel][var],
                     ls = '-', lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
            
            #Plot AddBiomechanics mean
            plt.plot(np.linspace(0,100,101), addBiomechMeanResiduals[runLabel][var],
                     ls = '-', lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)

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
        fig.savefig(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\figures\\{subject}_{runLabel}_residualsComparison.png',
                    format = 'png', dpi = 300)
        
        #Close figure
        plt.close('all')
        
        #Save residual data dictionaries
        #RRA data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraResiduals.pkl', 'wb') as writeFile:
            pickle.dump(rraResiduals, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanResiduals.pkl', 'wb') as writeFile:
            pickle.dump(rraMeanResiduals, writeFile)
        #RRA3 data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3Residuals.pkl', 'wb') as writeFile:
            pickle.dump(rra3Residuals, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanResiduals.pkl', 'wb') as writeFile:
            pickle.dump(rra3MeanResiduals, writeFile)
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoResiduals.pkl', 'wb') as writeFile:
            pickle.dump(mocoResiduals, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanResiduals.pkl', 'wb') as writeFile:
            pickle.dump(mocoMeanResiduals, writeFile)
        #AddBiomechanics data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechResiduals.pkl', 'wb') as writeFile:
            pickle.dump(addBiomechResiduals, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanResiduals.pkl', 'wb') as writeFile:
            pickle.dump(addBiomechMeanResiduals, writeFile)
    
# %% ----- end of 02_collateSimulations.py ----- %% #