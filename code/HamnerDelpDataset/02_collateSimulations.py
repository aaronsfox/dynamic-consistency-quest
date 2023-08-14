# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through the process of collating the RRA and Moco Tracking
    simulations run on the Hamner & Delp 2013 data. Specifically we load in results
    from each subject to compare the kinematics and store as a figure. Data is also
    stored in easy to load in formats for subsequent analyses.
    
    TODO: 
        > Incorporate AddBiomechanics results

"""

# %% Import packages

import opensim as osim
# import osimFunctions as helper
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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

#Set conditionals for analyses to evalaute
readAndCheckKinematics = True
readAndCheckResiduals = True
runInverseSimulations = False

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

#Set a list for residual variables
residualVars = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']

#Set variables for residual data in respective files for rra vs. moco
rraResidualVars = ['ground_pelvis_pelvis_offset_FX', 'ground_pelvis_pelvis_offset_FY',
                   'ground_pelvis_pelvis_offset_FZ', 'ground_pelvis_pelvis_offset_MX',
                   'ground_pelvis_pelvis_offset_MY', 'ground_pelvis_pelvis_offset_MZ']
mocoResidualVars = ['/forceset/pelvis_tx_actuator', '/forceset/pelvis_ty_actuator',
                    '/forceset/pelvis_tz_actuator','/forceset/pelvis_list_actuator',
                    '/forceset/pelvis_rotation_actuator', '/forceset/pelvis_tilt_actuator']


#Set dicitonary for plotting axes
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

# %% Loop through subject list

for subject in subList:
    
    #Load in the subjects gait timing data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\gaitTimes.pkl', 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
    
    # %% Read in and compare kinematics
    
    #Check whether to evaluate kinematics
    if readAndCheckKinematics:
    
        #Create dictionaries to store RRA and Moco data
        #Individual cycle data
        ikKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
        rraKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
        mocoKinematics = {run: {cyc: {var: np.zeros(101) for var in kinematicVars} for cyc in cycleList} for run in runList}
        #Mean data
        ikMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
        rraMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
        mocoMeanKinematics = {run: {var: np.zeros(101) for var in kinematicVars} for run in runList}
        
        #Load in original IK kinematics
        ikData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\ik\\{runName}.mot')
        ikTime = np.array(ikData.getIndependentColumn())
        
        #Loop through cycles, load and normalise gait cycle to 101 points
        for cycle in cycleList:
            
            #Load RRA kinematics
            rraData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_Kinematics_q.sto')
            rraTime = np.array(rraData.getIndependentColumn())
            
            #Load Moco kinematics
            mocoData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_mocoKinematics.sto')
            mocoTime = np.array(mocoData.getIndependentColumn())
            
            #Associate start and stop indices to IK data for this cycle
            #Get times
            initialTime = rraTime[0]
            finalTime = rraTime[-1]
            #Get IK indices
            initialInd = np.argmax(ikTime > initialTime)
            finalInd = np.argmax(ikTime > finalTime) - 1
            
            #Loop through kinematic variables to extract
            for var in kinematicVars:
                
                #Extract kinematic variable data
                rraKinematicVar = rraData.getDependentColumn(var).to_numpy()
                if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                    mocoKinematicVar = mocoData.getDependentColumn(var).to_numpy()
                else:
                    mocoKinematicVar = np.rad2deg(mocoData.getDependentColumn(var).to_numpy()) #still in radians for joint angles for some reason
                
                #Extract inverse kinematics over time period
                ikKinematicVar = ikData.getDependentColumn(var).to_numpy()[initialInd:finalInd]
                ikTimeCycle = ikTime[initialInd:finalInd]
                
                #Interpolate to 101 points
                #Create interpolation function
                rraInterpFunc = interp1d(rraTime, rraKinematicVar)
                mocoInterpFunc = interp1d(mocoTime, mocoKinematicVar)
                ikInterpFunc = interp1d(ikTimeCycle, ikKinematicVar)
                #Interpolate data and store in relevant dictionary
                rraKinematics[runLabel][cycle][var] = rraInterpFunc(np.linspace(rraTime[0], rraTime[-1], 101))
                mocoKinematics[runLabel][cycle][var] = mocoInterpFunc(np.linspace(mocoTime[0], mocoTime[-1], 101))
                ikKinematics[runLabel][cycle][var] = ikInterpFunc(np.linspace(ikTimeCycle[0], ikTimeCycle[-1], 101))
        
        #Create a plot of the kinematics
        
        #Set colours for plots
        #RRA = blue; Moco = red
        rraColMean = '#0106BE'
        mocoColMean = '#BE0101'
        rraColCycle = '#7D81FF'
        mocoColCycle = '#FF7E7E'
        ikColMean = "#000000"
        ikColCycle = '#8F8F8F'
        
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
                         ls = '-', lw = 0.5, c = rraColCycle, zorder = 2)
                
                #Plot Moco data
                plt.plot(np.linspace(0,100,101), mocoKinematics[runLabel][cycle][var],
                         ls = '-', lw = 0.5, c = mocoColCycle, zorder = 2)
                
                #Plot IK data
                plt.plot(np.linspace(0,100,101), ikKinematics[runLabel][cycle][var],
                         ls = '-', lw = 0.5, c = ikColCycle, zorder = 2)
                
            #Plot mean curves
            
            #Calculate mean for current kinematic variable
            
            #RRA data
            rraKinematicMean = np.mean(np.vstack((rraKinematics[runLabel]['cycle1'][var],
                                                  rraKinematics[runLabel]['cycle2'][var],
                                                  rraKinematics[runLabel]['cycle3'][var])),
                                       axis = 0)
            
            #Moco data
            mocoKinematicMean = np.mean(np.vstack((mocoKinematics[runLabel]['cycle1'][var],
                                                   mocoKinematics[runLabel]['cycle2'][var],
                                                   mocoKinematics[runLabel]['cycle3'][var])),
                                        axis = 0)
            
            #IK data
            ikKinematicMean = np.mean(np.vstack((ikKinematics[runLabel]['cycle1'][var],
                                                 ikKinematics[runLabel]['cycle2'][var],
                                                 ikKinematics[runLabel]['cycle3'][var])),
                                        axis = 0)
            
            #Plot means
            
            #Plot RRA mean
            plt.plot(np.linspace(0,100,101), rraKinematicMean,
                     ls = '-', lw = 1.5, c = rraColMean, zorder = 3)
            
            #Plot Moco mean
            plt.plot(np.linspace(0,100,101), mocoKinematicMean,
                     ls = '-', lw = 1.5, c = mocoColMean, zorder = 3)
            
            #Plot Ik mean
            plt.plot(np.linspace(0,100,101), ikKinematicMean,
                     ls = '-', lw = 1.5, c = ikColMean, zorder = 3)
            
            #Store mean data for later use
            rraMeanKinematics[runLabel][var] = rraKinematicMean
            mocoMeanKinematics[runLabel][var] = mocoKinematicMean
            ikMeanKinematics[runLabel][var] = ikKinematicMean
            
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
        fig.suptitle(f'{subject} Kinematics Comparison (IK = Black, RRA = Blue, Moco = Red)',
                     fontsize = 12, fontweight = 'bold', y = 0.99)
        
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
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematics.pkl', 'wb') as writeFile:
            pickle.dump(mocoKinematics, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.pkl', 'wb') as writeFile:
            pickle.dump(mocoMeanKinematics, writeFile)
        
        #Calculate RMSE of kinematics vs. IK data
        
        #Create dictionaries for RMSE data (inc. spot for mean data)
        rraKinematicsRMSE = {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList}
        mocoKinematicsRMSE = {run: {cyc: {var: np.zeros(1) for var in kinematicVars} for cyc in cycleList+['mean']} for run in runList}
        
        #Loop through variables
        for var in kinematicVars:        
            #Loop through cycles
            for cycle in cycleList:                    
                #RRA data        
                rraKinematicsRMSE[runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - rraKinematics[runLabel]['cycle1'][var])**2))                
                #Moco data        
                mocoKinematicsRMSE[runLabel][cycle][var] = np.sqrt(np.mean((ikKinematics[runLabel]['cycle1'][var] - mocoKinematics[runLabel]['cycle1'][var])**2))
            #Calculate mean RMSE across cycles
            #RRA data
            rraKinematicsRMSE[runLabel]['mean'][var] = np.mean([rraKinematicsRMSE[runLabel][cycle][var] for cycle in cycleList])
            #Moco data
            mocoKinematicsRMSE[runLabel]['mean'][var] = np.mean([mocoKinematicsRMSE[runLabel][cycle][var] for cycle in cycleList])
        
        #Save kinematic RMSE data dictionaries
        #RRA data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(rraKinematicsRMSE, writeFile)
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematicsRMSE.pkl', 'wb') as writeFile:
            pickle.dump(mocoKinematicsRMSE, writeFile)
            
    # %% Read in and compare residuals
    
    #Check whether to evaluate residuals
    if readAndCheckResiduals:
    
        #Create dictionaries to store RRA and Moco data
        #Individual cycle data
        rraResiduals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
        mocoResiduals = {run: {cyc: {var: np.zeros(101) for var in residualVars} for cyc in cycleList} for run in runList}
        #Mean data
        rraMeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
        mocoMeanResiduals = {run: {var: np.zeros(101) for var in residualVars} for run in runList}
        
        #Loop through cycles, load and normalise gait cycle to 101 points
        for cycle in cycleList:
            
            #Load RRA body forces
            rraData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_bodyForces.sto')
            rraTime = np.array(rraData.getIndependentColumn())
            
            #Load Moco solution
            mocoData = osim.TimeSeriesTable(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_mocoSolution.sto')
            mocoTime = np.array(mocoData.getIndependentColumn())

            #Loop through kinematic variables to extract
            for var in residualVars:
                
                #Map residual variable to appropriate column label in respective data
                rraVar = rraResidualVars[residualVars.index(var)]
                mocoVar = mocoResidualVars[residualVars.index(var)]
                
                #Extract residual data
                rraResidualVar = rraData.getDependentColumn(rraVar).to_numpy()
                mocoResidualVar = mocoData.getDependentColumn(mocoVar).to_numpy() #no need to multiply by optForce as it was 1

                #Normalise data to model mass
                
                #Load models
                rraModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{cycle}\\{subject}_{runLabel}_{cycle}_rraAdjusted.osim')
                mocoModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
                
                #Get body mass
                rraModelMass = np.sum([rraModel.updBodySet().get(ii).getMass() for ii in range(rraModel.updBodySet().getSize())])
                mocoModelMass = np.sum([mocoModel.updBodySet().get(ii).getMass() for ii in range(mocoModel.updBodySet().getSize())])
                
                #Normalise data
                rraResidualVarNorm = rraResidualVar / rraModelMass
                mocoResidualVarNorm = mocoResidualVar / mocoModelMass
                
                #Interpolate to 101 points
                #Create interpolation function
                rraInterpFunc = interp1d(rraTime, rraResidualVarNorm)
                mocoInterpFunc = interp1d(mocoTime, mocoResidualVarNorm)
                #Interpolate data and store in relevant dictionary
                rraResiduals[runLabel][cycle][var] = rraInterpFunc(np.linspace(rraTime[0], rraTime[-1], 101))
                mocoResiduals[runLabel][cycle][var] = mocoInterpFunc(np.linspace(mocoTime[0], mocoTime[-1], 101))
        
        #Create a plot of the residuals
        
        #Set colours for plots
        #RRA = blue; Moco = red
        rraColMean = '#0106BE'
        mocoColMean = '#BE0101'
        rraColCycle = '#7D81FF'
        mocoColCycle = '#FF7E7E'
        
        #Create the figure
        fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (9, 4))
        
        #Adjust subplots
        plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.085, top = 0.875,
                            hspace = 0.4, wspace = 0.35)
        
        #Loop through variables and plot data
        for var in residualVars:
            
            #Set the appropriate axis
            plt.sca(ax.flatten()[residualVars.index(var)])
                    
            #Loop through cycles to plot individual curves
            for cycle in cycleList:
                
                #Plot RRA data
                plt.plot(np.linspace(0,100,101), rraResiduals[runLabel][cycle][var],
                         ls = '-', lw = 0.5, c = rraColCycle, zorder = 2)
                
                #Plot Moco data
                plt.plot(np.linspace(0,100,101), mocoResiduals[runLabel][cycle][var],
                         ls = '-', lw = 0.5, c = mocoColCycle, zorder = 2)
                
            #Plot mean curves
            
            #Calculate mean for current kinematic variable
            
            #RRA data
            rraResidualMean = np.mean(np.vstack((rraResiduals[runLabel]['cycle1'][var],
                                                 rraResiduals[runLabel]['cycle2'][var],
                                                 rraResiduals[runLabel]['cycle3'][var])),
                                      axis = 0)
            
            #Moco data
            mocoResidualMean = np.mean(np.vstack((mocoResiduals[runLabel]['cycle1'][var],
                                                  mocoResiduals[runLabel]['cycle2'][var],
                                                  mocoResiduals[runLabel]['cycle3'][var])),
                                       axis = 0)
            
            #Plot means
            
            #Plot RRA mean
            plt.plot(np.linspace(0,100,101), rraResidualMean,
                     ls = '-', lw = 1.5, c = rraColMean, zorder = 3)
            
            #Plot Moco mean
            plt.plot(np.linspace(0,100,101), mocoResidualMean,
                     ls = '-', lw = 1.5, c = mocoColMean, zorder = 3)
            
            #Store mean data for later use
            rraMeanResiduals[runLabel][var] = rraResidualMean
            mocoMeanResiduals[runLabel][var] = mocoResidualMean
            
            #Clean up axis properties
            
            #Set x-limits
            plt.gca().set_xlim([0,100])
            
            #Add labels
            
            #X-axis (if bottom row)
            if var.startswith('M'):
                plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 8, fontweight = 'bold')
                
            #Y-axis (dependent on kinematic variable)
            if var.startswith('F'):
                plt.gca().set_ylabel('Residual Force (N.kg)', fontsize = 8, fontweight = 'bold')
            else:
                plt.gca().set_ylabel('Residual Moment (Nm.kg)', fontsize = 8, fontweight = 'bold')
    
            #Set title
            plt.gca().set_title(var, pad = 3, fontsize = 12, fontweight = 'bold')
                
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
        fig.suptitle(f'{subject} Residuals Comparison (RRA = Blue, Moco = Red)',
                     fontsize = 12, fontweight = 'bold', y = 0.99)
        
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
        #Moco data
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoResiduals.pkl', 'wb') as writeFile:
            pickle.dump(mocoResiduals, writeFile)
        with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanResiduals.pkl', 'wb') as writeFile:
            pickle.dump(mocoMeanResiduals, writeFile)
    
# %% ----- end of 02_collateSimulations.py ----- %% #