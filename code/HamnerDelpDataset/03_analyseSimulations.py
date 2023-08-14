# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through the process of reading in and analysing the RRA
    and Moco Tracking simulations run on the Hamner & Delp 2013 data to compare
    the solutions.
    
    TODO:
        > Include AddBiomechanics results in here

"""

# %% Import packages

import opensim as osim
# import osimFunctions as helper
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

#Set a list for generic kinematic vars
kinematicVarsGen = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                    'hip_flexion', 'hip_adduction', 'hip_rotation',
                    'knee_angle', 'ankle_angle',
                    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                    'arm_flex', 'arm_add', 'arm_rot',
                    'elbow_flex', 'pro_sup'
                    ]

#Set a list for residual variables
residualVars = ['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ']

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

#Set colours
rraCol = '#0106BE'
mocoCol = '#BE0101'
ikCol = "#000000"

# %% Extract solution times

#Set a place to store average solution times
solutionTimes = {'rra': np.zeros(len(subList)), 'moco': np.zeros(len(subList))}

#Loop through subject list
for subject in subList:
    
    #Load RRA and Moco solution time data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{subject}_rraRunTimeData.pkl', 'rb') as openFile:
        rraRunTime = pickle.load(openFile)
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{subject}_mocoRunTimeData.pkl', 'rb') as openFile:
        mocoRunTime = pickle.load(openFile)
        
    #Calculate averages and append to dictionary
    solutionTimes['rra'][subList.index(subject)] = np.array([rraRunTime[runLabel][cycle]['rraRunTime'] for cycle in cycleList]).mean()
    solutionTimes['moco'][subList.index(subject)] = np.array([mocoRunTime[runLabel][cycle]['mocoRunTime'] for cycle in cycleList]).mean()
    
#Average and display these results
# print(f'Average RRA run time (s): {np.round(solutionTimes["rra"].mean(),2)} +/- {np.round(solutionTimes["rra"].std(),2)}')
# print(f'Average Moco run time (s): {np.round(solutionTimes["moco"].mean(),2)} +/- {np.round(solutionTimes["moco"].std(),2)}')
print(f'Average RRA run time (s): {np.round((solutionTimes["rra"]/60).mean(),2)} +/- {np.round((solutionTimes["rra"]/60).std(),2)}')
print(f'Average Moco run time (s): {np.round((solutionTimes["moco"]/60).mean(),2)} +/- {np.round((solutionTimes["moco"]/60).std(),2)}')

#Convert to dataframe for plotting
#Convert to minutes here too
solutionTimes_df = pd.DataFrame(list(zip((list(solutionTimes['rra'] / 60) + list(solutionTimes['moco'] / 60)),
                                         (['RRA']*len(solutionTimes['rra']) + ['MOCO']*len(solutionTimes['moco'])))),
                                columns = ['Time', 'Solver'])

#Create figure
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))

#Add boxplot
bp = sns.boxplot(data = solutionTimes_df,
                 x = 'Solver', y = 'Time', order = ['RRA', 'MOCO'],
                 palette = [rraCol, mocoCol],
                 dodge = 0.5, width = 0.3, whis = [0,100],
                 ax = ax)

#Adjust colours of boxplot lines and fill
for ii in range(len(ax.patches)):
    
    #Get the current artist
    artist = ax.patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

    #Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    #Loop over them here, and use the same colour as above
    for jj in range(ii*6,ii*6+6):
        line = ax.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)
        
#Add the strip plot for points
sp = sns.stripplot(data = solutionTimes_df,
                   x = 'Solver', y = 'Time', order = ['RRA', 'MOCO'],
                   size = 5, palette = [rraCol, mocoCol], alpha = 0.5,
                   jitter = True, dodge = False,
                   ax = ax)

#Set y-axes limits
ax.set_ylim([0,ax.get_ylim()[1]])

#Remove x-label
ax.set_xlabel('')

#Set y-label
ax.set_ylabel('Solution Time (mins)', fontsize = 14, labelpad = 10)

#Set x-labels
ax.set_xticklabels(['RRA', 'MOCO'], fontsize = 12)
ax.set_xlabel('')

#Despine top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig('..\\..\\results\\figures\\averageSolutionTimes.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

# %% Extract peak residual forces and moments

#Set a place to store average solution times
peakResiduals = {'rra': {var: np.zeros(len(subList)) for var in residualVars},
                 'moco':{var: np.zeros(len(subList)) for var in residualVars}}

#Loop through subject list
for subject in subList:
    
    #Load RRA and Moco solution RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraResiduals.pkl', 'rb') as openFile:
        rraResiduals = pickle.load(openFile)
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoResiduals.pkl', 'rb') as openFile:
        mocoResiduals = pickle.load(openFile)

    #Loop through and extract peak residuals and average
    for var in residualVars:
        
        #Extract the peak from each cycle and place in dictionary for rra and moco data
        peakResiduals['rra'][var][subList.index(subject)] = np.array([np.abs(rraResiduals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()
        peakResiduals['moco'][var][subList.index(subject)] = np.array([np.abs(mocoResiduals[runLabel][cycle][var]).max() for cycle in cycleList]).mean()

#Average and display results for residual variables
for var in residualVars:
    print(f'Average RRA residuals for {var}: {np.round(peakResiduals["rra"][var].mean(),3)} +/- {np.round(peakResiduals["rra"][var].std(),3)}')
    print(f'Average Moco resdiauls for {var}: {np.round(peakResiduals["moco"][var].mean(),3)} +/- {np.round(peakResiduals["moco"][var].std(),3)}')

#Check X-fold increases in RRA vs. Moco residuals
for var in residualVars:
    print(f'X-fold increase in RRA residuals for {var}: {np.round(peakResiduals["rra"][var].mean() / peakResiduals["moco"][var].mean(),3)}')


#Convert to dataframe for plotting
#Forces
residuals = (list(peakResiduals['rra']['FX']) + list(peakResiduals['rra']['FY']) + list(peakResiduals['rra']['FZ']) + list(peakResiduals['moco']['FX']) + list(peakResiduals['moco']['FY']) + list(peakResiduals['moco']['FZ']))
solver = (['RRA']*len(peakResiduals['rra']['FX'])*3 + ['MOCO']*len(peakResiduals['rra']['FX'])*3)
axis = (['FX']*len(peakResiduals['rra']['FX']) + ['FY']*len(peakResiduals['rra']['FY']) + ['FZ']*len(peakResiduals['rra']['FZ']) + ['FX']*len(peakResiduals['moco']['FX']) + ['FY']*len(peakResiduals['moco']['FY']) + ['FZ']*len(peakResiduals['moco']['FZ']))
peakResidualForces_df = pd.DataFrame(list(zip(residuals, solver, axis)),
                                     columns = ['Peak Residual Force', 'Solver', 'Axis'])
#Moments
residuals = (list(peakResiduals['rra']['MX']) + list(peakResiduals['rra']['MY']) + list(peakResiduals['rra']['MZ']) + list(peakResiduals['moco']['MX']) + list(peakResiduals['moco']['MY']) + list(peakResiduals['moco']['MZ']))
solver = (['RRA']*len(peakResiduals['rra']['MX'])*3 + ['MOCO']*len(peakResiduals['rra']['MX'])*3)
axis = (['MX']*len(peakResiduals['rra']['MX']) + ['MY']*len(peakResiduals['rra']['MY']) + ['MZ']*len(peakResiduals['rra']['MZ']) + ['MX']*len(peakResiduals['moco']['MX']) + ['MY']*len(peakResiduals['moco']['MY']) + ['MZ']*len(peakResiduals['moco']['MZ']))
peakResidualMoments_df = pd.DataFrame(list(zip(residuals, solver, axis)),
                                      columns = ['Peak Residual Moment', 'Solver', 'Axis'])
    
#Create figure for forces
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))

#Add boxplot
bp = sns.boxplot(data = peakResidualForces_df,
                 x = 'Axis', y = 'Peak Residual Force', order = ['FX', 'FY', 'FZ'],
                 hue = 'Solver', hue_order = ['RRA', 'MOCO'],
                 palette = [rraCol, mocoCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax)

#Adjust colours of boxplot lines and fill
for ii in range(len(ax.patches)):
    
    #Get the current artist
    artist = ax.patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,6):
    if ii % 2 == 0:
        col = rraCol
    else:
        col = mocoCol
    #Loop through the 6 lines
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = peakResidualForces_df,
                   x = 'Axis', y = 'Peak Residual Force', order = ['FX', 'FY', 'FZ'],
                   hue = 'Solver', hue_order = ['RRA', 'MOCO'],
                   size = 5, palette = [rraCol, mocoCol], alpha = 0.5,
                   jitter = True, dodge = 0.01,
                   ax = ax)

#Turn off legend
ax.get_legend().remove()

#Set y-axes limits
ax.set_ylim([0.0,ax.get_ylim()[1]])

#Remove x-label
ax.set_xlabel('')

#Set y-label
ax.set_ylabel(r'Peak Residual Force (N$\cdot$kg$^{-1}$)', fontsize = 14, labelpad = 10)

#Set x-labels
ax.set_xticklabels(['FX', 'FY', 'FZ'], fontsize = 12)

#Despine top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig('..\\..\\results\\figures\\averagePeakResidualForces.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

#Create figure for moments
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,7))

#Add boxplot
bp = sns.boxplot(data = peakResidualMoments_df,
                 x = 'Axis', y = 'Peak Residual Moment', order = ['MX', 'MY', 'MZ'],
                 hue = 'Solver', hue_order = ['RRA', 'MOCO'],
                 palette = [rraCol, mocoCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax)

#Adjust colours of boxplot lines and fill
for ii in range(len(ax.patches)):
    
    #Get the current artist
    artist = ax.patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,6):
    if ii % 2 == 0:
        col = rraCol
    else:
        col = mocoCol
    #Loop through the 6 lines
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = peakResidualMoments_df,
                   x = 'Axis', y = 'Peak Residual Moment', order = ['MX', 'MY', 'MZ'],
                   hue = 'Solver', hue_order = ['RRA', 'MOCO'],
                   size = 5, palette = [rraCol, mocoCol], alpha = 0.5,
                   jitter = True, dodge = 0.01,
                   ax = ax)

#Turn off legend
ax.get_legend().remove()

#Set y-axes limits
ax.set_ylim([0.0,ax.get_ylim()[1]])

#Remove x-label
ax.set_xlabel('')

#Set y-label
ax.set_ylabel(r'Peak Residual Moments (Nm$\cdot$kg$^{-1}$)', fontsize = 14, labelpad = 10)

#Set x-labels
ax.set_xticklabels(['MX', 'MY', 'MZ'], fontsize = 12)

#Despine top and right axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig('..\\..\\results\\figures\\averagePeakResidualMoments.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()


# %% Extract root mean square deviations versus kinematic data

#Set a place to store average solution times
kinematicsRMSD = {'rra': {var: np.zeros(len(subList)) for var in kinematicVarsGen},
                  'moco':{var: np.zeros(len(subList)) for var in kinematicVarsGen}}

#Loop through subject list
for subject in subList:
    
    #Load RRA and Moco solution RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinematicsRMSE.pkl', 'rb') as openFile:
        rraRMSD = pickle.load(openFile)
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematicsRMSE.pkl', 'rb') as openFile:
        mocoRMSD = pickle.load(openFile)

    #Loop through and extract mean for generic kinematic variables
    for var in kinematicVarsGen:
        
        #Check for pelvis/lumbar variable
        if 'pelvis' in var or 'lumbar' in var:
            #Extract the mean and place in dictionary for rra and moco data
            kinematicsRMSD['rra'][var][subList.index(subject)] = rraRMSD[runLabel]['mean'][var]
            kinematicsRMSD['moco'][var][subList.index(subject)] = mocoRMSD[runLabel]['mean'][var]
        else:
            #Caclculate for both left and right variables
            kinematicsRMSD['rra'][var][subList.index(subject)] = np.array((rraRMSD[runLabel]['mean'][f'{var}_r'], rraRMSD[runLabel]['mean'][f'{var}_l'])).mean()
            kinematicsRMSD['moco'][var][subList.index(subject)] = np.array((mocoRMSD[runLabel]['mean'][f'{var}_r'], mocoRMSD[runLabel]['mean'][f'{var}_l'])).mean()

#Average and display results for variables
for var in kinematicVarsGen:
    if var in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
        #Convert to cm for translations
        print(f'Average RRA RMSD for {var}: {np.round(kinematicsRMSD["rra"][var].mean()*100,2)} +/- {np.round(kinematicsRMSD["rra"][var].std()*100,2)}')
        print(f'Average Moco RMSD for {var}: {np.round(kinematicsRMSD["moco"][var].mean()*100,2)} +/- {np.round(kinematicsRMSD["moco"][var].std()*100,2)}')
    else:
        #Present in degrees
        print(f'Average RRA RMSD for {var}: {np.round(kinematicsRMSD["rra"][var].mean(),2)} +/- {np.round(kinematicsRMSD["rra"][var].std(),2)}')
        print(f'Average Moco RMSD for {var}: {np.round(kinematicsRMSD["moco"][var].mean(),2)} +/- {np.round(kinematicsRMSD["moco"][var].std(),2)}')

# %% Create coloured models and average kinematic datafiles for each participant

#Create dictionaries to store group kinematic data
ikGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
rraGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
mocoGroupKinematics = {run: {var: np.zeros((len(subList),101)) for var in kinematicVars} for run in runList}
avgGroupTimes = {run: {'time': np.zeros((len(subList),101))} for run in runList}

#Convert kinematic variables to table labels
tableLabels = osim.StdVectorString()
for currLabel in kinematicVars:
    tableLabels.append(currLabel)

#Loop through subjects
for subject in subList:
    
    #Read in gait timings
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\gaitTimes.pkl', 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
    
    #Read in the kinematic data
    
    #IK
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.pkl', 'rb') as openFile:
        ikKinematics = pickle.load(openFile)
    #RRA
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.pkl', 'rb') as openFile:
        rraKinematics = pickle.load(openFile)
    #Moco
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.pkl', 'rb') as openFile:
        mocoKinematics = pickle.load(openFile)
        
    #Build a time series table with the mean kinematics from each category
    
    #Create the three tables
    ikTable = osim.TimeSeriesTable()
    rraTable = osim.TimeSeriesTable()
    mocoTable = osim.TimeSeriesTable()
    
    #Set the column labels
    ikTable.setColumnLabels(tableLabels)
    rraTable.setColumnLabels(tableLabels)
    mocoTable.setColumnLabels(tableLabels)
    
    #Create a time variable for the participant based on the average gait timings
    avgDur = np.array([gaitTimings[runLabel][cycle]['finalTime'] - gaitTimings[runLabel][cycle]['initialTime'] for cycle in cycleList]).mean()
    avgTime = np.linspace(0, avgDur, 101)
    
    #Fill the table with rows by looping through the rows and columns
    #By default OpenSim assumes radians, so convert those here (if not a translation)
    for iRow in range(len(avgTime)):
        #Create an array that gets each kinematic variable at the current point in time
        ikRowData = np.array([ikKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(ikKinematics[runLabel][var][iRow]) for var in kinematicVars])
        rraRowData = np.array([rraKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(rraKinematics[runLabel][var][iRow]) for var in kinematicVars])
        mocoRowData = np.array([mocoKinematics[runLabel][var][iRow] if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(mocoKinematics[runLabel][var][iRow]) for var in kinematicVars])
        #Convert each of these to an osim row variable
        ikRow = osim.RowVector.createFromMat(ikRowData)
        rraRow = osim.RowVector.createFromMat(rraRowData)
        mocoRow = osim.RowVector.createFromMat(mocoRowData)
        #Append to the relevant table
        ikTable.appendRow(iRow, ikRow)
        rraTable.appendRow(iRow, rraRow)
        mocoTable.appendRow(iRow, mocoRow)
        #Set the time for current row
        ikTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
        rraTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
        mocoTable.setIndependentValueAtIndex(iRow, avgTime[iRow])
        
    #Write to mot file format
    osim.STOFileAdapter().write(ikTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.sto')
    osim.STOFileAdapter().write(rraTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.sto')
    osim.STOFileAdapter().write(mocoTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.sto')
    
    #Append participants kinematics to the broader group dictionary
    for var in kinematicVars:
        ikGroupKinematics[runLabel][var][subList.index(subject)] = ikKinematics[runLabel][var]
        rraGroupKinematics[runLabel][var][subList.index(subject)] = rraKinematics[runLabel][var]
        mocoGroupKinematics[runLabel][var][subList.index(subject)] = mocoKinematics[runLabel][var]
        
    #Store average time in dictionary
    avgGroupTimes[runLabel]['time'][subList.index(subject)] = avgTime
        
    #Create colured versions of models for the categories
    
    #Read in three versions of the subject model
    ikModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    rraModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    mocoModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    
    #Delete the forceset in each model to get rid of muscles
    ikModel.updForceSet().clearAndDestroy()
    rraModel.updForceSet().clearAndDestroy()
    mocoModel.updForceSet().clearAndDestroy()
    
    #Delete marker set in model
    ikModel.updMarkerSet().clearAndDestroy()
    rraModel.updMarkerSet().clearAndDestroy()
    mocoModel.updMarkerSet().clearAndDestroy()

    #Loop through the bodies and set the colouring
    #Also adjust the opacity here
    #IK = black; RRA = blue; MOCO = red
    for bodyInd in range(ikModel.updBodySet().getSize()):
        
        #Loop through the attached geomtries on body
        for gInd in range(ikModel.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
            
                #Set the colours for the current body and geometry
                ikModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(0,0,0)) #black
                rraModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(0,0,1)) #blue
                mocoModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(1,0,0)) #red
                
                #Opacity
                ikModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4) 
                rraModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                mocoModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
                
    #Set names
    ikModel.setName(f'{subject}_IK')
    rraModel.setName(f'{subject}_RRA')
    mocoModel.setName(f'{subject}_MOCO')
    
    #Finalise model connections
    ikModel.finalizeConnections()
    rraModel.finalizeConnections()
    mocoModel.finalizeConnections()
    
    #Print to file
    ikModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikModel.osim')
    rraModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraModel.osim')
    mocoModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoModel.osim')

# %% Create mean models and kinematic files

#Create coloured mean models based on generic model

#Read in 3 versions of generic model
ikMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
rraMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
mocoMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')

#Delete the forceset in each model to get rid of muscles
ikMeanModel.updForceSet().clearAndDestroy()
rraMeanModel.updForceSet().clearAndDestroy()
mocoMeanModel.updForceSet().clearAndDestroy()

#Delete marker set in model
ikMeanModel.updMarkerSet().clearAndDestroy()
rraMeanModel.updMarkerSet().clearAndDestroy()
mocoMeanModel.updMarkerSet().clearAndDestroy()

#Loop through the bodies and set the colouring
#Also adjust the opacity here
#IK = black; RRA = blue; MOCO = red
for bodyInd in range(ikMeanModel.updBodySet().getSize()):
    
    #Loop through the attached geomtries on body
    for gInd in range(ikMeanModel.updBodySet().get(bodyInd).getPropertyByName('attached_geometry').size()):
    
        #Set the colours for the current body and geometry
        ikMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(0,0,0)) #black
        rraMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(0,0,1)) #blue
        mocoMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setColor(osim.Vec3(1,0,0)) #red
        
        #Opacity
        ikMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4) 
        rraMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
        mocoMeanModel.updBodySet().get(bodyInd).get_attached_geometry(gInd).setOpacity(0.4)
        
#Set names
ikMeanModel.setName('generic_IK')
rraMeanModel.setName('generic_RRA')
mocoMeanModel.setName('generic_MOCO')

#Finalise model connections
ikMeanModel.finalizeConnections()
rraMeanModel.finalizeConnections()
mocoMeanModel.finalizeConnections()

#Print to file
ikMeanModel.printToXML('..\\..\\results\\outputs\\generic_ikModel.osim')
rraMeanModel.printToXML('..\\..\\results\\outputs\\generic_rraModel.osim')
mocoMeanModel.printToXML('..\\..\\results\\outputs\\generic_mocoModel.osim')

#Build a mean time series table with the kinematics from each category

#Create the three tables
ikMeanTable = osim.TimeSeriesTable()
rraMeanTable = osim.TimeSeriesTable()
mocoMeanTable = osim.TimeSeriesTable()

#Set the column labels
ikMeanTable.setColumnLabels(tableLabels)
rraMeanTable.setColumnLabels(tableLabels)
mocoMeanTable.setColumnLabels(tableLabels)

#Create an average time variable
avgMeanTime = np.mean(avgGroupTimes[runLabel]['time'], axis = 0)

#Fill the table with rows by looping through the rows and columns
#By default OpenSim assumes radians, so convert those here (if not a translation)
for iRow in range(len(avgMeanTime)):
    #Create an array that gets each kinematic variable at the current point in time
    ikRowData = np.array([np.mean(ikGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(ikGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
    rraRowData = np.array([np.mean(rraGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(rraGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
    mocoRowData = np.array([np.mean(mocoGroupKinematics[runLabel][var][:,iRow]) if var in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.deg2rad(np.mean(mocoGroupKinematics[runLabel][var][:,iRow])) for var in kinematicVars])
    #Convert each of these to an osim row variable
    ikRow = osim.RowVector.createFromMat(ikRowData)
    rraRow = osim.RowVector.createFromMat(rraRowData)
    mocoRow = osim.RowVector.createFromMat(mocoRowData)
    #Append to the relevant table
    ikMeanTable.appendRow(iRow, ikRow)
    rraMeanTable.appendRow(iRow, rraRow)
    mocoMeanTable.appendRow(iRow, mocoRow)
    #Set the time for current row
    ikMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
    rraMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
    mocoMeanTable.setIndependentValueAtIndex(iRow, avgMeanTime[iRow])
    
#Write to mot file format
osim.STOFileAdapter().write(ikMeanTable, '..\\..\\results\\outputs\\group_ikMeanKinematics.sto')
osim.STOFileAdapter().write(rraMeanTable, '..\\..\\results\\outputs\\group_rraMeanKinematics.sto')
osim.STOFileAdapter().write(mocoMeanTable, '..\\..\\results\\outputs\\group_mocoMeanKinematics.sto')
    
# %% ----- end of 03_analyseSimulations.py ----- %% #