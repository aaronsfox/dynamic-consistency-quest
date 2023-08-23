# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through the process of reading in and analysing the data 
    from the various approaches tested for simulations run on the Hamner & Delp
    2013 data to compare the solutions.

"""

# %% Import packages

import opensim as osim
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

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
kinematicVarsPlot = {'pelvis_tilt': [0,0], 'pelvis_list': [0,1], 'pelvis_rotation': [0,2],
                     'hip_flexion': [1,0], 'hip_adduction': [1,1], 'hip_rotation': [1,2],
                     'knee_angle': [2,0], 'ankle_angle': [2,1],
                     'lumbar_extension': [3,0], 'lumbar_bending': [3,1], 'lumbar_rotation': [3,2],
                     'arm_flex': [4,0], 'arm_add': [4,1], 'arm_rot': [4,2],
                     'elbow_flex': [5,0], 'pro_sup': [5,1]
                     }

#Set a dictionary for plotting kinetic vars and their axes
kineticVarsPlot = {'hip_flexion': [0,0], 'hip_adduction': [0,1], 'hip_rotation': [0,2],
                   'knee_angle': [1,0], 'ankle_angle': [1,1],
                   'lumbar_extension': [2,0], 'lumbar_bending': [2,1], 'lumbar_rotation': [2,2],
                   'arm_flex': [3,0], 'arm_add': [3,1], 'arm_rot': [3,2],
                   'elbow_flex': [4,0], 'pro_sup': [4,1]
                   }

#Set a list for kinematic plot titles
kinematicVarsTitle = ['Pelvis Post. (+) / Ant. (-) Tilt', 'Pelvis Right (+) / Left (-) List', 'Pelvis Left (+) / Right (-) Rot.',
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

#Set colours for plots
ikCol = '#000000' #IK = black
rraCol = '#e569ce' #RRA = purple
rra3Col = '#ff6876' #RRA3 = pink
mocoCol = '#4885ed' #Moco = blue
addBiomechCol = '#ffa600' #AddBiomechanics = gold

#Set HEX as RGB colours (https://www.rapidtables.com/convert/color/hex-to-rgb.html)
#These are only used as osim Vec3 objects so they can be set that way here
ikColRGB = osim.Vec3(0,0,0) #IK = black
rraColRGB = osim.Vec3(0.8980392156862745,0.4117647058823529,0.807843137254902) #RRA = purple
rra3ColRGB = osim.Vec3(1,0.40784313725490196,0.4627450980392157) #RRA3 = pink
mocoColRGB = osim.Vec3(0.2823529411764706,0.5215686274509804,0.9294117647058824) #Moco = blue
addBiomechColRGB = osim.Vec3(1,0.6509803921568628,0) #AddBiomechanics = gold

# %% Extract solution times

#Set a place to store average solution times
solutionTimes = {'rra': np.zeros(len(subList)), 'rra3': np.zeros(len(subList)),
                 'moco': np.zeros(len(subList)), 'addBiomech': np.zeros(len(subList))}

#Loop through subject list
for subject in subList:
    
    #Load in the subjects gait timing data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\gaitTimes.pkl', 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
    
    #Load RRA solution time data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra\\{runLabel}\\{subject}_rraRunTimeData.pkl', 'rb') as openFile:
        rraRunTime = pickle.load(openFile)
        
    #Load RRA3 solution time data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\rra3\\{runLabel}\\{subject}_rra3RunTimeData.pkl', 'rb') as openFile:
        rra3RunTime = pickle.load(openFile)
        
    #Load Moco solution time data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\moco\\{runLabel}\\{subject}_mocoRunTimeData.pkl', 'rb') as openFile:
        mocoRunTime = pickle.load(openFile)
        
    #Extract AddBiomechanics processing time from logs
    
    #Read in the log file
    fid = open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\addBiomechanics\\{runLabel}\\processingLogs.txt', 'r')
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
    addBiomechTime = osim.TimeSeriesTableVec3(f'..\\..\\data\\HamnerDelp2013\\{subject}\\addBiomechanics\\{runLabel}\\{runName}.trc').getIndependentColumn()
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

#Add boxplot
bp = sns.boxplot(data = solutionTimes_df,
                 x = 'Solver', y = 'Time',
                 order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                 palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
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
                   x = 'Solver', y = 'Time',
                   order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                   palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                   size = 5, alpha = 0.5,
                   jitter = True, dodge = False,
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
fig.savefig('..\\..\\results\\HamnerDelpDataset\\figures\\averageSolutionTimes.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

#Export solution times dictionary to file
with open('..\\..\\results\\HamnerDelpDataset\\outputs\\solutionTimes.pkl', 'wb') as writeFile:
    pickle.dump(solutionTimes, writeFile)
    
#Export summary data to csv file
solutionTimes_df.to_csv('..\\..\\results\\HamnerDelpDataset\\outputs\\solutionTimes_summary.csv',
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
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraResiduals.pkl', 'rb') as openFile:
        rraResiduals = pickle.load(openFile)
        
    #Load RRA3 residuals data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3Residuals.pkl', 'rb') as openFile:
        rra3Residuals = pickle.load(openFile)
        
    #Load Moco residuals data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoResiduals.pkl', 'rb') as openFile:
        mocoResiduals = pickle.load(openFile)
        
    #Load AddBiomechanics residuals data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechResiduals.pkl', 'rb') as openFile:
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

#Add boxplot of average residual forces
bp = sns.boxplot(data = avgResidualForces_df,
                 x = 'Axis', y = 'Average Residual Force',
                 order = ['FX', 'FY', 'FZ'],
                 hue = 'Solver',
                 hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                 palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax[0])

#Adjust colours of boxplot lines and fill
for ii in range(len(ax[0].patches)):
    
    #Get the current artist
    artist = ax[0].patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,12):
    
    #Set colouring based on order of boxes
    if ii in [0, 4, 8]:
        col = rraCol
    elif ii in [1, 5, 9]:
        col = rra3Col
    elif ii in [2, 6, 10]:
        col = mocoCol
    elif ii in [3, 7, 11]:
        col = addBiomechCol
        
    #Loop through the 6 lines to recolour
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = avgResidualForces_df,
                   x = 'Axis', y = 'Average Residual Force',
                   order = ['FX', 'FY', 'FZ'],
                   hue = 'Solver',
                   hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                   palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                   size = 5, alpha = 0.5,
                   jitter = True, dodge = True,
                   ax = ax[0])

#Add the average recommended threshold for residuals
ax[0].axhline(y = residualThresholds['F'].mean(), color = 'black',
              linewidth = 1, ls = '--', zorder = 1)

#Turn off legend
ax[0].get_legend().remove()

#Set y-axes limits
ax[0].set_ylim([0.0,ax[0].get_ylim()[1]])

#Remove x-label
ax[0].set_xlabel('')

#Set y-label
ax[0].set_ylabel('Average Residual Force (N)', fontsize = 14, labelpad = 10)

#Set x-labels
ax[0].set_xticklabels(['FX', 'FY', 'FZ'], fontsize = 12, rotation = 45, ha = 'right')

#Despine top and right axes
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

#Add boxplot of peak residual forces
bp = sns.boxplot(data = peakResidualForces_df,
                 x = 'Axis', y = 'Peak Residual Force',
                 order = ['FX', 'FY', 'FZ'],
                 hue = 'Solver',
                 hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                 palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax[1])

#Adjust colours of boxplot lines and fill
for ii in range(len(ax[1].patches)):
    
    #Get the current artist
    artist = ax[1].patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,12):
    
    #Set colouring based on order of boxes
    if ii in [0, 4, 8]:
        col = rraCol
    elif ii in [1, 5, 9]:
        col = rra3Col
    elif ii in [2, 6, 10]:
        col = mocoCol
    elif ii in [3, 7, 11]:
        col = addBiomechCol
        
    #Loop through the 6 lines to recolour
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = peakResidualForces_df,
                   x = 'Axis', y = 'Peak Residual Force',
                   order = ['FX', 'FY', 'FZ'],
                   hue = 'Solver',
                   hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                   palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                   size = 5, alpha = 0.5,
                   jitter = True, dodge = True,
                   ax = ax[1])

#Add the average recommended threshold for residuals
ax[1].axhline(y = residualThresholds['F'].mean(), color = 'black',
              linewidth = 1, ls = '--', zorder = 1)

#Turn off legend
ax[1].get_legend().remove()

#Set y-axes limits so that peak values get included
ax[1].set_ylim([0.0,
                peakResidualForces_df['Peak Residual Force'].max() * 1.05])

#Remove x-label
ax[1].set_xlabel('')

#Set y-label
ax[1].set_ylabel('Peak Residual Force (N)', fontsize = 14, labelpad = 10)

#Set x-labels
ax[1].set_xticklabels(['FX', 'FY', 'FZ'], fontsize = 12, rotation = 45, ha = 'right')

#Despine top and right axes
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig('..\\..\\results\\HamnerDelpDataset\\figures\\residualForces.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

#Create figure for residual moments
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5), sharey = True)

#Add boxplot of average residual forces
bp = sns.boxplot(data = avgResidualMoments_df,
                 x = 'Axis', y = 'Average Residual Moment',
                 order = ['MX', 'MY', 'MZ'],
                 hue = 'Solver',
                 hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                 palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax[0])

#Adjust colours of boxplot lines and fill
for ii in range(len(ax[0].patches)):
    
    #Get the current artist
    artist = ax[0].patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,12):
    
    #Set colouring based on order of boxes
    if ii in [0, 4, 8]:
        col = rraCol
    elif ii in [1, 5, 9]:
        col = rra3Col
    elif ii in [2, 6, 10]:
        col = mocoCol
    elif ii in [3, 7, 11]:
        col = addBiomechCol
        
    #Loop through the 6 lines to recolour
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = avgResidualMoments_df,
                   x = 'Axis', y = 'Average Residual Moment',
                   order = ['MX', 'MY', 'MZ'],
                   hue = 'Solver',
                   hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                   palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                   size = 5, alpha = 0.5,
                   jitter = True, dodge = True,
                   ax = ax[0])

#Add the average recommended threshold for residuals
ax[0].axhline(y = residualThresholds['M'].mean(), color = 'black',
              linewidth = 1, ls = '--', zorder = 1)

#Turn off legend
ax[0].get_legend().remove()

#Set y-axes limits
ax[0].set_ylim([0.0,ax[0].get_ylim()[1]])

#Remove x-label
ax[0].set_xlabel('')

#Set y-label
ax[0].set_ylabel('Average Residual Moment (Nm)', fontsize = 14, labelpad = 10)

#Set x-labels
ax[0].set_xticklabels(['MX', 'MY', 'MZ'], fontsize = 12, rotation = 45, ha = 'right')

#Despine top and right axes
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

#Add boxplot of peak residual moments
bp = sns.boxplot(data = peakResidualMoments_df,
                 x = 'Axis', y = 'Peak Residual Moment',
                 order = ['MX', 'MY', 'MZ'],
                 hue = 'Solver',
                 hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                 palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                 dodge = 0.4, width = 0.5, whis = [0,100],
                 ax = ax[1])

#Adjust colours of boxplot lines and fill
for ii in range(len(ax[1].patches)):
    
    #Get the current artist
    artist = ax[1].patches[ii]
    
    #Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

#Loop through lines and set
#Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
for ii in range(0,12):
    
    #Set colouring based on order of boxes
    if ii in [0, 4, 8]:
        col = rraCol
    elif ii in [1, 5, 9]:
        col = rra3Col
    elif ii in [2, 6, 10]:
        col = mocoCol
    elif ii in [3, 7, 11]:
        col = addBiomechCol
        
    #Loop through the 6 lines to recolour
    for jj in range(ii*6,ii*6+6):
        line = bp.lines[jj]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

#Add the strip plot for points
sp = sns.stripplot(data = peakResidualMoments_df,
                   x = 'Axis', y = 'Peak Residual Moment',
                   order = ['MX', 'MY', 'MZ'],
                   hue = 'Solver',
                   hue_order = ['RRA', 'RRA3', 'Moco', 'AddBiomechanics'],
                   palette = [rraCol, rra3Col, mocoCol, addBiomechCol],
                   size = 5, alpha = 0.5,
                   jitter = True, dodge = True,
                   ax = ax[1])

#Add the average recommended threshold for residuals
ax[1].axhline(y = residualThresholds['M'].mean(), color = 'black',
              linewidth = 1, ls = '--', zorder = 1)

#Turn off legend
ax[1].get_legend().remove()

#Set y-axes limits so that peak values get included
ax[1].set_ylim([0.0,
                peakResidualMoments_df['Peak Residual Moment'].max() * 1.05])

#Remove x-label
ax[1].set_xlabel('')

#Set y-label
ax[1].set_ylabel('Peak Residual Moment (Nm)', fontsize = 14, labelpad = 10)

#Set x-labels
ax[1].set_xticklabels(['MX', 'MY', 'MZ'], fontsize = 12, rotation = 45, ha = 'right')

#Despine top and right axes
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

#Tight layout
plt.tight_layout()

#Save figure
fig.savefig('..\\..\\results\\HamnerDelpDataset\\figures\\residualMoments.png',
            format = 'png', dpi = 300)

#Close figure
plt.close()

#Export residual summary dataframes to file
avgResidualForces_df.to_csv('..\\..\\results\\HamnerDelpDataset\\outputs\\avgResidualForces.csv', index = False)
avgResidualMoments_df.to_csv('..\\..\\results\\HamnerDelpDataset\\outputs\\avgResidualMoments.csv', index = False)
peakResidualForces_df.to_csv('..\\..\\results\\HamnerDelpDataset\\outputs\\peakResidualForces.csv', index = False)
peakResidualMoments_df.to_csv('..\\..\\results\\HamnerDelpDataset\\outputs\\peakResidualMoments.csv', index = False)

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
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikKinematicsRMSE.pkl', 'rb') as openFile:
        rmseData['IK'] = pickle.load(openFile)
    
    #Load RRA RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraKinematicsRMSE.pkl', 'rb') as openFile:
        rmseData['RRA'] = pickle.load(openFile)
        
    #Load RRA3 RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3KinematicsRMSE.pkl', 'rb') as openFile:
        rmseData['RRA3'] = pickle.load(openFile)
    
    #Load Moco RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoKinematicsRMSE.pkl', 'rb') as openFile:
        rmseData['Moco'] = pickle.load(openFile)
        
    #Load AddBiomechanics RMSD data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechKinematicsRMSE.pkl', 'rb') as openFile:
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
with open('..\\..\\results\\HamnerDelpDataset\\outputs\\kinematicsRMSD.pkl', 'wb') as writeFile:
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
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.pkl', 'rb') as openFile:
        ikMeanKinematics = pickle.load(openFile)
        
    #Read in RRA kinematic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.pkl', 'rb') as openFile:
        rraMeanKinematics = pickle.load(openFile)
    
    #Read in RRA3 kinematic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinematics.pkl', 'rb') as openFile:
        rra3MeanKinematics = pickle.load(openFile)
        
    #Read in Moco kinematic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.pkl', 'rb') as openFile:
        mocoMeanKinematics = pickle.load(openFile)
        
    #Read in AddBiomechanics kinematic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinematics.pkl', 'rb') as openFile:
        addBiomechMeanKinematics = pickle.load(openFile)
        
    #Loop through and extract kinematic data
    for var in kinematicVars:
        meanKinematics['ik'][var][subList.index(subject),:] = ikMeanKinematics[runLabel][var]
        meanKinematics['rra'][var][subList.index(subject),:] = rraMeanKinematics[runLabel][var]
        meanKinematics['rra3'][var][subList.index(subject),:] = rra3MeanKinematics[runLabel][var]
        meanKinematics['moco'][var][subList.index(subject),:] = mocoMeanKinematics[runLabel][var]
        meanKinematics['addBiomech'][var][subList.index(subject),:] = addBiomechMeanKinematics[runLabel][var]
        
#Create figure of group kinematics across the different approaches
#Note that generic kinematic variablea are used here and right side values are presented

#Create the figure
fig, ax = plt.subplots(nrows = 6, ncols = 3, figsize = (8,11), sharex = True)

#Adjust subplots
plt.subplots_adjust(left = 0.075, right = 0.95, bottom = 0.05, top = 0.95,
                    hspace = 0.4, wspace = 0.5)

#Loop through variables and plot data
for var in kinematicVarsPlot.keys():
    
    #Set the appropriate axis
    plt.sca(ax[kinematicVarsPlot[var][0],kinematicVarsPlot[var][1]])
    
    #Set the plotting variable based on whether it is a general or side variable
    if 'pelvis' in var or 'lumbar' in var:
        plotVar = str(var)
    else:
        plotVar = var+'_r'
            
    #Plot mean and SD curves
    
    #IK mean
    plt.plot(np.linspace(0,100,101), meanKinematics['ik'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = ikCol, alpha = 1.0, zorder = 3)
    #IK sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinematics['ik'][plotVar].mean(axis = 0) + meanKinematics['ik'][plotVar].std(axis = 0),
                     meanKinematics['ik'][plotVar].mean(axis = 0) - meanKinematics['ik'][plotVar].std(axis = 0),
                     color = ikCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #RRA mean
    plt.plot(np.linspace(0,100,101), meanKinematics['rra'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
    #RRA sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinematics['rra'][plotVar].mean(axis = 0) + meanKinematics['rra'][plotVar].std(axis = 0),
                     meanKinematics['rra'][plotVar].mean(axis = 0) - meanKinematics['rra'][plotVar].std(axis = 0),
                     color = rraCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #RRA3 mean
    plt.plot(np.linspace(0,100,101), meanKinematics['rra3'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
    #RRA3 sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinematics['rra3'][plotVar].mean(axis = 0) + meanKinematics['rra3'][plotVar].std(axis = 0),
                     meanKinematics['rra3'][plotVar].mean(axis = 0) - meanKinematics['rra3'][plotVar].std(axis = 0),
                     color = rra3Col, alpha = 0.1, zorder = 2, lw = 0)
    
    #Moco mean
    plt.plot(np.linspace(0,100,101), meanKinematics['moco'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
    #Moco sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinematics['moco'][plotVar].mean(axis = 0) + meanKinematics['moco'][plotVar].std(axis = 0),
                     meanKinematics['moco'][plotVar].mean(axis = 0) - meanKinematics['moco'][plotVar].std(axis = 0),
                     color = mocoCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #AddBiomechanics mean
    plt.plot(np.linspace(0,100,101), meanKinematics['addBiomech'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
    #AddBiomechanics sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinematics['addBiomech'][plotVar].mean(axis = 0) + meanKinematics['addBiomech'][plotVar].std(axis = 0),
                     meanKinematics['addBiomech'][plotVar].mean(axis = 0) - meanKinematics['addBiomech'][plotVar].std(axis = 0),
                     color = addBiomechCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #Clean up axis properties
    
    #Set x-limits
    plt.gca().set_xlim([0,100])
    
    #Add labels
    
    #X-axis (if bottom row)
    if list(kinematicVarsPlot.keys()).index(var) >= 14:
        plt.gca().set_xlabel('0-100% Gait Cycle', fontsize = 10, fontweight = 'bold')
        
    #Y-axis
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
    
#Turn off un-used axes
ax[2,2].axis('off')
ax[5,2].axis('off')

#Save figure
fig.savefig('..\\..\\results\\HamnerDelpDataset\\figures\\meanKinematics.png',
            format = 'png', dpi = 300)

#Close figure
plt.close('all')

#Export mean kinematics dictionary to file
with open('..\\..\\results\\HamnerDelpDataset\\outputs\\meanKinematics.pkl', 'wb') as writeFile:
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
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinetics.pkl', 'rb') as openFile:
        rraMeanKinetics = pickle.load(openFile)
    
    #Read in RRA3 kinetic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinetics.pkl', 'rb') as openFile:
        rra3MeanKinetics = pickle.load(openFile)
        
    #Read in Moco kinetic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinetics.pkl', 'rb') as openFile:
        mocoMeanKinetics = pickle.load(openFile)
        
    #Read in AddBiomechanics kinetic data
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinetics.pkl', 'rb') as openFile:
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
fig, ax = plt.subplots(nrows = 5, ncols = 3, figsize = (8,10), sharex = True)

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
             ls = '-', lw = 1.5, c = rraCol, alpha = 1.0, zorder = 3)
    #RRA sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinetics['rra'][plotVar].mean(axis = 0) + meanKinetics['rra'][plotVar].std(axis = 0),
                     meanKinetics['rra'][plotVar].mean(axis = 0) - meanKinetics['rra'][plotVar].std(axis = 0),
                     color = rraCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #RRA3 mean
    plt.plot(np.linspace(0,100,101), meanKinetics['rra3'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = rra3Col, alpha = 1.0, zorder = 3)
    #RRA3 sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinetics['rra3'][plotVar].mean(axis = 0) + meanKinetics['rra3'][plotVar].std(axis = 0),
                     meanKinetics['rra3'][plotVar].mean(axis = 0) - meanKinetics['rra3'][plotVar].std(axis = 0),
                     color = rra3Col, alpha = 0.1, zorder = 2, lw = 0)
    
    #Moco mean
    plt.plot(np.linspace(0,100,101), meanKinetics['moco'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = mocoCol, alpha = 1.0, zorder = 3)
    #Moco sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinetics['moco'][plotVar].mean(axis = 0) + meanKinetics['moco'][plotVar].std(axis = 0),
                     meanKinetics['moco'][plotVar].mean(axis = 0) - meanKinetics['moco'][plotVar].std(axis = 0),
                     color = mocoCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #AddBiomechanics mean
    plt.plot(np.linspace(0,100,101), meanKinetics['addBiomech'][plotVar].mean(axis = 0),
             ls = '-', lw = 1.5, c = addBiomechCol, alpha = 1.0, zorder = 3)
    #AddBiomechanics sd
    plt.fill_between(np.linspace(0,100,101),
                     meanKinetics['addBiomech'][plotVar].mean(axis = 0) + meanKinetics['addBiomech'][plotVar].std(axis = 0),
                     meanKinetics['addBiomech'][plotVar].mean(axis = 0) - meanKinetics['addBiomech'][plotVar].std(axis = 0),
                     color = addBiomechCol, alpha = 0.1, zorder = 2, lw = 0)
    
    #Clean up axis properties
    
    #Set x-limits
    plt.gca().set_xlim([0,100])
    
    #Add labels
    
    #X-axis (if bottom row)
    if list(kinematicVarsPlot.keys()).index(var) >= 14:
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
    
#Turn off un-used axes
ax[1,2].axis('off')
ax[4,2].axis('off')
    
#Save figure
fig.savefig('..\\..\\results\\HamnerDelpDataset\\figures\\meanKinetics.png',
            format = 'png', dpi = 300)

#Close figure
plt.close('all')

#Export mean kinematics dictionary to file
with open('..\\..\\results\\HamnerDelpDataset\\outputs\\meanKinetics.pkl', 'wb') as writeFile:
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
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\expData\\gaitTimes.pkl', 'rb') as openFile:
        gaitTimings = pickle.load(openFile)
    
    #Read in the kinematic data
    
    #IK
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.pkl', 'rb') as openFile:
        ikKinematics = pickle.load(openFile)
    #RRA
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.pkl', 'rb') as openFile:
        rraKinematics = pickle.load(openFile)
    #RRA3
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinematics.pkl', 'rb') as openFile:
        rra3Kinematics = pickle.load(openFile)
    #Moco
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.pkl', 'rb') as openFile:
        mocoKinematics = pickle.load(openFile)
    #AddBiomechanics
    with open(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinematics.pkl', 'rb') as openFile:
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
    osim.STOFileAdapter().write(ikTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikMeanKinematics.sto')
    osim.STOFileAdapter().write(rraTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraMeanKinematics.sto')
    osim.STOFileAdapter().write(rra3Table, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3MeanKinematics.sto')
    osim.STOFileAdapter().write(mocoTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoMeanKinematics.sto')
    osim.STOFileAdapter().write(addBiomechTable, f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechMeanKinematics.sto')
    
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
    
    #Read in three versions of the subject model
    ikModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    rraModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    rra3Model = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    mocoModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    addBiomechModel = osim.Model(f'..\\..\\data\\HamnerDelp2013\\{subject}\\model\\{subject}_adjusted_scaled.osim')
    
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
    ikModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_ikModel.osim')
    rraModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rraModel.osim')
    rra3Model.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_rra3Model.osim')
    mocoModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_mocoModel.osim')
    addBiomechModel.printToXML(f'..\\..\\data\\HamnerDelp2013\\{subject}\\results\\outputs\\{subject}_addBiomechModel.osim')

# %% Create mean models and kinematic files

#Create coloured mean models based on generic model

#Read in 3 versions of generic model
ikMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
rraMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
rra3MeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
mocoMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')
addBiomechMeanModel = osim.Model('..\\..\\data\\HamnerDelp2013\\subject01\\model\\genericModel.osim')

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
ikMeanModel.printToXML('..\\..\\results\\HamnerDelpDataset\\outputs\\generic_ikModel.osim')
rraMeanModel.printToXML('..\\..\\results\\HamnerDelpDataset\\outputs\\generic_rraModel.osim')
rra3MeanModel.printToXML('..\\..\\results\\HamnerDelpDataset\\outputs\\generic_rra3Model.osim')
mocoMeanModel.printToXML('..\\..\\results\\HamnerDelpDataset\\outputs\\generic_mocoModel.osim')
addBiomechMeanModel.printToXML('..\\..\\results\\HamnerDelpDataset\\outputs\\generic_addBiomechModel.osim')

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
osim.STOFileAdapter().write(ikMeanTable, '..\\..\\results\\HamnerDelpDataset\\outputs\\group_ikMeanKinematics.sto')
osim.STOFileAdapter().write(rraMeanTable, '..\\..\\results\\HamnerDelpDataset\\outputs\\group_rraMeanKinematics.sto')
osim.STOFileAdapter().write(rra3MeanTable, '..\\..\\results\\HamnerDelpDataset\\outputs\\group_rra3MeanKinematics.sto')
osim.STOFileAdapter().write(mocoMeanTable, '..\\..\\results\\HamnerDelpDataset\\outputs\\group_mocoMeanKinematics.sto')
osim.STOFileAdapter().write(addBiomechMeanTable, '..\\..\\results\\HamnerDelpDataset\\outputs\\group_addBiomechMeanKinematics.sto')
    
# %% ----- end of 03_analyseSimulations.py ----- %% #