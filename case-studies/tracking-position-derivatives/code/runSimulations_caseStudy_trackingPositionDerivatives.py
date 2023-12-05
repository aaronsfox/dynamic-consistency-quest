# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:51:22 2023

@author:
    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs through a case study to investigate slightly different
    approaches to using Moco - whereby additional tracking of reference position
    derivatives (i.e. velocities) is added and compared to not doing so. The
    reasoning behind this is to see whether  noisy outputs from Moco are blunted
    by tracking speeds.
    
    This analysis focuses on a randomly selected subject (subject04) with confirmed
    oscillatory joint kinematics and kinetics.
    
    The dataset and analysis approaches originally implemented can be reviewed
    in greater detail in the main README associated with this repository.

"""

# %% Import packages

import opensim as osim
import osimFunctions as helper
import os
import shutil
import numpy as np
import pickle
import time

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

#Settings for whether to re-run the repeat Moco analysis or just to compare
runMocoSimulation = True
compareMocoSimulation = True

#Print out some info/warnings for certain things
if runMocoSimulation:
    print('***** You have selected to re-run the Moco analyses. *****')
    print('***** These analyses take some time, so prepare to be here for a while... *****')

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

#Set the subject to analyse
subject = 'subject04'

#Set run names list
#This is useful if you want to assess further running speeds
runList  = ['run2',
            'run3',
            'run4',
            'run5']

#Set the run label and trial to analyse
runLabel = 'run5'
runName = 'Run_5'

#Set run cycle list
cycleList = ['cycle1',
             'cycle2',
             'cycle3']

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

# %% Set-up for simulations

#Create dictionary to store timing data
mocoRunTimeData = {run: {cyc: {'mocoRunTime': [], 'nIters': [], 'solved': []} for cyc in cycleList} for run in runList}

#Create a directory to store simulation results to
os.makedirs(os.path.join('..','moco_noSpeeds'), exist_ok = True)
os.makedirs(os.path.join('..','moco_withSpeeds'), exist_ok = True)

#Create run trial specific directory as well
#Note this is currently just run5
os.makedirs(os.path.join('..','moco_noSpeeds',runLabel), exist_ok = True) 
os.makedirs(os.path.join('..','moco_withSpeeds',runLabel), exist_ok = True) 

#Load in the subjects gait timing data
with open(os.path.join('..','..','..','data','HamnerDelp2013',subject,'expData','gaitTimes.pkl'), 'rb') as openFile:
    gaitTimings = pickle.load(openFile)    

# %% Run the Moco approach but this time track reference position derivatives
    
#Check whether simulations are to be run
if runMocoSimulation:
    
    #Loop through the two cases
    for case in ['noSpeeds', 'withSpeeds']:
    
        #Change to Moco directory for ease of use with tools
        os.chdir(os.path.join('..',f'moco_{case}',runLabel))
        
        #Copy external load files across as there are issues with using these out of
        #directory with Moco tools
        shutil.copyfile(os.path.join('..','..','..','..','data','HamnerDelp2013',subject,'expData',f'{runName}_grf.xml'),
                        f'{runName}_grf.xml')
        shutil.copyfile(os.path.join('..','..','..','..','data','HamnerDelp2013',subject,'expData',f'{runName}_grf.mot'),
                        f'{runName}_grf.mot')
    
        #Copy across the already extracted coordinates data for the run trial
        shutil.copyfile(os.path.join('..','..','..','..','data','HamnerDelp2013',subject,'moco',f'{runLabel}',f'{runName}_coordinates.sto'),
                        f'{runName}_coordinates.sto')
    
        #Copy the desired model across
        shutil.copyfile(os.path.join('..','..','..','..','data','HamnerDelp2013',subject,'model',f'{subject}_adjusted_scaled.osim'),
                        f'{subject}_adjusted_scaled.osim')
        
        #Add in opensim logger
        osim.Logger.removeFileSink()
        osim.Logger.addFileSink('mocoLog.log')
    
        #Create a generic tracking tool to manipulate for the 3 cycles
        mocoTrack = osim.MocoTrack()
        mocoTrack.setName('mocoResidualReduction')
        
        # Construct a ModelProcessor and set it on the tool.
        modelProcessor = osim.ModelProcessor(f'{subject}_adjusted_scaled.osim')
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
                
        #Set whther or not to to track the position derivatives
        if case == 'noSpeeds':
            mocoTrack.set_track_reference_position_derivatives(False)
        elif case == 'withSpeeds':
            mocoTrack.set_track_reference_position_derivatives(True)
                
        #Set tracking mesh interval time
        mocoTrack.set_mesh_interval(0.01)
        
        #Set the coordinate reference task weights to match RRA
        
        #Create weight set for state tracking
        stateWeights = osim.MocoWeightSet()
        
        #Set a scaling factor for tracking the speeds
        if case == 'noSpeeds':
            speedsTrackingScale = 0
        elif case == 'withSpeeds':
            speedsTrackingScale = 0.01
        
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
                                                            rraTasks[coordName]*speedsTrackingScale))
                
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
            
        #Return to home directory
        os.chdir(homeDir)

# %% TODO: Continue with comparison...






# %% ----- End of runSimulations_caseStudy_trackingPositionDerivatives.py ----- %% #