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
    
    This uses the Hamner & Delp 2013 dataset available at:
        https://simtk.org/projects/nmbl_running
        
    There are four processes included with this script:
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

# %% Set-up

#Add OpenSim geometry path
#Can be helpful with running into any issues around geometry path
#Set this to OpenSim install directory
#### NOTE: change geometry path to OpenSim install directory
geomDir = os.path.join('C:', os.sep, 'OpenSim 4.3', 'Geometry')
osim.ModelVisualizer.addDirToGeometrySearchPaths(geomDir)
print(f'***** OpenSim Geometry installation directory set at {geomDir} *****')
print('***** Please change the geomDir variable in runSimulations.py if incorrect *****')

#Get home path
homeDir = os.getcwd()

#Set subject list
subList = ['subject01',
    'subject02',
    'subject03',
    'subject04',
    'subject08',
    'subject10',
    'subject11',
    'subject17',
    'subject19',
    'subject20']
    
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

# %% Modify the settings in this block to run desired analyses

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

##### TODO:

##### ...collating data flags and packages... #####

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

# %% ----- end of 01_runSimulations.py ----- %% #
