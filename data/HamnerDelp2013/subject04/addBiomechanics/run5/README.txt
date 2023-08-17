*** This data was generated with AddBiomechanics (www.addbiomechanics.org) ***
AddBiomechanics was written by Keenon Werling.

Automatic processing achieved the following marker errors (averaged
over all frames of all trials):

- Avg. Marker RMSE      = 1.29 cm
- Avg. Max Marker Error = 2.96 cm

Automatic processing reduced the residual loads needed for dynamic
consistency to the following magnitudes (averaged over all frames of
all trials):

- Avg. Residual Force  = 178.58 N
- Avg. Residual Torque = 64.15 N-m

Automatic processing found a new model mass to achieve dynamic
consistency:

  - Total mass = 79.35 kg (-0.81% change from original 80.0 kg)

Individual body mass changes:

  - pelvis    mass = 12.43 kg (-0.81% change from original 12.53 kg)
  - femur_r   mass = 9.82 kg (-0.81% change from original 9.90 kg)
  - tibia_r   mass = 3.91 kg (-0.81% change from original 3.95 kg)
  - talus_r   mass = 0.11 kg (-0.81% change from original 0.11 kg)
  - calcn_r   mass = 1.32 kg (-0.81% change from original 1.33 kg)
  - toes_r    mass = 0.23 kg (-0.81% change from original 0.23 kg)
  - femur_l   mass = 9.82 kg (-0.81% change from original 9.90 kg)
  - tibia_l   mass = 3.91 kg (-0.81% change from original 3.95 kg)
  - talus_l   mass = 0.11 kg (-0.81% change from original 0.11 kg)
  - calcn_l   mass = 1.32 kg (-0.81% change from original 1.33 kg)
  - toes_l    mass = 0.23 kg (-0.81% change from original 0.23 kg)
  - torso     mass = 28.32 kg (-0.81% change from original 28.55 kg)
  - humerus_r mass = 2.15 kg (-0.81% change from original 2.16 kg)
  - ulna_r    mass = 0.64 kg (-0.81% change from original 0.65 kg)
  - radius_r  mass = 0.64 kg (-0.81% change from original 0.65 kg)
  - hand_r    mass = 0.48 kg (-0.81% change from original 0.49 kg)
  - humerus_l mass = 2.15 kg (-0.81% change from original 2.16 kg)
  - ulna_l    mass = 0.64 kg (-0.81% change from original 0.65 kg)
  - radius_l  mass = 0.64 kg (-0.81% change from original 0.65 kg)
  - hand_l    mass = 0.48 kg (-0.81% change from original 0.49 kg)

The following trials were processed to perform automatic body scaling,
marker registration, and residual reduction:

trial: Run_5
  - Avg. Marker RMSE      = 1.29 cm
  - Avg. Marker Max Error = 2.96 cm
  - Avg. Residual Force   = 178.58 N
  - Avg. Residual Torque  = 64.15 N-m
  - WARNING: 82 joints hit joint limits!
  - WARNING: Automatic data processing required modifying TRC data from 2 marker(s)!
  - WARNING: 440 frame(s) with ground reaction force inconsistencies detected!
  --> See IK/Run_5_ik_summary.txt and ID/Run_5_id_summary.txt for more details.


The model file containing optimal body scaling, marker offsets, and
mass parameters is:

Models/final.osim

This tool works by finding optimal scale factors and marker offsets at
the same time. If specified, it also runs a second optimization to
find mass parameters to fit the model dynamics to the ground reaction
force data.

The model containing the optimal body scaling and marker offsets found
prior to the dynamics fitting step is:

Models/optimized_scale_and_markers.osim

If you want to manually edit the marker offsets, you can modify the
<MarkerSet> in "Models/unscaled_but_with_optimized_markers.osim" (by
default this file contains the marker offsets found by the optimizer).
If you want to tweak the Scaling, you can edit
"Models/rescaling_setup.xml". If you change either of these files,
then run (FROM THE "Models" FOLDER, and not including the leading ">
"):

 > opensim-cmd run-tool rescaling_setup.xml
           # This will re-generate Models/optimized_scale_and_markers.osim


You do not need to re-run Inverse Kinematics unless you change
scaling, because the output motion files are already generated for you
as "*_ik.mot" files for each trial, but you are welcome to confirm our
results using OpenSim. To re-run Inverse Kinematics with OpenSim, to
verify the results of AddBiomechanics, you can use the automatically
generated XML configuration files. Here are the command-line commands
you can run (FROM THE "IK" FOLDER, and not including the leading "> ")
to verify IK results for each trial:

 > opensim-cmd run-tool Run_5_ik_setup.xml
           # This will create a results file IK/Run_5_ik_by_opensim.mot


To re-run Inverse Dynamics using OpenSim, you can also use
automatically generated XML configuration files. WARNING: Inverse
Dynamics in OpenSim uses a different time-step definition to the one
used in AddBiomechanics (AddBiomechanics uses semi-implicit Euler,
OpenSim uses splines). This means that your OpenSim inverse dynamics
results WILL NOT MATCH your AddBiomechanics results, and YOU SHOULD
NOT EXPECT THEM TO. The following commands should work (FROM THE "ID"
FOLDER, and not including the leading "> "):

 > opensim-cmd run-tool Run_5_id_setup_segment_0.xml
           # This will create results on time range (0.09s to 0.18s) in file ID/Run_5_osim_segment_0_id.sto
 > opensim-cmd run-tool Run_5_id_setup_segment_1.xml
           # This will create results on time range (1.23s to 1.42s) in file ID/Run_5_osim_segment_1_id.sto
 > opensim-cmd run-tool Run_5_id_setup_segment_2.xml
           # This will create results on time range (3.11s to 3.29s) in file ID/Run_5_osim_segment_2_id.sto
 > opensim-cmd run-tool Run_5_id_setup_segment_3.xml
           # This will create results on time range (4.36s to 4.55s) in file ID/Run_5_osim_segment_3_id.sto
 > opensim-cmd run-tool Run_5_id_setup_segment_4.xml
           # This will create results on time range (4.99s to 5.18s) in file ID/Run_5_osim_segment_4_id.sto
 > opensim-cmd run-tool Run_5_id_setup_segment_5.xml
           # This will create results on time range (5.43s to 5.45s) in file ID/Run_5_osim_segment_5_id.sto


The original unscaled model file is present in:

Models/unscaled_generic.osim

There is also an unscaled model, with markers moved to spots found by
this tool, at:

Models/unscaled_but_with_optimized_markers.osim

If you encounter errors, please submit a post to the AddBiomechanics
user forum on SimTK.org:

   https://simtk.org/projects/addbiomechanics