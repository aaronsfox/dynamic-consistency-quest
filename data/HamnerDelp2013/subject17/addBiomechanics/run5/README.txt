*** This data was generated with AddBiomechanics (www.addbiomechanics.org) ***
AddBiomechanics was written by Keenon Werling.

Automatic processing achieved the following marker errors (averaged
over all frames of all trials):

- Avg. Marker RMSE      = 1.84 cm
- Avg. Max Marker Error = 5.74 cm

Automatic processing reduced the residual loads needed for dynamic
consistency to the following magnitudes (averaged over all frames of
all trials):

- Avg. Residual Force  = 9.34 N
- Avg. Residual Torque = 44.91 N-m

Automatic processing found a new model mass to achieve dynamic
consistency:

  - Total mass = 69.68 kg (+1.80% change from original 68.45 kg)

Individual body mass changes:

  - pelvis    mass = 11.52 kg (+7.40% change from original 10.72 kg)
  - femur_r   mass = 9.08 kg (+7.23% change from original 8.47 kg)
  - tibia_r   mass = 3.50 kg (+3.53% change from original 3.38 kg)
  - talus_r   mass = 0.29 kg (+216.73% change from original 0.09 kg)
  - calcn_r   mass = 0.53 kg (-53.29% change from original 1.14 kg)
  - toes_r    mass = 0.00 kg (-98.55% change from original 0.20 kg)
  - femur_l   mass = 9.08 kg (+7.23% change from original 8.47 kg)
  - tibia_l   mass = 3.50 kg (+3.53% change from original 3.38 kg)
  - talus_l   mass = 0.29 kg (+216.73% change from original 0.09 kg)
  - calcn_l   mass = 0.53 kg (-53.29% change from original 1.14 kg)
  - toes_l    mass = 0.00 kg (-98.55% change from original 0.20 kg)
  - torso     mass = 26.25 kg (+7.45% change from original 24.43 kg)
  - humerus_r mass = 1.95 kg (+5.47% change from original 1.85 kg)
  - ulna_r    mass = 0.29 kg (-48.07% change from original 0.55 kg)
  - radius_r  mass = 0.29 kg (-48.07% change from original 0.55 kg)
  - hand_r    mass = 0.03 kg (-92.93% change from original 0.42 kg)
  - humerus_l mass = 1.95 kg (+5.47% change from original 1.85 kg)
  - ulna_l    mass = 0.29 kg (-48.07% change from original 0.55 kg)
  - radius_l  mass = 0.29 kg (-48.07% change from original 0.55 kg)
  - hand_l    mass = 0.03 kg (-92.93% change from original 0.42 kg)

The following trials were processed to perform automatic body scaling,
marker registration, and residual reduction:

trial: Run_5
  - Avg. Marker RMSE      = 1.84 cm
  - Avg. Marker Max Error = 5.74 cm
  - Avg. Residual Force   = 9.34 N
  - Avg. Residual Torque  = 44.91 N-m
  - WARNING: 2 marker(s) with RMSE greater than 4 cm!
  - WARNING: Automatic data processing required modifying TRC data from 2 marker(s)!
  - WARNING: 175 frame(s) with ground reaction force inconsistencies detected!
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

 > opensim-cmd run-tool Run_5_id_setup.xml
           # This will create results on time range (0.025s to 4.708333s) in file ID/Run_5_osim_id.sto


The original unscaled model file is present in:

Models/unscaled_generic.osim

There is also an unscaled model, with markers moved to spots found by
this tool, at:

Models/unscaled_but_with_optimized_markers.osim

If you encounter errors, please submit a post to the AddBiomechanics
user forum on SimTK.org:

   https://simtk.org/projects/addbiomechanics