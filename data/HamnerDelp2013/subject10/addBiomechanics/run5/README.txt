*** This data was generated with AddBiomechanics (www.addbiomechanics.org) ***
AddBiomechanics was written by Keenon Werling.

Automatic processing achieved the following marker errors (averaged
over all frames of all trials):

- Avg. Marker RMSE      = 1.58 cm
- Avg. Max Marker Error = 3.84 cm

Automatic processing reduced the residual loads needed for dynamic
consistency to the following magnitudes (averaged over all frames of
all trials):

- Avg. Residual Force  = 4.36 N
- Avg. Residual Torque = 61.66 N-m

Automatic processing found a new model mass to achieve dynamic
consistency:

  - Total mass = 71.50 kg (+3.23% change from original 69.26 kg)

Individual body mass changes:

  - pelvis    mass = 12.04 kg (+10.99% change from original 10.85 kg)
  - femur_r   mass = 9.05 kg (+5.61% change from original 8.57 kg)
  - tibia_r   mass = 3.07 kg (-10.19% change from original 3.42 kg)
  - talus_r   mass = 0.30 kg (+220.25% change from original 0.09 kg)
  - calcn_r   mass = 0.98 kg (-14.85% change from original 1.15 kg)
  - toes_r    mass = 0.02 kg (-87.61% change from original 0.20 kg)
  - femur_l   mass = 9.05 kg (+5.61% change from original 8.57 kg)
  - tibia_l   mass = 3.07 kg (-10.19% change from original 3.42 kg)
  - talus_l   mass = 0.30 kg (+220.25% change from original 0.09 kg)
  - calcn_l   mass = 0.98 kg (-14.85% change from original 1.15 kg)
  - toes_l    mass = 0.02 kg (-87.61% change from original 0.20 kg)
  - torso     mass = 26.49 kg (+7.18% change from original 24.72 kg)
  - humerus_r mass = 2.23 kg (+18.99% change from original 1.87 kg)
  - ulna_r    mass = 0.26 kg (-53.49% change from original 0.56 kg)
  - radius_r  mass = 0.26 kg (-53.49% change from original 0.56 kg)
  - hand_r    mass = 0.31 kg (-26.44% change from original 0.42 kg)
  - humerus_l mass = 2.23 kg (+18.99% change from original 1.87 kg)
  - ulna_l    mass = 0.26 kg (-53.49% change from original 0.56 kg)
  - radius_l  mass = 0.26 kg (-53.49% change from original 0.56 kg)
  - hand_l    mass = 0.31 kg (-26.44% change from original 0.42 kg)

The following trials were processed to perform automatic body scaling,
marker registration, and residual reduction:

trial: Run_5
  - Avg. Marker RMSE      = 1.58 cm
  - Avg. Marker Max Error = 3.84 cm
  - Avg. Residual Force   = 4.36 N
  - Avg. Residual Torque  = 61.66 N-m
  - WARNING: Automatic data processing required modifying TRC data from 2 marker(s)!
  - WARNING: 328 frame(s) with ground reaction force inconsistencies detected!
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
           # This will create results on time range (0.04s to 5.45s) in file ID/Run_5_osim_id.sto


The original unscaled model file is present in:

Models/unscaled_generic.osim

There is also an unscaled model, with markers moved to spots found by
this tool, at:

Models/unscaled_but_with_optimized_markers.osim

If you encounter errors, please submit a post to the AddBiomechanics
user forum on SimTK.org:

   https://simtk.org/projects/addbiomechanics