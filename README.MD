# The Quest for Dynamic Consistency

This repository holds the data related to the work:

***[TODO: add paper details]***

## data

The `data` folder holds the original experimental data used in this work. It is separated into folders based on the original dataset.

### HamnerDelp2013

The data in the `HamnerDelp2013` folder comes from a [publicly available dataset](https://simtk.org/projects/nmbl_running) released with the paper:

Hamner SR & Delp SL (2013). Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds. *J Biomech*, 46(4): 780-787. 

The folder also contains a Python script (`extractData.py`) which was used to extract the relevant data from the original downloaded dataset. There are further comments in this script on how to use it, however it can be completely ignored in running the associated code that processes this dataset (i.e. the data has already been extracted and compiled to a useable format with the other code included in this repository).

Each subject from the original dataset has their own folder, which contains the original experimental data alongside the processed outputs from the present work. Specifically, each subject folder contains:

- **addBiomechanics:** The original generic `.osim`, and `.trc` and `.mot` files from the experimental data which were uploaded to the [AddBiomechanics server](https://addbiomechanics.org/) for processing. The processed results were then downloaded and added to this folder.
- **expData:** The original `.trc` and `.mot` files from the experimental data, alongside a Python dictionary object (in `.pkl` format) containing the timings of the gait cycles analysed.
- **ik:** The results from the original inverse kinematics procedures run on the data.
- **moco:** The outputs from the Moco tracking processes run on the experimental data. This contains the residual forces and moments that were generated from the Moco tracking approach.
- **model:** The scaled model based on the original experimental approaches used. This model was used in the Moco and RRA procedures. 
- **results:** Contains subfolders of `figures` which contains some summary figures of the subjects results, and `outputs` which contains the compiled and mean results from the various approaches tested.
- **rra:** The outputs from the residual reduction algorithm (RRA) processes run on the experimental data. This contains the residual forces and moments that were generated from a single iteration of the RRA approach.
- **rra3:** The outputs from the residual reduction algorithm (RRA) processes run on the experimental data. This contains the residual forces and moments that were generated from three iterations of the RRA approach.

## code

The `code` folder contains the Python scripts used to process and analyse the experimental data through the various processes tested. It is separated into folders based on the dataset analysed.

### HamnerDelpDataset

The code in the `HamnerDelpDataset` folder relates to analysis of the data in `HamnerDelp2013` data folder (see comments in above section). The code is separated into scripts associated with different steps which are sequentially ordered:

- **01_runSimulations.py:** Loops through subjects included in the experimental dataset and runs their data through the various processes examined. 
- **02_collateSimulations.py:** Collates each individual subjects results from the various processes examined into a compiled dataset that can be used in subsequent analysis scripts.
- **03_analyseSimulations.py:** Groups all subject data form the various processes examined together to generate summary statistics.
- **osimFunctions:** Contains assistive 'helper' functions used regularly throughout the abovementioned scripts.

