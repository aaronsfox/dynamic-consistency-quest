# The Quest for Dynamic Consistency

This repository holds the data related to the work:

Fox A (2023). The Quest for Dynamic Consistency: A Comparison of OpenSim Tools for Residual Reduction in Simulations of Human Running. *BioRxiv*.

Please cite the above work accordingly if using data or code from this repository.

## code

The `code` folder contains the Python scripts used to process and analyse the experimental data through the various processes tested. It is separated into folders based on the dataset analysed. In the below described Python scripts, the OpenSim install Geometry paths are assumed to be at `C:/OpenSim 4.3/Geometry` and hence if installed at a different location this aspect of the scripts should be edited.

### HamnerDelpDataset

The `runSimulations.py` script in the `HamnerDelpDataset` folder contains all of the necessary code to process and analyse the simulations for this dataset. It contains a series of options depending on what parts of the code wish to be re-run:

- **runRRA** / **runRRA3** / **runMoco** / **runAddBiomech**: Each of these variables can be set to `True` which results in the simulations using the named tool to be re-run. When set to `False`, the existing data in the repository can still be used for subsequent steps.
- **compileData**: When set to `True`, the data from each of the tools processed simulations is compiled and saved to file for easier use later in the script or for creating new scripts using these data. *NOTE:* this flag should be set to `True` if any of the simulations are re-run, so that accurate up-to-date data is used in the script. Within this compiling data section, there is an option for which data to read in and check (by creating relevant figures) by setting the flag associated with that data type to `True` — those being `readAndCheckKinematics`, `readAndCheckKinetics`, and `readAndCheckResiduals`.
- **analyseData**: When set to `True`, the collated data from each individual subject will be used to generate group descriptive data and the figures associated with these data. *NOTE:* this flag should be set to `True` if any of the simulations are re-run, so that accurate up-to-date data is used in the script and in generating figures.

The `osimFunctions.py` script contains assistive 'helper' functions used regularly in the main script for running simulations.

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

## preprint

The `preprint` folder contains the associated *BioRxiv* preprint associated with this work. Please cite this accordingly if using data or code from this repository.

## results

The `results` folder holds the summary group results relative to the dataset examined. `figures` and `outputs` subfolders are included to hold the different summary formats.

## tools

The `tools` folder contains any additional miscellaneous resources required for the code in this repository to work.

