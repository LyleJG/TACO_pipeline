# Welcome to TACO pipeline!

This python-based pipeline has been designed for the treatment and analysis of current-clamp recordings using long-currents steps protocols, in order to extract biophysical properties coherently across databases while minimizing experimentally induced variability.

## How does it work?
To ensure a coherent and database-independent analysis while minimizing the need for users to adapt their databases, the TACO pipeline extracts raw traces using a user-provided script and associated database information. These elements are specified in a `config_json_file`, which the pipeline helps the user generate (see How to use the TACO pipeline). The script and configuration are then automatically integrated into the pipeline, and the data proceed through the different analysis steps.  
For each cell in the database, the pipeline produces an `.h5` file containing the analysis results. These files can also be loaded back into the TACO pipeline for visual inspection.

## Defining some common terms
Before digging into the details of the pipeline functioning, it is important to start by giving a set of definitions used in the context of the pipeline and referring to different aspect of the current-clamp experiment.  
Notably we define a “trace” as an array of recorded values representing a unique modality of the experiment (i.e. voltage trace, or current trace).  
A “sweep”, is a set of corresponding voltage and current traces for a given cell. A sweep is referred to by a single identifier “sweep id” which is unique for a given cell but can be used for different cell.  
A “protocol” is constituted by a train of sweeps grouped together as they are done one after the other and have unique level of stimulation.  
Finally, an “experiment” refers to the ensemble of the protocols that have been performed on the cell.

## Pre-requisites
Before using the pipeline, it is highly recommended to create a separated python environment (https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)  
This environment must contain the following python package to enable the proper functioning of the pipeline:
- pandas
- numpy
- matplotlib
- scipy
- lmfit
- tqdm
- importlib
- json
- ast
- traceback
- concurrent
- shiny
- shiny widget
- scikit-learn
- plotly
- h5py
- re
- inspect
- plotnine
- anywidget

To help the user to configure the environment, it is possible to create it from the `TACO_pipeline_env.yml` file here provided. By following the instructions indicated here (https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) the user can automatically configure a fully functioning environment for the TACO pipeline.  
Once the pipeline is created, do not forget to activate the environment before using the TACO pipeline.

The pipeline is composed of 8 different scripts:
- `TACO_pipeline_App.py`
- `Sweep_QC_analysis.py`
- `Analysis_pipeline.py`
- `Ordinary_functions.py`
- `Sweep_analysis.py`
- `Spike_analysis.py`
- `Firing_analysis.py`
- `globals_module.py`

All scripts must be installed in a same directory.

# How to prepare a database to be used in the TACO pipeline?
To be analysed in the TACO pipeline, a database must be documented (apart from the raw data) with some specific information, organized in 3 different components. As each database is created and stored in different ways, these database's descriptors are crucial to ensure their successfull processing by the TACO pipeline. We'll explain these different components, their utility, and give example for concrete databases:
## Quick description of example databases:
**(1) Allen Cell Type Database (Gouwens et al, 2019)**
The different cells can be accessed using the allen software development kit, and CellTypeCache. After dowloading the specimen files of interest, a manifest json file is created to access the data through the allen SDK (see https://allensdk.readthedocs.io/en/latest/cell_types.html#cell-types-cache). The folder is then organized as follows:
```
\Allen_CTB_folder
    manifest.json
    \specimen_565871768
        ephys.nwb
        ephys_sweeps.json
    \specimen_605889373
        ephys.nwb
        ephys_sweeps.json     
```
For each specimen (=Cell_id), the different sweeps of the experiment (and their characteristics) can be accessed using the CellTypeCache (see https://allensdk.readthedocs.io/en/latest/cell_types.html#feature-extraction). Each sweep therefore has information about the stimulus start and end times, stimulus amplitude, ... accessible in the files.
    
**(2) Da Silva Lantyer Database (Da Silva Lantyer et al., 2018)**
The raw traces are stored as matlab files (one file per cell). The folder can then be organized as follows:
```
\Lantyer_Data
    \Original_matlab_file
        \161214_AL_113_CC.mat
        \170130_AL_133_CC.mat
```
For each cell, the file contains the input current and membrane potential traces for the different sweeps. Each trace is named : Trace_a_b_c_d (with a=cell identifier, b=Protocol id, c= trace id and d =1 for input current and d= 2 for membrane potential)

**(3) NVC database**

The host lab database is composed of protocol-based binary files (.ivb), organized by animal and cell recorded such that :
```
\NVC_database
    \2016.03.28
        \cell 3
            \G_clamp
                28175818.ivb
                28180420.ivb
```
Such that animal folder is "2016.03.28", in which "cell 3" has been recorded, for which 2 protocols  have been recorded (i.e.: 28175818.ivb and 28180420.ivb)
Each ivb file contains a header containing information about the stimulus amplitude, the increment of stimulus amplitude between sweeps, sampling rate... followed by the raw membrane potential and input current traces.

For each database to be analyzed, 3 different files must be prepared:

### 1- Population class table (.csv)
***** LG Rename to "Cell Table"?
***** LG When would there be more than one "Database" for a given Population class table?
***** LG What is the first column for, e.g. 
	Cell_id	General_area	Sub_area
42	170811NC97_4	SS	Unknown


This CSV file specifies and describes each cell present in the database. The only required information in this file is: 
- Cell_id
- Database
Apart from the cell_id (which should be unique across all databases) and the name of the database it belongs to, no information is mandatory.
Therefore, the file should have as many rows than there are cells in the database, and at least 2 columns (named *Cell_id* and *Database*)

However, the more details can be attributed to each cell, regarding cellular information (e.g.: cell type, custom classification, cortical area, . . . ) or recording conditions (e.g.: age of the animal, recording temperature. . . ), the more details can be used in further analysis with other databases.

### 2- Cell-Sweep table (.csv)
***** LG Rename to "Sweep Table"?

This csv file describes, for each cell present in the database, the sweep to consider for the analysis. This information mainly concerns the organization of the database. The idea is that in any lab, any experiment performed on a cell is fundamentally a collection of sweeps. Most of the time, databases of current-clamp recordings organize their data so that different sweeps can easily be accessed, notably by indexing them separately by giving them a unique sweep id. However, it should be noted that some databases may not directly provide such information directly but rather provide method to understand the storage of the experiment, which results in the a-posteriori definition of sweep ids. Also, some databases may store in a same file, multiple kind of recordings. This is notably the case for the Allen Cell Type Database which recorded for each cell different protocols of current-clamp experiments (i.e.: long square stimulus, short square stimulus, noise…). By specifying this information for any cell of the database, it allows to ensure that only the same kind of protocols are considered, to keep track of the sweeps present in an experiment and organize the analysis in a trace-based fashion.
The only required information in this file is:
- Cell_id
- Sweep_id
Apart from the cell_id (which should be unique across all databases, and match those present in the "Population class table") and the sweep_id to consider, no information is mandatory.
Therefore, the file should have as many rows than there are sweeps to use in the database, and at least 2 columns (named "Cell_id" and "Sweep_id")
Other sweep-related information that can be useful to access the traces (e.g.: stimulus amplitude, stimulus start and end times, ...) can be stored in other columns

***** LG What is the first column for, e.g. 

,Cell_id,Sweep_id,Original_file
3197,161214AL113_3,CC_2_10,161214_AL_113_CC
3198,161214AL113_3,CC_2_1,161214_AL_113_CC
3199,161214AL113_3,CC_2_2,161214_AL_113_CC
3200,161214AL113_3,CC_2_3,161214_AL_113_CC
3201,161214AL113_3,CC_2_4,161214_AL_113_CC



### 3- Trace extraction Python script (.py)
The last required element for the integration of a database, is a Python code defining a function, which given a) a path toward a cell file, b) the cell_id, c) the list of sweep_id of interest, and d) the database's cell sweep table; and returns at least the lists of the corresponding voltage, current and time traces; and if applicable the lists of stimulus start and end times (if it doesn’t, the pipeline will automatically estimate the stimulus start and end times by performing autocorrelation on the current trace first time derivative, based on the stimulus duration provided by the database).

This requirement is at the core of gathering different databases together, as it is the only portion of the pipeline relative to the structure of the original databases. The purpose of this function is to describe how to access the raw data for a given database, without any need to manually integrate it into the pipeline. Indeed, the pipeline automatically import the function, provide it with relevant inputs, and receives the lists corresponding traces which will undergo the analysis. 

For each of these databases, the first step consists in gathering for each cell, the id of the sweep of interest, so that the combination of cell_id+sweep_id enables to directly access the appropriate file, and extract the corresponding traces.

## How to design the files?
Here are some guidelines to help you create the database's "*Population Class table*", "*Cell Sweep table*" and "*Python script*". 
- Start by gathering the cell_id of the cells you want to analyze. Create a CSV table with columns "*Cell_id*" and "*Database*". This will correspond to the "*Population Class table*". You can add other columns that may be useful in subsequent analysis (e.g., "*Recording temperature*", "*Animal*", "*Cortical area*", "*Cortical layer*"...)
- For each cell, gather the sweep_id you want to analyze. You can design the sweep_id as you like (e.g.: ivb_2344_3; 54; Sweep_01,...). This will serve as a basis to create the "Cell Sweep table". You can add other columns that may be useful to access the traces of a specific sweep, or information about stimulus start and end times...
- To design the *Python script*, keep in mind that the purpose of the function is **at least** to extract for the different sweeps, the time traces, the membrane potential traces and the input current traces. Remember that the function takes as an input the database's *Cell sweep table*, notably to access the list of sweep_id. Thus, the function can benefit from the way you constructed the sweep_id to get to the correct file or trace location.

## Be careful when designing these files:
- The cell_id indicated in the "Population Class table" and "Cell Sweep table" **must be the same**, and **unique** across the different databases
- the database's name indicated in the "Population Class table" **must be the same** than the name of the database indicated in the config json file (see later: (1) Preparing the analysis)
- Be careful that for a specific cell, each sweep_id is **unique**
- Be careful of the names of the mandatory columns used in the "Population Class table" and the "Cell Sweep table". These should be "Cell_id" and "Database" for "Population Class table" and "Cell_id" and "Sweep_id" for the "Cell Sweep table".
- When writing the database's specific python script, if any library is required it should be imported at the top of the file.
- Be careful that the order of database's function **inputs** follows the order : cell_file_folder, cell_id, sweep_list, cell_sweep_table
- Be careful that the order of database's function **outputs** follows the order : time_trace_list, potential_trace_list, current_trace_list
- In the case where the database's files contain stimulus start and end files, be careful that the order of database's function **outputs** follows the order : time_trace_list, potential_trace_list, current_trace_list

## Example files for Allen CTD, Da Silva Lantyer Database, and NVC Database
You can find in the folder *Example databases files* example files for each of the database we described earlier.


# How to use the TACO pipeline
The pipeline is used through a Shiny-based GUI.  
To start the application, the user can follow the steps:
1. Open the Terminal
2. Start the environment with appropriate python packages
3. Go to the folder in which the different python scripts are installed
4. Type :  
   ```bash
   shiny run --reload --launch-browser TACO_pipeline_App.py
5. The application will automatically open in the web browser 

## Application panels

The app is composed of three main panels which help the user **prepare (1), run (2), and investigate (3) the analysis**:

### (1) Preparing the analysis

The TACO pipeline operates based on information specified in a **JSON configuration file**. To prepare this file, the user can rely on the first panel of the TACO app. In this panel, the user can enter multiple information regarding the overall analysis and rely on the app to ensure the paths and files indicated are correctly written (ensure paths/files exist):

- Full path to save the JSON configuration file (the path must contain the name of the JSON file to be created, e.g., `/Path/to/save/JSON_file.json`)
- Path of the folder in which cell-related analysis files will be saved (e.g., `/path/to/saving/folder/`)
- Path to the user-defined Python QC file (e.g., `/Path/to/save/User_defined_QC_file.py`)

For each database to analyze, the user must provide the following information:

- Name of the database
- Path to folder containing original files (e.g., `/Path/to/folder/with/original/files/`)
- Path to database-specific Python script (e.g., `/Path/to/database_python_script.py`)
- Path to database-specific population class table (e.g., `/Path/to/database/Population_class_table.csv`)
- Path to database-specific cell-sweep table (e.g., `/Path/to/database/Cell_Sweep_table.csv`)
- Whether the stimulus times are provided in the database (check the box if yes)
- Stimulus duration (in seconds)

Once all the information is provided, the user can click the **Add Database** button. If an error has been made in the information, the user can re-enter the details for the database; the JSON file will be automatically corrected.  
Once ready, the user can click the **Save JSON** button to save the JSON configuration file.

---

### (2) Running the analysis

To run the analysis, the user can go to the second panel **Run analysis**.  

The pipeline analyzes the data of multiple cells simultaneously using parallel processing. The user can choose the number of CPU cores to use; by default, the analysis will use half of the computer’s CPU cores.  

The user can then enter the path of the JSON configuration file (e.g., `/Path/to/save/JSON_file.json`), choose which analyses to perform (Spike, Sweep, Firing, or Metadata), and indicate whether to overwrite any existing analysis files.  

Once all information has been entered, the analysis can be launched by clicking the **Run analysis** button. Progress can be monitored via the progress bar displayed in the app. During the analysis, the user should **not close the Terminal or the application page in the browser**.  

After the analysis, the user can specify a path to save summary files (e.g., `/path/to/saving/folder/`). Tables summarizing fit parameters, firing properties, and linear properties will be saved at this location.

---

### (3) Visual inspection

To visually inspect the analysis, the user can go (after completion of the analysis) to the third panel **Cell Visualization**. In this panel, the user can select the JSON configuration file. The list of available cell files is generated based on the cells indicated in the different databases’ population class tables.

***** LG Is this font/format **XXX** specifically for app panes/buttons? +++

---

## Example use of the TACO pipeline

We provide a Test_folder.zip containing few example cells originating from **two open-access databases**:


- Da Silva Lantyer et al., 2018  
- Harrison et al., 2015  

The following files are provided:

- Original cell files  
- Database-specific scripts  
- Population and cell-sweep tables  
- A ready-to-use `config_json_file_test.json`  

You can directly use this file in the Analysis part of the pipeline.  

**Note:** Before using the test data, do not forget to update the `config_json_file_test.json` with correct file and folder paths.
