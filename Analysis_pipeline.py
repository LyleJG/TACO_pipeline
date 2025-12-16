#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:06:37 2023

@author: julienballbe
"""



import numpy as np
import pandas as pd

import os


import warnings
import traceback

import time

import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import Ordinary_functions as ordifunc
import Sweep_QC_analysis as sw_qc
import TACO_pipeline_App as TACO_pipeline_App
import importlib

['170811NC97_4', 0,
            [{'name': 'config_json_file_test_lg.json', 'size': 909,
              'type': 'application/json',
              'datapath': '/tmp/fileupload-t62z5qet/tmptfq9hpkk/0.json'}],
            ('All',), True]

def cell_processing(cell_id, config_files_idx=0, config_files=None, analysis=['All'], overwrite_files=False):
    """
    Central function of the TACO pipeline
    Runs the analysis on a specified cell.

    Parameters
    ----------
    cell_id (str),  
    config_files: paths of TACO config files
    config_files_idx: which file in that list to use, default = 0.
    overwrite_files: overwrite existing cell file, default = False.
    analysis: list of analyses to perform, default = ['All'].

    Returns
    -------
    Either saves the analysis or return the cell_id if the analysis fails

    """
    print(f'{cell_id=}, {config_files_idx=},{config_files=},{analysis=}, ')
    config_df = TACO_pipeline_App.import_json_config_files(config_files)
    current_db = config_df.loc[config_files_idx,:].to_dict()
    database_name = current_db["database_name"]
    path_to_python_folder = current_db["path_to_db_script_folder"]
    python_file=current_db['python_file_name']
    module=python_file.replace('.py',"")
    full_path_to_python_script=str(path_to_python_folder+python_file)
    spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
    DB_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DB_module)
    path_to_saving_file = config_df.loc[0,'path_to_saving_file']
    path_to_QC_file = config_df.loc[0,'path_to_QC_file']
    
    db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")

    db_original_file_directory = current_db['original_file_directory']
    db_population_class = pd.read_csv(current_db['db_population_class_file'],sep =',',encoding = "unicode_escape")

    db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],
                                     sep =',',encoding = "unicode_escape")
    cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])

    #Get cell's raw traces
    start_time=time.time()
    cell_TVC_table, cell_stim_time_table, warning_TVC = sw_an.get_cell_TVC_table(
         module=module,
         full_path_to_python_script=full_path_to_python_script,
         db_function_name=current_db["db_function_name"],
         db_original_file_directory=db_original_file_directory,
         cell_id=cell_id,
         cell_sweep_list=cell_sweep_list,
         db_cell_sweep_file=db_cell_sweep_file,
         stimulus_time_provided=current_db["stimulus_time_provided"],
         db_stimulus_duration=current_db["db_stimulus_duration"])
    processing_time = time.time()-start_time
    processing_table = append_processing_table(
        "Get traces", processing_table, warning_TVC, processing_time)

    if 'All' in analysis :
        # analysis_choices is taken from TACO_pipeline_App.py
        analysis_choices={"All":"All" ,"Metadata":"Metadata", "Sweep analysis": "Sweep analysis",
                      "Spike analysis":"Spike analysis","Firing analysis":"Firing analysis"}
        analysis = [key for key in analysis_choices if key != "All"]
    try:
        saving_file_cell = str(path_to_saving_file+"Cell_"+str(cell_id)+".h5")
       
        if overwrite_files == True or overwrite_files == False and os.path.exists(saving_file_cell) == False:
            # db_original_file_directory = current_db['original_file_directory']
            # db_population_class = pd.read_csv(current_db['db_population_class_file'],sep =',',encoding = "unicode_escape")
            # db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
            if 'All' in analysis :
                saving_dict = {}
                current_process = "Sweep analysis"
                cell_sweep_info_table, processing_table = perform_sweep_related_analysis(cell_TVC_table, cell_stim_time_table, processing_table)
                cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                current_process = "Spike analysis"
                Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(cell_TVC_table, cell_sweep_info_table, processing_table)
                current_process = "Firing analysis"
                cell_feature_table, cell_fit_table,cell_adaptation_table, processing_table = perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table, processing_table)
                current_process = "Metadata"
                Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
                Sweep_analysis_dict = {"Sweep info" : cell_sweep_info_table,
                                       "Sweep QC" : cell_Sweep_QC_table}
                firing_dict={"Cell_feature" : cell_feature_table,
                             "Cell_fit" : cell_fit_table,
                             "Cell_Adaptation" : cell_adaptation_table}
                saving_dict = {"Sweep analysis" : Sweep_analysis_dict,
                               "Spike analysis" : Full_SF_dict,
                               "Firing analysis" : firing_dict,
                               "Metadata" : Metadata_dict}
            else:
                saving_dict = {}
                # cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
                # perform sequentially the different part of the analysis
                if "Sweep analysis" in analysis:
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    current_process = "Sweep analysis"
                    is_Processing_report_df = pd.DataFrame()
                    if os.path.exists(saving_file_cell) == True:
                        #If this analysis has already been performed, get exisiting cell processing df
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Processing_report'])
                        is_Processing_report_df = cell_dict['Processing_table']
                        
                    if is_Processing_report_df.shape[0]!=0:
                        #if the processing df is not empty, remove entries corresponding to sweep analysis and sweep QC
                        processing_table = is_Processing_report_df
                        processing_table = processing_table[processing_table['Processing_step']!="Sweep analysis"]
                        processing_table = processing_table[processing_table['Processing_step']!="Sweep QC"]

                    #Get cell's raw traces
                    # cell_TVC_table, cell_stim_time_table, processing_table = get_cell_TVC_table(
                    #     get_cell_TVC_table_args, processing_table)
                    #perform sweep analysis
                    cell_sweep_info_table, processing_table = perform_sweep_related_analysis(cell_TVC_table, cell_stim_time_table, processing_table)
                    #perform sweep QC analysis
                    cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                
                    Sweep_analysis_dict = {"Sweep info" : cell_sweep_info_table,
                                           "Sweep QC" : cell_Sweep_QC_table}
                    dict_to_add = {"Sweep analysis":Sweep_analysis_dict}
                    saving_dict.update(dict_to_add)
                    saving_dict.update({"Processing report" : processing_table})
                    
                if "Spike analysis" in analysis:
                    current_process = "Spike analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])

                    #Get cell's raw traces
                    # cell_TVC_table, cell_stim_time_table, processing_table = get_cell_TVC_table(
                    #     get_cell_TVC_table_args, processing_table)

                    is_sweep_info_table = pd.DataFrame()
                    is_sweep_QC_table = pd.DataFrame()
                    is_Processing_report_df = pd.DataFrame()
                    if os.path.exists(saving_file_cell) == True:
                        #If this analysis has already been performed, get exisiting cell processing df
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Sweep analysis','Sweep QC','Processing_report'])
                        is_sweep_info_table = cell_dict["Sweep_info_table"]
                        is_sweep_QC_table = cell_dict['Sweep_QC_table']
                        is_Processing_report_df = cell_dict['Processing_table']
                        
                    if is_sweep_info_table.shape[0]==0 or is_sweep_QC_table.shape[0]==0 or is_Processing_report_df.shape[0]==0 or os.path.exists(saving_file_cell) == False:
                        #If sweep analysis,or QC analysis haven't been performed, or if there is no existing file for this cell, then perfrom Sweep analysis and QC analysis before performing Spike analysis
                        cell_sweep_info_table, processing_table = perform_sweep_related_analysis(cell_TVC_table, cell_stim_time_table, processing_table)
                        
                        cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)

                    else: 
                        #Otherwise, get exisiting analysis, and remove existing entries of spike analysis in processing report
                        cell_sweep_info_table = is_sweep_info_table
                        cell_Sweep_QC_table = is_sweep_QC_table
                        processing_table = is_Processing_report_df
                        processing_table = processing_table[processing_table['Processing_step'] != "Spike analysis"]
                        
                    #perform spike analysis
                    Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(cell_TVC_table, cell_sweep_info_table, processing_table)
                    
                    saving_dict.update({"Spike analysis" : Full_SF_dict})
                    saving_dict.update({"Processing report" : processing_table})
                    
                if "Firing analysis" in analysis:
                    current_process = "Firing analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])

                    # Get raw traces
                    # cell_TVC_table, cell_stim_time_table, processing_table = get_cell_TVC_table(args_list, processing_table)
                    
                    is_sweep_info_table = pd.DataFrame()
                    is_sweep_QC_table = pd.DataFrame()
                    is_SF_table = pd.DataFrame()
                    is_Processing_report_df = pd.DataFrame()
    
                    if os.path.exists(saving_file_cell) == True:
    
                        #if a cell file already exists, get the different part of the analysis (Sweep, Spike and QC analysis)
                        current_db_df = pd.DataFrame(current_db,index=[0])
                        cell_dict = ordifunc.read_cell_file_h5(cell_id, current_db_df, selection = ['Sweep analysis','Sweep QC', 'TVC_SF','Processing_report'])
                        is_SF_table = cell_dict['Full_SF_table']
                        is_sweep_info_table = cell_dict["Sweep_info_table"]
                        is_sweep_QC_table = cell_dict['Sweep_QC_table']
                        is_Processing_report_df = cell_dict['Processing_table']
                    
                    if is_SF_table.shape[0]==0 or is_sweep_info_table.shape[0]==0 or is_sweep_QC_table.shape[0]==0 or is_Processing_report_df.shape[0]==0 or os.path.exists(saving_file_cell) == False:
                            #If spike analysis, sweep analysis,or QC analysis haven't been performed, or if there is no existing file for this cell, then perform Spike analysis, Sweep analysis and QC analysis before performing Spike analysis
                            cell_sweep_info_table, processing_table = perform_sweep_related_analysis(cell_TVC_table, cell_stim_time_table, processing_table)
                            
                            cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                            
                            Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(cell_TVC_table, cell_sweep_info_table, processing_table)
                    else: 
                        #Otherwise, get exisiting analysis, and remove existing entries of firing analysis in processing report
                        cell_sweep_info_table = is_sweep_info_table
                        cell_Sweep_QC_table = is_sweep_QC_table
                        processing_table = is_Processing_report_df
                        Full_SF_table = is_SF_table
                        processing_table = processing_table[processing_table['Processing_step'] != "Firing analysis"]
                        
                    #Perform firing analysis
                    cell_feature_table, cell_fit_table, cell_adaptation_table, processing_table = perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table, processing_table)
                    
                    firing_dict={"Cell_feature" : cell_feature_table,
                                 "Cell_fit" : cell_fit_table,
                                 "Cell_Adaptation" : cell_adaptation_table}
                    
                    saving_dict.update({"Firing analysis" : firing_dict})
                    saving_dict.update({"Processing report" : processing_table})
                if 'Metadata' in analysis:
                    
                    current_process = "Metadata"

                    #Get relevant information from population class table
                    Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
                    
                    saving_dict.update({"Metadata" : Metadata_dict})
                    

            #Write the different analysis in the cell file
            ordifunc.write_cell_file_h5(saving_file_cell,
                               saving_dict,
                               overwrite=overwrite_files,
                               selection = analysis)

            
                

                
        
    except:

        error= traceback.format_exc()

        #write message corresponding to the analysis failure in the process report
        message = str('Error in '+str(current_process)+': '+str(error))
        new_line = pd.DataFrame([current_process,'--',message]).T

        new_line.columns=processing_table.columns
        processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
        saving_dict.update({"Processing report" : processing_table})
        #Write the different analysis in the cell file
        ordifunc.write_cell_file_h5(saving_file_cell,
                           saving_dict,
                           overwrite=overwrite_files,
                           selection = analysis)
        return cell_id
        

def append_processing_table(current_process,processing_table, warning_list, processing_time):
    """
    Update the processing report with the observations and processing time for a given analysis
    """
    if len(warning_list) == 0:
        new_line = pd.DataFrame([current_process,str(str(round((processing_time),3))+'s'),'--']).T
        new_line.columns=processing_table.columns
        processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
    else:
        for current_warning in warning_list:
            message = str('Warnings in '+str(current_warning.filename)+str(' line:')+str(current_warning.lineno)+str(':')+str(current_warning.message))
            new_line = pd.DataFrame([current_process,str(str(round((processing_time),3))+'s'),message]).T
            new_line.columns=processing_table.columns
            processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
    
    return processing_table
        
# LG Change to handle processing_table outside of function call.
# def get_db_traces(args_list, processing_table):


# LG change Full_TVC_table to cell_TVC_table for consistency (other plances?)
def perform_sweep_related_analysis (cell_TVC_table, cell_stim_time_table, processing_table):
    """
    Organize the sweep analysis
    Requires the cell_TVC_table, cell_stim_time_table, processing_table (from get_cell_TVC_table)

    

    """
    current_process = "Sweep Analysis"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_cell_sweep_table:
        
        cell_sweep_info_table = sw_an.sweep_analysis_processing(cell_TVC_table, cell_stim_time_table)

    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_sweep_table, processing_time)
    
    return cell_sweep_info_table, processing_table
        
def perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table):
    """
    Organize the sweep analysis
    Requires the cell_TVC_table (from get_cell_TVC_table) , cell_sweep_info_table (from perform_sweep_related_analysis), path_to_QC_file, processing_table(from perform_sweep_related_analysis)

    """
    current_process = "Sweep QC"
    start_time=time.time()
    QC_function_module = os.path.basename(path_to_QC_file)
    with warnings.catch_warnings(record=True) as warning_cell_QC_table:
        
        cell_Sweep_QC_table, error_message = sw_qc.run_QC_for_cell(cell_TVC_table, cell_sweep_info_table, QC_function_module, path_to_QC_file)
        
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_QC_table, processing_time)
    
    return cell_Sweep_QC_table, processing_table
    
 

def perform_spike_related_analysis(cell_TVC_table, cell_sweep_info_table, processing_table):
    """
    Organize the spike analysis
    Requires the cell_TVC_table (from get_cell_TVC_table) , cell_sweep_info_table (from perform_sweep_related_analysis), processing_table (from perform_QC_analysis)

    """
    current_process = "Spike analysis"
    start_time=time.time()
    
    with warnings.catch_warnings(record=True) as warning_Full_SF_table:
        Full_SF_dict = sp_an.create_cell_Full_SF_dict_table(
            cell_TVC_table.copy(), cell_sweep_info_table.copy())

        Full_SF_table = sp_an.create_Full_SF_table(
            cell_TVC_table.copy(), Full_SF_dict.copy(), cell_sweep_info_table.copy())


        Full_SF_table.index = Full_SF_table.index.astype(str)
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_Full_SF_table, processing_time)
    
    return Full_SF_dict, Full_SF_table, processing_table
    
   

def perform_firing_related_analysis(Full_SF_table, cell_sweep_info_table,cell_Sweep_QC_table, processing_table):
    """
    Organize the firing analysis
    Requires the Full_SF_table (from perform_spike_related_analysis) , cell_sweep_info_table (from perform_sweep_related_analysis), cell_Sweep_QC_table (perform_QC_analysis), processing_table (from perform_spike_related_analysis)

    """
    current_process = "Firing analysis"
    start_time=time.time()

    with warnings.catch_warnings(record=True) as warning_cell_feature_table:
        response_duration_dictionnary={
            'Time_based':[.005, .010, .025, .050, .100, .250, .500],
            'Index_based':list(np.arange(2,18)),
            'Interval_based':list(np.arange(1,17))}

        cell_feature_table, cell_fit_table = fir_an.compute_cell_features(Full_SF_table,
                                                                   cell_sweep_info_table,
                                                                   response_duration_dictionnary,
                                                                   cell_Sweep_QC_table)
        
        cell_adaptation_table = fir_an.compute_cell_adaptation_behavior(Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table)
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_feature_table, processing_time)
    
    return cell_feature_table, cell_fit_table, cell_adaptation_table ,processing_table
    
    
