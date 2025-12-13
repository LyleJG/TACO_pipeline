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




     
                    
def cell_processing(args_list):
    """
    Central function of the TACO pipeline
    Given necessary information are provided, the function runs the analysis

    Parameters
    ----------
    args_list : list
        contains in this specific order : Cell_id (str), database name (str), name of python script (str), path to QC file (str), path to saving file (str), wether we overwrite existing cell file (Bool), list of analysis to perform (list)

    Returns
    -------
    Either saves the analysis or return the cell_id if the analysis fails

    """
    
    cell_id,current_db, module, full_path_to_python_script, path_to_QC_file,  path_to_saving_file,overwrite_cell_files, analysis_to_perform = args_list
    
    try:
        
       
        
        saving_file_cell = str(path_to_saving_file+"Cell_"+str(cell_id)+".h5")
       
        if overwrite_cell_files == True or overwrite_cell_files == False and os.path.exists(saving_file_cell) == False:
            

            db_original_file_directory = current_db['original_file_directory']
            
            db_population_class = pd.read_csv(current_db['db_population_class_file'],sep =',',encoding = "unicode_escape")
            
            db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
            
            
            if 'All' in analysis_to_perform : 
                saving_dict = {}
                processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
            
    # LG
                # Full_SF_dict='--'
                # cell_sweep_info_table='--'
                # cell_Sweep_QC_table='--'
                # cell_adaptation_table = '--'
                # cell_feature_table='--'
                # Metadata_dict='--'
                
                current_process = "Get traces"
                
                current_cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
                # Gather for the different sweeps of the cell, time, voltage and current traces, as weel as stimulus start and end times
                
               
                # LG
                get_db_traces_args = [module,
                              full_path_to_python_script,
                              current_db["db_function_name"],
                              db_original_file_directory,
                              cell_id,
                              current_cell_sweep_list,
                              db_cell_sweep_file,
                              current_db["stimulus_time_provided"],
                              current_db["db_stimulus_duration"]]
                
                
                Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(get_db_traces_args, processing_table)
                
                current_process = "Sweep analysis"
                cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, processing_table)
                
                
                #cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_sweep_info_table, processing_table)
                cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                
                
                current_process = "Spike analysis"
                Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
                
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
                current_cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()
                # perform sequentially the different part of the analysis
                
                if "Sweep analysis" in analysis_to_perform:
                    #Perform sweep analysis
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
                        
                   
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    
                    #Get cell's raw traces
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, processing_table)
                    #perform sweep analysis
                    cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, processing_table)
                    #perform sweep QC analysis
                    cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                
                    Sweep_analysis_dict = {"Sweep info" : cell_sweep_info_table,
                                           "Sweep QC" : cell_Sweep_QC_table}
                    dict_to_add = {"Sweep analysis":Sweep_analysis_dict}
                    saving_dict.update(dict_to_add)
                    saving_dict.update({"Processing report" : processing_table})
                    
                if "Spike analysis" in analysis_to_perform:
                    current_process = "Spike analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, processing_table)
                    
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
                        cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, processing_table)
                        
                        cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                        
                    else: 
                        #Otherwise, get exisiting analysis, and remove existing entries of spike analysis in processing report
                        cell_sweep_info_table = is_sweep_info_table
                        cell_Sweep_QC_table = is_sweep_QC_table
                        processing_table = is_Processing_report_df
                        processing_table = processing_table[processing_table['Processing_step'] != "Spike analysis"]
                        
                    #perform spike analysis
                    Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
                    
                    saving_dict.update({"Spike analysis" : Full_SF_dict})
                    saving_dict.update({"Processing report" : processing_table})
                    
                if "Firing analysis" in analysis_to_perform:
                    current_process = "Firing analysis"
                    processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
                    
                    
                    args_list = [module,
                                  full_path_to_python_script,
                                  current_db["db_function_name"],
                                  db_original_file_directory,
                                  cell_id,
                                  current_cell_sweep_list,
                                  db_cell_sweep_file,
                                  current_db["stimulus_time_provided"],
                                  current_db["db_stimulus_duration"]]
                    # Get raw traces
                    Full_TVC_table, cell_stim_time_table, processing_table = get_db_traces(args_list, processing_table)
                    
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
                            cell_sweep_info_table, processing_table = perform_sweep_related_analysis(Full_TVC_table, cell_stim_time_table, processing_table)
                            
                            cell_Sweep_QC_table, processing_table = perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)
                            
                            Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table)
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
                if 'Metadata' in analysis_to_perform:
                    
                    current_process = "Metadata"

                    #Get relevant information from population class table
                    Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
                    
                    saving_dict.update({"Metadata" : Metadata_dict})
                    

            #Write the different analysis in the cell file
            ordifunc.write_cell_file_h5(saving_file_cell,
                               saving_dict,
                               overwrite=overwrite_cell_files,
                               selection = analysis_to_perform)

            
                
                
                
        
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
                           overwrite=overwrite_cell_files,
                           selection = analysis_to_perform)
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
        
    
def get_db_traces(args_list, processing_table):
    """
    Organize the access to raw traces for a given cel, given :
    args_list = [module, #name of python script
                  full_path_to_python_script, #path to python script
                  current_db["db_function_name"], #Database name
                  db_original_file_directory, #path to original cell files
                  cell_id, # Cell id
                  current_cell_sweep_list, #list of sweeps
                  db_cell_sweep_file, # Cell sweep table
                  current_db["stimulus_time_provided"], #Wether the stimulus time are provided
                  current_db["db_stimulus_duration"]] # stimulus duration

    

    """
    current_process = "get_TVC_tables"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_TVC:
        Full_TVC_table = pd.DataFrame(columns=['Sweep','TVC'])
        cell_stim_time_table = pd.DataFrame(columns=['Sweep','Stim_start_s', 'Stim_end_s'])
        
        
        
        Full_TVC_table, cell_stim_time_table = sw_an.get_TVC_table(args_list)
        
        Full_TVC_table.index = Full_TVC_table.loc[:,"Sweep"]
        Full_TVC_table.index = Full_TVC_table.index.astype(str)
        
        cell_stim_time_table.index = cell_stim_time_table.loc[:,"Sweep"]
        cell_stim_time_table.index = cell_stim_time_table.index.astype(str)
        cell_stim_time_table=cell_stim_time_table.astype({'Stim_start_s':float, 'Stim_end_s':float})
    end_time=time.time()
    processing_time = end_time-start_time
    processing_table = append_processing_table(current_process, processing_table, warning_TVC, processing_time)

        
    return Full_TVC_table, cell_stim_time_table, processing_table

# LG change Full_TVC_table to cell_Full_TVC_table for consistency (other plances?)
def perform_sweep_related_analysis (cell_Full_TVC_table, cell_stim_time_table, processing_table):
    """
    Organize the sweep analysis
    Requires the cell_Full_TVC_table, cell_stim_time_table, processing_table (from get_db_traces)

    

    """
    current_process = "Sweep Analysis"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_cell_sweep_table:
        
        cell_sweep_info_table = sw_an.sweep_analysis_processing(cell_Full_TVC_table, cell_stim_time_table)

    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_sweep_table, processing_time)
    
    return cell_sweep_info_table, processing_table
        
def perform_QC_analysis(Full_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table):
    """
    Organize the sweep analysis
    Requires the Full_TVC_table (from get_db_traces) , cell_sweep_info_table (from perform_sweep_related_analysis), path_to_QC_file, processing_table(from perform_sweep_related_analysis)

    """
    current_process = "Sweep QC"
    start_time=time.time()
    QC_function_module = os.path.basename(path_to_QC_file)
    with warnings.catch_warnings(record=True) as warning_cell_QC_table:
        
        cell_Sweep_QC_table, error_message = sw_qc.run_QC_for_cell(Full_TVC_table, cell_sweep_info_table, QC_function_module, path_to_QC_file)
        
        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = append_processing_table(current_process, processing_table, warning_cell_QC_table, processing_time)
    
    return cell_Sweep_QC_table, processing_table
    
 

def perform_spike_related_analysis(Full_TVC_table, cell_sweep_info_table, processing_table):
    """
    Organize the spike analysis
    Requires the Full_TVC_table (from get_db_traces) , cell_sweep_info_table (from perform_sweep_related_analysis), processing_table (from perform_QC_analysis)

    """
    current_process = "Spike analysis"
    start_time=time.time()
    
    with warnings.catch_warnings(record=True) as warning_Full_SF_table:
        Full_SF_dict = sp_an.create_cell_Full_SF_dict_table(
            Full_TVC_table.copy(), cell_sweep_info_table.copy())

        Full_SF_table = sp_an.create_Full_SF_table(
            Full_TVC_table.copy(), Full_SF_dict.copy(), cell_sweep_info_table.copy())
        

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
    
    
