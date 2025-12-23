#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:06:37 2023

@author: julienballbe
"""



import numpy as np
import pandas as pd

import os, sys


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

def cell_processing(cell_id, TACO_config_file='', db_spec_idx=0, analysis=['All'], overwrite_files=False):
    """
    Central function of the TACO pipeline
    Runs the analysis on a specified cell.

    Parameters
    ----------
    cell_id (str),  
    db_spec: paths of TACO config files, or dicts from ui paths input
    config_files_idx: which file in that list to use, default = 0.
    overwrite_files: overwrite existing cell file, default = False.
    analysis: list of analyses to perform, default = ['All'].

    Returns
    -------
    Either saves the analysis or return the cell_id if the analysis fails

    """
    config_df = ordifunc.get_TACO_config_file_df(TACO_config_file)
    db_spec = config_df.loc[db_spec_idx,:].to_dict()    
    db_spec_df = pd.DataFrame(db_spec,index=[0])
    cell_saving_file = str(db_spec['path_to_saving_file']+"Cell_"+str(cell_id)+".h5")
    
    # Punt if no overwrite of existing analysis.
    if not (overwrite_files == True or (overwrite_files == False and os.path.exists(cell_saving_file) == False)):
        print(f'Existing {cell_saving_file=} preserved, processing {cell_id=} abandoned.')
        return
         
    module=db_spec['python_file_name'].replace('.py',"")
    full_path_to_python_script=str(db_spec["path_to_db_script_folder"]+db_spec['python_file_name'])
    spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
    DB_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DB_module)
    path_to_QC_file = db_spec['path_to_QC_file']
    db_population_class = pd.read_csv(db_spec['db_population_class_file'],sep =',',encoding = "unicode_escape")
    db_cell_sweep_file = pd.read_csv(db_spec['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
    cell_sweep_list = db_cell_sweep_file.loc[db_cell_sweep_file['Cell_id'] == cell_id, "Sweep_id"].to_list()    
    
    # if 'All' in analysis : # analysis_choices is taken from TACO_pipeline_App.py -> change this to a global var
    #     analysis_choices={'All':'All' ,'Metadata':'Metadata', 'Sweep analysis': 'Sweep analysis',
    #                   'Spike analysis':'Spike analysis','Firing analysis':'Firing analysis'}
    #     analysis = [key for key in analysis_choices if key != 'All']
    
    # Get cell_dict if cell file exists.
    if os.path.exists(cell_saving_file) == True:
         #If this analysis has already been performed, get existing cell processing df
         cell_dict = ordifunc.read_cell_file_h5(cell_id, db_spec_df, selection = ['All'])
         processing_table = cell_dict['Processing_table']
         if 'Processing_step' not in processing_table.keys():
             processing_table['Processing_step']=''
    else:
         cell_dict=None
         processing_table = pd.DataFrame(columns=['Processing_step','Processing_time','Warnings_encontered'])
    if cell_dict:
         cell_sweep_info_table=cell_dict['Sweep_info_table']
         cell_Sweep_QC_table=cell_dict['Sweep_QC_table']
         Full_SF_table=cell_dict['Full_SF_table']
    existing_sweep_analysis_P=cell_dict and cell_dict['Sweep_info_table'].shape[0]>0
    existing_QC_analysis_P=cell_dict and cell_dict['Sweep_QC_table'].shape[0]>0
    existing_spike_analysis_P=cell_dict and cell_dict['Full_SF_table'].shape[0]>0
    
    # breakpoint()
    # foo=TACO_pipeline_App.get_firing_analysis(cell_dict,'Time_based')
    # foo[1]['Full_stim_freq_table']['Frequency_Hz']

    if cell_id not in list(db_cell_sweep_file['Cell_id']):
       raise ValueError(f"{cell_id=} not found from {TACO_config_file=}, {db_spec_idx=}")
       
    print(f"Processing {cell_id=}")            
    #Get cell's raw traces
    start_time=time.time()
    cell_TVC_table, cell_stim_time_table, warning_TVC = sw_an.get_cell_TVC_table(
         module=module,
         full_path_to_python_script=full_path_to_python_script,
         db_function_name=db_spec["db_function_name"],
         db_original_file_directory=db_spec['original_file_directory'],
         cell_id=cell_id,
         cell_sweep_list=cell_sweep_list,
         db_cell_sweep_file=db_cell_sweep_file,
         stimulus_time_provided=db_spec["stimulus_time_provided"],
         db_stimulus_duration=db_spec["db_stimulus_duration"])            
    processing_time = time.time()-start_time
    processing_table = append_processing_table("Get traces", processing_table, warning_TVC, processing_time)

    try:
        saving_dict = {}      
        # perform sequentially the different part of the analysis
        if ('Sweep analysis' in analysis or 'All' in analysis
            or (('Spike analysis' in analysis or 'Firing analysis' in analysis)
                and not (existing_QC_analysis_P and existing_sweep_analysis_P))):
             current_process = 'Sweep and QC analysis'
             if processing_table.shape[0]!=0: #if processing_table not empty, remove sweep analysis and sweep QC entries
                 processing_table = processing_table[processing_table['Processing_step']!='Sweep analysis']
                 processing_table = processing_table[processing_table['Processing_step']!='Sweep QC']
             cell_sweep_info_table, processing_table = perform_sweep_related_analysis(cell_TVC_table, cell_stim_time_table, processing_table)
             cell_Sweep_QC_table, processing_table = perform_QC_analysis(cell_TVC_table, cell_sweep_info_table, path_to_QC_file, processing_table)                
             saving_dict.update({'Sweep analysis':{'Sweep info' : cell_sweep_info_table, 'Sweep QC' : cell_Sweep_QC_table}})
             print(f'Done with {current_process=}')
        if ('Spike analysis' in analysis or 'All' in analysis
            or ('Firing analysis' in analysis and not existing_spike_analysis_P)):
             current_process = 'Spike analysis'
             if processing_table.shape[0]!=0: #if processing_table not empty, remove spike analysis entry
                 processing_table = processing_table[processing_table['Processing_step']!=current_process]
             Full_SF_dict, Full_SF_table, processing_table = perform_spike_related_analysis(
                 cell_TVC_table, cell_sweep_info_table, processing_table)
             saving_dict.update({'Spike analysis' : Full_SF_dict})
             print(f'Done with {current_process=}')
             
        if 'Firing analysis' in analysis or 'All' in analysis:
             current_process = 'Firing analysis'
             # breakpoint()
             if processing_table.shape[0]!=0: #if processing_table not empty, remove firing analysis entry
                 processing_table = processing_table[processing_table['Processing_step']!=current_process]
             cell_feature_table, cell_fit_table, cell_adaptation_table, processing_table = perform_firing_related_analysis(
                 Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table, processing_table)               
             saving_dict.update({'Firing analysis' : {'Cell_feature' : cell_feature_table,
                                                      'Cell_fit' : cell_fit_table,
                                                      'Cell_Adaptation' : cell_adaptation_table}})
             print(f'Done with {current_process=}')
        # LG Always add 'Metadata'
        if True or 'Metadata' in analysis:
             current_process = 'Metadata'
             #Get relevant information from population class table
             Metadata_dict = db_population_class.loc[db_population_class['Cell_id'] == cell_id,:].iloc[0,:].to_dict()
             saving_dict.update({'Metadata' : Metadata_dict})
             print(f'Done with {current_process=}')
             # breakpoint()
    except:
        error= traceback.format_exc()
        #write message corresponding to the analysis failure in the process report
        message = str('Error in '+str(current_process)+': '+str(error))
        print(f'{message=}')
        new_line = pd.DataFrame([current_process,'--',message]).T
        new_line.columns=processing_table.columns
        processing_table = pd.concat([processing_table,new_line],ignore_index=True,axis=0)
        return cell_id

    saving_dict.update({'Processing report' : processing_table})
    #Write the different analysis in the cell file    
    ordifunc.write_cell_file_h5(cell_saving_file,saving_dict, overwrite=overwrite_files, selection = analysis)
        
    # Full_SF_table = cell_dict['Full_SF_table']
    # cell_sweep_info_table = cell_dict['Sweep_info_table']
    # sweep_QC_table = cell_dict['Sweep_QC_table']
    # import Firing_analysis as fir_an
    # # breakpoint()
    # stim_freq_table = fir_an.get_stim_freq_table(
    #     Full_SF_table.copy(), 
    #     cell_sweep_info_table.copy(),
    #     cell_Sweep_QC_table.copy(), 
    #     float(0.5),
    #     str('Time_based'))

    

def append_processing_table(current_process, processing_table, warning_list, processing_time):
    """
    Update the processing report with the observations and processing time for a given analysis.
    Ensures proper alignment with processing_table columns to avoid ValueError.
    """
    # Prepare rows to append
    rows_to_append = []
    if len(warning_list) == 0:
        rows_to_append.append({'Processing_step': current_process,
                               'Processing_time': f"{round(processing_time, 3)}s",'Warnings_encontered': '--' })
    else:
        for current_warning in warning_list:
            message = f"Warnings in {current_warning.filename} line:{current_warning.lineno}: {current_warning.message}"
            rows_to_append.append({'Processing_step': current_process, 
                                   'Processing_time': f"{round(processing_time, 3)}s",'Warnings_encontered': message})
    # Create DataFrame directly from list of dicts
    new_lines = pd.DataFrame(rows_to_append)
    # Ensure column order matches processing_table
    new_lines = new_lines[processing_table.columns]
    # Concatenate safely
    processing_table = pd.concat([processing_table, new_lines], ignore_index=True)
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
    processing_table = processing_table[processing_table['Processing_step'] != current_process]
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
    processing_table = processing_table[processing_table['Processing_step'] != current_process]
    processing_table = append_processing_table(current_process, processing_table, warning_cell_QC_table, processing_time)    
    return cell_Sweep_QC_table, processing_table
    
def perform_spike_related_analysis(cell_TVC_table, cell_sweep_info_table, processing_table):
    """
    Organize the spike analysis
    Requires the cell_TVC_table (from get_cell_TVC_table) , cell_sweep_info_table (from perform_sweep_related_analysis), processing_table (from perform_QC_analysis)
    """
    current_process = "Spike analysis"
    start_time=time.time()    
    with warnings.catch_warnings(record=True) as warning_cell_Full_SF_table:
        cell_Full_SF_dict = sp_an.create_cell_Full_SF_dict_table(
            cell_TVC_table.copy(), cell_sweep_info_table.copy())
        cell_Full_SF_table = sp_an.create_Full_SF_table(
            cell_TVC_table.copy(), cell_Full_SF_dict.copy(), cell_sweep_info_table.copy())
        cell_Full_SF_table.index = cell_Full_SF_table.index.astype(str)        
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = processing_table[processing_table['Processing_step'] != current_process]
    processing_table = append_processing_table(current_process, processing_table, warning_cell_Full_SF_table, processing_time)    
    return cell_Full_SF_dict, cell_Full_SF_table, processing_table
    
def perform_firing_related_analysis(cell_Full_SF_table, cell_sweep_info_table,cell_Sweep_QC_table, processing_table):
    """
    Organize the firing analysis
    Requires the cell_Full_SF_table (from perform_spike_related_analysis) , cell_sweep_info_table (from perform_sweep_related_analysis), cell_Sweep_QC_table (perform_QC_analysis), processing_table (from perform_spike_related_analysis)
    """
    current_process = "Firing analysis"
    start_time=time.time()
    with warnings.catch_warnings(record=True) as warning_cell_feature_table:
        response_duration_dictionary={
            'Time_based':[.005, .010, .025, .050, .100, .250, .500],
            'Index_based':list(np.arange(2,18)),
            'Interval_based':list(np.arange(1,17))}
        # response_duration_dictionary={
        #     'Time_based':[.500]}
        cell_feature_table, cell_fit_table = fir_an.compute_cell_features(cell_Full_SF_table,
                                                                   cell_sweep_info_table,
                                                                   response_duration_dictionary,
                                                                   cell_Sweep_QC_table)       
        cell_adaptation_table = fir_an.compute_cell_adaptation_behavior(cell_Full_SF_table, cell_sweep_info_table, cell_Sweep_QC_table)
    end_time=time.time()
    processing_time = end_time - start_time
    processing_table = processing_table[processing_table['Processing_step'] != current_process]
    processing_table = append_processing_table(current_process, processing_table, warning_cell_feature_table, processing_time)   
    return cell_feature_table, cell_fit_table, cell_adaptation_table ,processing_table
    
    
