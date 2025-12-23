#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:19:08 2024

@author: julienballbe
"""



# Sweep_QC_analysis.py
#from Defne_decorators import variables


import inspect
import numpy as np
import pandas as pd
import importlib
import warnings
import traceback
import Ordinary_functions as ordi_func
import globals_module

def set_globals(sweep_variable_dict):
    globals_module.Bridge_Error_extrapolated = sweep_variable_dict['Bridge_Error_extrapolated']
    globals_module.Bridge_Error_GOhms = sweep_variable_dict['Bridge_Error_GOhms']
    globals_module.Cell_Input_Resistance_GOhms = sweep_variable_dict['Cell_Input_Resistance_GOhms']
    globals_module.Cell_Resting_potential_mV = sweep_variable_dict['Cell_Resting_potential_mV']
    globals_module.Cell_Time_constant_ms = sweep_variable_dict['Cell_Time_constant_ms']
    globals_module.Holding_current_pA = sweep_variable_dict['Holding_current_pA']
    globals_module.Holding_potential_mV = sweep_variable_dict['Holding_potential_mV']
    globals_module.Input_Resistance_GOhms = sweep_variable_dict['Input_Resistance_GOhms']
    globals_module.Raw_current_pA_trace = sweep_variable_dict['Raw_current_pA_trace']
    globals_module.Raw_potential_mV_trace = sweep_variable_dict['Raw_potential_mV_trace']
    globals_module.Resting_potential_mV = sweep_variable_dict['Resting_potential_mV']
    globals_module.Sampling_Rate_Hz = sweep_variable_dict['Sampling_Rate_Hz']
    globals_module.SS_potential_mV = sweep_variable_dict['SS_potential_mV']
    globals_module.Stim_amp_pA = sweep_variable_dict['Stim_amp_pA']
    globals_module.Stim_end_s = sweep_variable_dict['Stim_end_s']
    globals_module.Stim_SS_pA = sweep_variable_dict['Stim_SS_pA']
    globals_module.Stim_start_s = sweep_variable_dict['Stim_start_s']
    globals_module.Time_constant_ms = sweep_variable_dict['Time_constant_ms']
    globals_module.Time_s_trace = sweep_variable_dict['Time_s_trace']
   

def run_QC_for_cell(Full_TVC, sweep_info_table, QC_function_module, path_to_QC_function_module):
    
    sweep_list = np.array(sweep_info_table.loc[:,'Sweep'])
    QC_table = pd.DataFrame()
    
    #Check that requirment made by function are correct
    
    spec=importlib.util.spec_from_file_location(QC_function_module,path_to_QC_function_module)
    QC_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(QC_module)
    
    functions_list = inspect.getmembers(QC_module, inspect.isfunction)
    
    error = ''
    sub_error = []
             
    #Run QC analysis   
    
    
    for current_sweep in sweep_list:
        
        # Get sweep variable dictionary
        sweep_variable_dict = get_sweep_variable_dict(Full_TVC, sweep_info_table, current_sweep)

        # Set global variables
        set_globals(sweep_variable_dict)

        # Load the QC function module
        spec = importlib.util.spec_from_file_location(QC_function_module, path_to_QC_function_module)
        QC_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(QC_module)

        # Get functions from the QC module
        functions_list = inspect.getmembers(QC_module, inspect.isfunction)

        results = []

        # Call functions and collect results
        for func_name, func in functions_list:
            try:
                result = func()  # Call the function
                results.append(result)
            except:
                error_message = traceback.format_exc()
                if error_message not in sub_error:
                    
                    error = str(error+ '\n'+error_message)
                    sub_error.append(error_message)
                else:
                    print('okok')
                
        data_dict = {key: [value] for key, value in results}
        sweep_QC_table = pd.DataFrame(data_dict)
        
        sweep_QC_table.loc[:,'Sweep']=str(current_sweep)
        
        QC_table = pd.concat([QC_table, sweep_QC_table], ignore_index = True)
        
    if len(error)==0:
        error='No error'
    else:
        warnings.warn(error, UserWarning)
    
    QC_table.index = QC_table['Sweep']
    QC_table.index = QC_table.index.astype(str)
    QC_table.index.name = 'Index'
    
    boolean_columns = QC_table.drop(columns='Sweep')

    QC_table['Passed_QC'] = boolean_columns.all(axis=1).astype(bool)

    return QC_table, error

def get_sweep_variable_dict(Full_TVC, sweep_info_table, sweep):
    
    variable_dict = {}
    raw_traces_TVC = (Full_TVC.loc[Full_TVC["Sweep"] == sweep, "TVC"].iloc[0])

    variable_dict['Raw_potential_mV_trace'] = np.array(raw_traces_TVC.loc[:,'Membrane_potential_mV'])
    variable_dict["Raw_current_pA_trace"] = np.array(raw_traces_TVC.loc[:,'Input_current_pA'])
    variable_dict["Time_s_trace"] = np.array(raw_traces_TVC.loc[:,'Time_s'])
    
    
    
    for col in sweep_info_table.columns:
        if col not in ['Sweep', 'Trace_id', 'Protocol_id']:
            variable_dict[col] = sweep_info_table.loc[sweep, col]
    
    cell_input_resistance = np.nanmean(sweep_info_table.loc[:,'Input_Resistance_GOhms'])
    variable_dict["Cell_Input_Resistance_GOhms"] = cell_input_resistance
    
    cell_time_constant = np.nanmean(sweep_info_table.loc[:,'Time_constant_ms'])
    variable_dict["Cell_Time_constant_ms"] = cell_time_constant
    
    cell_Resting_potential = np.nanmean(sweep_info_table.loc[:,'Resting_potential_mV'])
    variable_dict["Cell_Resting_potential_mV"] = cell_Resting_potential
    
    
    return variable_dict
    
    
    


