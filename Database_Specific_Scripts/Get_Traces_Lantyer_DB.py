#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:35:26 2023

@author: julienballbe
"""

import scipy
import pandas as pd
import numpy as np


# LG Change names so that "current" means electrical current, not temporal current (and others)
def get_traces_Lantyer(original_cell_file_folder,cell_id,sweep_id,cell_sweep_table):
    
    #Get file name
    file_name = cell_sweep_table.loc[cell_sweep_table['Cell_id'] == cell_id,'Original_file'].unique()[0]

    #open File
    cell_file = scipy.io.loadmat(str(
        str(original_cell_file_folder)+str(file_name)))
    # LG Why is cell_id a list?
    _cell_id = cell_id[-1]
    
    #Get sweep from file 
    experiment,sweep_protocol,current_sweep = sweep_id.split("_")
    current_id = str('Trace_'+ _cell_id +'_' +
                     sweep_protocol+'_'+str(current_sweep))
    #Get traces
    stim_trace = pd.DataFrame(cell_file[str(current_id+'_1')], columns=['Time_s', 'Input_current_pA'])
    stim_trace.loc[:, 'Input_current_pA'] *= 1e12 #To pA

    membrane_trace = pd.DataFrame(cell_file[str(current_id+'_2')], columns=['Time_s', 'Membrane_potential_mV'])
    membrane_trace.loc[:, 'Membrane_potential_mV'] *= 1e3 #to mV

    # LG Fix these names....
    time_trace = np.array(membrane_trace.loc[:, 'Time_s'])
    potential_trace = np.array(membrane_trace.loc[:, 'Membrane_potential_mV'])
    
    current_trace = np.array(stim_trace.loc[:, 'Input_current_pA'])
    
    return time_trace , potential_trace, current_trace

def get_traces_Allen_CTD(original_cell_file_folder,cell_id,sweep_id,cell_sweep_table):
    ctc = CellTypesCache(
        manifest_file=str(str(original_cell_file_folder)+"manifest.json"))
    cell_id = int(cell_id)
    my_Cell_data = ctc.get_ephys_data(cell_id)
    index_range = my_Cell_data.get_sweep(int(sweep_id))["index_range"]
    sampling_rate = my_Cell_data.get_sweep(int(sweep_id))["sampling_rate"]
    current_trace = (my_Cell_data.get_sweep(int(sweep_id))["stimulus"][0:index_range[1]+1]) * 1e12  # to pA
    potential_trace = (my_Cell_data.get_sweep(int(sweep_id))["response"][0:index_range[1]+1]) * 1e3  # to mV
    potential_trace = np.array(potential_trace)
    current_trace = np.array(current_trace)
    stim_start_index = index_range[0]+next( x for x, val in enumerate(current_trace[index_range[0]:]) if val != 0)
    time_trace = np.arange(0, len(current_trace)) * (1.0 / sampling_rate)
    stim_start_time = time_trace[stim_start_index]
    stim_end_time = stim_start_time+1.
    
    return time_trace, potential_trace, current_trace, stim_start_time, stim_end_time


def get_traces_NVC(original_cell_file_folder,cell_id,sweep_id,cell_sweep_table):
    
    experiment,current_Protocol,current_trace = sweep_id.split("_")
    current_Protocol = str(current_Protocol)
    current_trace=int(current_trace)
    current_trace_df = cell_sweep_table.loc[(cell_sweep_table['Cell_id']==cell_id)& (cell_sweep_table['Sweep_id']==sweep_id),:]
    current_trace_df=current_trace_df.reset_index()
    species = current_trace_df.loc[0,"Species"]
    date = current_trace_df.loc[0,"Date"]
    cell = current_trace_df.loc[0,"Cell/Electrode"]


    Full_cell_file=original_cell_file_folder+species+'/'+date+'/'+cell+'/G clamp/'+current_Protocol+'.'+experiment
    
    
    float_array=np.fromfile(Full_cell_file,dtype='>f')
    separator_indices=np.argwhere(float_array==max(float_array))   
    header=float_array[:separator_indices[0][0]]

    metadata={"Iterations":header[0],
              "I_start_pulse_pA":header[1],
              "I_increment_pA":header[2],
              "I_pulse_width_ms":header[3],
              "I_delay_ms":header[4],
              "Sweep_duration_ms":header[5],
              "Sample_rate_Hz":header[6],
              "Cycle_time_ms":header[7],
              "Bridge_MOhms":header[8],
              "Cap_compensation":header[9]}
    
    if current_trace==len(separator_indices):
        previous_index=separator_indices[int(current_trace)-1][0]
        raw_traces=float_array[previous_index:]
    else:
        previous_index=separator_indices[int(current_trace)-1][0]
        index=separator_indices[current_trace][0]
        raw_traces=float_array[previous_index:index]
        

    traces=raw_traces[1:]
    potential_current_sep_index=int(len(traces)/2)
    potential_trace=traces[:potential_current_sep_index]
    
    if current_trace_df.shape[0]>1:
        raise ValueError(f'Cell {cell_id} has more than one row for sweep {sweep_id} in cell_sweep_csv')
    else:
        if current_trace_df.loc[0,"Synthesize Current"] == True:
            stim_start_time = float(current_trace_df.loc[0,"Stimulus Start (msec)"])
            stim_start_time*=1e-3
            
            stim_end_time = float(current_trace_df.loc[0,"Stimulus End (msec)"])
            stim_end_time*=1e-3
            
            stim_amp = float(current_trace_df.loc[0,"Stimulus Amplitude (pA)"])
            sampling_rate = int(current_trace_df.loc[0,"Sample Rate (Hz)"])
            
            sweep_duration = len(potential_trace)/sampling_rate
            time_trace=np.arange(0,sweep_duration,1/sampling_rate)
            
            current_trace=np.zeros(len(potential_trace))
            
            stim_start_index=np.argmin(abs(time_trace-stim_start_time))
            stim_end_index = np.argmin(abs(time_trace- stim_end_time))
            current_trace[stim_start_index:stim_end_index]+=stim_amp
        
        else:
            current_trace=traces[potential_current_sep_index:]
            
            sweep_duration=len(current_trace)/metadata["Sample_rate_Hz"]
            
            time_trace=np.arange(0,sweep_duration,(1/metadata["Sample_rate_Hz"]))
            
            stim_start_time=metadata["I_delay_ms"]*1e-3
            stim_end_time=stim_start_time+metadata["I_pulse_width_ms"]*1e-3
            
    return time_trace, potential_trace, current_trace, stim_start_time, stim_end_time
            
            
    