#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:36:59 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import plotnine as p9
import scipy
import h5py
import importlib
import re
import tqdm
import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import traceback
import json
import concurrent.futures


def get_upstroke_dowstroke_and_intervals(args_list):
    """
    This function detect the fist sweep with at least 5 spikes and gather the first spike upstroke and downstroke derivatives
    as well as the first sweep with at least 10 spikes and gather the first and 10th ISI

    Parameters
    ----------
    args_list : List
        list containting cell_id and config_line to open cell file.

    Returns
    -------
    new_line : pd.DataFrame
        

    """
    cell_id,config_line = args_list
    try:
        
        cell_dict = read_cell_file_h5(str(cell_id),config_line,["All"])
        Full_SF_table = cell_dict['Full_SF_table']
        sweep_info_table = cell_dict['Sweep_info_table']
        
        
        sweep_info_table = sweep_info_table.sort_values(by=['Stim_amp_pA'])
        sweep_list = np.array(sweep_info_table.loc[:,'Sweep'])
        first_upstroke_deriv, first_downstroke_deriv, first_interval, tenth_interval, first_sweep_with_five, first_sweep_with_ten = [np.nan]*6
        Obs='--'
        
        for current_sweep in sweep_list:
            SF_table = Full_SF_table.loc[current_sweep, "SF"]
            
            peak_table = SF_table.loc[SF_table['Feature']=='Peak',:]
            if peak_table.shape[0] ==0:
                continue
            elif peak_table.shape[0] >=5 : # Get forst sweep with at least 5 spikes, and get forst spike's Up/Downstroke ratio
                SF_table = SF_table.sort_values(by=['Time_s'])
                upstroke_table = SF_table.loc[SF_table['Feature']=="Upstroke",:]
                upstroke_table=upstroke_table.reset_index(drop=True)
                first_upstroke_deriv = upstroke_table.loc[0,"Potential_first_time_derivative_mV/s"]
                
                downstroke_table = SF_table.loc[SF_table['Feature']=="Downstroke",:]
                downstroke_table=downstroke_table.reset_index(drop=True)
                first_downstroke_deriv = downstroke_table.loc[0,"Potential_first_time_derivative_mV/s"]
                
                first_sweep_with_five = current_sweep
                break
                
        for current_sweep in sweep_list:
            SF_table = Full_SF_table.loc[current_sweep, "SF"]
            
            peak_table = SF_table.loc[SF_table['Feature']=='Peak',:]
            
            if peak_table.shape[0] ==0:
                continue
            
            elif peak_table.shape[0] >=10 : #Get first sweep with at least 10 spikes and get forst and tenth ISI
                SF_table = SF_table.sort_values(by=['Time_s'])
                peak_table = peak_table.reset_index(drop=True)
                first_interval = peak_table.loc[1,'Time_s'] - peak_table.loc[0,'Time_s']
                
                tenth_interval = peak_table.loc[9,'Time_s'] - peak_table.loc[8,'Time_s']
                
                first_sweep_with_ten = current_sweep
                break
    except:
        error= traceback.format_exc()
        first_upstroke_deriv, first_downstroke_deriv, first_interval, tenth_interval, first_sweep_with_five, first_sweep_with_ten = [np.nan]*6
        Obs = error
    new_line = pd.DataFrame([str(cell_id), Obs, first_sweep_with_five, first_upstroke_deriv, first_downstroke_deriv, first_sweep_with_ten,first_interval, tenth_interval]).T
    new_line.columns = ['Cell_id','Obs','First_sweep_with_five_spikes', 'First_upstroke_Potential_derivative','First_downstroke_Potential_derivative', 'First_sweep_with_ten_spikes',"First_spike_interval", "Tenth_spike_interval"]
    
   
    
    return new_line

# LG change
def first(x):
    """Return first element of array/list/Series safely."""
    if hasattr(x, "iloc"):      # Pandas Series or Index
        return x.iloc[0]
    return x[0]                 # NumPy array or list


def last(x):
    """Return last element of array/list/Series safely."""
    if hasattr(x, "iloc"):      # Pandas Series or Index
        return x.iloc[-1]
    return x[-1]                # NumPy array or list

def time_slice_of_trace(time_trace, data_trace, start_time=None, end_time=None):
    '''
    Return slices of data_trace and time_trace according to the specified time window.
    Parameters
    ----------
    time_trace : num array
    data_trace : num array
    start_time : num
        DESCRIPTION. The default is None, thus window start at beginning of traces
        Otherwise, window start in units of time_trace
    end_time : num
        DESCRIPTION. The default is None, thus window end at end of traces
        Otherwise, window end in units of time_trace

    Returns
    -------
    time_trace_slice, data_trace_slice
        Windowed copies of original traces
    '''
    if start_time is None:
        start_time = first(time_trace)
    if end_time is None:
        end_time = last(time_trace)

    # Build mask for the desired time interval
    mask = (time_trace >= start_time) & (time_trace <= end_time)

    # Apply mask to both arrays
    return time_trace[mask], data_trace[mask]


def get_filtered_TVC_table(
        original_cell_full_TVC_table,sweep,do_filter=True,filter=5.,do_plot=False):
    '''
    From the cell Full TVC table, get the sweep related TVC table, and if required, with filtered Potential and Current values
    

    Parameters
    ----------
    original_cell_full_TVC_table : pd.DataFrame
        2 columns DataFrame, containing in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time (sec), Current and Potential Traces
    sweep : str
        Sweep id.
    do_filter : Bool, optional
        Wether or not to Filter Membrane voltage and Current traces. The default is True.
    do_plot : Bool, optional
       The default is False.

    Returns
    -------
    TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.

    '''
    
    cell_full_TVC_table = original_cell_full_TVC_table.copy()
    TVC_table=cell_full_TVC_table.loc[str(sweep),'TVC'].copy()

    if do_filter:
        #Filter membrane potential and input current traces
        
        TVC_table['Membrane_potential_mV']=np.array(filter_trace(TVC_table['Membrane_potential_mV'],
                                                                            TVC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
        
        TVC_table['Input_current_pA']=np.array(filter_trace(TVC_table['Input_current_pA'],
                                                                            TVC_table['Time_s'],
                                                                            filter=filter,
                                                                            do_plot=do_plot))
    
    
    #Get first and second time derivative of membrane potential trace
    first_derivative=get_derivative(np.array(TVC_table['Membrane_potential_mV']),np.array(TVC_table['Time_s']))
    second_derivative=get_derivative(first_derivative,np.array(TVC_table['Time_s']))

    
    TVC_table['Potential_first_time_derivative_mV/s'] = first_derivative
    TVC_table["Potential_second_time_derivative_mV/s/s"] = second_derivative
    TVC_table=TVC_table.astype({'Time_s':float,
                               'Membrane_potential_mV':float,
                               'Input_current_pA':float,
                               'Potential_first_time_derivative_mV/s':float,
                               'Potential_second_time_derivative_mV/s/s':float}) 
    
    
    if do_plot:
        voltage_plot = p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Membrane_potential_mV'))+p9.geom_line()
        voltage_plot+=p9.ggtitle(str('Sweep:'+str(sweep)+'Membrane_Potential_mV'))
        print(voltage_plot)
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y='Input_current_pA'))+p9.geom_line()
        current_plot+=p9.ggtitle(str('Sweep:'+str(sweep)+'Input_current_pA'))
        print(current_plot)


    return TVC_table

def subsample_TVC_table(original_TVC_table, subsampling_freq):
    """
    Subsamples a time-varying signal table (TVC table) to a specified frequency.

    This function reduces the sampling frequency of the provided time-varying 
    signal table by selecting every nth row based on the subsampling factor 
    determined by the original and target frequencies. If the original sampling 
    frequency is less than or equal to the target frequency, no subsampling is performed.

    Parameters
    ----------
    original_TVC_table : pandas.DataFrame
        A DataFrame containing the original time-varying signal data. 
        Must include a 'Time_s' column representing time in seconds.
    subsampling_freq : int
        The desired subsampling frequency in Hz.

    Returns
    -------
    subsampled_TVC_table : pandas.DataFrame
        A new DataFrame containing the subsampled data. If no subsampling is 
        performed, the function returns the original table.

    Notes
    -----
    - The original sampling frequency is computed as the reciprocal of the time 
      step (`delta_t`) between consecutive rows in the 'Time_s' column.
    - If the original sampling frequency is less than or equal to the target frequency, 
      the function returns the original table without changes and prints a message.
    - If the original sampling frequency is not an integer multiple of the target frequency, 
      the function performs approximate subsampling and informs the user of the resulting 
      actual frequency.
    - The function resets the index of the resulting subsampled DataFrame to maintain consistency.

    Warnings
    --------
    - If the original sampling frequency divided by the target frequency is not an integer, 
      the actual resulting frequency may differ slightly from the requested subsampling frequency.


    """
    subsampled_TVC_table = original_TVC_table.copy()
    time_trace = subsampled_TVC_table.loc[:,'Time_s']
    delta_t = time_trace[1] - time_trace[0]
    sample_freq = round(1. / delta_t)
    
    if sample_freq <= subsampling_freq:
        print (f'Original sampling frequency ( {sample_freq}Hz) lower or equal to subsampling frequency ({subsampling_freq}Hz). No subsampling performed')
        return original_TVC_table
    else:
        subsample_factor = sample_freq // subsampling_freq
        
        if sample_freq % subsampling_freq !=0:
            resulting_freq = sample_freq//subsample_factor
            not_integer_subsample_factor = sample_freq / subsampling_freq
            print(f'Subsampling process not exact. sample_freq / subsampling_freq not integer ({not_integer_subsample_factor}). The actual resulting frequency will be {resulting_freq}Hz. ')

        subsampled_TVC_table = subsampled_TVC_table.iloc[::subsample_factor,:]
        subsampled_TVC_table = subsampled_TVC_table.reset_index(drop=True)
        return subsampled_TVC_table

#  LG change
def filter_trace(value_trace, time_trace, filter=5., filter_order = 2, zero_phase = False,do_plot=False,
                 start_time_sec=None,end_time_sec=None):
    '''
    Apply a Butterworth Low-pass filters time-varying signal.

    Parameters
    ----------
    value_trace : np.array
        array of time_varying signal to filter.
        
    time_trace : np.array
        Array of time in second.
        
    filter : Float, optional
        Cut-off frequency in kHz . The default is 5.
        
    filter_order : int, optional
        Order of the filter to apply. The default is 2.
        
    zero_phase : bool, optional
        Shoudl the fileter be a zero_phase filter. The default is False
        
    do_plot : Boolean, optional
        Do Plot. The default is False.

    start_time_sec : Num, optional
          The default is None, thus start at beginning of trace.

    end_time_sec : Num, optional
          The default is None, thus end at end of trace.
    Raises
    ------
    ValueError
        Raise error if the sampling frequency of the time varying signal is lower thant the Nyquist frequency.

    Returns
    -------
    filtered_signal : np.array
        Value trace filtered.

    '''

    time_trace_slice,value_trace_slice=time_slice_of_trace(time_trace, value_trace,
                                                   start_time=start_time_sec, end_time=end_time_sec)


    delta_t = time_trace_slice[1] - time_trace_slice[0]
    sample_freq = 1. / delta_t

    filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency

    if filt_coeff < 0 or filt_coeff >= 1:
        raise ValueError("Butterworth coeff ({:f}) is outside of valid range [0,1]; cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
    # Design a 4th order low-pass Butterworth filter
    b, a = scipy.signal.butter(filter_order, filt_coeff, btype='low')
    
    if zero_phase == True:
        # Apply the filter to the signal using filtfilt for zero-phase filtering
        filtered_signal = scipy.signal.filtfilt(b, a, value_trace_slice)
    else:
        zi = scipy.signal.lfilter_zi(b, a)
    
        filtered_signal =  scipy.signal.lfilter(b, a, value_trace_slice,zi=zi*value_trace_slice[0], axis=0)[0]
   
    
    if do_plot:
        signal_df = pd.DataFrame({'Time_s':np.array(time_trace_slice),
                                  'Values':np.array(value_trace_slice)})
        filtered_df = pd.DataFrame({'Time_s':np.array(time_trace_slice),
                                  'Values':np.array(filtered_signal)})
        
        signal_df ['Trace']='Original_Trace'
        filtered_df ['Trace']='Filtered_Trace'
        

        signal_df = pd.concat([signal_df, filtered_df], ignore_index = True)
        filter_plot = p9.ggplot(signal_df, p9.aes(x='Time_s',y='Values',color='Trace',group='Trace'))+p9.geom_line()+p9.xlim(2.1,2.4)
        print(filter_plot)
        
    return filtered_signal

# LG change
def get_derivative(value_trace, time_trace, start_time_sec=None,end_time_sec=None):
    '''
    Get time derivative of a signal trace

    Parameters
    ----------
    value_trace : np.array
        Array of time_varying signal.
        
    time_trace : np.array
        Array of time in second.

    start_time_sec : Num, optional
          The default is None, thus start at beginning of trace.

    end_time_sec : Num, optional
          The default is None, thus end at end of trace.

    Returns
    -------
    dvdt : np.array
        Time derivative of the time varying signal trace.

    '''
    time_trace_slice,value_trace_slice=time_slice_of_trace(time_trace, value_trace,
                                                   start_time=start_time_sec, end_time=end_time_sec)

    trace_derivative = np.gradient(value_trace_slice, time_trace_slice)
    trace_derivative *= 1e-3 # in mV/s = mV/ms
    

    return trace_derivative

def write_cell_file_h5(cell_file_path,
                       saving_dict,
                       overwrite=False,
                       selection=['All']):
    '''
    Create a hdf5 file for a given cell file path (or overwrite existing one if required) and store results of the analysis.

    Parameters
    ----------
    cell_file_path : str
        File path to which store cell file.
        
    original_Full_SF_dict : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    original_cell_sweep_QC : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        
    original_cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.
        
    original_cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
        
    original_Metadata_table : pd.DataFrame
        DataFrame containing cell metadata.
    
    original_processing_table : pd.DataFrame
        DataFrame containing for each part of the analysis the processing time 
        as well as one row per Error or Warning encountered during the analysis
        
    overwrite : Bool, optional
        Wether to overwrite already existing information, if for a given cell_id, a cell file already exists in saving_folder_path
        The default is False.



    '''


    file = h5py.File(cell_file_path, "a")
    
    if "All" in selection:
        selection = ["Metadata","Sweep analysis", "Spike analysis", "Firing analysis"]
    
    if "Metadata" in saving_dict.keys():
        original_Metadata_table = saving_dict["Metadata"]
        
        if 'Metadata' in file.keys() and overwrite == True:
            del file["Metadata"]
            if isinstance(original_Metadata_table,dict) == True:
                Metadata_group = file.create_group('Metadata')
                for elt in original_Metadata_table.keys():
        
                    Metadata_group.create_dataset(
                        str(elt), data=original_Metadata_table[elt])
        elif 'Metadata' not in file.keys():
            if isinstance(original_Metadata_table,dict) == True:
                Metadata_group = file.create_group('Metadata')
                for elt in original_Metadata_table.keys():
        
                    Metadata_group.create_dataset(
                        str(elt), data=original_Metadata_table[elt])
                
            
    if "Spike analysis" in saving_dict.keys():
        original_Full_SF_dict = saving_dict["Spike analysis"]
        
        if 'Spike analysis' in file.keys() and overwrite == True:
            del file["Spike analysis"]
            if isinstance(original_Full_SF_dict,pd.DataFrame) == True:
                SF_group = file.create_group("Spike analysis")
                
                sweep_list = np.array(original_Full_SF_dict['Sweep'])
                for current_sweep in sweep_list:
                    
                    current_SF_dict = original_Full_SF_dict.loc[current_sweep, "SF_dict"]
                    current_SF_group = SF_group.create_group(str(current_sweep))
                    for elt in current_SF_dict.keys():
        
                        if len(current_SF_dict[elt]) != 0:
                            current_SF_group.create_dataset(
                                str(elt), data=current_SF_dict[elt])
    
        elif 'Spike analysis' not in file.keys():
            if isinstance(original_Full_SF_dict,pd.DataFrame) == True:
                sweep_list = np.array(original_Full_SF_dict['Sweep'])
                SF_group = file.create_group("Spike analysis")
                
                for current_sweep in sweep_list:
        
                    current_SF_dict = original_Full_SF_dict.loc[current_sweep, "SF_dict"]
                    current_SF_group = SF_group.create_group(str(current_sweep))
                    for elt in current_SF_dict.keys():
        
                        if len(current_SF_dict[elt]) != 0:
                            current_SF_group.create_dataset(
                                str(elt), data=current_SF_dict[elt])
               
    if "Sweep analysis" in saving_dict.keys():
        sweep_analysis_dict = saving_dict["Sweep analysis"]
        original_cell_sweep_info_table = sweep_analysis_dict['Sweep info']
        original_cell_sweep_QC = sweep_analysis_dict['Sweep QC']
        
        if 'Sweep analysis' in file.keys() and overwrite == True:
            
            del file["Sweep analysis"]
            if isinstance(original_cell_sweep_info_table,pd.DataFrame) == True:
                cell_sweep_info_table_group = file.create_group('Sweep analysis')
                sweep_list=np.array(original_cell_sweep_info_table['Sweep'])
        
                for elt in np.array(original_cell_sweep_info_table.columns):
                    cell_sweep_info_table_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_info_table[elt]))
                   
    
        elif 'Sweep analysis' not in file.keys():
            if isinstance(original_cell_sweep_info_table,pd.DataFrame) == True:
                cell_sweep_info_table_group = file.create_group('Sweep analysis')
                for elt in np.array(original_cell_sweep_info_table.columns):
                    cell_sweep_info_table_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_info_table[elt]))
                    
            
        
        if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
            convert_dict = {}
            for col in original_cell_sweep_QC.columns:
                if col == "Sweep":
                    convert_dict[col]=str
                else:
                    convert_dict[col]=bool
          
        
            original_cell_sweep_QC = original_cell_sweep_QC.astype(convert_dict)
    
        if 'Sweep_QC' in file.keys() and overwrite == True:
            del file['Sweep_QC']
            if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
                cell_sweep_QC_group = file.create_group("Sweep_QC")
                for elt in np.array(original_cell_sweep_QC.columns):
                    
                    cell_sweep_QC_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_QC[elt]))
        elif 'Sweep_QC' not in file.keys():
            if isinstance(original_cell_sweep_QC,pd.DataFrame) == True:
                cell_sweep_QC_group = file.create_group("Sweep_QC")
                for elt in np.array(original_cell_sweep_QC.columns):
                    
                    cell_sweep_QC_group.create_dataset(
                        elt, data=np.array(original_cell_sweep_QC[elt]))
            
    
    if 'Firing analysis' in saving_dict.keys():
        Firing_analysis_dict = saving_dict["Firing analysis"]
        original_cell_feature_table = Firing_analysis_dict['Cell_feature']
        original_cell_fit_table = Firing_analysis_dict['Cell_fit']
        original_cell_adaptation_table = Firing_analysis_dict['Cell_Adaptation']
        if 'Cell_Feature' in file.keys() and overwrite == True:
            del file["Cell_Feature"]
            if isinstance(original_cell_feature_table,pd.DataFrame) == True:
                cell_feature_group = file.create_group('Cell_Feature')
                for elt in np.array(original_cell_feature_table.columns):
                    cell_feature_group.create_dataset(
                        elt, data=np.array(original_cell_feature_table[elt]))
            
            del file['Cell_Fit']
            if isinstance(original_cell_fit_table,pd.DataFrame) == True:
                cell_fit_group = file.create_group('Cell_Fit')
                for elt in np.array(original_cell_fit_table.columns):
                    cell_fit_group.create_dataset(
                        elt, data=np.array(original_cell_fit_table[elt]))
                    
            del file['Cell_Adaptation']
            if isinstance(original_cell_adaptation_table,pd.DataFrame) == True:
                cell_adaptation_group = file.create_group('Cell_Adaptation')
                for elt in np.array(original_cell_adaptation_table.columns):
                    if elt in ['Obs','Feature','Measure']:
                        original_cell_adaptation_table = original_cell_adaptation_table.astype({elt:str})
                    else:
                        original_cell_adaptation_table = original_cell_adaptation_table.astype({elt:float})
                    cell_adaptation_group.create_dataset(
                        elt, data=np.array(original_cell_adaptation_table[elt]))
                
    
            
    
        elif 'Cell_Feature' not in file.keys():
            if isinstance(original_cell_feature_table,pd.DataFrame) == True:
                cell_feature_group = file.create_group('Cell_Feature')
                for elt in np.array(original_cell_feature_table.columns):
                    cell_feature_group.create_dataset(
                        elt, data=np.array(original_cell_feature_table[elt]))
           
    
            # Store Cell fit table
            if isinstance(original_cell_fit_table,pd.DataFrame) == True:
                cell_fit_group = file.create_group('Cell_Fit')
                
        
                for elt in np.array(original_cell_fit_table.columns):
                    
                    cell_fit_group.create_dataset(
                        elt, data=np.array(original_cell_fit_table[elt]))
                    
            if isinstance(original_cell_adaptation_table,pd.DataFrame) == True:
                cell_adaptation_group = file.create_group('Cell_Adaptation')
                for elt in np.array(original_cell_adaptation_table.columns):
                    if elt in ['Obs','Feature','Measure']:
                        original_cell_adaptation_table = original_cell_adaptation_table.astype({elt:str})
                    else:
                        original_cell_adaptation_table = original_cell_adaptation_table.astype({elt:float})
                    cell_adaptation_group.create_dataset(
                        elt, data=np.array(original_cell_adaptation_table[elt]))
                
    #store processing report
    if "Processing report" in saving_dict.keys():
        original_processing_table = saving_dict["Processing report"]
        if 'Processing_report' in file.keys() and overwrite == True:
            del file["Processing_report"]
            if isinstance(original_processing_table,pd.DataFrame) == True:
                processing_report_group = file.create_group('Processing_report')
                for elt in np.array(original_processing_table.columns):
                    processing_report_group.create_dataset(
                        elt, data=np.array(original_processing_table[elt]))
                
        elif 'Processing_report' not in file.keys():
            if isinstance(original_processing_table,pd.DataFrame) == True:
                processing_report_group = file.create_group('Processing_report')
                for elt in np.array(original_processing_table.columns):
                    processing_report_group.create_dataset(
                        elt, data=np.array(original_processing_table[elt]))
           
                
    

    file.close()
    
# LG Change name
def open_json_config_file(config_file):
    '''
    Open JSON configuration file and return a DataFrame

    Parameters
    ----------
    config_file : str
        Path to JSON configuration file (ending .json).

    Returns
    -------
    config_df : pd.DataFrame
        DataFrame containing the information of the JSON configuration file.

    '''
    # Read and parse JSON
    
    with open(config_file, "r", encoding="utf-8-sig") as f:
        config_json = json.load(f) 
    
    # Turn dict into DataFrame
    colnames = list(config_json.keys())[:-1]
    colnames += list(pd.DataFrame(config_json["DB_parameters"][0], index=[0]).columns)

    config_df = pd.DataFrame(columns=colnames)
    for db in config_json["DB_parameters"]:
        new_line = pd.DataFrame(db, index=[0])
        new_line["path_to_saving_file"] = config_json["path_to_saving_file"]
        new_line["path_to_QC_file"] = config_json["path_to_QC_file"]
        config_df = pd.concat([config_df, new_line], axis=0, ignore_index=True)

    return config_df

def create_TVC(time_trace,voltage_trace,current_trace):
    '''
    Create table containing Time, Voltage, Current traces

    Parameters
    ----------
    time_trace : np.array
        Time points array in s.
        
    voltage_trace : np.array
        Array of membrane voltage recording in mV.
        
    current_trace : np.array
        Array of input current in pA.

    Raises
    ------
    ValueError
        All array must be of the same length, if not raises ValueError.

    Returns
    -------
    TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.

    '''
    length=len(time_trace)
    
    if length!=len(voltage_trace) or length!=len(current_trace):

        raise ValueError('All lists must be of equal length. Current lengths: Time {}, Potential {}, Current {}'.format(len(time_trace),len(voltage_trace),len(current_trace)))
    
    
    TVC_table=pd.DataFrame({'Time_s':time_trace,
                               'Membrane_potential_mV':voltage_trace,
                               'Input_current_pA':current_trace},
                           dtype=np.float64) 
    return TVC_table

def estimate_trace_stim_limits(TVC_table_original,stimulus_duration,do_plot=False):
    '''
    Estimate for a given trace the start time and end of the stimulus, by performing autocorrelation

    Parameters
    ----------
    TVC_table_original : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stimulus_duration : float
        Duration of the stimulus in second.
        
    do_plot : Bool, optional
        Do plot. The default is False.

    Returns
    -------
    best_stim_start : float
        Start time of the stimulus in second.
        
    best_stim_end : float
        End time of the stimulus in second.

    '''
    
    TVC_table=TVC_table_original.copy()
    
    TVC_table['Input_current_pA']=np.array(filter_trace(TVC_table['Input_current_pA'],
                                                                        TVC_table['Time_s'],
                                                                        filter=5,
                                                                        do_plot=False))
    current_derivative = get_derivative(np.array(TVC_table['Input_current_pA']),
                                        np.array(TVC_table['Time_s']))

    TVC_table["Filtered_Stimulus_trace_derivative_pA/ms"]=np.array(current_derivative)
    # remove last 50ms of signal (potential step)
    limit = TVC_table.shape[0] - \
        int(0.05/(TVC_table.iloc[1, 0]-TVC_table.iloc[0, 0]))

    TVC_table.loc[limit:,
                   "Filtered_Stimulus_trace_derivative_pA/ms"] = np.nan
    
    TVC_table = get_autocorrelation(TVC_table, stimulus_duration, do_plot=do_plot)
    
    best_stim_start = TVC_table[TVC_table['Autocorrelation'] == np.nanmin(
        TVC_table['Autocorrelation'])].iloc[0, 0]

    best_stim_end = best_stim_start+stimulus_duration
    
    return best_stim_start, best_stim_end

def get_autocorrelation(table, time_shift, do_plot=False):
    '''
    Compute autocorrelation at each time point for a stimulus trace table

    Parameters
    ----------
    table : DataFrame
        Stimulus trace table, 3 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
    time_shift : float
        Time shift in s .
    do_plot : Bool, optional
        The default is False.

    Returns
    -------
    table : DataFrame
        Stimulus trace table, 6 columns
        First column = 'Time_s'
        Second column = 'Input_current_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
        4th column = "Filtered_Stimulus_trace_derivative_pA/ms"
        5th column = "Shifted_trace" --> Filtered stiumulus trace derivatove shifted by 'time_shift'
        6th column = 'Autocorrelation --> Autocorrelation between 4th and 5th column
    '''

    shift = int(time_shift/(table.iloc[1, 0]-table.iloc[0, 0]))

    table["Shifted_trace"] = table['Filtered_Stimulus_trace_derivative_pA/ms'].shift(
        -shift)

    table['Autocorrelation'] = table['Filtered_Stimulus_trace_derivative_pA/ms'] * \
        table["Shifted_trace"]

    if do_plot == True:

        myplot = p9.ggplot(table, p9.aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Filtered_Stimulus_trace_derivative_pA/ms']))+p9.geom_line(
            color='blue')+p9.geom_line(table, p9.aes(x=table.loc[:, 'Time_s'], y=table.loc[:, 'Autocorrelation']), color='red')

        myplot += p9.xlab(str("Time_s; Time_shift="+str(time_shift)))
        print(myplot)

    return table

def find_time_index(t, t_0):
    """ 
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """

    assert np.nanmin([t[0],t[-1]]) <= t_0 <= np.nanmax([t[0],t[-1]]), "Given time ({:f}) is outside of time range ({:f}, {:f})".format(t_0, t[0], t[-1])

    idx = np.argmin(abs(t - t_0))
    return idx

def read_cell_file_h5(cell_id, config_line, selection=['All']):
    '''
    Open cell h5 and returns the different elements

    Parameters
    ----------
    cell_file : str
        Path to the cell HDF5 file (ending .h5).
        
    config_json_file : str
        Path to JSON configuration file (ending .json).
        
        
    selection : List, optional
        Indicates which elements from the files to return (Elements must correspond to h5 groups). The default is ['All'].

    Returns
    -------
    Full_TVC_table : pd.DataFrame
        2 columns DataFrame, containing in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces.
        
    Full_SF_dict_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)
    
    Full_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    Metadata_table : pd.DataFrame
        DataFrame containing cell metadata.
        
    sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    Sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit
        
    cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.
        
    cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
        
    Processing_report_df : pd.DataFrame
        DataFrame containing for each part of the analysis the processing time 
        as well as one row per Error or Warning encountered during the analysis

    '''
    
    cell_dict = dict()
    if 'Metadata' in selection and len(selection)==1:
        ## Metadata ##
        ## Specific case for Cell vizualization app, cell_id represent full path to cell file
        current_file = h5py.File(cell_id, 'r')
        if 'Metadata' not in current_file.keys():
            print('File does not contains Metadata group')
            return pd.DataFrame()
        Metadata_group = current_file['Metadata']
        Metadata_dict = {}
        for data in Metadata_group.keys():
            if type(Metadata_group[data][()]) == bytes:
                print(Metadata_group[data][()])
                Metadata_dict[data] = Metadata_group[data][()].decode('ascii')
            else:
                Metadata_dict[data] = Metadata_group[data][()]
        Metadata_table = pd.DataFrame(Metadata_dict, index=[0])
        return Metadata_table
    
    saving_folder_path = config_line['path_to_saving_file'].values[0]

    cell_file_path = str(saving_folder_path+'Cell_'+str(cell_id)+'.h5')

    current_file = h5py.File(cell_file_path, 'r')

    Full_TVC_table = pd.DataFrame()
    Full_SF_dict_table = pd.DataFrame()
    Full_SF_table = pd.DataFrame()
    Metadata_table = pd.DataFrame()
    sweep_info_table = pd.DataFrame()
    Sweep_QC_table = pd.DataFrame()
    cell_fit_table = pd.DataFrame()
    cell_adaptation_table = pd.DataFrame()
    cell_feature_table = pd.DataFrame()
    Processing_report_df = pd.DataFrame()
    
    if 'All' in selection:
        selection = ['TVC_SF', 'Sweep analysis','Sweep QC', 'Metadata', 'Firing analysis','Processing_report']
    
    
    
    if 'Metadata' in selection and 'Metadata' in current_file.keys():
        ## Metadata ##

        Metadata_group = current_file['Metadata']
        Metadata_dict = {}
        for data in Metadata_group.keys():
            if type(Metadata_group[data][()]) == bytes:
                Metadata_dict[data] = Metadata_group[data][()].decode('ascii')
            else:
                Metadata_dict[data] = Metadata_group[data][()]
        Metadata_table = pd.DataFrame(Metadata_dict, index=[0])
        
    
    elif 'Metadata' in selection and 'Metadata' not in current_file.keys():
        print('File does not contains Metadata group')

    if 'TVC_SF' in selection and 'Spike analysis' in current_file.keys():
        
        SF_group = current_file['Spike analysis']

        sweep_list = list(SF_group.keys())

        
        
        if isinstance(config_line,pd.DataFrame) == True:
            config_line = config_line.reset_index()
            config_line = config_line.to_dict('index')
            config_line = config_line[0]

        path_to_python_folder = config_line['path_to_db_script_folder']
        python_file = config_line['python_file_name']
        module=python_file.replace('.py',"")
        full_path_to_python_script=str(path_to_python_folder+python_file)
     
        spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)

        DB_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(DB_module)
        
        
        
        
        db_original_file_directory = config_line['original_file_directory']
        

        
        db_cell_sweep_file = pd.read_csv(config_line['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
        
        
        Full_SF_dict_table = pd.DataFrame(columns=['Sweep', 'SF_dict'])
        
        
        args_list = [module,
                      full_path_to_python_script,
                      config_line["db_function_name"],
                      db_original_file_directory,
                      cell_id,
                      sweep_list,
                      db_cell_sweep_file,
                      config_line["stimulus_time_provided"],
                      config_line["db_stimulus_duration"]]
        
        
        
        Full_TVC_table = sw_an.get_TVC_table(args_list)[0]
        
            
        for current_sweep in sweep_list:

            current_SF_table = SF_group[str(current_sweep)]
            SF_dict = {}
    
            for feature in current_SF_table.keys():
                SF_dict[feature] = np.array(current_SF_table[feature])
    
            new_line_SF = pd.DataFrame([str(current_sweep), SF_dict]).T
            new_line_SF.columns=['Sweep', 'SF_dict']
            Full_SF_dict_table = pd.concat([Full_SF_dict_table,
                        new_line_SF], ignore_index=True)
           
        Full_TVC_table.index = Full_TVC_table.loc[:,"Sweep"]
        Full_TVC_table.index = Full_TVC_table.index.astype(str)
        
        Full_SF_dict_table.index = Full_SF_dict_table["Sweep"]
        Full_SF_dict_table.index = Full_SF_dict_table.index.astype(str)
        
        Sweep_analysis_group = current_file['Sweep analysis']
        Sweep_analysis_dict = {}
        for data in Sweep_analysis_group.keys():
            Sweep_analysis_dict[data] = Sweep_analysis_group[data][()]
        Sweep_analysis_dict['Sweep']=Sweep_analysis_dict['Sweep'].astype(str)
        sweep_info_table = pd.DataFrame(
            Sweep_analysis_dict, index=Sweep_analysis_dict['Sweep'])

        Full_SF_table = sp_an.create_Full_SF_table(
            Full_TVC_table, Full_SF_dict_table.copy(), sweep_info_table.copy())
        
        

    if 'Sweep analysis' in selection and 'Sweep analysis' in current_file.keys():

        ## Sweep_analysis_table ##

        Sweep_analysis_group = current_file['Sweep analysis']
        Sweep_analysis_dict = {}
        for data in Sweep_analysis_group.keys():
            Sweep_analysis_dict[data] = Sweep_analysis_group[data][()]
        Sweep_analysis_dict['Sweep']=Sweep_analysis_dict['Sweep'].astype(str)
        sweep_info_table = pd.DataFrame(
            Sweep_analysis_dict, index=Sweep_analysis_dict['Sweep'])
        
        if 'Sweep_QC' in current_file.keys():
            Sweep_QC_group = current_file['Sweep_QC']
            Sweep_QC_dict = {}
            for data in Sweep_QC_group.keys():
                Sweep_QC_dict[data] = Sweep_QC_group[data][()]
            Sweep_QC_dict['Sweep']=Sweep_QC_dict['Sweep'].astype(str)
            Sweep_QC_table = pd.DataFrame(
                Sweep_QC_dict, index=Sweep_QC_dict['Sweep'])
            for col in Sweep_QC_table.columns:
                if col != 'Sweep':
                    Sweep_QC_table=Sweep_QC_table.astype({str(col):"bool"})
        else:
            print('File does not contain Sweep_QC group')

    elif 'Sweep analysis' in selection and 'Sweep analysis' not in current_file.keys():
        print('File does not contain Sweep analysis group')
        
    

    if 'Firing analysis' in selection and 'Cell_Fit' in current_file.keys():
        ## Cell_fit ##

        Cell_fit_group = current_file['Cell_Fit']
        Cell_fit_dict = {}
        for data in Cell_fit_group.keys():
            if type(Cell_fit_group[data][(0)]) == bytes:
                Cell_fit_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_fit_group[data][()]], dtype='str')
            else:
                Cell_fit_dict[data] = Cell_fit_group[data][()]

        cell_fit_table = pd.DataFrame(
            Cell_fit_dict)
        cell_fit_table.index = cell_fit_table.index.astype(int)

        ## Cell_feature ##

        Cell_feature_group = current_file['Cell_Feature']
        cell_feature_dict={}
        for data in Cell_feature_group.keys():
            if type(Cell_feature_group[data][(0)]) == bytes:
                cell_feature_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_feature_group[data][()]], dtype='str')
            else:
                cell_feature_dict[data] = Cell_feature_group[data][()]

        cell_feature_table = pd.DataFrame(
            cell_feature_dict)
        cell_feature_table.index = cell_feature_table.index.astype(int)
        
        ## Cell_Adaptation ##
        
        Cell_adaptation_group = current_file['Cell_Adaptation']
        cell_adaptation_dict={}
        for data in Cell_adaptation_group.keys():
            if type(Cell_adaptation_group[data][(0)]) == bytes:
                cell_adaptation_dict[data] = np.array(
                    [x.decode('ascii') for x in Cell_adaptation_group[data][()]], dtype='str')
            else:
                cell_adaptation_dict[data] = Cell_adaptation_group[data][()]

        cell_adaptation_table = pd.DataFrame(
            cell_adaptation_dict)
        cell_adaptation_table.index = cell_adaptation_table.index.astype(int)

        
        
    elif 'Firing analysis' in selection and 'Cell_Fit' not in current_file.keys():
        print('File does not contains Cell_Fit group')
        
        
    if 'Processing_report' in selection and 'Processing_report' in current_file.keys():
        Processing_report_group = current_file['Processing_report']
        Processing_report_dict={} 
        for data in Processing_report_group.keys():
            if type(Processing_report_group[data][(0)]) == bytes:
                Processing_report_dict[data] = np.array(
                    [x.decode('ascii') for x in Processing_report_group[data][()]], dtype='str')
        Processing_report_df = pd.DataFrame(Processing_report_dict)

    current_file.close()
    
    cell_dict['Full_TVC_table'] = Full_TVC_table
    cell_dict["Full_SF_table"] = Full_SF_table
    cell_dict["Full_SF_dict_table"] = Full_SF_dict_table
    cell_dict['Sweep_info_table'] = sweep_info_table
    cell_dict['Sweep_QC_table'] = Sweep_QC_table
    cell_dict['Metadata_table'] = Metadata_table
    cell_dict['Cell_fit_table'] = cell_fit_table
    cell_dict['Cell_feature_table'] = cell_feature_table
    cell_dict['Cell_Adaptation'] = cell_adaptation_table
    cell_dict['Processing_table'] = Processing_report_df
    return cell_dict
    

def compute_cell_input_resistance(cell_dict):
    '''
    Define the method to compute the cell's input resistance from sweep-based measurements
    Cell's input Resistance is defined as the mean of sweep based inut resistance for all sweeps that passed the QC

    Parameters
    ----------
    cell_dict 
    
    Returns
    -------
    Mean_IR (in GOhms)
    SD_IR (in GOhms)

    '''
    
    sweep_info_table = cell_dict['Sweep_info_table']
    sweep_QC_table = cell_dict['Sweep_QC_table']
    sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
    sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
    sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
    
    Mean_IR = np.nanmean(sweep_info_QC_table.loc[:,'Input_Resistance_GOhms'])
    SD_IR = np.nanstd(sweep_info_QC_table.loc[:,'Input_Resistance_GOhms'])
    
    
    return Mean_IR, SD_IR

def compute_cell_time_constant(cell_dict):
    '''
    Define the method to compute the cell's time constant  from sweep-based measurements
    Cell's time constant is defined as the mean of sweep-based time constant of sweeps which validated the quality criteria analysis 

    Parameters
    ----------
    cell_dict 
    
    Returns
    -------
    Time_cst_mean, Time_cst_SD : floats
        Mean time constant and standard deviation of sweep based time constants

    '''
    sweep_info_table = cell_dict['Sweep_info_table']
    sweep_QC_table = cell_dict['Sweep_QC_table']
    sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
    sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
    sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
    Time_cst_mean = np.nanmean(sub_sweep_info_QC_table['Time_constant_ms'])
    Time_cst_SD = np.nanstd(sub_sweep_info_QC_table['Time_constant_ms'])
    
    return Time_cst_mean, Time_cst_SD
    
def compute_cell_resting_potential(cell_dict):
    '''
    Define the method to compute the cell's Resting potential  from sweep-based measurements
    Cell's Resting potential is defined as the mean of sweep-based Resting potential of sweeps which validated the quality criteria analysis 

    Parameters
    ----------
    cell_dict 
    
    Returns
    -------
    Time_cst_mean, Time_cst_SD : floats
        Mean time constant and standard deviation of sweep based time constants

    '''
    sweep_info_table = cell_dict['Sweep_info_table']
    sweep_QC_table = cell_dict['Sweep_QC_table']
    sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
    sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
    sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
    Resting_potential_mean = np.nanmean(sub_sweep_info_QC_table['Resting_potential_mV'])
    Resting_potential_SD = np.nanstd(sub_sweep_info_QC_table['Resting_potential_mV'])
    
    return Resting_potential_mean, Resting_potential_SD
    
    

def create_summary_tables(config_json_file_path, saving_path):
    '''
    Gather the information, and features of each cells analysed

    Parameters
    ----------
    cell_id_list : List or array
        List of cells to get information from.
        
    config_json_file : str
        Path to JSON configuration file (ending .json).
        
    saving_path : str
        Path to folder in which summary tables will be saved (ending with / or \)

    Returns
    -------
    problem_cell : list
        Return the cell_id for which the function encountered a problem.

    '''
    
    #Prepare dataframes with columns names, and units
    
    unit_line=pd.DataFrame(['--','--','--','Hz/pA','pA','Hz','pA','Hz', "pA"]).T
    Full_feature_table=pd.DataFrame(columns=['Cell_id','Obs','I_O_NRMSE','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Response_Fail_Frequency','Response_Fail_Stimulus','Response_type',"Output_Duration"])
    
    unit_fit_line = pd.DataFrame(['--','--','--','--','--','--', "--",'--']).T
    Full_fit_table=pd.DataFrame(columns=['Cell_id','Obs','Hill_amplitude','Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_k','Response_type',"Output_Duration"])

    cell_linear_values = pd.DataFrame(columns=['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD'])
    linear_values_unit = pd.DataFrame(['--','GOhms','GOhms','ms','ms','mV','mV']).T
    linear_values_unit.columns=['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD']
    cell_linear_values = pd.concat([cell_linear_values,linear_values_unit],ignore_index=True)
    
    processing_time_table = pd.DataFrame(columns=['Cell_id','Processing_step','Processing_time'])
    processing_time_unit_line = pd.DataFrame(['--','--','s']).T
    processing_time_unit_line.columns=processing_time_table.columns
    processing_time_table = pd.concat([processing_time_table,processing_time_unit_line],ignore_index=True)
    
    
    Adaptation_table = pd.DataFrame(columns = ["Cell_id", "Adaptation_Obs", "Adaptation_Instantaneous_Frequency_Hz",
                                               'Adaptation_Spike_width_at_half_heigth_s',
                                               "Adaptation_Spike_heigth_mV", 
                                               "Adaptation_Threshold_mV",
                                               'Adaptation_Upstroke_mV/s',
                                               "Adaptation_Peak_mV", 
                                               'Adaptation_Downstroke_mV/s',
                                               "Adaptation_Fast_Trough_mV",
                                               'Adaptation_fAHP_mV',
                                               'Adaptation_Trough_mV'])
    Adaptation_table_unit_line = pd.DataFrame(["--","--","Index","Index","Index","Index","Index","Index","Index","Index","Index","Index"]).T
    Adaptation_table_unit_line.columns = Adaptation_table.columns
    Adaptation_table = pd.concat([Adaptation_table, Adaptation_table_unit_line],ignore_index = True)
    problem_cell=[]
    problem_df = pd.DataFrame(columns = ['Cell_id','Error_message'])
    
    
    config_json_file = open_json_config_file(config_json_file_path)
    Full_population_calss_table = pd.DataFrame()
    
    #Create full population class table, and resulting cell id list
    for line in config_json_file.index:
        current_db_population_class_table = pd.read_csv(config_json_file.loc[line,'db_population_class_file'])
        Full_population_calss_table= pd.concat([Full_population_calss_table,current_db_population_class_table],ignore_index = True)
    Full_population_calss_table=Full_population_calss_table.astype({'Cell_id':'str'})
    cell_id_list = Full_population_calss_table.loc[:,'Cell_id'].unique()
    for cell_id in tqdm.tqdm(cell_id_list):
        try:
            #For each cell, compute the mean IR, time constant, resting potential
            current_DB = Full_population_calss_table.loc[Full_population_calss_table['Cell_id']==cell_id,'Database'].values[0]
            config_line = config_json_file.loc[config_json_file['database_name']==current_DB,:]
            cell_dict = read_cell_file_h5(str(cell_id),config_line,['Sweep analysis','Firing analysis','Processing_report', "Sweep QC"])
            sweep_info_table = cell_dict['Sweep_info_table']
            cell_fit_table = cell_dict['Cell_fit_table']
            cell_feature_table = cell_dict['Cell_feature_table']
            Processing_df = cell_dict['Processing_table']
            cell_adaptation_table = cell_dict['Cell_Adaptation']
            sweep_QC_table = cell_dict['Sweep_QC_table']
            sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
            sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
            sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
            
           
            
            Cell_IR, IR_SD = compute_cell_input_resistance(cell_dict)
            
            
            
            if Cell_IR <=0:
                Cell_IR = np.nan
                IR_SD = np.nan
           
            Time_cst_mean, Time_cst_SD = compute_cell_time_constant(cell_dict)
            
            
            
            Resting_potential_mean, Resting_potential_SD = compute_cell_resting_potential(cell_dict)
            
            
            cell_linear_values_line = pd.DataFrame([str(cell_id),Cell_IR,IR_SD,Time_cst_mean,Time_cst_SD,Resting_potential_mean, Resting_potential_SD ]).T
            cell_linear_values_line.columns = ['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD']
            cell_linear_values=pd.concat([cell_linear_values,cell_linear_values_line],ignore_index=True)
            
            response_duration_dictionnary={
                'Time_based':[.005, .010, .025, .050, .100, .250, .500],
                'Index_based':list(np.arange(2,18)),
                'Interval_based':list(np.arange(1,17))}
            
            if cell_fit_table.shape[0]==1:

                for response_type in response_duration_dictionnary.keys():
                    output_duration_list=response_duration_dictionnary[response_type]
                    
                    for output_duration in output_duration_list:
                        I_O_obs=cell_fit_table.loc[(cell_fit_table['Response_type']==response_type) & (cell_fit_table['Output_Duration']==output_duration),"I_O_obs"]
                        if len(I_O_obs)!=0:
                            
                            Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = np.array(cell_feature_table.loc[(cell_feature_table['Response_type']==response_type )&
                                                                                                                                       (cell_feature_table['Output_Duration']==output_duration),
                                                                                                                                ["Gain","Threshold","Saturation_Frequency","Saturation_Stimulus","Response_Fail_Frequency", "Response_Fail_Stimulus"]]).tolist()[0]
                            I_O_obs=I_O_obs.tolist()[0]
                            
                            Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = np.array(cell_fit_table.loc[(cell_fit_table['Response_type']==response_type )&
                                                                                                                                       (cell_fit_table['Output_Duration']==output_duration),
                                                                                                                                ["Hill_Half_cst", "Hill_amplitude", "Hill_coef", "Hill_x0", "Output_Duration", "Response_type", "Sigmoid_k","Sigmoid_x0"]]).tolist()[0]
                        else:
                            I_O_obs="No_I_O_Adapt_computed"
                            empty_array = np.empty(6)
                            empty_array[:] = np.nan
                            Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = empty_array
                            
                            empty_array = np.empty(8)
                            empty_array[:] = np.nan
                            Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = empty_array
                        
                            
                        if Gain<=0:
                            Gain=np.nan

                            
                            
                        new_line=pd.DataFrame([str(cell_id),I_O_obs,Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,response_type,output_duration]).T
                        new_line.columns=Full_feature_table.columns
                        Full_feature_table = pd.concat([Full_feature_table,new_line],ignore_index=True)
                        
                        new_fit_line = pd.DataFrame([str(cell_id),I_O_obs,Hill_amplitude, Hill_coef, Hill_Half_cst, Hill_x0, Sigmoid_x0, Sigmoid_k, response_type, output_duration]).T
                        new_fit_line.columns = Full_fit_table.columns
                        Full_fit_table = pd.concat([Full_fit_table, new_fit_line], ignore_table = True)
                        
                        
            else:
                
                cell_feature_table = cell_feature_table.merge(
            cell_fit_table.loc[:,['I_O_obs','Response_type','Output_Duration','I_O_NRMSE']], how='inner', on=['Response_type','Output_Duration'])
    
                cell_feature_table['Cell_id']=str(cell_id)
                cell_feature_table=cell_feature_table.rename(columns={"I_O_obs": "Obs"})
                cell_feature_table=cell_feature_table.reindex(columns=Full_feature_table.columns)
                Full_feature_table = pd.concat([Full_feature_table,cell_feature_table],ignore_index=True)
                
                cell_fit_table['Cell_id'] = str(cell_id)
                cell_fit_table=cell_fit_table.rename(columns={"I_O_obs": "Obs"})
                cell_fit_table=cell_fit_table.reindex(columns=Full_fit_table.columns)
                Full_fit_table = pd.concat([Full_fit_table, cell_fit_table], ignore_index = True)
                
            
            cell_adaptation_table_copy = cell_adaptation_table.copy()
            cell_adaptation_table_copy['Feature'] = cell_adaptation_table_copy.apply(lambda row: append_last_measure(row['Feature'], row['Measure']), axis=1)
            feature_dict = cell_adaptation_table_copy.set_index('Feature')['Adaptation_Index'].to_dict()
            
            result_df = pd.DataFrame([feature_dict])
            result_df = result_df.add_prefix('Adaptation_')
            result_df.loc[:,'Cell_id']=cell_id
            result_df.loc[:,'Obs']="--"
            result_df = result_df.rename(columns={'Adaptation_Instantaneous_Frequency_mV':"Adaptation_Instantaneous_Frequency_Hz", 
                                                  "Obs":"Adaptation_Obs"})

            Adaptation_table = pd.concat([Adaptation_table, result_df], ignore_index=True)
            

            for step in Processing_df.loc[:,'Processing_step'].unique():
                sub_processing_table = Processing_df.loc[Processing_df['Processing_step']==step,:]
                sub_processing_table=sub_processing_table.reset_index()
                sub_processing_time = sub_processing_table.loc[0,"Processing_time"]
                sub_processing_time = float(sub_processing_time.replace("s",""))
                new_line = pd.DataFrame([cell_id,step,sub_processing_time]).T
                
                new_line.columns = processing_time_table.columns
                processing_time_table=pd.concat([processing_time_table,new_line],ignore_index=True,axis=0)
        except:

            try:
                cell_dict = read_cell_file_h5(str(cell_id),config_line,['Processing_report'])
                
                Processing_df = cell_dict['Processing_table']
                for elt in Processing_df.index:
                    if "Error" in Processing_df.loc[elt,'Warnings_encontered'] :
                        new_line = pd.DataFrame([cell_id, Processing_df.loc[elt,'Warnings_encontered']]).T
                        new_line.columns = problem_df.columns
                        problem_df = pd.concat([problem_df, new_line], ignore_index=True)
                        break
            except:

                problem_cell.append(cell_id)
                
            
    for response_type in response_duration_dictionnary.keys():
        output_duration_list=response_duration_dictionnary[response_type]
        for output_duration in output_duration_list:
            
            new_table = pd.DataFrame(columns=['Cell_id','Obs','I_O_NRMSE','Gain','Threshold','Saturation_Frequency','Saturation_Stimulus','Response_Fail_Frequency','Response_Fail_Stimulus',])
            unit_line.columns=new_table.columns
            new_table = pd.concat([new_table,unit_line],ignore_index=True)
            
            sub_table=Full_feature_table.loc[(Full_feature_table['Response_type']==response_type)&(Full_feature_table['Output_Duration']==output_duration),]
            sub_table=sub_table.drop(['Response_type','Output_Duration'], axis=1)
            sub_table=sub_table.reindex(columns=new_table.columns)
            sub_table['Gain'] = sub_table['Gain'].where(sub_table['Gain'] >= 0, np.nan)
            
            new_table = pd.concat([new_table,sub_table],ignore_index=True)
            if len(new_table['Cell_id'].unique())!=new_table.shape[0]:
                return str('problem with table'+str(response_type)+'_'+str(output_duration))
            
            
            
            new_fit_table = pd.DataFrame(columns = ['Cell_id','Obs','Hill_amplitude','Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_k'])
            unit_fit_line.columns = new_fit_table.columns
            new_fit_table = pd.concat([new_fit_table, unit_fit_line], ignore_index = True)
            
            sub_fit_table = Full_fit_table.loc[(Full_fit_table['Response_type']==response_type)&(Full_fit_table['Output_Duration']==output_duration),]
            sub_fit_table = sub_fit_table.drop(['Response_type','Output_Duration'], axis=1)
            sub_fit_table = sub_fit_table.reindex(columns=new_fit_table.columns)
            new_fit_table = pd.concat([new_fit_table,sub_fit_table],ignore_index=True)
            
            
            if response_type == 'Time_based':
                output_duration*=1000
                output_duration=str(str(int(output_duration))+'ms')
            

            new_table.to_csv(f"{saving_path}Full_Feature_Table_{response_type}_{output_duration}.csv")
            
            
            if len(new_fit_table['Cell_id'].unique())!=new_fit_table.shape[0]:
                return str('problem with table'+str(response_type)+'_'+str(output_duration))
            
            new_fit_table.to_csv(f"{saving_path}Full_Fit_Table_{response_type}_{output_duration}.csv")
            
            
    cell_linear_values.to_csv(str(saving_path+ 'Full_Cell_linear_values.csv')) 
    
    processing_time_table.to_csv(str(saving_path+'Full_Processing_Times.csv'))
    Adaptation_table.to_csv(f'{saving_path}Full_Adaptation_Table_Time_based_500ms.csv')
    problem_df.to_csv(f'{saving_path}Problem_report.csv')
    for col in Full_population_calss_table.columns:
        if "Unnamed" in col:
            Full_population_calss_table = Full_population_calss_table.drop(columns=[col])
    Full_population_calss_table.to_csv(f'{saving_path}Full_Population_Class.csv')
    return problem_df, problem_cell

def append_last_measure(feature, measure):
    last_element = measure.split('_')[-1]
    return f"{feature}_{last_element}"

def gather_all_features_table(folder):
    
    file_dict={'Time_based':['5ms','10ms','25ms','50ms','100ms','250ms','500ms'],
               "Index_based":['Index_2','Index_3','Index_4','Index_5','Index_6','Index_7','Index_8','Index_9','Index_10','Index_11','Index_12','Index_13','Index_14','Index_15','Index_16','Index_17'],
               "Interval_based":['Interval_1','Interval_2','Interval_3','Interval_4','Interval_5','Interval_6','Interval_7','Interval_8','Interval_9','Interval_10','Interval_11','Interval_12','Interval_13','Interval_14','Interval_15','Interval_16']}
    folder_directory = str(folder+ 'Full_Feature_Table_')
    dfs = []
    for output_type in file_dict.keys():
        response_duration_list = file_dict[output_type]
        for response_duration in response_duration_list:
            numeric_values = re.findall(r'\d+', response_duration)
        
            # Convert the extracted numeric values to integers
            numeric_values = [int(num) for num in numeric_values]
            if output_type !='Time_based':
                file_str = str(folder_directory+output_type+"_"+str(numeric_values[0])+'.csv')
            else:
                file_str = str(folder_directory+output_type+"_"+response_duration+'.csv')
            df = pd.read_csv(file_str,index_col=0)
            # Add file name suffix to each column
            df.columns = ['Cell_id' if col =='Cell_id' else col + '_' + str(response_duration) for col in df.columns ]
            dfs.append(df)
    Full_I_O_features_table = dfs[0]  # Initialize merged DataFrame with the first DataFrame
    for df in dfs[1:]:
        Full_I_O_features_table = pd.merge(Full_I_O_features_table, df, on='Cell_id')
        
    return Full_I_O_features_table

def summarize_features_evolution(Full_I_O_features_table):
    file_dict={'Time_based':['5ms','10ms','25ms','50ms','100ms','250ms','500ms'],
               "Index_based":['Index_2','Index_3','Index_4','Index_5','Index_6','Index_7','Index_8','Index_9','Index_10','Index_11','Index_12','Index_13','Index_14','Index_15','Index_16','Index_17'],
               "Interval_based":['Interval_1','Interval_2','Interval_3','Interval_4','Interval_5','Interval_6','Interval_7','Interval_8','Interval_9','Interval_10','Interval_11','Interval_12','Interval_13','Interval_14','Interval_15','Interval_16']}


    
    Full_I_O_features_table_without_units = Full_I_O_features_table.iloc[1:,:]
    cell_features_slope_int_table = pd.DataFrame(index=Full_I_O_features_table_without_units.loc[:,'Cell_id'].unique())
    cell_features_slope_int_table.loc[:,'Test']=0
    for output_type in file_dict.keys():
        if output_type == 'Time_based':
            filtered_cols = ['Cell_id'] + [col for col in Full_I_O_features_table_without_units.columns if 'ms' in col]
        elif output_type == 'Index_based':
            filtered_cols = ['Cell_id'] + [col for col in Full_I_O_features_table_without_units.columns if 'Index' in col]
        elif output_type == 'Interval_based':
            filtered_cols = ['Cell_id'] + [col for col in Full_I_O_features_table_without_units.columns if 'Interval' in col]
        sub_IO_table = Full_I_O_features_table_without_units[filtered_cols]
        
        features_list = ['Gain','Threshold','Saturation_Stimulus','Saturation_Amplitude']
        
        for feature in features_list:
            filtered_features_cols = ['Cell_id'] + [col for col in sub_IO_table.columns if feature in col]
            sub_IO_feature_table = sub_IO_table[filtered_features_cols]
            for col in sub_IO_feature_table.columns:
                if col != 'Cell_id':
                    sub_IO_feature_table[col] = sub_IO_feature_table[col].astype(float)
                    
            for idx, row in tqdm.tqdm(sub_IO_feature_table.iterrows()):
                # Extract x values (column names) and y values (row values)
                cell_id = row['Cell_id']
                row_data = row.drop('Cell_id')
                row_data = row_data.dropna()  # Drop columns with NaN values
                if len(row_data) > 0:
                    
                    x_values = [int(''.join(filter(str.isdigit, col))) * 1e-3 for col in row_data.index]
                    y_values = row_data.tolist()
            
                    # Perform linear regression using custom function
                    slope,intercept=fir_an.linear_fit(x_values,y_values)
                else:
                    slope, intercept = np.nan, np.nan
                x_values = [int(''.join(filter(str.isdigit, col))) for col in sub_IO_feature_table.columns if col != 'Cell_id']  # Extract numerical value from column name
                y_values = row.drop('Cell_id').tolist()
                
                
                
                cell_features_slope_int_table.loc[cell_id,str(feature+'_'+output_type+'_Slope')]=slope
                cell_features_slope_int_table.loc[cell_id,str(feature+'_'+output_type+'_Intercept')]=intercept
                
    cell_features_slope_int_table_with_cell_ids = cell_features_slope_int_table.reset_index()
    cell_features_slope_int_table_with_cell_ids=cell_features_slope_int_table_with_cell_ids.drop(columns=['Test'])
    cell_features_slope_int_table_with_cell_ids=cell_features_slope_int_table_with_cell_ids.rename(columns={'index':'Cell_id'})
    unit_line = pd.DataFrame(["--","Hz/pA/ms","Hz/pA","pA/ms","pA","pA/ms","pA","Hz/ms","Hz","Hz/pA/Index","Hz/pA","pA/Index","pA","pA/Index","pA","Hz/Index","Hz","Hz/pA/Interval","Hz/pA","pA/Interval","pA","pA/Interval","pA","Hz/Interval","Hz"]).T
    unit_line.columns = cell_features_slope_int_table_with_cell_ids.columns
    
    cell_features_slope_int_table_with_cell_ids=pd.concat([unit_line,cell_features_slope_int_table_with_cell_ids],axis=0,ignore_index=True)
    return cell_features_slope_int_table_with_cell_ids



    
def get_first_spike_features_hat(config_json_file):
    '''
    

    Extracts and compiles the features of the first spike for a collection of cells across databases.

    This function processes spike feature data for cells listed in a configuration file. It iterates 
    over databases, reads population and configuration information, and extracts spike features 
    (threshold, height, upstroke, downstroke, etc.) for the first spike in each sweep of each cell. 
    Results from all cells are combined into a single table, including a row for units of measurement.

    Parameters
    ----------
    config_json_file : pandas.DataFrame
        A DataFrame containing configuration information for multiple databases.
        Must include the following columns:
        - 'database_name': Name of the database.
        - 'db_population_class_file': Path to a CSV file containing cell population information.

    Returns
    -------
    Full_spike_table : pandas.DataFrame
        A DataFrame containing the first spike features for all cells in all databases, with the following columns:
        - 'Cell_id': Identifier of the cell.
        - 'Sweep': Sweep ID where the spike was observed.
        - 'Stim_amp_pA': Stimulation amplitude in pA.
        - 'Spike_threshold': Spike threshold in mV.
        - 'Spike_heigth': Spike height in mV.
        - 'Spike_Upstroke': Upstroke velocity in mV/s.
        - 'Spike_Downstroke': Downstroke velocity in mV/s.
        - 'Spike_peak': Spike peak value in mV.
        - 'Spike_trough': Spike trough value in mV.
        - 'Spike_width': Spike width in seconds.
        The first row contains the units of measurement for each column.

    Notes
    -----
    - Uses parallel processing to speed up the computation of spike features for multiple cells using `concurrent.futures.ProcessPoolExecutor`.
    - Each database's cell population table is read from the file specified in the configuration file.
    - Results are compiled into a single DataFrame with units added as the first row.
    - A progress bar (`tqdm`) is displayed to track processing progress for each database.


    '''
    
    unit_line=pd.DataFrame(['--','--','pA','mV','mV','mV/s','mV/s','mV', "mV", 's']).T
    Full_spike_table=pd.DataFrame(columns=['Cell_id','Sweep','Stim_amp_pA','Spike_threshold','Spike_heigth', "Spike_Upstroke", "Spike_Downstroke", "Spike_peak", 'Spike_trough', 'Spike_width'])
    results_lists = []
    for database in config_json_file.loc[:,'database_name'].unique():
        database_cell_pop_table_file = config_json_file.loc[config_json_file['database_name']==database,"db_population_class_file"].values[0]
        database_config_line = config_json_file.loc[config_json_file['database_name']==database,:]
        database_cell_pop_table = pd.read_csv(database_cell_pop_table_file)
        cell_id_list = list(database_cell_pop_table.loc[:,'Cell_id'].unique())
        
        args_list = [[x, 
                      database_config_line] for x in cell_id_list]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = 7) as executor:
            problem_cell_list = {executor.submit(sp_an.get_first_spike_features,x): x for x in args_list}
            for f in tqdm.tqdm(concurrent.futures.as_completed(problem_cell_list),total = len(cell_id_list), desc=f'Processing {database}'):
                spike_lines = f.result()
                if spike_lines is not None:
                    results_lists.append(f.result())
        
        
    Full_spike_table = pd.concat(results_lists)
    unit_line.columns = Full_spike_table.columns
    Full_spike_table = pd.concat([unit_line, Full_spike_table], ignore_index = True)
    
    return Full_spike_table


def get_cells_sampling_freq(config_json_file):
    
    result_list = []
    for elt in config_json_file.index:
        pop_class_table = pd.read_csv(config_json_file.loc[elt,"db_population_class_file"])
        cell_id_list = pop_class_table.loc[:,"Cell_id"].unique()
        current_db = config_json_file.loc[elt,"database_name"]
        current_line = config_json_file.loc[config_json_file['database_name']==current_db,:]
        for cell_id in tqdm.tqdm(cell_id_list,desc=f"Processing database : {current_db}"):
            try:
                cell_dict = read_cell_file_h5(str(cell_id), current_line,"Sweep analysis")
                cell_sweep_info_table = cell_dict['Sweep_info_table']
                sampling_freq = np.nanmean(cell_sweep_info_table.loc[:,"Sampling_Rate_Hz"])
                result_df = pd.DataFrame([cell_id, current_db, sampling_freq]).T
                result_df.columns = ['Cell_id', 'Database', 'Sampling_Frequency_Hz']
                
                result_list.append(result_df)

            
            except:
                print(cell_id)
    
    full_result = pd.concat(result_list, ignore_index = True)
    
    return full_result

def get_max_frequency(config_json_file):
    
    
    results_lists = []
    for database in config_json_file.loc[:,'database_name'].unique():
        database_cell_pop_table_file = config_json_file.loc[config_json_file['database_name']==database,"db_population_class_file"].values[0]
        database_config_line = config_json_file.loc[config_json_file['database_name']==database,:]
        database_cell_pop_table = pd.read_csv(database_cell_pop_table_file)
        cell_id_list = list(database_cell_pop_table.loc[:,'Cell_id'].unique())
        
        args_list = [[x, 
                      database_config_line] for x in cell_id_list]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = 7) as executor:
            problem_cell_list = {executor.submit(sw_an.get_max_frequency_parallel,x): x for x in args_list}
            for f in tqdm.tqdm(concurrent.futures.as_completed(problem_cell_list),total = len(cell_id_list), desc=f'Processing {database}'):
                spike_lines = f.result()
                if spike_lines is not None:
                    results_lists.append(f.result())
        
                    
    Full_max_freq_table = pd.concat(results_lists, ignore_index = True)
    
    unit_line = pd.DataFrame(["--", "Hz", "pA", 'Hz', 'pA', 'Hz',"pA",'--']).T
    unit_line.columns = ['Cell_id','Maximum_Frequency_Hz', "Maximum_Frequency_Stimulus_pA", "Maximum_Frequency_step_Hz", "Stimulus_for_Maximum_freq_Step_pA", "Second_Maximum_Frequency_Step_Hz","Stimulus_for_Second_Maximum_freq_Step_pA", "Maximum_frequency_Step_ratio"]
    
    Full_max_freq_table = pd.concat([unit_line, Full_max_freq_table], ignore_index = True)
    
    return Full_max_freq_table

def gather_Full_population_cell_Sweep_info(config_json_file):
    
    full_result_cell_sweep_info_list = []
    full_result_cell_QC_list = []
    
    for current_db in config_json_file.loc[:,"database_name"].unique():
        
        current_db_config_line = config_json_file.loc[config_json_file['database_name']==current_db,:].reset_index(drop=True)
        current_db_pop_class_table = pd.read_csv(current_db_config_line.loc[0,"db_population_class_file"])
        
        
        for current_cell_id in tqdm.tqdm(current_db_pop_class_table.loc[:,'Cell_id'].unique(), desc = f'Current DB = {current_db}'):
            
            try:
                
                cell_dict = read_cell_file_h5(current_cell_id,
                                              current_db_config_line,
                                              selection = ["Sweep analysis"])
                
                current_cell_sweep_info_table = cell_dict["Sweep_info_table"]
                current_cell_sweep_info_table.loc[:,'Cell_id'] = str(current_cell_id)
                current_cell_sweep_info_table.loc[:,'Database'] = str(current_db)
                
                current_cell_QC_table = cell_dict["Sweep_QC_table"]
                current_cell_QC_table.loc[:,'Cell_id'] = str(current_cell_id)
                current_cell_QC_table.loc[:,'Database'] = str(current_db)
                
                full_result_cell_sweep_info_list.append(current_cell_sweep_info_table)
                full_result_cell_QC_list.append(current_cell_QC_table)
                
            except:
                print(f'Problem with cell : {current_cell_id} in database {current_db}')
                
    Full_cell_sweep_info = pd.concat(full_result_cell_sweep_info_list, ignore_index = True)
    Full_cell_sweep_QC = pd.concat(full_result_cell_QC_list, ignore_index = True)
    
    return Full_cell_sweep_info, Full_cell_sweep_QC
                
                
                
    
     
    
    
    
    
            
   