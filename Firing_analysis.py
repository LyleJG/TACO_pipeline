#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:03:38 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
from lmfit.models import PolynomialModel, Model, ConstantModel
from lmfit import Parameters
import plotnine as p9
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

class CustomFitError(Exception):
    """Exception raised for errors in the fitting process."""
    pass
class NotEnoughValueError(Exception):
    """Exception raised when there are not enough values to perform fit"""
    pass
class StimulusSpaceSamplingNotSparseEnough(Exception):
    """Exception raised when the stimulus space is not sampled sparsely enough"""
    pass

class EmptyTrimmedPolynomialFit(Exception):
    """Exception raised when the trimmed polynomial fit is empty"""
    pass

def compute_cell_features(Full_SF_table,cell_sweep_info_table,response_duration_dictionnary,sweep_QC_table):
    '''
    Compute cell I/O features, for a dictionnnary of response_type:output_duration.
    Compute Adaptation for 500ms response.
     and adapation (cell_fit_table) as well as the different features computed (cell_feature_table)

    Parameters
    ----------
    
    Full_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    response_duration_dictionnary : dict
        Dictionnary containing as key the response type (Time_based, Index_based, Interval_based),
        and for each key a list of output_duration.
        
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit

    Returns
    -------
    cell_feature_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the Adaptation and I/O features.
    cell_fit_table : pd.DataFrame
        DataFrame containing for each pair of response type - output duration the fit parameters to reconstruct I/O curve and adaptation curve.

    '''
    
    fit_columns = ['Response_type','Output_Duration', 'I_O_obs', 'I_O_NRMSE', 'Hill_amplitude',
                   'Hill_coef', 'Hill_Half_cst','Hill_x0','Sigmoid_x0','Sigmoid_k']
    cell_fit_table = pd.DataFrame(columns=fit_columns)

    feature_columns = ['Response_type','Output_Duration', 'Gain','Threshold', 'Saturation_Frequency',"Saturation_Stimulus","Response_Fail_Frequency", "Response_Fail_Stimulus"]
    cell_feature_table = pd.DataFrame(columns=feature_columns)
    # print(f'{response_duration_dictionnary.keys()=}')

    for response_type in response_duration_dictionnary.keys():
        output_duration_list=response_duration_dictionnary[response_type]

        
        for output_duration in output_duration_list:

    
            stim_freq_table = get_stim_freq_table(
                Full_SF_table.copy(), cell_sweep_info_table.copy(),sweep_QC_table.copy(), output_duration,response_type)
    
            pruning_obs, do_fit, condition_table = data_pruning_I_O(stim_freq_table,cell_sweep_info_table)

            if do_fit == True:
                
                I_O_obs, Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_k,I_O_NRMSE, Gain, Threshold, Saturation_freq,Saturation_stim, IO_fail_stim, IO_fail_freq, parameters_table = get_IO_features(stim_freq_table, response_type, output_duration, False)
            else:

                I_O_obs = pruning_obs
                empty_array = np.empty(13)
                empty_array[:] = np.nan
                Hill_Amplitude, Hill_coef, Hill_Half_cst,Hill_x0, sigmoid_x0,sigmoid_k,I_O_NRMSE, Gain, Threshold, Saturation_freq,Saturation_stim,IO_fail_stim, IO_fail_freq = empty_array
    
    
    
            new_fit_table_line = pd.DataFrame([response_type,
                                               output_duration,
                                            I_O_obs,
                                            I_O_NRMSE,
                                            Hill_Amplitude,
                                            Hill_coef,
                                            Hill_Half_cst,
                                            Hill_x0,
                                            sigmoid_x0,
                                            sigmoid_k]).T
            new_fit_table_line.columns=fit_columns
            cell_fit_table=pd.concat([cell_fit_table,new_fit_table_line],ignore_index=True)
    
    
            new_feature_table_line = pd.DataFrame([response_type,
                                               output_duration,
                                                Gain,
                                                Threshold,
                                                Saturation_freq,
                                                Saturation_stim,
                                                IO_fail_stim, 
                                                IO_fail_freq]).T
            new_feature_table_line.columns=feature_columns
            cell_feature_table=pd.concat([cell_feature_table,new_feature_table_line],ignore_index=True)

        

    cell_fit_table=cell_fit_table.astype({'Response_type': 'str',
                                          'Output_Duration':'float',
                                          'I_O_NRMSE':'float',
                                          'Hill_amplitude':'float',
                                          'Hill_coef':'float',
                                          'Hill_Half_cst':'float',
                                          'Hill_x0':'float',
                                          'Sigmoid_x0':'float',
                                          'Sigmoid_k':'float'})
    
    cell_feature_table_convert_dict={'Response_type': str,
                                   'Output_Duration':float,
                    'Gain':float,
                    'Threshold':float,
                    'Saturation_Frequency':float,
                    'Saturation_Stimulus':float,
                    'Response_Fail_Frequency' : float,
                    'Response_Fail_Stimulus' : float}
    cell_feature_table=cell_feature_table.astype(cell_feature_table_convert_dict)

    if cell_fit_table.shape[0] == 0:
        cell_fit_table.loc[0, :] = np.nan
        cell_fit_table = cell_fit_table.apply(pd.to_numeric)
    if cell_feature_table.shape[0] == 0:
        cell_feature_table.loc[0, :] = np.nan
        cell_feature_table = cell_feature_table.apply(pd.to_numeric)
    
    return cell_feature_table,cell_fit_table


def get_stim_freq_table(original_SF_table, original_cell_sweep_info_table,sweep_QC_table,response_duration, response_based="Time_based"):
    '''
    Given a table containing the spike features for the different sweep, compute the frequency for a given response

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_duration : float
        Duration of teh response to consider.
        
    response_based : str, optional
        Type of response to consider (can be 'Index_based' or 'Interval_based'. The default is "Time_based".

    Returns
    -------
    stim_freq_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.

    '''
    

    SF_table=original_SF_table.copy()


    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_list=np.array(original_SF_table.loc[:,"Sweep"])
    # print(f'{sweep_list=}')

    stim_freq_table=pd.DataFrame(columns=["Sweep","Stim_amp_pA","Frequency_Hz"])
    # stim_freq_table=cell_sweep_info_table.copy()
    # stim_freq_table['Frequency_Hz']=0


    for current_sweep in sweep_list:
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'].copy())
        df=df[df['Feature']=='Upstroke']
        stim_amp = cell_sweep_info_table.loc[current_sweep,"Stim_amp_pA"]

        if response_based == 'Time_based':
            #stim_freq_table.loc[current_sweep,'Frequency_Hz']=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_duration)].shape[0])/response_duration
            frequency = (df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_duration)].shape[0])/response_duration
            
            # print(f'{current_sweep=} {frequency=}')
            # print('asdfasdf')
            # # breakpoint()
            
        elif response_based == 'Index_based':
            df=df.sort_values(by=["Time_s"])
            if df.shape[0] < response_duration: # if there are less spikes than response duration required, then set frequency to NaN
                #e.g.: If we want spike 3, we need 3 spikes at least

                continue
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=np.nan
                
            
            else:
                spike_time=np.array(df['Time_s'])[int(response_duration-1)]
                
                #stim_freq_table.loc[current_sweep,'Frequency_Hz'] = df.shape[0]/(spike_time-cell_sweep_info_table.loc[current_sweep,'Stim_start_s'])
                frequency = df.shape[0]/(spike_time-cell_sweep_info_table.loc[current_sweep,'Stim_start_s'])
        elif response_based == 'Interval_based':
            df=df.sort_values(by=["Time_s"])
            
            
            if df.shape[0]<=int(response_duration): # if there are less interval than response duration required, then set frequency to NaN
                #e.g.: If we want interval 3, we need 4 spikes at least
                continue
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=np.nan
            else:
                spike_time=np.array(df['Time_s'])
                response_duration = int(response_duration)
                #As python indexing starts at 0,, and interval number starts at 1 then interval_n = [spike_(n)- spike_(n-1)]
                # So that interval 1=spike_1-spike_0 (the interval between the second spike and the first spike)
                #stim_freq_table.loc[current_sweep,'Frequency_Hz']=1/(spike_time[response_duration]-spike_time[response_duration-1])
                frequency = 1/(spike_time[response_duration]-spike_time[response_duration-1])
                
        
        new_line=pd.DataFrame([current_sweep,stim_amp,frequency]).T
        new_line.columns = stim_freq_table.columns

        stim_freq_table = pd.concat([stim_freq_table,new_line],ignore_index=True)
        
    stim_freq_table=pd.merge(stim_freq_table,sweep_QC_table,on='Sweep')
    stim_freq_table=stim_freq_table.astype({"Stim_amp_pA":float,
                                            "Frequency_Hz":float})
    
    return stim_freq_table

def data_pruning_I_O_old(original_stim_freq_table,cell_sweep_info_table):
    '''
    For a given set of couple (Input_current - Stim_Freq), test wether there is enough information before proceeding to IO fit.
    Test the number of non-zero response, the number of different non-zero response
    Estimate the original frequency step, to avoid fitting continuous IO curve to Type II neurons

    Parameters
    ----------
    original_stim_freq_table : pd.DataFrame
        DataFrame containing one row per sweep, with the corresponding input current and firing frequency.
        
    cell_sweep_info_table : pd.DataFraem
        DESCRIPTION.

    Returns
    -------
    obs : str
        If do_fit == False, then obs contains the reason why no to proceed to IO fit .
    do_fit : bool
        Wether or not proceeding to IO fit.

    '''
    stim_freq_table=original_stim_freq_table.copy()
    stim_freq_table=stim_freq_table[stim_freq_table["Passed_QC"]==True]
    stim_freq_table = stim_freq_table.sort_values(
        by=['Stim_amp_pA', 'Frequency_Hz'])
    
    frequency_array = np.array(stim_freq_table.loc[:, 'Frequency_Hz'])
    obs = '-'
    do_fit = True
    
    
    
    
    non_zero_freq = frequency_array[np.where(frequency_array > 0)]
   
    if np.count_nonzero(frequency_array) < 4:
        obs = 'Less_than_4_response'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

    
    
    if len(np.unique(non_zero_freq)) < 3:
        obs = 'Less_than_3_different_frequencies'
        do_fit = False
        return obs, do_fit  # , max_freq,max_freq_step,(max_freq_step/max_freq)

   
    minimum_frequency_step = get_min_freq_step(stim_freq_table,do_plot=False)[0]

    if minimum_frequency_step >30:
        obs = 'Minimum_frequency_step_higher_than_30Hz'
        do_fit = False
        return obs, do_fit
        
    
    obs = '--'
    do_fit = True
    return obs, do_fit

def data_pruning_I_O(original_stim_freq_table,cell_sweep_info_table):
    '''
    For a given set of couple (Input_current - Stim_Freq), test wether there is enough information before proceeding to IO fit.
    Test the number of non-zero response, the number of different non-zero response
    Estimate the original frequency step, to avoid fitting continuous IO curve to Type II neurons

    Parameters
    ----------
    original_stim_freq_table : pd.DataFrame
        DataFrame containing one row per sweep, with the corresponding input current and firing frequency.
        
    cell_sweep_info_table : pd.DataFraem
        DESCRIPTION.

    Returns
    -------
    obs : str
        If do_fit == False, then obs contains the reason why no to proceed to IO fit .
    do_fit : bool
        Wether or not proceeding to IO fit.

    '''
    stim_freq_table=original_stim_freq_table.copy()
    stim_freq_table=stim_freq_table[stim_freq_table["Passed_QC"]==True]
    stim_freq_table = stim_freq_table.sort_values(
        by=['Stim_amp_pA', 'Frequency_Hz'])
    
    frequency_array = np.array(stim_freq_table.loc[:, 'Frequency_Hz'])
    obs = '-'
    do_fit = True
    condition_lines_list = []
    
    non_zero_freq = frequency_array[np.where(frequency_array > 0)]
   
    ##### -- Minimum 4 non-zero responses --
    condition = "Minimum 4 non zero responses"
    non_zero_responses = np.count_nonzero(frequency_array)
    
    if non_zero_responses < 4:
        condition_result = False
        obs = 'Less_than_4_response'
        do_fit = False
        
        condition_line = pd.DataFrame([condition, 
                                     non_zero_responses,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
        
        condition_table = pd.concat(condition_lines_list, ignore_index = True)
        return obs, do_fit, condition_table
    else:
        condition_result = True
        condition_line = pd.DataFrame([condition, 
                                     non_zero_responses,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
        condition_result = True
    
    
    
    ##### -- Minimum 3 different non zero frequency --

    condition = "Minimum 3 different frequencies"
    nb_non_zero_freq_diff = len(np.unique(non_zero_freq))
    
    
    if nb_non_zero_freq_diff < 3:
        condition_result = False
        obs = 'Less_than_3_different_frequencies'
        do_fit = False
        condition_line = pd.DataFrame([condition, 
                                     nb_non_zero_freq_diff,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
        
        condition_table = pd.concat(condition_lines_list, ignore_index = True)
        return obs, do_fit, condition_table  
    else:
        condition_result = True
        condition_line = pd.DataFrame([condition, 
                                     nb_non_zero_freq_diff,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
        condition_result = True
        
    
    
    
    ##### -- Maximum frequency jump = 30Hz
   
    condition = "Initial frequency jump <= 30Hz"
    minimum_frequency_step = get_min_freq_step(stim_freq_table,do_plot=False)[0]
    
    if minimum_frequency_step >30:
        condition_result = False
        obs = 'Minimum_frequency_step_higher_than_30Hz'
        do_fit = False
        condition_line = pd.DataFrame([condition, 
                                     minimum_frequency_step,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
        condition_table = pd.concat(condition_lines_list, ignore_index = True)
        return obs, do_fit, condition_table 
        
        #return obs, do_fit
    else:
        condition_result = True
        condition_line = pd.DataFrame([condition, 
                                     minimum_frequency_step,
                                     condition_result]).T
        condition_line.columns = ['Condition', 'Value', 'Condition passed']
        condition_lines_list.append(condition_line)
    
    
    try:
        original_data_table =  original_stim_freq_table.copy()
        original_data_table = original_data_table.dropna()
        
        original_data_subset_QC = original_data_table.copy()
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        
    
        original_data_subset_QC = original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        trimmed_stimulus_frequency_table=original_data_subset_QC.copy()
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.reset_index(drop=True)
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        
        ### 0 - Trim Data
        
        response_threshold_ascending = np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])/10
        response_threshold_descending = np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])/2 
        

        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.reset_index(drop=True)
        last_zero_response_ascending = np.nan
        for elt in trimmed_stimulus_frequency_table.index:
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] == 0.:
                last_zero_response_ascending = elt
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] >= response_threshold_ascending:
                n_ref = elt
                break
            n_ref = np.nan
        
        for elt in trimmed_stimulus_frequency_table.index[::-1]:
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] >= response_threshold_descending:
                N_theta_one = elt
                break
            N_theta_one = np.nan
            
        #Either trim fromthe last zero response before n_ref or if no zero response, from the last sweep before threshold(n_ref-1)
        if np.isnan(last_zero_response_ascending):
            N_theta_zero = int(n_ref-1)
        else:
            N_theta_zero = int(last_zero_response_ascending)
            
        trimmed_space_start = int(np.nanmax([0,N_theta_zero]))
        trimmed_space_end = int(np.nanmin([original_data_subset_QC.shape[0],int(N_theta_one+1)]))
        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.loc[trimmed_space_start:trimmed_space_end]
        
        
        
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.reset_index(drop=True)

        trimmed_x_data=np.array(trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA'])
        trimmed_y_data=np.array(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        #Check for iverfitting conditions before polynomial fit
        
        non_zero_trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.copy()
        non_zero_trimmed_stimulus_frequency_table = non_zero_trimmed_stimulus_frequency_table.loc[non_zero_trimmed_stimulus_frequency_table['Frequency_Hz']!=0,:]
        
        condition = "Polynomial fit requires more than 3 values"
        nb_of_values = non_zero_trimmed_stimulus_frequency_table.shape[0]
        
        if nb_of_values < 4:
            condition_result = False
            obs = 'Trimmed data has less than 3 data to fit polynomial'
            do_fit = False
            
            condition_line = pd.DataFrame([condition, 
                                         nb_of_values,
                                         condition_result]).T
            condition_line.columns = ['Condition', 'Value', 'Condition passed']
            condition_lines_list.append(condition_line)
            condition_table = pd.concat(condition_lines_list, ignore_index = True)
            return obs, do_fit, condition_table 
            
            
        else:
            condition_result = True
            condition_line = pd.DataFrame([condition, 
                                         nb_of_values,
                                         condition_result]).T
            condition_line.columns = ['Condition', 'Value', 'Condition passed']
            condition_lines_list.append(condition_line)
        
            

        
        min_trimmed_x_data = np.nanmin(trimmed_x_data)
        max_trimmed_x_data = np.nanmax(trimmed_x_data)
        
        intervals = np.linspace(min_trimmed_x_data, max_trimmed_x_data, 9) #Devide the stimulus-space into 8 windows (9 boundaries)
        count_values_per_interval = np.histogram(trimmed_x_data, bins = intervals)[0]
       
        non_zero_count_interval = np.sum(count_values_per_interval > 0)
        
        condition = "Stimulus span over 4 different intervals"
        if non_zero_count_interval < 4:
            condition_result = False
            obs = 'Stimulus span over less than 4 different intervals'
            do_fit = False
            condition_line = pd.DataFrame([condition, 
                                         non_zero_count_interval,
                                         condition_result]).T
            condition_line.columns = ['Condition', 'Value', 'Condition passed']
            condition_lines_list.append(condition_line)
            condition_table = pd.concat(condition_lines_list, ignore_index = True)
            return obs, do_fit, condition_table 
        else:
            condition_result = True
            condition_line = pd.DataFrame([condition, 
                                         non_zero_count_interval,
                                         condition_result]).T
            condition_line.columns = ['Condition', 'Value', 'Condition passed']
            condition_lines_list.append(condition_line)
    except Exception as e:
        condition_table = pd.concat(condition_lines_list, ignore_index = True)
        return obs, do_fit, condition_table 
    
    obs = '--'
    do_fit = True
    condition_table = pd.concat(condition_lines_list, ignore_index = True)
    return obs, do_fit, condition_table


def get_IO_features(original_stimulus_frequency_table,response_type, response_duration, cell_id='--', do_plot = False, print_plot = False):
    
    
    obs, fit_model_table, Amp, H_x0, H_Half_cst, H_Hill_coef, S_x0, S_k, Legend, Fit_NRMSE ,plot_list, parameters_table = fit_IO_relationship(original_stimulus_frequency_table,cell_id = cell_id,do_plot=do_plot,print_plot=print_plot)
    
    if obs == 'Hill-Sigmoid':

        Descending_segment = True
        
        feature_obs,Gain,Threshold,Saturation_frequency,Saturation_stimulation,IO_fail_stim, IO_fail_freq ,plot_list = extract_IO_features(original_stimulus_frequency_table,response_type, fit_model_table, Descending_segment, Legend, response_duration,cell_id = cell_id, do_plot = do_plot ,plot_list=plot_list, print_plot=print_plot)
    
    elif obs == 'Hill':
        Descending_segment = False
        feature_obs,Gain,Threshold,Saturation_frequency,Saturation_stimulation,IO_fail_stim, IO_fail_freq ,plot_list = extract_IO_features(original_stimulus_frequency_table,response_type, fit_model_table, Descending_segment, Legend, response_duration, cell_id = cell_id, do_plot = do_plot,plot_list=plot_list, print_plot=print_plot)
    
    else:

        empty_array = np.empty(7)
        empty_array[:] = np.nan
        feature_obs=obs
        Fit_NRMSE,Gain,Threshold,Saturation_frequency,Saturation_stimulation,IO_fail_stim, IO_fail_freq = empty_array
    
    if do_plot:
        return plot_list
    return feature_obs, Amp, H_Hill_coef, H_Half_cst, H_x0, S_x0, S_k, Fit_NRMSE, Gain, Threshold, Saturation_frequency, Saturation_stimulation, IO_fail_stim, IO_fail_freq, parameters_table
 
    
    
    
def fit_IO_relationship(original_stimulus_frequency_table,cell_id='--',do_plot=False,print_plot=False):
    '''
    Fit to the Input Output (stimulus-frequency) relationship  a continuous curve, and compute I/O features (Gain, Threshold, Saturation...)
    If a saturation is detected, then the IO relationship is fit to a Hill-Sigmoid function
    Otherwise the IO relationship is fit to a Hill function

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
        
    do_plot : Boolean, optional
        Do plot. The default is False.
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    feature_obs, Amp, Hill_coef, Hill_Half_cst, Hill_x0, sigmoid_x0, sigmoid_sigma : float
        Results of the fitting procedure to reproduce the I/O curve fit
        
    best_QNRMSE : float
        Godness of fit
        
    Gain, Threshold, Saturation_frequency, Saturation_stimulation : float
        Results of neuronal firing features computation.

    '''
    
    
    scale_dict = {'3rd_order_poly' : 'black',
                  '2nd_order_poly' : 'black',
                  'Trimmed_asc_seg' : 'pink',
                  "Trimmed_desc_seg" : "red",
                  'Ascending_Sigmoid' : 'blue',
                  "Descending_Sigmoid"  : 'orange',
                  'Amplitude' : 'yellow',
                  "Asc_Hill_Desc_Sigmoid": "red",
                  "First_Hill_Fit" : 'green',
                  'Hill_Sigmoid_Fit' : 'green',
                  'Sigmoid_fit' : "blue",
                  'Hill_Fit' : 'green',
                  True : 'o',
                  False : 's'
                  
                  }
    
    
    
    Gain=np.nan
    Threshold=np.nan
    Saturation_frequency=np.nan
    Saturation_stimulation=np.nan
    try:
        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        
        
        if do_plot==True:
            plot_list=dict()
            i=1
            original_data_table=original_data_table.astype({'Passed_QC':bool})
    
        else:
            plot_list=None
        
        original_data_subset_QC = original_data_table.copy()
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        
    
        original_data_subset_QC = original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        trimmed_stimulus_frequency_table=original_data_subset_QC.copy()
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.reset_index(drop=True)
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        
        ### 0 - Trim Data
        
        #response_threshold = (np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20

        #response_threshold = np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])/20
        response_threshold_ascending = np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])/10
        response_threshold_descending = np.nanmax(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])/2 
        

        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.reset_index(drop=True)
        last_zero_response_ascending = np.nan
        for elt in trimmed_stimulus_frequency_table.index:
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] == 0.:
                last_zero_response_ascending = elt
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] >= response_threshold_ascending:
                n_ref = elt
                break
            n_ref = np.nan
        
        for elt in trimmed_stimulus_frequency_table.index[::-1]:
            if trimmed_stimulus_frequency_table.loc[elt,"Frequency_Hz"] >= response_threshold_descending:
                N_theta_one = elt
                break
            N_theta_one = np.nan
            
        #Either trim fromthe last zero response before n_ref or if no zero response, from the last sweep before threshold(n_ref-1)
        if np.isnan(last_zero_response_ascending):
            N_theta_zero = int(n_ref-1)
        else:
            N_theta_zero = int(last_zero_response_ascending)
            
        trimmed_space_start = int(np.nanmax([0,N_theta_zero]))
        trimmed_space_end = int(np.nanmin([original_data_subset_QC.shape[0],int(N_theta_one+1)]))
        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.loc[trimmed_space_start:trimmed_space_end]
        
        
        
        
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        trimmed_stimulus_frequency_table=trimmed_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.reset_index(drop=True)

        trimmed_x_data=np.array(trimmed_stimulus_frequency_table.loc[:,'Stim_amp_pA'])
        trimmed_y_data=np.array(trimmed_stimulus_frequency_table.loc[:,"Frequency_Hz"])
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
        #Check for iverfitting conditions before polynomial fit
        
        non_zero_trimmed_stimulus_frequency_table = trimmed_stimulus_frequency_table.copy()
        non_zero_trimmed_stimulus_frequency_table = non_zero_trimmed_stimulus_frequency_table.loc[non_zero_trimmed_stimulus_frequency_table['Frequency_Hz']!=0,:]
        
        if non_zero_trimmed_stimulus_frequency_table.shape[0] < 4:
            raise NotEnoughValueError(f"Fit procedure requires more than 3 values, get {non_zero_trimmed_stimulus_frequency_table.shape[0]}")

        
        min_trimmed_x_data = np.nanmin(trimmed_x_data)
        max_trimmed_x_data = np.nanmax(trimmed_x_data)
        
        intervals = np.linspace(min_trimmed_x_data, max_trimmed_x_data, 9) #Devide the stimulus-space into 8 windows (9 boundaries)
        count_values_per_interval = np.histogram(trimmed_x_data, bins = intervals)[0]
       
        non_zero_count_interval = np.sum(count_values_per_interval > 0)
        
        if non_zero_count_interval < 4:
            raise StimulusSpaceSamplingNotSparseEnough(f"Fit procedure requires the stimulus to be in at least four different stimulus bins, currently only in {non_zero_count_interval}")
        
        #If non of the above condition are met, proceed with the fit
        if len(trimmed_y_data)>=4: # Need to have more than 4 non-zero responses
            third_order_poly_model = PolynomialModel(degree = 3)
    
            pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
            third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
            best_c0 = third_order_poly_model_results.best_values['c0']
            best_c1 = third_order_poly_model_results.best_values['c1']
            best_c2 = third_order_poly_model_results.best_values['c2']
            best_c3 = third_order_poly_model_results.best_values['c3']
            
            extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            
            trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_3rd_poly_model})
            trimmed_3rd_poly_table['Legend']='3rd_order_poly'
            
            third_order_fit_params_table = get_parameters_table(pars, third_order_poly_model_results)
            
            third_order_fit_params_table.loc[:,'Fit'] = "3rd_order_polynomial_fit"
            
            
                
            extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
            
            
            
            
            
            zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
            if len(zero_crossings_3rd_poly_model_freq_diff)==0: # if the derivative of the 3rd order polynomial does not change sign --> Assume there is no Descending Segment
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table.copy()
                
            elif len(zero_crossings_3rd_poly_model_freq_diff)==1:# if the derivative of the 3rd order polynomial changes sign 1 time
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    ## if before the change of sign the derivative of the 3rd order poly is positive, 
                    ## then we know there is a Descending Segment after the change of sign
                    Descending_segment=True
                    
                    # ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    # descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    ## if before the change of sign the derivative of the 3rd order poly is negative, 
                    ## then we know the "Descending Segment" is fitted to the beginning of the data = artifact --> there is no Descending segment after the Acsending Segment
                    Descending_segment=False
                    #ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                    
            elif len(zero_crossings_3rd_poly_model_freq_diff)==2:# if the derivative of the 3rd order polynomial changes sign 2 times
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
                
                
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    ## if before the first change of sign the derivative of the 3rd order poly is positive
                    ## then we consider the Ascending Segment before the first root and the Descending Segment after the First root
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
                    descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
                    
                    
                    trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
                    
                    descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                          trimmed_descending_segment['Frequency_Hz'])
                    
                    if descending_linear_slope_init<=0:
                        Descending_segment = True
                        # ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                        # descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    else:
                        Descending_segment=False
                        #ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=True
                    # ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                    # descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
            
            ### 1 - end
        
        else:
            Descending_segment=False
            ascending_segment = trimmed_3rd_poly_table
        
        
        original_data_table =  original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool})
            
        
        original_data_subset_QC = original_data_table.copy()
        
    
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        
        original_data_subset_QC=original_data_subset_QC.reset_index(drop=True)
        original_data_subset_QC=original_data_subset_QC.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        
        original_x_data = np.array(original_data_subset_QC['Stim_amp_pA'])
        original_y_data = np.array(original_data_subset_QC['Frequency_Hz'])
        extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
        
        
        
        ### 2 - Derive fitting parameters for Hill from polynomial

        extended_poly_fit = best_c3*(extended_x_data)**3+best_c2*(extended_x_data)**2+best_c1*(extended_x_data)+best_c0
        
        extended_poly_fit_diff = 3*best_c3*(extended_x_data)**2 + 2*best_c2*(extended_x_data)+best_c1
        extended_polynomial_fit_table = pd.DataFrame({"Stim_amp_pA":extended_x_data,
                                                      "Frequency_Hz":extended_poly_fit,
                                                      "First_Derivative" : extended_poly_fit_diff})
        
        
        
        trimmed_extended_polynomial_fit_table = extended_polynomial_fit_table.loc[(extended_polynomial_fit_table["Stim_amp_pA"]>=min(trimmed_x_data))&(extended_polynomial_fit_table["Stim_amp_pA"]<=max(trimmed_x_data)),:]
        stimulus_for_max_freq = trimmed_extended_polynomial_fit_table.loc[trimmed_extended_polynomial_fit_table['Frequency_Hz'].idxmax(), 'Stim_amp_pA']
        poly_min = np.nanmin(trimmed_extended_polynomial_fit_table.loc[:,'Frequency_Hz'])
        poly_max = np.nanmax(trimmed_extended_polynomial_fit_table.loc[:,'Frequency_Hz'])
        A_init = 2*poly_max
        A_min = 0
        if Descending_segment :
            A_max = 2*A_init
        else:
            A_max = 10*A_init
        
        poly_up_table = extended_polynomial_fit_table.loc[(extended_polynomial_fit_table['First_Derivative']>0)&(extended_polynomial_fit_table['Stim_amp_pA']<stimulus_for_max_freq),:]
        if poly_up_table.shape[0]==0:
            raise EmptyTrimmedPolynomialFit("No Ascending portion of polynomial fit before maximum frequency")
        
        
        
        if poly_up_table.loc[poly_up_table['Frequency_Hz']<=0,:].shape[0] > 0 or extended_poly_fit_diff[0] <= 0:
        # check if polynomial up fit intersects X_axis over the input space, so if there are any frequency lower than 0 
        #or check if the first derivative of the polynomial fit is negative at  the start of the stimulus space
            

            x0_init = trimmed_x_data[0]
            
        else:
        #if not, an estimate  for x0 is provided by a linear extrapolation backwards from polynomial fit, evaluated at the start of stimulus space

            x_min = np.nanmin(original_x_data)
            poly_x_min = best_c3*(x_min)**3+best_c2*(x_min)**2+best_c1*(x_min)+best_c0
            deriv_x_min = 3*best_c3*(x_min)**2 + 2*best_c2*(x_min)+best_c1
            x0_init = x_min - (poly_x_min/deriv_x_min)
            
        x0_max = x0_init
        
        mid_response = 0.5*poly_max
        
        
        
        if poly_min < 0.5*poly_max:

            stim_for_mid_response = poly_up_table.loc[(poly_up_table['Frequency_Hz'] - mid_response).abs().idxmin(), 'Stim_amp_pA']
            Ka_init = stim_for_mid_response-x0_init
        else:
            Ka_init = 0.5 * (stimulus_for_max_freq-x0_init)
        
        Ka_min = 1e-9
        
        if Descending_segment :
            poly_down_table = extended_polynomial_fit_table.loc[extended_polynomial_fit_table['Stim_amp_pA'] > stimulus_for_max_freq,:]
            poly_down_slope_init,poly_down_intercept_init = linear_fit(poly_down_table["Stim_amp_pA"],
                                                  poly_down_table['Frequency_Hz'])
            
            Sigmoid_x0_init = 0.5 * ((-poly_down_intercept_init/poly_down_slope_init) + stimulus_for_max_freq)
            

            Sigmoid_k_init = 4*poly_down_slope_init/(poly_down_slope_init*stimulus_for_max_freq + poly_down_intercept_init)


            
            if do_plot:
                
                polynomial_plot = p9.ggplot()
                polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
                polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
                polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
                polynomial_plot += p9.geom_abline(slope=poly_down_slope_init,
                                             intercept=poly_down_intercept_init,
                                             colour="red",
                                             linetype='dashed')
                polynomial_plot += p9.ggtitle('3rd order polynomial fit to trimmed_data')
                polynomial_plot += p9.scale_color_manual(values=scale_dict)
                polynomial_plot += p9.scale_shape_manual(values=scale_dict)
                polynomial_fit_dict_scale_dict = {'3rd_order_poly' : 'black',
                              '2nd_order_poly' : 'black',
                              True : 'o',
                              False : 's'
                              
                              }
                polynomial_fit_dict = {"original_data_table":original_data_table,
                                       'trimmed_stimulus_frequency_table' : trimmed_stimulus_frequency_table,
                                       'trimmed_3rd_poly_table': trimmed_3rd_poly_table,
                                       'descending_linear_slope_init':poly_down_slope_init,
                                       "stimulus_for_maximum_frequency": stimulus_for_max_freq,
                                       'descending_linear_intercept_init':poly_down_intercept_init,
                                       'color_shape_dict' : polynomial_fit_dict_scale_dict}
                
                plot_list[f'{i}-Polynomial_fit']=polynomial_fit_dict
                i+=1
                if print_plot==True:
                    polynomial_plot.show()
        else:
            if do_plot:
                
                polynomial_plot = p9.ggplot()
                polynomial_plot += p9.geom_point(original_data_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour="grey")
                polynomial_plot += p9.geom_point(trimmed_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",shape='Passed_QC'),colour='black')
                polynomial_plot += p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz",color='Legend',group='Legend'))
                polynomial_plot += p9.ggtitle('3rd order polynomial fit to trimmed_data')
                polynomial_plot += p9.scale_color_manual(values=scale_dict)
                polynomial_plot += p9.scale_shape_manual(values=scale_dict)
                polynomial_fit_dict_scale_dict = {'3rd_order_poly' : 'black',
                              '2nd_order_poly' : 'black',
                              True : 'o',
                              False : 's'
                              
                              }
                polynomial_fit_dict = {"original_data_table":original_data_table,
                                       'trimmed_stimulus_frequency_table' : trimmed_stimulus_frequency_table,
                                       'trimmed_3rd_poly_table': trimmed_3rd_poly_table,
                                       'color_shape_dict' : polynomial_fit_dict_scale_dict}
                
                
                plot_list[f'{i}-Polynomial_fit']=polynomial_fit_dict
                i+=1
                if print_plot==True:
                    polynomial_plot.show()
        # Final fit
        #Fit Amplitude*Hill*Descending Sigmoid to original data points
        
        Final_amplitude_fit = ConstantModel(prefix='Final_Amp_')
        Final_amplitude_fit_pars = Final_amplitude_fit.make_params()
        Final_amplitude_fit_pars['Final_Amp_c'].set(value=A_init, min= A_min, max = A_max)
        
        
        Hill_model = Model(hill_function, prefix='Final_Hill_')
        Hill_pars = Parameters()
        Hill_pars.add("Final_Hill_x0",value=x0_init, max = x0_max, min = np.nanmin(original_x_data)-100)
        Hill_pars.add("Final_Hill_Half_cst",value=Ka_init, min = Ka_min)
        Hill_pars.add('Final_Hill_Hill_coef',value=1.2, min=1, max= 5)
        
        Final_fit_model =  Final_amplitude_fit*Hill_model
        Final_fit_model_pars = Final_amplitude_fit_pars+Hill_pars
        
        if Descending_segment:
            Sigmoid_model = Model(sigmoid_function_second, prefix="Final_Sigmoid_")
            Sigmoid_pars = Parameters()
            
            Sigmoid_pars.add("Final_Sigmoid_x0",value=Sigmoid_x0_init )
            Sigmoid_pars.add("Final_Sigmoid_k",value=Sigmoid_k_init, max = 0)
            
            Final_fit_model *= Sigmoid_model
            Final_fit_model_pars += Sigmoid_pars
        
        try:
            #Start by doing fit with Least-Square method

            Final_Fit_results_Least_Square = Final_fit_model.fit(original_y_data, Final_fit_model_pars, x=original_x_data,method = 'least_squares' )

            Least_Square_fit_done = True
            Least_Square_Final_fit_parameters = get_parameters_table(Final_fit_model_pars, Final_Fit_results_Least_Square)
            Least_Square_Final_fit_parameters.loc[:,'Fit'] = "Least_Square Final fit to data"
            
            Least_Square_Final_Amp = Final_Fit_results_Least_Square.best_values['Final_Amp_c']
            Least_Square_Final_H_Half_cst=Final_Fit_results_Least_Square.best_values['Final_Hill_Half_cst']
            Least_Square_Final_H_Hill_coef=Final_Fit_results_Least_Square.best_values['Final_Hill_Hill_coef']
            Least_Square_Final_H_x0 = Final_Fit_results_Least_Square.best_values['Final_Hill_x0']
            
            Least_Square_Final_fit_extended = Least_Square_Final_Amp * hill_function(extended_x_data, Least_Square_Final_H_x0, Least_Square_Final_H_Hill_coef, Least_Square_Final_H_Half_cst) 
            Least_Square_Final_fit_original = Least_Square_Final_Amp * hill_function(original_x_data, Least_Square_Final_H_x0, Least_Square_Final_H_Hill_coef, Least_Square_Final_H_Half_cst) 
            if Descending_segment:
                Least_Square_Final_S_x0 = Final_Fit_results_Least_Square.best_values['Final_Sigmoid_x0']
                Least_Square_Final_S_k = Final_Fit_results_Least_Square.best_values['Final_Sigmoid_k']
                Least_Square_Final_fit_extended = Least_Square_Final_Amp * hill_function(extended_x_data, Least_Square_Final_H_x0, Least_Square_Final_H_Hill_coef, Least_Square_Final_H_Half_cst) * sigmoid_function_second(extended_x_data, Least_Square_Final_S_x0, Least_Square_Final_S_k)
                Least_Square_Final_fit_original *= sigmoid_function_second(original_x_data, Least_Square_Final_S_x0, Least_Square_Final_S_k)
            
            Least_square_NRMSE = normalized_root_mean_squared_error(original_y_data, Least_Square_Final_fit_original, Least_Square_Final_fit_extended)
            
            if is_fit_stuck_at_initial_conditions(Least_Square_Final_fit_parameters,'Least_Square Final fit to data') == True:

                raise Exception()
                
        except:

            Least_Square_fit_done = False
            
        try:
            Final_Fit_results_Levenberg = Final_fit_model.fit(original_y_data, Final_fit_model_pars, x=original_x_data, method = "leastsq")
            Levenberg_fit_done = True
            Levenberg_Final_fit_parameters = get_parameters_table(Final_fit_model_pars, Final_Fit_results_Levenberg)
            Levenberg_Final_fit_parameters.loc[:,'Fit'] = "Levenberg Final fit to data"
            
            Levenberg_Final_Amp = Final_Fit_results_Levenberg.best_values['Final_Amp_c']
            Levenberg_Final_H_Half_cst=Final_Fit_results_Levenberg.best_values['Final_Hill_Half_cst']
            Levenberg_Final_H_Hill_coef=Final_Fit_results_Levenberg.best_values['Final_Hill_Hill_coef']
            Levenberg_Final_H_x0 = Final_Fit_results_Levenberg.best_values['Final_Hill_x0']
            
            Levenberg_Final_fit_extended = Levenberg_Final_Amp * hill_function(extended_x_data, Levenberg_Final_H_x0, Levenberg_Final_H_Hill_coef, Levenberg_Final_H_Half_cst) 
            Levenberg_Final_fit_original = Levenberg_Final_Amp * hill_function(original_x_data, Levenberg_Final_H_x0, Levenberg_Final_H_Hill_coef, Levenberg_Final_H_Half_cst) 
            if Descending_segment:
                Levenberg_Final_S_x0 = Final_Fit_results_Levenberg.best_values['Final_Sigmoid_x0']
                Levenberg_Final_S_k = Final_Fit_results_Levenberg.best_values['Final_Sigmoid_k']
                Levenberg_Final_fit_extended *= sigmoid_function_second(extended_x_data, Levenberg_Final_S_x0, Levenberg_Final_S_k)
                Levenberg_Final_fit_original *= sigmoid_function_second(original_x_data, Levenberg_Final_S_x0, Levenberg_Final_S_k)
            
            Levenberg_NRMSE = normalized_root_mean_squared_error(original_y_data, Levenberg_Final_fit_original, Levenberg_Final_fit_extended)
            
            if is_fit_stuck_at_initial_conditions(Levenberg_Final_fit_parameters,'Levenberg Final fit to data') == True:
                
                raise Exception()
        except:
            Levenberg_fit_done = False

            
            
        if Least_Square_fit_done == False and Levenberg_fit_done == False:
            # Both methods failed
            raise CustomFitError("Both fitting methods failed, stucked at initial conditions")
            
            
        elif Least_Square_fit_done == False and Levenberg_fit_done == True:
            # if Least_Square method failed and Levenberg method worked, use the last one as final fit
            Final_Fit_results = Final_Fit_results_Levenberg 
            Fit_NRMSE = Levenberg_NRMSE

        
        elif Least_Square_fit_done == True and Levenberg_fit_done == False:
            # if Least_Square method Worked and Levenberg method failed, use the first one as final fit
            Final_Fit_results = Final_Fit_results_Least_Square 
            Fit_NRMSE = Least_square_NRMSE

        else:
            # if both fit work, use the fit with best (minimum) error
            if Least_square_NRMSE <= Levenberg_NRMSE:
                Fit_NRMSE = Least_square_NRMSE
                Final_Fit_results = Final_Fit_results_Least_Square 

            else:
                Final_Fit_results = Final_Fit_results_Levenberg 
                Fit_NRMSE = Levenberg_NRMSE

        
        
        
        Final_fit_parameters = get_parameters_table(Final_fit_model_pars, Final_Fit_results)
        Final_fit_parameters.loc[:,'Fit'] = "Final fit to data"
        
        Final_Amp = Final_Fit_results.best_values['Final_Amp_c']

        
        Final_H_Half_cst=Final_Fit_results.best_values['Final_Hill_Half_cst']
        Final_H_Hill_coef=Final_Fit_results.best_values['Final_Hill_Hill_coef']
        Final_H_x0 = Final_Fit_results.best_values['Final_Hill_x0']
        

        
        if Descending_segment:
            Legend = "Final_Hill_Sigmoid_fit"
            obs = 'Hill-Sigmoid'
            Final_S_x0 = Final_Fit_results.best_values['Final_Sigmoid_x0']
            Final_S_k = Final_Fit_results.best_values['Final_Sigmoid_k']
            Final_fit_extended = Final_Amp * hill_function(extended_x_data, Final_H_x0, Final_H_Hill_coef, Final_H_Half_cst) * sigmoid_function_second(extended_x_data, Final_S_x0, Final_S_k)
            Final_fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Final_fit_extended})
            Final_fit_table['Legend'] = Legend
            
            Initial_Hill_sigmoid_fit = A_init * hill_function(extended_x_data, x0_init, 1.2, Ka_init) * sigmoid_function_second(extended_x_data, Sigmoid_x0_init, Sigmoid_k_init)
            Initial_Fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Initial_Hill_sigmoid_fit})
            Initial_Fit_table['Legend'] = "Initial conditions"
            
        else:
            Legend = "Final_Hill_fit"
            obs = 'Hill'
            Final_fit_extended = Final_Amp * hill_function(extended_x_data, Final_H_x0, Final_H_Hill_coef, Final_H_Half_cst) 
            Final_fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Final_fit_extended})
            Final_fit_table['Legend'] = Legend
            
            Initial_Hill_fit = A_init * hill_function(extended_x_data, x0_init, 1.2, Ka_init)
            Initial_Fit_table = pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                                      "Frequency_Hz" : Initial_Hill_fit})
            Initial_Fit_table['Legend'] = "Initial conditions"
            Final_S_x0 = np.nan
            Final_S_k = np.nan
            
        
        Hill_sigmoid_color_dict = {"Final_Hill_Fit":"blue",
                                   "Final_Hill_Sigmoid_fit" : "blue",
                                   "Initial conditions" : "red"}
        
        parameters_table = pd.concat([third_order_fit_params_table, Final_fit_parameters], ignore_index= True)
        
        
        Hill_Sigmoid_plot =  p9.ggplot()
        Hill_Sigmoid_plot += p9.geom_point(original_data_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
        Hill_Sigmoid_plot += p9.geom_line(Initial_Fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_Sigmoid_plot += p9.geom_line(Final_fit_table,p9.aes(x='Stim_amp_pA',y="Frequency_Hz", group='Legend', color='Legend'))
        Hill_Sigmoid_plot += p9.scale_color_manual(Hill_sigmoid_color_dict)
        Hill_Sigmoid_plot += p9.ggtitle(f"Final_Hill_Sigmoid_Fit cell {cell_id}")
        if do_plot:
            
            Hill_Sigmoid_fit_dict = {"original_data_table":original_data_table,
                                       'Initial_Fit_table':Initial_Fit_table, 
                                       'Final_fit_table' : Final_fit_table,
                                       
                                   
                                   'color_shape_dict' : Hill_sigmoid_color_dict,
                                   }
            
            
            plot_list[f'{i}-Final_Hill_Sigmoid_Fit']=Hill_Sigmoid_fit_dict
            i+=1
            
            if print_plot:
                Hill_Sigmoid_plot.show()
        return obs,Final_fit_table,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_k,Legend, Fit_NRMSE ,plot_list, parameters_table
    
    except(StopIteration):
          obs='Error_Iteration'

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_k=np.nan
          Legend=np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_k,Legend, Fit_NRMSE,plot_list, parameters_table
             
    except (ValueError):
          obs='Error_Value'
          
          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_k=np.nan
          Legend=np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_k,Legend, Fit_NRMSE ,plot_list, parameters_table
              
              
    except (RuntimeError):
          obs='Error_Runtime'

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_k=np.nan
          Legend=np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_k,Legend, Fit_NRMSE ,plot_list, parameters_table
    
    except CustomFitError as e:
            # Handle the custom exception
          obs = str(e)  # Capture the custom error message
          Final_fit_extended = np.nan
          Final_Amp = np.nan
          Final_H_Hill_coef = np.nan
          Final_H_Half_cst = np.nan
          Final_H_x0 = np.nan
          Final_S_x0 = np.nan
          Final_S_k = np.nan
          Legend = np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs, Final_fit_extended, Final_Amp, Final_H_x0, Final_H_Half_cst, Final_H_Hill_coef, Final_S_x0, Final_S_k, Legend, Fit_NRMSE, plot_list, parameters_table
      
    except NotEnoughValueError as e:
            # Handle the custom exception
          obs = str(e)  # Capture the custom error message
          Final_fit_extended = np.nan
          Final_Amp = np.nan
          Final_H_Hill_coef = np.nan
          Final_H_Half_cst = np.nan
          Final_H_x0 = np.nan
          Final_S_x0 = np.nan
          Final_S_k = np.nan
          Legend = np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs, Final_fit_extended, Final_Amp, Final_H_x0, Final_H_Half_cst, Final_H_Hill_coef, Final_S_x0, Final_S_k, Legend, Fit_NRMSE, plot_list, parameters_table

    except StimulusSpaceSamplingNotSparseEnough as e:
            # Handle the custom exception
          obs = str(e)  # Capture the custom error message
          Final_fit_extended = np.nan
          Final_Amp = np.nan
          Final_H_Hill_coef = np.nan
          Final_H_Half_cst = np.nan
          Final_H_x0 = np.nan
          Final_S_x0 = np.nan
          Final_S_k = np.nan
          Legend = np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs, Final_fit_extended, Final_Amp, Final_H_x0, Final_H_Half_cst, Final_H_Hill_coef, Final_S_x0, Final_S_k, Legend, Fit_NRMSE, plot_list, parameters_table
    except EmptyTrimmedPolynomialFit as e:
          obs = str(e)  # Capture the custom error message
          Final_fit_extended = np.nan
          Final_Amp = np.nan
          Final_H_Hill_coef = np.nan
          Final_H_Half_cst = np.nan
          Final_H_x0 = np.nan
          Final_S_x0 = np.nan
          Final_S_k = np.nan
          Legend = np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs, Final_fit_extended, Final_Amp, Final_H_x0, Final_H_Half_cst, Final_H_Hill_coef, Final_S_x0, Final_S_k, Legend, Fit_NRMSE, plot_list, parameters_table
    except (TypeError) as e:
          obs=str(e)

          Final_fit_extended=np.nan
          Final_Amp=np.nan
          Final_H_Hill_coef=np.nan
          Final_H_Half_cst=np.nan
          Final_H_x0=np.nan
          Final_S_x0=np.nan
          Final_S_k=np.nan
          Legend=np.nan
          Fit_NRMSE = np.nan
          parameters_table = pd.DataFrame()
          return obs,Final_fit_extended,Final_Amp,Final_H_x0, Final_H_Half_cst,Final_H_Hill_coef,Final_S_x0, Final_S_k,Legend, Fit_NRMSE ,plot_list, parameters_table
    
        
            
            
def is_fit_stuck_at_initial_conditions(parameters_table,fit):
    sub_parameters_table = parameters_table.loc[parameters_table['Fit'] == fit,:]
    number_params = sub_parameters_table.shape[0]
    stucked_parameters = 0
    for elt in sub_parameters_table.index:
        initial_value = sub_parameters_table.loc[elt, "Initial Value"]
        resulting_value = sub_parameters_table.loc[elt, "Resulting Value"]
        
        if round(initial_value,4) == round(resulting_value,4):
            stucked_parameters+=1

    
    if stucked_parameters == number_params:
        return True
    else:
        return False
    
    
def extract_IO_features(original_stimulus_frequency_table,response_type,fit_model_table,Descending_segment,Legend,response_duration,cell_id = '--', do_plot=False,plot_list=None,print_plot=False):
    '''
    From the fitted IO relationship, computed neuronal firing features 

    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        Table gathering for each sweep the stimulus amplitude, and the frequency of spikes over the response.
            
    fit_model_table : pd.DataFrame
        Table of resulting fit (Stimulus-Frequency) .
        
        
    Descending_segment : Boolean
        Wether a Descending segment has been detected.
        
    Legend : str

    do_plot : Boolean, optional
        Do plot. The default is False.
   	        
    plot_list : List, optional
        List of plot during the fitting procedures (required do_plot == True). The default is None.
   	        
    print_plot : Boolean, optional
        Print plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the result of the fit.
        
    best_QNRMSE : Float
        Godness of fit.
        
    Gain : Float
        Neuronal Gain, slope of the linear portion of the ascending segment .
        
    Threshold : Float
        Firing threshold, x-axis intercept of the fit to the linear portion of the ascending segment .
        
    Saturation_frequency : Float
        Firing Saturation (if any, otherwise nan), maximum frequency of the IO fit.
        
    Saturation_stimulation : Float
        Stimulus eliciting Saturation (if any, otherwise nan), Stimulus amplitude eleiciting maximum frequency of the IO fit.
        
    plot_list : List
        List of plot during the fitting procedures (required do_plot == True).

    '''
    try:
        scale_dict={True:"o",
                    False:'s'}
        if Descending_segment:
            obs='Hill-Sigmoid'
        else:
            obs='Hill'


        original_data_table=original_stimulus_frequency_table.copy()
        original_data_table = original_data_table.dropna()
        
        if do_plot == True:
            original_data_table=original_data_table.astype({'Passed_QC':bool})
        original_data_subset_QC = original_data_table.copy()
        
        if 'Passed_QC' in original_data_subset_QC.columns:
            original_data_subset_QC=original_data_subset_QC[original_data_subset_QC['Passed_QC']==True]
        

        extended_x_data= fit_model_table['Stim_amp_pA']
        predicted_y_data = fit_model_table['Frequency_Hz']
        
        IO_min = np.nanmin(fit_model_table.loc[:,'Frequency_Hz'])
        IO_max = np.nanmax(fit_model_table.loc[:,'Frequency_Hz'])
        
        fit_start_frequency = 0.5 * (1.25*IO_min + 0.75*IO_max)
        fit_end_frequency = 0.5 * (0.75*IO_min + 1.25*IO_max)

        fit_model_table = fit_model_table.sort_values(by=['Stim_amp_pA'])
        fit_model_table = fit_model_table.reset_index(drop=True)
        
        for elt in fit_model_table.index:
            if fit_model_table.loc[elt, "Frequency_Hz"] >= fit_start_frequency:
                start_index = elt
                break
        
        for elt in fit_model_table.index:
            if fit_model_table.loc[elt, "Frequency_Hz"] >= fit_end_frequency:
                end_index = elt
                break
            
        linear_portion_table = fit_model_table.loc[start_index:end_index,:].copy()
        Gain,Intercept=linear_fit(linear_portion_table.loc[:,"Stim_amp_pA"],linear_portion_table.loc[:,"Frequency_Hz"])
        
        
        gain_table = pd.DataFrame({'Stim_amp_pA':np.array(fit_model_table.loc[start_index:end_index,'Stim_amp_pA']),
                                   "Frequency_Hz" : np.array(fit_model_table.loc[start_index:end_index,'Frequency_Hz'])})
        stimulus_for_max_freq = fit_model_table.loc[fit_model_table['Frequency_Hz'].idxmax(), 'Stim_amp_pA']
        
        fit_derivative = np.diff(fit_model_table.loc[:,'Frequency_Hz'])
        fit_derivative = np.insert(fit_derivative, 0, np.nan)
        fit_model_table.loc[:,'First_derivative'] = fit_derivative
        
        maximum_slope = np.nanmax(fit_model_table.loc[:,'First_derivative'])
        final_slope = fit_derivative[-1]
        max_frequency_index = fit_model_table['Frequency_Hz'].idxmax()
        if final_slope/maximum_slope <= .2:
            #The fit flattens enough compared to the maximum slope
            #Then consider there is Saturation
            
            Saturation_frequency = fit_model_table.loc[max_frequency_index,'Frequency_Hz']
            Saturation_stimulation = fit_model_table.loc[max_frequency_index,'Stim_amp_pA']
            
        else:
            Saturation_frequency = np.nan
            Saturation_stimulation = np.nan
        
        
        #The threshold is defined as the stimulus eliciting the minimum detectable response measure over the duration of the response considered
        if response_type == 'Time_based':
            minimum_response = 1/response_duration
            
        else:
            minimum_response = 1
        filtered_df = fit_model_table.loc[fit_model_table['Frequency_Hz'] > minimum_response,:]
        Threshold = filtered_df['Stim_amp_pA'].min()


        model_table=pd.DataFrame(np.column_stack((extended_x_data,predicted_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
        model_table['Legend']=Legend
        
        #Check if there is a response failure
        Maximum_stimulus_index = fit_model_table['Stim_amp_pA'].idxmax()
        
        frequency_at_max_stim =  fit_model_table.loc[Maximum_stimulus_index, 'Frequency_Hz']
        maximum_frequency = np.nanmax(fit_model_table.loc[:,'Frequency_Hz'])

        if frequency_at_max_stim <= 0.5*maximum_frequency:
            descending_portion_table = fit_model_table.loc[max_frequency_index:,:]
            for elt in descending_portion_table.index:
                if descending_portion_table.loc[elt,"Frequency_Hz"] <= 0.5*maximum_frequency:
                    IO_fail_stim = descending_portion_table.loc[elt,"Stim_amp_pA"]
                    IO_fail_freq = descending_portion_table.loc[elt,"Frequency_Hz"]
                    break
        else:
            IO_fail_stim = np.nan
            IO_fail_freq = np.nan
            
        
        

        if do_plot:

            feature_plot=p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point(p9.aes(shape='Passed_QC'))

            feature_plot+=p9.geom_line(model_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz',color='Legend',group='Legend'))
            feature_plot += p9.geom_line(gain_table, p9.aes(x='Stim_amp_pA',y='Frequency_Hz'), color="green")
            feature_plot+=p9.geom_abline(p9.aes(intercept=Intercept,slope=Gain), color = "green")
            Threshold_table = pd.DataFrame({'Stim_amp_pA':[Threshold],'Frequency_Hz':[0]})
            
            feature_plot+=p9.geom_point(Threshold_table,p9.aes(x=Threshold_table["Stim_amp_pA"],y=Threshold_table["Frequency_Hz"]),color='green')
            feature_plot_dict={'original_data_table':original_data_table,
                               "model_table" : model_table,
                               "gain_table" : gain_table,
                               'intercept':Intercept,
                               "stimulus_for_maximum_frequency": stimulus_for_max_freq,
                               'Gain':Gain,
                               "Threshold" : Threshold_table}
            
            if not np.isnan(Saturation_frequency):
                sat_model_table = pd.DataFrame({'Stim_amp_pA':[Saturation_stimulation],'Frequency_Hz':[Saturation_frequency]})
                feature_plot += p9.geom_point(sat_model_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'),color="green")
                feature_plot_dict['Saturation'] = sat_model_table
                
            if not np.isnan(IO_fail_stim):
                IO_fail_table = pd.DataFrame({'Stim_amp_pA':[IO_fail_stim],'Frequency_Hz':[IO_fail_freq]})
                feature_plot += p9.geom_point(IO_fail_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'),color="red")
                feature_plot_dict['Response_Failure'] = IO_fail_table
                
            feature_plot += p9.scale_shape_manual(values=scale_dict)
            
            
            plot_list[f"{len(plot_list.keys())+1}-IO_fit"]=feature_plot_dict


            if print_plot==True:
                print(feature_plot)
    

            
        return obs,Gain,Threshold,Saturation_frequency,Saturation_stimulation, IO_fail_stim, IO_fail_freq ,plot_list
            
    except (TypeError) as e:
        
        obs=str(e)
        
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        IO_fail_stim = np.nan
        IO_fail_freq = np.nan
        
        return obs,Gain,Threshold,Saturation_frequency,Saturation_stimulation,IO_fail_stim, IO_fail_freq ,plot_list
    except (ValueError) as e:
        
        obs=str(e)
        
        Gain=np.nan
        Threshold=np.nan
        Saturation_frequency=np.nan
        Saturation_stimulation=np.nan
        IO_fail_stim = np.nan
        IO_fail_freq = np.nan
        return obs,Gain,Threshold,Saturation_frequency,Saturation_stimulation,IO_fail_stim, IO_fail_freq ,plot_list

    
    
def get_parameters_table(fit_model_params, result):
    
    initial_params = {param: fit_model_params[param].value for param in fit_model_params}
    result_params = {param: result.params[param].value for param in result.params} 
    min_params = {param: fit_model_params[param].min for param in fit_model_params}
    max_params = {param: fit_model_params[param].max for param in fit_model_params}
    parameters_table = pd.DataFrame({
        'Parameter': initial_params.keys(),
        'Initial Value': initial_params.values(),
        'Min Value': min_params.values(),
        'Max Value': max_params.values(),
        'Resulting Value': result_params.values()

    })
    
    return parameters_table
    
    
    



def plot_IO_fit(plot_list, plot_to_do, return_plot=False):
    

    symbol_map = {True: 'circle', False: 'circle-x-open'}
    if "-Polynomial_fit" in plot_to_do:
    
        polynomial_fit_dict = plot_list["1-Polynomial_fit"]
        # Extract data from dictionary
        original_data_table = polynomial_fit_dict["original_data_table"]
        trimmed_stimulus_frequency_table = polynomial_fit_dict['trimmed_stimulus_frequency_table']
        trimmed_3rd_poly_table = polynomial_fit_dict['trimmed_3rd_poly_table']
        color_shape_dict = polynomial_fit_dict['color_shape_dict']

        
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]

        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle'),
            name='Original',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open',color="orange"),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add trimmed stimulus frequency points
        fig.add_trace(go.Scatter(
            x=trimmed_stimulus_frequency_table['Stim_amp_pA'],
            y=trimmed_stimulus_frequency_table['Frequency_Hz'],
            mode='markers',
            marker=dict(color='black'),
            name='Trimmed Stimulus Frequency',
            text=trimmed_stimulus_frequency_table['Passed_QC']
        ))
        
        # Add polynomial fit lines
        for legend in trimmed_3rd_poly_table['Legend'].unique():
            legend_data = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Legend'] == legend]
            fig.add_trace(go.Scatter(
                x=legend_data['Stim_amp_pA'],
                y=legend_data['Frequency_Hz'],
                mode='lines',
                name=legend,
                line=dict(color=color_shape_dict[legend])
            ))
        
        # Update layout
        fig.update_layout(
            title='3rd order polynomial fit to trimmed_data',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
    elif '-Final_Hill_Sigmoid_Fit' in plot_to_do:
        
        Hill_Sigmoid_fit_dict = plot_list['2-Final_Hill_Sigmoid_Fit']
        # Extract data from the dictionary
        original_data_table = Hill_Sigmoid_fit_dict['original_data_table']
        Initial_Fit_table = Hill_Sigmoid_fit_dict['Initial_Fit_table']
        Final_fit_table = Hill_Sigmoid_fit_dict['Final_fit_table']
        color_shape_dict = Hill_Sigmoid_fit_dict['color_shape_dict']
        
        # Create the plotly figure
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
        
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='black'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='orange'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        fig.add_trace(go.Scatter(x=Initial_Fit_table['Stim_amp_pA'], 
                                 y=Initial_Fit_table['Frequency_Hz'], 
                                 mode='lines', 
                                 name="Initial conditions", 
                                 line=dict(color=color_shape_dict["Initial conditions"])))
        
        fig.add_trace(go.Scatter(x=Final_fit_table['Stim_amp_pA'], 
                                 y=Final_fit_table['Frequency_Hz'], 
                                 mode='lines', 
                                 name="Final Fit", 
                                 line=dict(color=color_shape_dict["Final_Hill_Sigmoid_fit"])))
        
        
        # Update the layout
        fig.update_layout(
            title='Final_Hill_Sigmoid_Fit',
            xaxis_title='Stim_amp_pA',
            yaxis_title='Frequency_Hz',
            legend_title='Legend'
        )
        
    elif "-IO_fit" in plot_to_do:
        
        feature_plot_dict = plot_list['3-IO_fit']
        
        # Extract data from the dictionary
        original_data_table = feature_plot_dict['original_data_table']
        model_table = feature_plot_dict['model_table']
        gain_table = feature_plot_dict['gain_table']
        Intercept = feature_plot_dict['intercept']
        Gain = feature_plot_dict['Gain']
        Threshold_table = feature_plot_dict['Threshold']
        stimulus_for_max_freq = feature_plot_dict["stimulus_for_maximum_frequency"]
        
        # Create the plotly figure
        passed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==True,:]
        failed_QC_table = original_data_table.loc[original_data_table['Passed_QC']==False,:]
       
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=passed_QC_table['Stim_amp_pA'],
            y=passed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle',color='black'),
            name='Original Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        
        fig.add_trace(go.Scatter(
            x=failed_QC_table['Stim_amp_pA'],
            y=failed_QC_table['Frequency_Hz'],
            mode='markers',
            marker=dict(symbol='circle-x-open', color='orange'),
            name='Failed QC Data',
            text=original_data_table['Sweep'],
            hoverinfo='text+x+y'
        ))
        
        # Add model line
        fig.add_trace(go.Scatter(x=model_table['Stim_amp_pA'], 
                                  y=model_table['Frequency_Hz'], 
                                  mode='lines', 
                                  name='IO Fit',
                                  line=dict(color='blue')))
        
        # Add model line
        fig.add_trace(go.Scatter(x=gain_table['Stim_amp_pA'], 
                                  y=gain_table['Frequency_Hz'], 
                                  mode='lines', 
                                  name='Linear IO portion',
                                  line=dict(color='green', 
                                            width = 6,
                                            dash='dot')))
        
        # Add the abline (slope and intercept)
        x_range = np.arange(original_data_table['Stim_amp_pA'].min(), stimulus_for_max_freq, 1)

        fig.add_trace(go.Scatter(x=x_range, 
                                  y=Intercept + Gain * x_range, 
                                  mode='lines', 
                                  name='Linear Fit', 
                                  line=dict(color='red', dash='dash')))
        
        # Add the Threshold points
        fig.add_trace(go.Scatter(x=Threshold_table["Stim_amp_pA"], 
                                  y=Threshold_table["Frequency_Hz"], 
                                  mode='markers', 
                                  name='Threshold', 
                                  marker=dict(color='green', size = 10, symbol='cross')))
        
        # Add Saturation points if not NaN
        if "Saturation" in feature_plot_dict.keys():
            Saturation_table = feature_plot_dict['Saturation']
            fig.add_trace(go.Scatter(x=Saturation_table['Stim_amp_pA'], 
                                      y=Saturation_table['Frequency_Hz'], 
                                      mode='markers', 
                                      name='Saturation', 
                                      marker=dict(color='green', size = 10, symbol='triangle-up')))
        
        if "Response_Failure" in feature_plot_dict.keys():
            response_failure_table = feature_plot_dict['Response_Failure']
            fig.add_trace(go.Scatter(x=response_failure_table['Stim_amp_pA'], 
                                      y=response_failure_table['Frequency_Hz'], 
                                      mode='markers', 
                                      name='Response Failure', 
                                      marker=dict(color='red', size = 10, symbol='x')))
        
    
    if return_plot == False:
        fig.show()
    else:
        return fig

        



def extract_inst_freq_table(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table_inst : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        the sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_time : float
        Response duration in s to consider
    

    Returns
    -------
    interval_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''
    
    maximum_nb_interval =0
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes

            
            
    new_columns=["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]

    

    SF_table = SF_table.reindex(SF_table.columns.tolist() + new_columns ,axis=1)

    for current_sweep in sweep_list:


        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)])
        spike_time_list=np.array(df.loc[:,'Time_s'])
        
        # Put a minimum number of spikes to compute adaptation
        if len(spike_time_list) >2:
            for current_spike_time_index in range(1,len(spike_time_list)):
                current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])

                SF_table.loc[current_sweep,str('Interval_'+str(current_spike_time_index))]=current_inst_frequency

                
            SF_table.loc[current_sweep,'Interval_1':]/=SF_table.loc[current_sweep,'Interval_1']

    interval_freq_table=pd.DataFrame(columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    isnull_table.columns=SF_table.columns
    isnull_table.index=SF_table.index
    
    for interval,col in enumerate(new_columns):
        for line in sweep_list:
            if isnull_table.loc[line,col] == False:

                new_line=pd.DataFrame([int(interval)+1, # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line]).T# Sweep id
                                   
                new_line.columns=['Spike_Interval','Normalized_Inst_frequency','Stimulus_amp_pA','Sweep']
                interval_freq_table=pd.concat([interval_freq_table,new_line],ignore_index=True)
                
    
    interval_freq_table = pd.merge(interval_freq_table,sweep_QC_table,on='Sweep')
    return interval_freq_table

def get_min_freq_step(original_stimulus_frequency_table,do_plot=False):
    '''
    Estimate the initial frequency step, in order to decipher between Type I and TypeII neurons
    Fit the non-zero response, and use the fit value at the first stimulus amplitude eliciting a response.


    Parameters
    ----------
    original_stimulus_frequency_table : pd.DataFrame
        DataFrame containing one row per sweep, with the corresponding input current and firing frequency.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    minimum_freq_step : float
        Estimation of the initial frequency step
    
    minimum_freq_step_stim,
        Stimulus amplitude corresponding to the first frequency step
    
    noisy_spike_freq_threshold,
        Frequency value considered as 'noisy' at the beginning of the non-zero responses values 
    
    np.nanmax(original_data_table['Frequency_Hz'])
        Maximum frequency observed

    '''
    
    original_data_table =  original_stimulus_frequency_table.copy()
   
    original_data = original_data_table.copy()
    
    original_data = original_data.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    original_x_data = np.array(original_data['Stim_amp_pA'])
    original_y_data = np.array(original_data['Frequency_Hz'])
    extended_x_data=pd.Series(np.arange(min(original_x_data),max(original_x_data),.1))
    
    first_stimulus_frequency_table=original_data_table.copy()
    first_stimulus_frequency_table=first_stimulus_frequency_table.reset_index(drop=True)
    first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
    
    try:
        ### 0 - Trim Data
        response_threshold = (np.nanmax(first_stimulus_frequency_table.loc[:,"Frequency_Hz"])-np.nanmin(first_stimulus_frequency_table.loc[:,"Frequency_Hz"]))/20
    
        
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'],ascending=False )
        for elt in range(first_stimulus_frequency_table.shape[0]):
            if first_stimulus_frequency_table.iloc[0,2] < response_threshold :
                first_stimulus_frequency_table=first_stimulus_frequency_table.drop(first_stimulus_frequency_table.index[0])
            else:
                break
        
        
            
        ### 0 - end
        
        ### 1 - Try to fit a 3rd order polynomial
        
        first_stimulus_frequency_table=first_stimulus_frequency_table.sort_values(by=['Stim_amp_pA','Frequency_Hz'])
        trimmed_x_data=first_stimulus_frequency_table.loc[:,'Stim_amp_pA']
        trimmed_y_data=first_stimulus_frequency_table.loc[:,"Frequency_Hz"]
        extended_x_trimmed_data=pd.Series(np.arange(min(trimmed_x_data),max(trimmed_x_data),.1))
        
       
    
        third_order_poly_model = PolynomialModel(degree = 3)
        pars = third_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
        
        if len(trimmed_y_data)>=4:
            
            third_order_poly_model_results = third_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
        
            best_c0 = third_order_poly_model_results.best_values['c0']
            best_c1 = third_order_poly_model_results.best_values['c1']
            best_c2 = third_order_poly_model_results.best_values['c2']
            best_c3 = third_order_poly_model_results.best_values['c3']
            
            
            
            
        
            extended_trimmed_3rd_poly_model = best_c3*(extended_x_trimmed_data)**3+best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            
            extended_trimmed_3rd_poly_model_freq_diff=np.diff(extended_trimmed_3rd_poly_model)
            trimmed_3rd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_3rd_poly_model})
            trimmed_3rd_poly_table['Legend']='3rd_order_poly'
            my_plot=p9.ggplot(first_stimulus_frequency_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point()
            my_plot+=p9.geom_line(trimmed_3rd_poly_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))
            if do_plot:
                
                print(my_plot)
            
            zero_crossings_3rd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_3rd_poly_model_freq_diff)))[0] # detect last index before change of sign
           
          
            
            if len(zero_crossings_3rd_poly_model_freq_diff)==0:
                Descending_segment=False
                ascending_segment = trimmed_3rd_poly_table.copy()
                
            elif len(zero_crossings_3rd_poly_model_freq_diff)==1:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=False
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] >= first_stim_root ]
                    
            elif len(zero_crossings_3rd_poly_model_freq_diff)==2:
                first_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[0]]
                second_stim_root = extended_x_trimmed_data[zero_crossings_3rd_poly_model_freq_diff[1]]
                
                
                if extended_trimmed_3rd_poly_model_freq_diff[0]>=0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] <= first_stim_root ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root]
                    
                elif extended_trimmed_3rd_poly_model_freq_diff[0]<0:
                    Descending_segment=True
                    ascending_segment = trimmed_3rd_poly_table[(trimmed_3rd_poly_table['Stim_amp_pA'] > first_stim_root) & (trimmed_3rd_poly_table['Stim_amp_pA'] <= second_stim_root) ]
                    descending_segment = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Stim_amp_pA'] > second_stim_root]
        
        else:
            Descending_segment=False
                
                
                
                
                
            
       
            ### 1 - end
            

        
        ### 2 - Trim polynomial fit; keep [mean-0.5*poly_amplitude ; [mean+0.5*poly_amplitude]
        if Descending_segment:
            
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            descending_segment_mean_freq = np.mean(descending_segment['Frequency_Hz'])
            descending_segment_amplitude_freq = np.nanmax(descending_segment['Frequency_Hz']) - np.nanmin(descending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            trimmed_descending_segment = descending_segment[(descending_segment['Frequency_Hz'] >= (descending_segment_mean_freq-0.5*descending_segment_amplitude_freq)) & (descending_segment['Frequency_Hz'] <= (descending_segment_mean_freq+0.5*descending_segment_amplitude_freq))]
            
            max_freq_poly_fit=np.nanmax(trimmed_3rd_poly_table['Frequency_Hz'])
            trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
            trimmed_descending_segment.loc[trimmed_descending_segment.index,'Legend']="Trimmed_desc_seg"
           
        
        else:
            #end_slope_positive --> go for 2nd order polynomial
            second_order_poly_model = PolynomialModel(degree = 2)
            pars = second_order_poly_model.guess(trimmed_y_data, x=trimmed_x_data)
            second_order_poly_model_results = second_order_poly_model.fit(trimmed_y_data, pars, x=trimmed_x_data)
    
            best_c0 = second_order_poly_model_results.best_values['c0']
            best_c1 = second_order_poly_model_results.best_values['c1']
            best_c2 = second_order_poly_model_results.best_values['c2']
            
            extended_trimmed_2nd_poly_model = best_c2*(extended_x_trimmed_data)**2+best_c1*(extended_x_trimmed_data)+best_c0
            extended_trimmed_2nd_poly_model_freq_diff=np.diff(extended_trimmed_2nd_poly_model)
            trimmed_2nd_poly_table = pd.DataFrame({'Stim_amp_pA' : extended_x_trimmed_data,
                                                   "Frequency_Hz" : extended_trimmed_2nd_poly_model})
            
           
            
            zero_crossings_2nd_poly_model_freq_diff = np.where(np.diff(np.sign(extended_trimmed_2nd_poly_model_freq_diff)))[0] # detect last index before change of sign
            
            if len(zero_crossings_2nd_poly_model_freq_diff) == 1:
    
                if extended_trimmed_2nd_poly_model_freq_diff[0]<0:
                    trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] >= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
                else:
                    trimmed_2nd_poly_table = trimmed_2nd_poly_table[trimmed_2nd_poly_table['Stim_amp_pA'] <= extended_x_trimmed_data[(zero_crossings_2nd_poly_model_freq_diff[0]+1)] ]
             
                
            ascending_segment = trimmed_2nd_poly_table.copy()
            ascending_segment_mean_freq = np.mean(ascending_segment['Frequency_Hz'])
            ascending_segment_amplitude_freq = np.nanmax(ascending_segment['Frequency_Hz']) - np.nanmin(ascending_segment['Frequency_Hz'])
            
            trimmed_ascending_segment = ascending_segment[(ascending_segment['Frequency_Hz'] >= (ascending_segment_mean_freq-0.5*ascending_segment_amplitude_freq)) & (ascending_segment['Frequency_Hz'] <= (ascending_segment_mean_freq+0.5*ascending_segment_amplitude_freq))]
            max_freq_poly_fit=np.nanmax(trimmed_2nd_poly_table['Frequency_Hz'])
            
            
            trimmed_2nd_poly_table.loc[trimmed_2nd_poly_table.index,'Legend']='2nd_order_poly'
            trimmed_ascending_segment.loc[trimmed_ascending_segment.index,'Legend']="Trimmed_asc_seg"
            
        ### 2 - end 
        
        ### 3 - Linear fit on polynomial trimmed data
        ascending_linear_slope_init,ascending_linear_intercept_init=linear_fit(trimmed_ascending_segment["Stim_amp_pA"],
                                             trimmed_ascending_segment['Frequency_Hz'])
        
        ascending_sigmoid_slope=max_freq_poly_fit/(4*ascending_linear_slope_init)
        
        if Descending_segment:

            descending_linear_slope_init,descending_linear_intercept_init=linear_fit(trimmed_descending_segment["Stim_amp_pA"],
                                                  trimmed_descending_segment['Frequency_Hz'])
        
        
        
        
        ### 3 - end
        
        ### 4 - Fit single or double Sigmoid
        
        ascending_sigmoid_fit= Model(sigmoid_function,prefix='asc_')
        ascending_segment_fit_params  = ascending_sigmoid_fit.make_params()
        
        
        ascending_segment_fit_params.add("asc_x0",value=np.nanmean(trimmed_ascending_segment['Stim_amp_pA']))
        ascending_segment_fit_params.add("asc_sigma",value=ascending_sigmoid_slope,min=1e-9,max=5*ascending_sigmoid_slope)
        
        amplitude_fit = ConstantModel(prefix='Amp_')
        amplitude_fit_pars = amplitude_fit.make_params()
        amplitude_fit_pars['Amp_c'].set(value=max_freq_poly_fit,min=0,max=10*max_freq_poly_fit)
        
        ascending_sigmoid_fit *= amplitude_fit
        ascending_segment_fit_params+=amplitude_fit_pars
        
        if Descending_segment:

            
            
           
            descending_sigmoid_slope = max_freq_poly_fit/(4*descending_linear_slope_init)
            
            descending_sigmoid_fit = Model(sigmoid_function,prefix='desc_')
            descending_sigmoid_fit_pars = descending_sigmoid_fit.make_params()
            descending_sigmoid_fit_pars.add("desc_x0",value=np.nanmean(trimmed_descending_segment['Stim_amp_pA']))
            descending_sigmoid_fit_pars.add("desc_sigma",value=descending_sigmoid_slope,min=3*descending_sigmoid_slope,max=-1e-9)
            
            ascending_sigmoid_fit *=descending_sigmoid_fit
            ascending_segment_fit_params+=descending_sigmoid_fit_pars

        ascending_sigmoid_fit_results = ascending_sigmoid_fit.fit(original_y_data, ascending_segment_fit_params, x=original_x_data)
        best_value_Amp = ascending_sigmoid_fit_results.best_values['Amp_c']
        
        best_value_x0_asc = ascending_sigmoid_fit_results.best_values["asc_x0"]
        best_value_sigma_asc = ascending_sigmoid_fit_results.best_values['asc_sigma']
        
        

        full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data , best_value_x0_asc, best_value_sigma_asc)
        full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                    "Frequency_Hz" : full_sigmoid_fit})
        
       
        full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
        
        
        if Descending_segment:
            
            best_value_x0_desc = ascending_sigmoid_fit_results.best_values["desc_x0"]
            best_value_sigma_desc = ascending_sigmoid_fit_results.best_values['desc_sigma']
            
            full_sigmoid_fit = best_value_Amp * sigmoid_function(extended_x_data, best_value_x0_asc, best_value_sigma_asc) * sigmoid_function(extended_x_data , best_value_x0_desc, best_value_sigma_desc)
            full_sigmoid_fit_table=pd.DataFrame({'Stim_amp_pA' : extended_x_data,
                                        "Frequency_Hz" : full_sigmoid_fit})
            full_sigmoid_fit_table['Legend'] = 'Sigmoid_fit'
            
            
        
        if do_plot:
            
            
            double_sigmoid_comps = ascending_sigmoid_fit_results.eval_components(x=original_x_data)
            asc_sig_comp = double_sigmoid_comps['asc_']
            asc_sig_comp_table = pd.DataFrame({'Stim_amp_pA' : original_x_data,
                                        "Frequency_Hz" : asc_sig_comp,
                                        'Legend' : 'Ascending_Sigmoid'})
            
            asc_sig_comp_table['Frequency_Hz'] = asc_sig_comp_table['Frequency_Hz']*max(original_data['Frequency_Hz'])/max(asc_sig_comp_table['Frequency_Hz'])

           
            sigmoid_fit_plot =  p9.ggplot(original_data,p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))+p9.geom_point()
            sigmoid_fit_plot += p9.geom_line(full_sigmoid_fit_table, p9.aes(x='Stim_amp_pA',y="Frequency_Hz"))
            sigmoid_fit_plot += p9.ggtitle("Sigmoid_fit_to original_data")
            
            
            print(sigmoid_fit_plot)
         ### 4 - end

        maximum_fit_frequency=np.nanmax(full_sigmoid_fit)
        maximum_fit_frequency_index = np.nanargmax(full_sigmoid_fit)
        maximum_fit_stimulus = extended_x_data[maximum_fit_frequency_index]
        original_data_table = original_data_table.sort_values(by=['Stim_amp_pA'])
        
        noisy_spike_freq_threshold = np.nanmax([.04*maximum_fit_frequency,2.])
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        

        minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
        minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
        minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]


    except Exception :

        x_data=np.array(original_data.loc[:,'Stim_amp_pA'])
        y_data=np.array(original_data.loc[:,'Frequency_Hz'])
        
        maximum_fit_frequency_index = np.nanargmax(y_data)
        maximum_fit_stimulus = x_data[maximum_fit_frequency_index]
        noisy_spike_freq_threshold = np.nanmax([.04*np.nanmax(y_data),2.])
        until_maximum_data = original_data_table[original_data_table['Stim_amp_pA']<=maximum_fit_stimulus].copy()
       
        until_maximum_data = until_maximum_data[until_maximum_data['Frequency_Hz']>noisy_spike_freq_threshold]
        
        if until_maximum_data.shape[0] == 0:
            without_zero_table = original_data_table[original_data_table['Frequency_Hz']!=0]

            minimum_freq_step_index = np.nanargmin(without_zero_table['Frequency_Hz'])
            minimum_freq_step = without_zero_table.iloc[minimum_freq_step_index,2]
            minimum_freq_step_stim = without_zero_table.iloc[minimum_freq_step_index,1]
        else:
            minimum_freq_step_index = np.nanargmin(until_maximum_data['Frequency_Hz'])
            minimum_freq_step = until_maximum_data.iloc[minimum_freq_step_index,2]
            minimum_freq_step_stim = until_maximum_data.iloc[minimum_freq_step_index,1]



        
    
    finally:

        if do_plot:
            minimum_freq_step_plot = p9.ggplot(original_data_table,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'))+p9.geom_point()
            minimum_freq_step_plot += p9.geom_point(until_maximum_data,p9.aes(x='Stim_amp_pA',y='Frequency_Hz'),color='red')
            minimum_freq_step_plot += p9.geom_vline(xintercept = minimum_freq_step_stim )
            minimum_freq_step_plot += p9.geom_hline(yintercept = minimum_freq_step)
            minimum_freq_step_plot += p9.geom_hline(yintercept = noisy_spike_freq_threshold,linetype='dashed',alpha=.4)
            
            print(minimum_freq_step_plot)
            
        return minimum_freq_step,minimum_freq_step_stim,noisy_spike_freq_threshold,np.nanmax(original_data_table['Frequency_Hz'])
    
def linear_fit(x, y):
    """
    Fit x-y to a line and return the slope of the fit.

    Parameters
    ----------
    x: array of values
    y: array of values
    Returns
    -------
    m: f-I curve slope for the specimen
    c:f-I curve intercept for the specimen

    """

   
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c


def sigmoid_function(x,x0,sigma):
    return (1-(1/(1+np.exp((x-x0)/sigma))))


def sigmoid_function_second(x,x0,k):
    return (1-(1/(1+np.exp(k*(x-x0)))))


def hill_function (x, x0, Hill_coef, Half_cst):
    y=np.empty(x.size)
        
        
        
    if len(max(np.where( x <= x0))) !=0:


        x0_index = max(np.where( x <= x0)[0])+1
        
    
        y[:x0_index]=0.

        y[x0_index:] =(((x[x0_index:]-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x[x0_index:]-x0)**(Hill_coef))))

    else:

        y = (((x-x0)**(Hill_coef))/((Half_cst**Hill_coef)+((x-x0)**(Hill_coef))))

    return y






    
def fit_adaptation_curve(interval_frequency_table_init,do_plot=False):
    '''
    Fit exponential curve to Spike-Interval - Instantaneous firing frequency
    

    Parameters
    ----------
    interval_frequency_table_init : pd.DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the fitting procedure.
        
    best_A , best_B , best_C : Float 
        Fitting results.
        
    RMSE : Float
        Godness of fit.

    '''

    try:
        interval_frequency_table = interval_frequency_table_init.copy()
        interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
        if interval_frequency_table.shape[0]==0:
            obs='Not_enough_spike'

            best_A=np.nan
            best_B=np.nan
            best_C=np.nan
            RMSE=np.nan
            return obs,best_A,best_B,best_C,RMSE
        interval_frequency_table.loc[:,'Spike_Interval']=interval_frequency_table.loc[:,'Spike_Interval'].astype(float)
        interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                                  "Normalized_Inst_frequency":"float",
                                                                  "Stimulus_amp_pA":'float'})
        x_data=interval_frequency_table.loc[:,'Spike_Interval']
        x_data=x_data.astype(float)
        y_data=interval_frequency_table.loc[:,'Normalized_Inst_frequency']
        y_data=y_data.astype(float)

        
        median_table=interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).median(numeric_only=True)
        median_table["Count_weights"]=pd.DataFrame(interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
        median_table["Spike_Interval"]=median_table.index
        median_table["Spike_Interval"]=np.float64(median_table["Spike_Interval"])  
        
        y_delta=y_data.iloc[-1]-y_data.iloc[0]
       
        y_delta_two_third=y_data.iloc[0]-.66*y_delta
        
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        
        initial_time_cst_guess=np.array(x_data)[initial_time_cst_guess_idx]
        
        
        
        
        
        initial_A=(y_data.iloc[0]-y_data.iloc[-1])/np.exp(-x_data.iloc[0]/initial_time_cst_guess)

        decayModel=Model(exponential_decay_function)

        decay_parameters=Parameters()

        decay_parameters.add("A",value=initial_A)
        decay_parameters.add('B',value=initial_time_cst_guess,min=0)
        decay_parameters.add('C',value=median_table["Normalized_Inst_frequency"][max(median_table["Spike_Interval"])])

        result = decayModel.fit(y_data, decay_parameters, x=x_data)

        best_A=result.best_values['A']
        best_B=result.best_values['B']
        best_C=result.best_values['C']


        pred=exponential_decay_function(np.array(x_data),best_A,best_B,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)

        A_norm=best_A/(best_A+best_C)
        C_norm=best_C/(best_A+best_C)
        interval_range=np.arange(1,max(median_table["Spike_Interval"])+1,.1)

        simulation=exponential_decay_function(interval_range,best_A,best_B,best_C)
        norm_simulation=exponential_decay_function(interval_range,A_norm,best_B,C_norm)
        sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])
        norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Spike_Interval","Normalized_Inst_frequency"])



        my_plot=np.nan
        if do_plot==True:

            my_plot=p9.ggplot(interval_frequency_table,p9.aes(x=interval_frequency_table["Spike_Interval"],y=interval_frequency_table["Normalized_Inst_frequency"]))+p9.geom_point(p9.aes(color=interval_frequency_table["Stimulus_amp_pA"]))

            my_plot=my_plot+p9.geom_point(median_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
            my_plot=my_plot+p9.geom_line(sim_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color='black')
            my_plot=my_plot+p9.geom_line(norm_sim_table,p9.aes(x='Spike_Interval',y='Normalized_Inst_frequency'),color="green")

            print(my_plot)


        obs='--'
        return obs,best_A,best_B,best_C,RMSE

    except (StopIteration):
        obs='Error_Iteration'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except (ValueError):
        obs='Error_Value'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        best_A=np.nan
        best_B=np.nan
        best_C=np.nan
        RMSE=np.nan
        return obs,best_A,best_B,best_C,RMSE
    
def normalized_root_mean_squared_error(true, pred,pred_extended):
    '''
    Compute the Root Mean Squared Error, normalized to the fit amplitude(max-min)

    Parameters
    ----------
    true : np.array
        Observed values.
    pred : np.array
        Values predicted by the fit.
    pred_extended : np.array
        Values predicted by the fit with higer number of points .

    Returns
    -------
    nrmse_loss : float
        Fit Error.

    '''
    #Normalization by the interquartile range
    squared_error = np.square((true - pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / true.size)
    min_val = np.nanmin(pred_extended)
    max_val = np.nanmax(pred_extended)

    nrmse_loss = rmse/(max_val-min_val)
    return nrmse_loss

def exponential_decay_function(x,A,B,C):
    '''
    Parameters
    ----------
    x : Array
        interspike interval index array.
    A: flt
        initial instantanous frequency .
    B : flt
        Adaptation index constant.
    C : flt
        intantaneous frequency limit.

    Returns
    -------
    y : array
        Modelled instantanous frequency.

    '''
   
    
    return  A*np.exp(-(x)/B)+C


def fit_adaptation_test(interval_frequency_table_init,do_plot=False):
    '''
    Fit exponential curve to Spike-Interval - Instantaneous firing frequency
    

    Parameters
    ----------
    interval_frequency_table_init : pd.DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    obs : str
        Observation of the fitting procedure.
        
    best_A , best_B , best_C : Float 
        Fitting results.
        
    RMSE : Float
        Godness of fit.

    '''

    try:
        interval_frequency_table = interval_frequency_table_init.copy()
        interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
        

        
        
        interval_frequency_table.loc[:,'Spike_Interval']=interval_frequency_table.loc[:,'Spike_Interval'].astype(float)

        interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                                  "Normalized_feature":"float",
                                                                  "Stimulus_amp_pA":'float'})
       
        median_table = get_median_feature_table_test(interval_frequency_table)

        if median_table.shape[0] < 3:
            raise NotEnoughValueError(f"Fit procedure requires more than 2 values, get {median_table.shape[0]}")
        
        x_data=median_table.loc[:,'Spike_Interval']
        x_data=x_data.astype(float)
        y_data=median_table.loc[:,'Normalized_feature']
        y_data=y_data.astype(float)
        weight_array = median_table.loc[:,'Count_weigths']
        weight_array = weight_array.astype(float)
    
        
        #Get initial condition 
        y_delta=y_data.iloc[-1]-y_data.iloc[0]
        y_delta_two_third=y_data.iloc[0]+.66*y_delta
        
        initial_time_cst_guess_idx=np.argmin(abs(y_data - y_delta_two_third))
        
        initial_time_cst_guess=np.array(x_data)[initial_time_cst_guess_idx]
        
        
        initial_alpha=(y_data.iloc[0]-y_data.iloc[-1])/np.exp(-x_data.iloc[0]/initial_time_cst_guess)
        
        decayModel=Model(exponential_decay_function)
        
        decay_parameters=Parameters()
    
        decay_parameters.add("A",value=initial_alpha)
        decay_parameters.add('B',value=initial_time_cst_guess,min=0, max=np.nanmax(x_data))
        decay_parameters.add('C',value=median_table["Normalized_feature"][max(median_table["Spike_Interval"])])
        
        result = decayModel.fit(y_data, decay_parameters, x=x_data)
    
        best_alpha=result.best_values['A']
        best_beta=result.best_values['B']
        best_gamma=result.best_values['C']

        
    
        y_data_pred=exponential_decay_function(np.array(x_data),best_alpha,best_beta,best_gamma)
        
        squared_error = np.square((y_data - y_data_pred))
        sum_squared_error = np.sum(squared_error)
        RMSE = np.sqrt(sum_squared_error / y_data.size)
        
        if np.abs(best_alpha/best_gamma) < 0.001: # consider the response as constant 
            
            y_data_pred_mean = np.nanmean(y_data_pred)
            average_list = [y_data_pred_mean]*len(y_data_pred)
            y_data_pred = np.array(average_list)
            
            best_alpha = 0
            best_gamma = 0
        
        #retain positive part of the fit, replace negative values by 0.
        positive_y_data_pred = []
        for i in y_data_pred:
            positive_y_data_pred.append(max(0.0,i))
        
        Na = int(np.nanmin([30,np.nanmax(x_data)])) #Select either the maximum between the 30th index and the maximum index of the data 


    
        C_ref = np.nanmin(positive_y_data_pred[:int(Na-1)]) #C_ref is the positive value of the predicted expoential at Na-1
        
        C = (Na-1)*C_ref # the constant part corresponds to the rectangle delimited by Na-1 its  corresponding y predicted value; and the plot's origin (0,0)

        modulated_y_data_pred = []
        for i in positive_y_data_pred:
            new_value = i - positive_y_data_pred[(Na-1)]
            modulated_y_data_pred.append(new_value)
            
        M = np.abs(np.nansum(modulated_y_data_pred))
        
        #we describe the adaptation index as the relative amount a certain characteritic changes during a spike train versus a constant component
        Adaptation_index = M/(C+M)
        obs='--'

        if do_plot:
            
            
            original_sim_table = pd.DataFrame(np.column_stack((x_data,y_data_pred)),columns=["Spike_Interval","Normalized_feature"])
            original_sim_table=original_sim_table.iloc[:int(Na-1),:]
            interval_range=np.arange(0,max(median_table["Spike_Interval"]),.1)
    
            simulation_extended=exponential_decay_function(interval_range,best_alpha,best_beta,best_gamma)
            
            sim_table=pd.DataFrame(np.column_stack((interval_range,simulation_extended)),columns=["Spike_Interval","Normalized_feature"])
            
            plot_dict = {"sim_table":sim_table, 
                         "original_sim_table" : original_sim_table,
                         "interval_frequency_table" : interval_frequency_table,
                         "median_table" : median_table,
                         "Na" : Na,
                         "C_ref" : C_ref,
                         'M' : M,
                         "C" : C}
            return plot_dict
        return obs, Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    
    except (StopIteration):
        obs='Error_Iteration'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan

        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    
    except NotEnoughValueError as e:
            # Handle the custom exception
          obs = str(e)  # Capture the custom error message
          Adaptation_index = np.nan
          M = np.nan
          C = np.nan
          median_table = np.nan
          
          best_alpha=np.nan
          best_beta=np.nan
          best_gamma=np.nan
          RMSE=np.nan

          return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
      
    except (ValueError):
        obs='Error_Value'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan
        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    except(RuntimeError):
        obs='Error_RunTime'
        Adaptation_index = np.nan
        M = np.nan
        C = np.nan
        median_table = np.nan
        
        best_alpha=np.nan
        best_beta=np.nan
        best_gamma=np.nan
        RMSE=np.nan

        return obs,Adaptation_index, M, C, median_table, best_alpha, best_beta, best_gamma, RMSE
    
def plot_adaptation(plot_dict):
    import plotly.graph_objects as go
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict['C_ref']
    interval_frequency_table = plot_dict['interval_frequency_table']
    median_table = plot_dict['median_table']
    M = plot_dict['M']
    C = plot_dict['C']
    
    fig = go.Figure()

    # Line plot
    fig.add_trace(go.Scatter(x=original_sim_table['Spike_Interval'], y=original_sim_table['Normalized_feature'], mode='lines', name='Original Sim Table'))
    
    # Area plot
    fig.add_trace(go.Scatter(x=original_sim_table['Spike_Interval'], y=original_sim_table['Normalized_feature'], fill='tozeroy', fillcolor='#e5c8d6', line=dict(color='rgba(0,0,0,0)')))
    
    # Rect plot (as shape)
    fig.add_shape(type='rect', x0=np.nanmin(sim_table['Spike_Interval']), x1=(Na-1)-1, y0=0, y1=C_ref, fillcolor='gray', opacity=0.7, line=dict(width=0))
    
    # Line plot for sim_table
    fig.add_trace(go.Scatter(x=sim_table['Spike_Interval'], y=sim_table['Normalized_feature'], mode='lines', name='Sim Table'))
    
    # Points for interval_frequency_table with hovertext
    hover_text = [
        f'Sweep: {s}<br>Stim_amp_pA: {p}' 
        for s, p in zip(interval_frequency_table['Sweep'], interval_frequency_table['Stimulus_amp_pA'])
    ]
    
    fig.add_trace(go.Scatter(
        x=interval_frequency_table['Spike_Interval'], 
        y=interval_frequency_table['Normalized_feature'], 
        mode='markers', 
        marker=dict(color=interval_frequency_table['Stimulus_amp_pA'], colorscale='Viridis'),
        name='Interval Frequency Table',
        hovertext=hover_text,
        hoverinfo='text'
    ))
        # Normalize sizes
    max_size = 25
    sizes = (median_table['Count_weigths'] / median_table['Count_weigths'].max()) * max_size
    
   # Points for median_table with normalized sizes and hovertext for Count_weigths
    hover_text_median = [
        f'Count_weight: {w}' 
        for w in median_table['Count_weigths']
    ]
    
    fig.add_trace(go.Scatter(
        x=median_table['Spike_Interval'], 
        y=median_table['Normalized_feature'], 
        mode='markers', 
        marker=dict(size=sizes, color='red', symbol='square'), 
        name='Median Table',
        hovertext=hover_text_median,
        hoverinfo='text'
    ))
    
        # Invisible traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='#e5c8d6'),
        name=f'M: {M}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='gray'),
        name=f'C: {C}'
    ))

    
    fig.update_layout(xaxis_title='Spike Interval', yaxis_title='Normalized Feature')
    
    fig.show()
                
def get_maximum_number_of_spikes_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time):
    maximum_nb_interval =0
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()

    sweep_list=np.array(SF_table.loc[:,"Sweep"])
    for current_sweep in sweep_list:
        
        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        nb_spikes=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)].shape[0])

        if nb_spikes>maximum_nb_interval:
            maximum_nb_interval=nb_spikes
            
    return maximum_nb_interval

def compute_cell_adaptation_behavior(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst):
    
    adaptation_dict = {'Instantaneous_Frequency':'Membrane_potential_mV',
                       "Spike_width_at_half_heigth" : "Time_s",
                       "Spike_heigth" : "Membrane_potential_mV", 
                       'Threshold' : "Membrane_potential_mV",
                       'Upstroke':"Potential_first_time_derivative_mV/s",
                       "Peak" : "Membrane_potential_mV",
                       "Downstroke" : "Potential_first_time_derivative_mV/s",
                       "Fast_Trough" : "Membrane_potential_mV",
                       "fAHP" : "Membrane_potential_mV",
                       "Trough" : "Membrane_potential_mV"}
    
    Adaptation_table = pd.DataFrame(columns = ["Obs", 'Feature', "Measure", "Adaptation_Index", 'M', 'C', 'alpha', 'beta', 'gamma', 'RMSE'])
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()

    
    for feature, measure in adaptation_dict.items():
        
        interval_based_feature = collect_interval_based_features_test(SF_table, cell_sweep_info_table, sweep_QC_table, 0.5, feature, measure)
        
        current_obs, current_Adaptation_index, current_M, current_C, current_median_table, current_best_alpha, current_best_beta, current_best_gamma, current_RMSE = fit_adaptation_test(interval_based_feature, False)

        new_line = pd.DataFrame([current_obs, feature, measure, current_Adaptation_index, current_M, current_C, current_best_alpha, current_best_beta, current_best_gamma, current_RMSE]).T
        new_line.columns = Adaptation_table.columns
        Adaptation_table = pd.concat([Adaptation_table, new_line], ignore_index = True)
    
    return Adaptation_table
        

    
    
def collect_interval_based_features_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time, feature, measure):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    original_SF_table : pd.DataFrame
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).
        
    sweep_QC_table_inst : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        the sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit.
        
    response_time : float
        Response duration in s to consider
    

    Returns
    -------
    interval_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''

    
   
    SF_table=original_SF_table.copy()
    cell_sweep_info_table=original_cell_sweep_info_table.copy()
    sweep_QC_table=sweep_QC_table_inst.copy()
    sweep_list=np.array(SF_table.loc[:,"Sweep"])
   

    maximum_number_of_spikes = get_maximum_number_of_spikes_test(original_SF_table, original_cell_sweep_info_table, sweep_QC_table_inst, response_time)

    if feature == "Instantaneous_Frequency":
        #start interval indexing at 0 --> first interval between spike 0 and spike 1
        new_columns=["Interval_"+str(i) for i in range((maximum_number_of_spikes))]
        
    else:
        # start spike indexing at 0
        new_columns=["Spike_"+str(i) for i in range((maximum_number_of_spikes))]
    
    

    SF_table = SF_table.reindex(SF_table.columns.tolist() + new_columns ,axis=1)
    
    for current_sweep in sweep_list:


        df=pd.DataFrame(SF_table.loc[current_sweep,'SF'])
        df=df[df['Feature']=='Upstroke']
        df=(df[df['Time_s']<(cell_sweep_info_table.loc[current_sweep,'Stim_start_s']+response_time)])

        spike_time_list=np.array(df.loc[:,'Time_s'])
        spike_index_in_time_range = np.array(df.loc[:,'Spike_index'].unique())
        
        # Put a minimum number of spikes to compute adaptation
        if feature == "Instantaneous_Frequency":
            if len(spike_time_list) >2: # strictly greater than 2 so that we have at least 2 intervals
                for current_spike_time_index in range(1,len(spike_time_list)): # start at 0 so the first substraction is index 1 - index 0...
                    current_inst_frequency=1/(spike_time_list[current_spike_time_index]-spike_time_list[current_spike_time_index-1])
    
                    SF_table.loc[current_sweep,str('Interval_'+str(int(current_spike_time_index-1)))]=current_inst_frequency
    
                    
                SF_table.loc[current_sweep,'Interval_0':]/=SF_table.loc[current_sweep,'Interval_0']
        else:
            if len(spike_time_list) >2:
                sub_SF = pd.DataFrame(SF_table.loc[current_sweep,'SF'])

                sub_SF = sub_SF.loc[sub_SF['Spike_index'].isin(spike_index_in_time_range),:]

                feature_df = sub_SF.loc[sub_SF['Feature']==feature,:]
                feature_df = feature_df.sort_values(by=['Time_s'])
                feature_list = np.array(feature_df.loc[:,measure])
                
                
                for current_spike_index in range(len(feature_list)): # start at 0 so the first substraction is index 1 - index 0...
                    if measure == "Membrane_potential_mV":
                        current_feature_normalized = feature_list[0]-feature_list[current_spike_index]
                    else:
                        current_feature_normalized = feature_list[current_spike_index]/feature_list[0]
                        
                    
                    SF_table.loc[current_sweep,str('Spike_'+str(int(current_spike_index)))]=current_feature_normalized
    
                if measure == "Membrane_potential_mV":
                    for col in new_columns:
                        SF_table.loc[current_sweep,col]=SF_table.loc[current_sweep,'Spike_0']-SF_table.loc[current_sweep,col]
                else:
                    
                    SF_table.loc[current_sweep,'Spike_0':]/=SF_table.loc[current_sweep,'Spike_0']
                

    interval_freq_table=pd.DataFrame(columns=['Spike_Interval','Normalized_feature','Stimulus_amp_pA','Sweep'])
    isnull_table=SF_table.isnull()
    isnull_table.columns=SF_table.columns
    isnull_table.index=SF_table.index
    
    for interval,col in enumerate(new_columns):
        for line in sweep_list:
            if isnull_table.loc[line,col] == False:

                new_line=pd.DataFrame([int(interval), # Interval#
                                    SF_table.loc[line,col], # Instantaneous frequency
                                    np.float64(cell_sweep_info_table.loc[line,'Stim_amp_pA']), # Stimulus amplitude
                                    line]).T# Sweep id
                                   
                new_line.columns=['Spike_Interval','Normalized_feature','Stimulus_amp_pA','Sweep']
                interval_freq_table=pd.concat([interval_freq_table,new_line],ignore_index=True)
                
    
    interval_freq_table = pd.merge(interval_freq_table,sweep_QC_table,on='Sweep')
    return interval_freq_table

def get_median_feature_table_test(interval_freq_table_init):
    interval_frequency_table = interval_freq_table_init.copy()
    interval_frequency_table=interval_frequency_table[interval_frequency_table['Passed_QC']==True]
    
    interval_frequency_table=interval_frequency_table.astype({"Spike_Interval":"float",
                                                              "Normalized_feature":"float",
                                                              "Stimulus_amp_pA":'float'})
    
    median_table=interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).median(numeric_only=True)
    median_table["Count_weigths"]=pd.DataFrame(interval_frequency_table.groupby(by=["Spike_Interval"],dropna=True).count()).loc[:,"Sweep"] #count number of sweep containing a response in interval#
    median_table["Spike_Interval"]=median_table.index
    median_table["Spike_Interval"]=np.float64(median_table["Spike_Interval"])  
    
    return median_table
    


     

