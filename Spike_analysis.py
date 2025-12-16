#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:46:44 2023

@author: julienballbe
"""
import pandas as pd
import numpy as np
import plotnine as p9
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import Ordinary_functions as ordifunc

def create_cell_Full_SF_dict_table(original_Full_TVC_table, original_cell_sweep_info_table,do_filter=True,BE_correct =True):
    '''
    Identify for all TVC table contained in a Full_TVC_table, all spike related_features

    Parameters
    ----------
    original_Full_TVC_table : pd.DataFrame
        2 columns DataFrame, containing in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces.
        
    original_cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    Returns
    -------
    Full_SF_dict_table : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)

    '''
    
    Full_TVC_table = original_Full_TVC_table.copy()
    cell_sweep_info_table = original_cell_sweep_info_table.copy()
    Full_TVC_table = Full_TVC_table.sort_values(by=['Sweep'])
    cell_sweep_info_table = cell_sweep_info_table.sort_values(by=["Sweep"])
    
    sweep_list = np.array(Full_TVC_table['Sweep'])
    

    Full_SF_dict_table=pd.DataFrame(columns=['Sweep',"SF_dict"])
    for current_sweep in sweep_list:
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        current_TVC=ordifunc.get_sweep_TVC_table(Full_TVC_table,current_sweep,do_filter=do_filter,filter=5.,do_plot=False)
        current_TVC_copy = current_TVC.copy()
        if BE_correct ==True:
            BE=original_cell_sweep_info_table.loc[current_sweep,'Bridge_Error_GOhms']
            
            if not np.isnan(BE):
                current_TVC_copy.loc[:,'Membrane_potential_mV'] = current_TVC_copy.loc[:,'Membrane_potential_mV']-BE*current_TVC_copy.loc[:,'Input_current_pA']
            
        
        membrane_trace = np.array(current_TVC_copy['Membrane_potential_mV'])
        time_trace = np.array(current_TVC_copy['Time_s'])
        current_trace = np.array(current_TVC_copy ['Input_current_pA'])
        stim_start = cell_sweep_info_table.loc[current_sweep, 'Stim_start_s']
        stim_end = cell_sweep_info_table.loc[current_sweep, 'Stim_end_s']

        SF_dict = identify_spike(
            membrane_trace, time_trace, current_trace, stim_start, stim_end, do_plot=False)
        new_line = pd.DataFrame([str(current_sweep), SF_dict]).T
        
        new_line.columns=['Sweep', 'SF_dict']
        Full_SF_dict_table=pd.concat([Full_SF_dict_table,new_line],ignore_index=True)
        
    Full_SF_dict_table.index = Full_SF_dict_table.loc[:, 'Sweep']
    Full_SF_dict_table.index = Full_SF_dict_table.index.astype(str)
    Full_SF_dict_table.index.name = 'Index'
    
    return Full_SF_dict_table

def create_Full_SF_table(original_Full_TVC_table, original_Full_SF_dict, cell_sweep_info_table,do_filter=True,BE_correct =True):
    '''
    Create for each Dict contained in original_Full_SF_dict a DataFrame containing the time, voltage and current values of the spike related features (if any, otherwise, empty dataframe)

    Parameters
    ----------
    original_Full_TVC_table : pd.DataFrame
        2 columns DataFrame, cotaining in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces.
        
    original_Full_SF_dict : pd.DataFrame
        DataFrame, two columns. For each row, first column ('Sweep') contains Sweep_id, and 
        second column ('SF') contains a Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)
        
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    Returns
    -------
    Full_SF_table : TYPE
        Two columns DataFrame containing in column 'Sweep' the sweep_id and 
        in the column 'SF' a pd.DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).

    '''

    Full_TVC_table = original_Full_TVC_table.copy()
    Full_SF_dict = original_Full_SF_dict.copy()

    sweep_list = np.array(Full_TVC_table['Sweep'])
    Full_SF_table = pd.DataFrame(columns=["Sweep", "SF"])

    for current_sweep in sweep_list:

        current_sweep = str(current_sweep)
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        current_TVC=ordifunc.get_sweep_TVC_table(Full_TVC_table,current_sweep,do_filter=do_filter,filter=5.,do_plot=False)
        BE = cell_sweep_info_table.loc[cell_sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
        if BE_correct == True and not np.isnan(BE) :
            
            current_TVC.loc[:,'Membrane_potential_mV'] -= BE*current_TVC.loc[:,'Input_current_pA']
                

        
        current_SF_table = create_SF_table(current_TVC, Full_SF_dict.loc[current_sweep, 'SF_dict'].copy())
        new_line = pd.DataFrame(
            [str(current_sweep), current_SF_table]).T
        new_line.columns=["Sweep", "SF"]
        Full_SF_table=pd.concat([Full_SF_table,new_line],ignore_index=True)


    Full_SF_table.index = Full_SF_table['Sweep']
    Full_SF_table.index = Full_SF_table.index.astype(str)
    Full_SF_table.index.name = 'Index'

    return Full_SF_table

def create_SF_table(original_TVC_table, SF_dict):
    '''
    From traces and spike features indexes, the function returns the Time, Potential and Current values of the different spike features 

    Parameters
    ----------
    original_TVC_table : pd.DataFrame
        Contains the Time, voltage, Current arranged in columns.
        
    SF_dict : Dict
        Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table).

    Returns
    -------
    SF_table : pd.DataFrame
        DataFrame containing the voltage and time values of the different spike features if any (otherwise empty DataFrame).

    '''
    TVC_table = original_TVC_table.copy()
    SF_table = pd.DataFrame(columns=['Time_s', 'Membrane_potential_mV', 'Input_current_pA',
                                 'Potential_first_time_derivative_mV/s', 'Potential_second_time_derivative_mV/s/s', 'Feature'])
    SF_table = SF_table.astype({'Time_s':'float',
                                'Membrane_potential_mV' : 'float',
                                'Input_current_pA' : 'float',
                                'Potential_first_time_derivative_mV/s' : 'float', 
                                'Potential_second_time_derivative_mV/s/s' : 'float', 
                                'Feature':'object'})
                           
  
    for feature in SF_dict.keys():
        
        current_feature_table = TVC_table.loc[SF_dict[feature], :].copy()
        current_feature_table['Feature'] = feature

        if current_feature_table.shape[0]!=0:
            
            SF_table = pd.concat([SF_table, current_feature_table],ignore_index = True)
        
    SF_table = SF_table.sort_values(by=['Time_s'])
    
    
    if SF_table.shape[0]!=0:
        spike_index=0
        SF_table.loc[:,"Spike_index"] = 0
        had_threshold = False
        for elt in range(SF_table.shape[0]):

            if SF_table.iloc[elt,5]=="Threshold" and had_threshold==False:
                had_threshold = True
                SF_table.iloc[elt,6] = spike_index
                
            elif SF_table.iloc[elt,5]!="Threshold":
                SF_table.iloc[elt,6] = spike_index
                
            elif SF_table.iloc[elt,5]=="Threshold" and had_threshold==True:
                spike_index+=1
                SF_table.iloc[elt,6] = spike_index
        
        SF_table = get_spike_half_width(TVC_table, SF_table)
        
    else:
        SF_table = pd.DataFrame(columns=['Time_s', 'Membrane_potential_mV', 'Input_current_pA',
                                     'Potential_first_time_derivative_mV/s', 'Potential_second_time_derivative_mV/s/s', 'Feature','Spike_index'])
    return SF_table

   
def get_spike_half_width(TVC_table_original, SF_table_original):
    TVC_table = TVC_table_original.copy()
    stim_amp_pA = np.nanmean(SF_table_original.loc[:,'Input_current_pA'])
    threshold_table = SF_table_original.loc[SF_table_original['Feature'] == 'Threshold',:]
    threshold_table = threshold_table.sort_values(by=['Time_s'])
    threshold_time = list(threshold_table.loc[:,'Time_s'])
    
    trough_table = SF_table_original.loc[SF_table_original['Feature'] == 'Trough',:]
    trough_table = trough_table.sort_values(by=['Time_s'])
    trough_time = list(trough_table.loc[:,'Time_s'])
    
    peak_table = SF_table_original.loc[SF_table_original['Feature'] == 'Peak',:]
    peak_table = peak_table.sort_values(by=['Time_s'])
    peak_time = list(peak_table.loc[:,'Time_s'])
    
    upstroke_table = SF_table_original.loc[SF_table_original['Feature'] == 'Upstroke',:]
    upstroke_table = upstroke_table.sort_values(by=['Time_s'])
    spike_time_list = list(upstroke_table.loc[:,'Time_s'])
    
    if len(threshold_time) != len(trough_time):
        if len(threshold_time)> len(trough_time):
            while len(threshold_time)> len(trough_time):
                threshold_time=threshold_time[:-1]
                
        elif len(threshold_time) < len(trough_time):
            while len(threshold_time) < len(trough_time):
                trough_time=trough_time[:-1]
                
    spike_index = 0
    
    for threshold, trough, peak, spike_time in zip(threshold_time, trough_time, peak_time, spike_time_list):
        
        threshold_to_peak_table = TVC_table.loc[(TVC_table['Time_s']>=threshold)&(TVC_table['Time_s']<=peak), :]
        membrane_voltage_array = np.array(threshold_to_peak_table.loc[:,'Membrane_potential_mV'])
        time_array = np.array(threshold_to_peak_table.loc[:,'Time_s'])
        
        spike_heigth = threshold_to_peak_table.loc[threshold_to_peak_table['Time_s'] == peak,'Membrane_potential_mV'].values[0] - threshold_to_peak_table.loc[threshold_to_peak_table['Time_s'] == threshold,'Membrane_potential_mV'].values[0]
        spike_height_line = pd.DataFrame([spike_time, spike_heigth, stim_amp_pA, np.nan, np.nan, "Spike_heigth", spike_index ]).T
        spike_height_line.columns = SF_table_original.columns
        spike_height_line = spike_height_line.astype({'Time_s':'float',
                                    'Membrane_potential_mV' : 'float',
                                    'Input_current_pA' : 'float',
                                    'Potential_first_time_derivative_mV/s' : 'float', 
                                    'Potential_second_time_derivative_mV/s/s' : 'float', 
                                    'Feature':'object',
                                    "Spike_index" : 'int'})
        
        SF_table_original = pd.concat([SF_table_original,spike_height_line ],ignore_index=True)
        
        putative_half_spike_heigth = (threshold_to_peak_table.loc[threshold_to_peak_table['Time_s'] == peak,'Membrane_potential_mV'].values[0]+threshold_to_peak_table.loc[threshold_to_peak_table['Time_s'] == threshold,'Membrane_potential_mV'].values[0])/2
        
        
        putative_half_width_start_index = ordifunc.find_time_index(membrane_voltage_array, putative_half_spike_heigth)
        putative_half_width_start = time_array[putative_half_width_start_index]
        
        peak_to_trough_table = TVC_table.loc[(TVC_table['Time_s']>=peak)&(TVC_table['Time_s']<=trough), :]

        membrane_voltage_array = np.array(peak_to_trough_table.loc[:,'Membrane_potential_mV'])
        time_array = np.array(peak_to_trough_table.loc[:,'Time_s'])
        
        if putative_half_spike_heigth < membrane_voltage_array[-1]: # if spike do not decreases enough to reach half spike computed in ascending phase, then 
            half_spike_heigth_end = membrane_voltage_array[-2]


            half_width_start_index = ordifunc.find_time_index(membrane_voltage_array, half_spike_heigth_end)
            half_width_start = time_array[half_width_start_index]
            
        else:
            half_spike_heigth_end = putative_half_spike_heigth
            half_width_start = putative_half_width_start
            
        
        #assert membrane_voltage_array[0] >= half_spike_heigth_end >= membrane_voltage_array[-1], "Given potential ({:f}) is outside of potential range ({:f}, {:f})".format(half_spike_heigth, membrane_voltage_array[0], membrane_voltage_array[-1])

        half_width_end_index = np.argmin(abs(membrane_voltage_array - half_spike_heigth_end))
        #half_width_end_index = ordifunc.find_time_index(membrane_voltage_array, half_spike_heigth)
        half_width_end = time_array[half_width_end_index]
        
        half_height_width = half_width_end - half_width_start
        
        half_width_line = pd.DataFrame([half_height_width, np.nan, stim_amp_pA, np.nan, np.nan, "Spike_width_at_half_heigth",spike_index ]).T
        
        half_width_line.columns = SF_table_original.columns

        half_width_line = half_width_line.astype({'Time_s':'float',
                                    'Membrane_potential_mV' : 'float',
                                    'Input_current_pA' : 'float',
                                    'Potential_first_time_derivative_mV/s' : 'float', 
                                    'Potential_second_time_derivative_mV/s/s' : 'float', 
                                    'Feature':'object',
                                    "Spike_index" : 'int'})
        SF_table_original = pd.concat([SF_table_original,half_width_line ],ignore_index=True)
        
        spike_index+=1
        
    return SF_table_original

def identify_spike(membrane_trace_array,time_array, current_trace, stim_start_time, stim_end_time,  do_plot=False):
    '''
    Based on AllenSDK.EPHYS_EPHYS_FEATURES module
    Identify spike and their related features based on membrane voltage trace and time traces

    Parameters
    ----------
    membrane_trace_array : np.array
        Time_varying membrane_voltage in mV.
        
    time_array : np.array
        Time array in s.
        
    current_trace : np.array
        Time_varying input current in pA.
        
    stim_start_time : float
        Stimulus start time.
        
    stim_end_time : float
        Stimulus end time.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    spike_feature_dict : Dict
        Dict in which the keys correspond to a spike-related feature, and the value to an array of time index (correspondance to sweep related TVC table)..

    '''

    
                        

    first_derivative=ordifunc.get_derivative(membrane_trace_array,time_array)
    

    second_derivative=ordifunc.get_derivative(first_derivative,time_array)
    filtered_second_derivative = ordifunc.filter_trace(second_derivative,time_array,filter=1.,do_plot=False)
    
    TVC_table=pd.DataFrame({'Time_s':time_array,
                               'Membrane_potential_mV':membrane_trace_array,
                               'Input_current_pA':current_trace,
                               'Potential_first_time_derivative_mV/s':first_derivative,
                               'Potential_second_time_derivative_mV/s/s':second_derivative},
                           dtype=np.float64) 

    time_derivative=first_derivative
    second_time_derivative=second_derivative
    
   
    
    preliminary_spike_index=detect_putative_spikes(v=membrane_trace_array,
                                                         t=time_array,
                                                         start=stim_start_time,
                                                         end=stim_end_time,
                                                         filter=5.,
                                                         dv_cutoff=18.,
                                                         dvdt=time_derivative)


    if do_plot:
        preliminary_spike_table=TVC_table.iloc[preliminary_spike_index[~np.isnan(preliminary_spike_index)],:]
        preliminary_spike_table['Feature']='A-Preliminary_spike_threshold'
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()+p9.geom_point(preliminary_spike_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+p9.xlim(stim_start_time,stim_end_time)
        current_plot += p9.xlab('1 - After detect_putative_spikes')
        print(current_plot)
        
    peak_index_array=find_peak_indexes(v=membrane_trace_array,
                                       t=time_array,
                                       spike_indexes=preliminary_spike_index,
                                       end=stim_end_time)

    if do_plot:
        peak_spike_table=TVC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:]

        peak_spike_table['Feature']='B-Preliminary_spike_peak'
        Full_table=pd.concat([preliminary_spike_table,peak_spike_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()+p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))+p9.xlim(stim_start_time,stim_end_time)
        current_plot += p9.xlab('2 - After find_peak_index')
        print(current_plot)

    spike_threshold_index,peak_index_array=filter_putative_spikes(v=membrane_trace_array,
                                                                  t=time_array,
                                                                  spike_indexes=preliminary_spike_index,
                                                                  peak_indexes=peak_index_array,
                                                                  min_height=20.,
                                                                  min_peak=-30.,
                                                                  filter=5.,
                                                                  dvdt=time_derivative)

    if do_plot:
        peak_spike_table=TVC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        spike_threshold_table=TVC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        
        Full_table=pd.concat([spike_threshold_table,peak_spike_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot += p9.xlab('3 - After filter_putative_spikes')
        print(current_plot)
    
    upstroke_index=find_upstroke_indexes(v=membrane_trace_array,
                                         t=time_array,
                                         spike_indexes=spike_threshold_index,
                                         peak_indexes=peak_index_array,filter=5.,
                                         dvdt=time_derivative)

    if do_plot:
        upstroke_table=TVC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        Full_table=pd.concat([Full_table,upstroke_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot += p9.xlab('4 - After find_upstroke_indexes')
        print(current_plot)
    
        
    spike_threshold_index=refine_threshold_indexes(v=membrane_trace_array,
                                                   t=time_array,
                                                   peak_indexes=peak_index_array,
                                                   upstroke_indexes=upstroke_index,
                                                   method = 'Mean_upstroke_fraction',
                                                   thresh_frac=0.05,
                                                   filter=5.,
                                                   dvdt=time_derivative,
                                                   dv2dt2=filtered_second_derivative)
    

    if do_plot:
        refined_threshold_table=TVC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; refined_threshold_table['Feature']='D-Refined_spike_threshold'
        Full_table=pd.concat([Full_table,refined_threshold_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot += p9.xlab('5 - After refine_threshold_indexes')
        print(current_plot)
        
    
    
    spike_threshold_index,peak_index_array,upstroke_index,clipped=check_thresholds_and_peaks(v=membrane_trace_array,
                                                                                            t=time_array,
                                                                                            spike_indexes=spike_threshold_index,
                                                                                            peak_indexes=peak_index_array,
                                                                                            upstroke_indexes=upstroke_index, 
                                                                                            start=stim_start_time, 
                                                                                            end=stim_end_time,
                                                                                            max_interval=0.01, 
                                                                                            thresh_frac=0.05, 
                                                                                            filter=5., 
                                                                                            dvdt=time_derivative,
                                                                                            tol=1.0, 
                                                                                            reject_at_stim_start_interval=0.)

    if do_plot:
        
        spike_threshold_table=TVC_table.iloc[spike_threshold_index[~np.isnan(spike_threshold_index)],:]; spike_threshold_table['Feature']='B-Spike_threshold'
        peak_spike_table=TVC_table.iloc[peak_index_array[~np.isnan(peak_index_array)],:];peak_spike_table['Feature']='A-Spike_peak'
        upstroke_table=TVC_table.iloc[upstroke_index[~np.isnan(upstroke_index)],:]; upstroke_table['Feature']='C-Upstroke'
        
        Full_table=pd.concat([spike_threshold_table,peak_spike_table,upstroke_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot += p9.xlab('6 - After check_thresholds_and_peaks')
        print(current_plot)
    
    if len(clipped)==1:
        if clipped[0] == True:
            spike_feature_dict={'Threshold':np.array([]).astype(int),
                                'Peak':np.array([]).astype(int),
                                'Upstroke':np.array([]).astype(int),
                                'Downstroke':np.array([]).astype(int),
                                'Trough':np.array([]).astype(int),
                                'Fast_Trough':np.array([]).astype(int),
                                'Slow_Trough':np.array([]).astype(int),
                                'ADP':np.array([]).astype(int),
                                'fAHP':np.array([]).astype(int)}
            
            return spike_feature_dict
    
    
    trough_index=find_trough_indexes(v=membrane_trace_array,
                                     t=time_array,
                                     spike_indexes=spike_threshold_index,
                                     peak_indexes=peak_index_array,
                                     clipped=clipped,
                                     end=stim_end_time)

    if do_plot:

        trough_table=TVC_table.iloc[trough_index[~np.isnan(trough_index)].astype(int),:]; trough_table['Feature']='D-Trough'
        Full_table=pd.concat([Full_table,trough_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot += p9.xlab('7 - After find_trough_indexes')
        print(current_plot)
    
    

    fast_AHP_index=find_fast_AHP_indexes(v=membrane_trace_array,
                                     t=time_array,
                                     spike_indexes=spike_threshold_index,
                                     peak_indexes=peak_index_array, 
                                     clipped=clipped, 
                                     end=stim_end_time,
                                     dvdt=time_derivative)

    if do_plot:
        fast_AHP_table=TVC_table.iloc[fast_AHP_index[~np.isnan(fast_AHP_index)],:]; fast_AHP_table['Feature']='E-Fast_AHP'
        Full_table=pd.concat([Full_table,fast_AHP_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot+=p9.xlab('8 - After find_fast_AHP_indexes')
        print(current_plot)
    
    
    
    
    downstroke_index=find_downstroke_indexes(v=membrane_trace_array,
                                             t=time_array, 
                                             peak_indexes=peak_index_array,
                                             trough_indexes=trough_index,
                                             clipped=clipped,
                                             filter=5.,
                                             dvdt=time_derivative)

    if do_plot:
        downstroke_table=TVC_table.iloc[downstroke_index[~np.isnan(downstroke_index)],:]; downstroke_table['Feature']='F-Downstroke'
        Full_table=pd.concat([Full_table,downstroke_table])
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot+=p9.xlab('9 - After find_trough_indexes')
        print(current_plot)
        
    
        

    fast_trough_index, adp_index, slow_trough_index, clipped=find_fast_trough_adp_slow_trough(v=membrane_trace_array, 
                                                                                                    t=time_array, 
                                                                                                    spike_indexes=spike_threshold_index,
                                                                                                    peak_indexes=peak_index_array, 
                                                                                                    downstroke_indexes=downstroke_index,
                                                                                                    clipped=clipped,
                                                                                                    end=stim_end_time,
                                                                                                    filter=5.,
                                                                                                    heavy_filter=1.,
                                                                                                    downstroke_frac=.01,
                                                                                                    adp_thresh=1.5,
                                                                                                    tol=1.,
                                                                                                    flat_interval=.002,
                                                                                                    adp_max_delta_t=.005,
                                                                                                    adp_max_delta_v=10,
                                                                                                    dvdt=time_derivative)

    if do_plot:
        fast_trough_table=TVC_table.iloc[fast_trough_index[~np.isnan(fast_trough_index)],:]; fast_trough_table['Feature']='G-Fast_Trough'
        Full_table=pd.concat([Full_table,fast_trough_table])
        
        adp_table=TVC_table.iloc[adp_index[~np.isnan(adp_index)],:]; adp_table['Feature']='H-ADP'
        Full_table=pd.concat([Full_table,adp_table])
        
        slow_trough_table=TVC_table.iloc[slow_trough_index[~np.isnan(slow_trough_index)],:]; slow_trough_table['Feature']='I-Slow_Trough'
        Full_table=pd.concat([Full_table,slow_trough_table])
        
        current_plot=p9.ggplot(TVC_table,p9.aes(x='Time_s',y="Membrane_potential_mV"))+p9.geom_line()
        current_plot+=p9.geom_point(Full_table,p9.aes(x='Time_s',y="Membrane_potential_mV",color='Feature'))
        current_plot+=p9.xlim(stim_start_time-.01,stim_end_time+.01)
        current_plot+=p9.xlab('10 - After find_fast_trough/ADP/slow_trough_indexes')
        print(current_plot)
        
    

    spike_threshold_index = spike_threshold_index[~np.isnan(spike_threshold_index)].astype(int)
    peak_index_array = peak_index_array[~np.isnan(peak_index_array)].astype(int)
    upstroke_index = upstroke_index[~np.isnan(upstroke_index)].astype(int)
    downstroke_index = downstroke_index[~np.isnan(downstroke_index)].astype(int)
    trough_index = trough_index[~np.isnan(trough_index)].astype(int)
    fast_trough_index = fast_trough_index[~np.isnan(fast_trough_index)].astype(int)
    slow_trough_index = slow_trough_index[~np.isnan(slow_trough_index)].astype(int)
    adp_index = adp_index[~np.isnan(adp_index)].astype(int)
    fast_AHP_index = fast_AHP_index[~np.isnan(fast_AHP_index)].astype(int)

    
    
    
     
    
    
    spike_feature_dict={'Threshold':np.array(spike_threshold_index),
                        'Peak':np.array(peak_index_array),
                        'Upstroke':np.array(upstroke_index),
                        'Downstroke':np.array(downstroke_index),
                        'Trough':np.array(trough_index),
                        'Fast_Trough':np.array(fast_trough_index),
                        'Slow_Trough':np.array(slow_trough_index),
                        'ADP':np.array(adp_index),
                        'fAHP':np.array(fast_AHP_index)}
    
    
    
    return spike_feature_dict

def detect_putative_spikes(v, t, start=None, end=None, filter=5., dv_cutoff=20., dvdt=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Perform initial detection of spikes and return their indexes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional)
    end : end of time window for spike detection (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    putative_spikes : numpy array of preliminary spike indexes
    """

    if not isinstance(v, np.ndarray):
        raise TypeError("v is not an np.ndarray")

    if not isinstance(t, np.ndarray):
        raise TypeError("t is not an np.ndarray")

    

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = ordifunc.find_time_index(t, start)
    end_index = ordifunc.find_time_index(t, end)
    v_window = v[start_index:end_index + 1]
    t_window = t[start_index:end_index + 1]

    if dvdt is None:
        
        
        dvdt = ordifunc.get_derivative(v_window, t_window)
       
    else:
        dvdt = dvdt[start_index:end_index]

    # Find positive-going crossings of dV/dt cutoff level
    

    putative_spikes = np.flatnonzero(np.diff(np.greater_equal(dvdt, dv_cutoff).astype(int)) == 1)

    if len(putative_spikes) <= 1:
        # Set back to original index space (not just window)
        return np.array(putative_spikes) + start_index

    # Only keep spike times if dV/dt has dropped all the way to zero between putative spikes
    putative_spikes = [putative_spikes[0]] + [s for i, s in enumerate(putative_spikes[1:])
        if np.any(dvdt[putative_spikes[i]:s] < 0)]

    # Set back to original index space (not just window)
    return np.array(putative_spikes) + start_index

def find_peak_indexes(v, t, spike_indexes, end=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find indexes of spike peaks.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    end : end of time window for spike detection (optional)
    """

    if not end:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)

    spks_and_end = np.append(spike_indexes, end_index)
    peak_indexes = [np.argmax(v[spk:next]) + spk for spk, next in
                    zip(spks_and_end[:-1], spks_and_end[1:])]
    #finds index of maximum value between two consecutive spikes index
    #spk represents the first spike index (going through the list of spike_index without the last one)
    #next respesent the second spike index (going through the list of spike index without the first one)
    #np.argmax(v[spk:next]) + spk --> add spk because is initilized at 0 in (v[spk:next])

    peak_indexes = np.array(peak_indexes)
    #peak_indexes = peak_indexes.astype(int)    

    return np.array(peak_indexes)

def filter_putative_spikes(v, t, spike_indexes, peak_indexes, min_height=2.,
                           min_peak=-30., filter=5., dvdt=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Filter out events that are unlikely to be spikes based on:
        * Height (threshold to peak)
        * Absolute peak level

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
    min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    spike_indexes : numpy array of threshold indexes
    peak_indexes : numpy array of peak indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([]), np.array([])

    if dvdt is None:
        
        dvdt = ordifunc.get_derivative(v, t)

    diff_mask = [np.any(dvdt[peak_ind:spike_ind] < 0)
                 for peak_ind, spike_ind
                 in zip(peak_indexes[:-1], spike_indexes[1:])]
    #diif_mask --> check if the derivative between a peak and the next threshold ever goes negative
    
    peak_indexes = peak_indexes[np.array(diff_mask + [True])]
    spike_indexes = spike_indexes[np.array([True] + diff_mask)]
    #keep only peak indexes where diff mask was True (same for spike index)

    peak_level_mask = v[peak_indexes] >= min_peak
    #check if identified peaks are higher than minimum peak values (defined)
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]
    #keep only spike and peaks if spike_peak is higher than minimum value

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= min_height
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]

    #keep only events where the voltage difference between peak and threshold is higher than minimum height (defined)
    spike_indexes = np.array(spike_indexes)
    #spike_indexes = spike_indexes.astype(int)
    
    peak_indexes = np.array(peak_indexes)
    #peak_indexes = peak_indexes.astype(int)
    
    return spike_indexes, peak_indexes

def find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=5., dvdt=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find indexes of maximum upstroke of spike.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    upstroke_indexes : numpy array of upstroke indexes

    """

    if dvdt is None:
        
        dvdt = ordifunc.get_derivative(v, t)
    upstroke_indexes = [np.argmax(dvdt[spike:peak]) + spike for spike, peak in
                        zip(spike_indexes, peak_indexes)]

    upstroke_indexes = np.array(upstroke_indexes)
    
    return upstroke_indexes


def refine_threshold_indexes(v, t, peak_indexes,upstroke_indexes,method="Mean_upstroke_fraction",  thresh_frac=0.05, filter=5., dvdt=None,dv2dt2=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Refine threshold detection of previously-found spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    upstroke_indexes : numpy array of indexes of spike upstrokes (for threshold target calculation)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    threshold_indexes : numpy array of threshold indexes
    """

    if not peak_indexes.size:
        return np.array([])

    if dvdt is None:
        
        
        dvdt = ordifunc.get_derivative(v, t)
    
    if method == "Second_Derivative":
    ##########
    # Here the threshold is defined as the local maximum of the second voltage derivative
        opening_window=np.append(np.array([0]), peak_indexes[:-1])
        closing_window=peak_indexes
        
        
        threshold_indexes = [np.argmax(dv2dt2[prev_peak:nex_peak])+prev_peak for prev_peak, nex_peak in
                            zip(opening_window, closing_window)]
        return np.array(threshold_indexes)
    ##########
    elif method == "Mean_upstroke_fraction":
        ## Here the threshold is defined as the last index where dvdt= avg_upstroke * thresh_frac
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * thresh_frac
    
        upstrokes_and_start = np.append(np.array([0]), upstroke_indexes)
        threshold_indexes = []
        for upstk, upstk_prev in zip(upstrokes_and_start[1:], upstrokes_and_start[:-1]):
    
            voltage_indexes = np.flatnonzero(dvdt[upstk:upstk_prev:-1] <= target)
    
            if not voltage_indexes.size:
                # couldn't find a matching value for threshold,
                # so just going to the start of the search interval
                threshold_indexes.append(upstk_prev)
            else:
                threshold_indexes.append(upstk - voltage_indexes[0])
    
        threshold_indexes = np.array(threshold_indexes)
        
        return threshold_indexes

def check_thresholds_and_peaks(v, t, spike_indexes, peak_indexes, upstroke_indexes, start=None, end=None,
                               max_interval=0.01, thresh_frac=0.05, filter=5., dvdt=None,
                               tol=1.0, reject_at_stim_start_interval=0.):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Validate thresholds and peaks for set of spikes

    Check that peaks and thresholds for consecutive spikes do not overlap
    Spikes with overlapping thresholds and peaks will be merged.

    Check that peaks and thresholds for a given spike are not too far apart.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    upstroke_indexes : numpy array of indexes of spike upstrokes
    start : start of time window for feature analysis (optional)
    end : end of time window for feature analysis (optional)
    max_interval : maximum allowed time between start of spike and time of peak in sec (default 0.005)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)
    tol : tolerance for returning to threshold in mV (optional, default 1)
    reject_at_stim_start_interval : duration of window after start to reject voltage spikes (optional, default 0)

    Returns
    -------
    spike_indexes : numpy array of modified spike indexes
    peak_indexes : numpy array of modified spike peak indexes
    upstroke_indexes : numpy array of modified spike upstroke indexes
    clipped : numpy array of clipped status of spikes
    """

    if start is not None and reject_at_stim_start_interval > 0:
        mask = t[spike_indexes] > (start + reject_at_stim_start_interval)
        spike_indexes = spike_indexes[mask]
        peak_indexes = peak_indexes[mask]
        upstroke_indexes = upstroke_indexes[mask]
    
    overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1] + 1)
    
    if overlaps.size:
        spike_mask = np.ones_like(spike_indexes, dtype=bool)
        spike_mask[overlaps + 1] = False
        spike_indexes = spike_indexes[spike_mask]

        peak_mask = np.ones_like(peak_indexes, dtype=bool)
        peak_mask[overlaps] = False
        peak_indexes = peak_indexes[peak_mask]

        upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
        upstroke_mask[overlaps] = False
        upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []

    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):

        if t[peak] - t[spk] >= max_interval:
            logging.info("Need to recalculate threshold-peak pair that exceeds maximum allowed interval ({:f} s)".format(max_interval))
            too_long_spikes.append(i)

    if too_long_spikes:
        if dvdt is None:
            dvdt = ordifunc.get_derivative(v, t)
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * thresh_frac

        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = ordifunc.find_time_index(t, t[peak] - max_interval)
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            
            if not below_target.size:

                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]
                if t[spike] + 2 * max_interval >= t[-1]:
                    t_0=ordifunc.find_time_index(t, t[-1])
                else:
                    t_0 = ordifunc.find_time_index(t, t[spike] + 2 * max_interval)
                    
                new_peak = np.argmax(v[spike:t_0]) + spike
               
                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                
                if t[new_peak] - t[spike] < max_interval and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak

                else:
                    # Otherwise, log and get rid of the spike
                    logging.info("Could not redetermine threshold-peak pair - dropping that pair")

                    drop_spikes.append(i)

#                     raise FeatureError("Could not redetermine threshold")
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]

        if drop_spikes:

            spike_indexes = np.delete(spike_indexes, drop_spikes)
            peak_indexes = np.delete(peak_indexes, drop_spikes)
            upstroke_indexes = np.delete(upstroke_indexes, drop_spikes)

    if not end:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)

    clipped = find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol)
    
    spike_indexes = np.array(spike_indexes)
    
    
    peak_indexes = np.array(peak_indexes)
    
    
    upstroke_indexes = np.array(upstroke_indexes)
    
    

    
    return spike_indexes, peak_indexes, upstroke_indexes, clipped

def find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol, time_tol=0.005):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Check that last spike was not cut off too early by end of stimulus
    by checking that the membrane voltage returned to at least the threshold
    voltage - otherwise, drop it

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    end_index: int index of the end of time window for feature analysis

    tol: float tolerance to returning to threshold
    time_tol: float specify the time window in which
    Returns
    -------
    clipped: Boolean np.array
    """
    clipped = np.zeros_like(spike_indexes, dtype=bool)

    if len(spike_indexes)>0:
        
        if t[peak_indexes[-1]]>= t[end_index]-time_tol:
            vtail = v[peak_indexes[-1]:end_index + 1]
            if not np.any(vtail <= v[spike_indexes[-1]] + tol):
               
                logging.debug(
                    "Failed to return to threshold voltage + tolerance (%.2f) after last spike (min %.2f) - marking last spike as clipped",
                    v[spike_indexes[-1]] + tol, vtail.min())
                clipped[-1] = True
                logging.debug("max %f, min %f, t(end_index):%f" % (np.max(vtail), np.min(vtail), t[end_index]))

    return clipped


def find_fast_AHP_indexes(v, t, spike_indexes, peak_indexes, clipped=None, end=None,dvdt=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Detect the minimum membrane voltage value in the 5ms time window after a spike peak. 
    For a given spike, fAHP is defined if during the 5ms following the peak:
        -	The time derivative of the membrane voltage went positive
        and
        -   The membrane voltage went below the spike threshold
    If either if this condition was not observed, fAHP was not defined for this spike



    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)

    Returns
    -------
    fast_AHP_indexes : numpy array of threshold indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)
    
    if dvdt is None:
        dvdt = ordifunc.get_derivative(v, t)
    
    
    window_closing_time=t[peak_indexes]+.005
    window_closing_idx=[]
    for current_time in window_closing_time:
        if current_time <= t[-1]:
            window_closing_idx.append(ordifunc.find_time_index(t, current_time))
        else:
            window_closing_idx.append(ordifunc.find_time_index(t, t[-1]))
    #window_closing_idx=find_time_index(t, window_closing_time)
    
    
    fast_AHP_indexes = np.zeros_like(window_closing_idx, dtype=float)
    # trough_indexes[:-1] = [v[peak:spk].argmin() + peak for peak, spk
    #                        in zip(peak_indexes[:-1], spike_indexes[1:])]
    fast_AHP_indexes=[]
    for peak , wnd, thresh in zip(peak_indexes, window_closing_idx,spike_indexes):
        if np.flatnonzero(dvdt[peak:wnd] >= 0).size and min(v[peak:wnd])<=thresh :
            fast_AHP_indexes.append(int(v[peak:wnd].argmin() + peak ))
            
        
   
    if len(fast_AHP_indexes) > 0:
        if fast_AHP_indexes[-1]>=end_index: #If the last fast trough detected is after end_time index (because of the closing window of 5ms), redefine it at end_index
            fast_AHP_indexes[-1]=end_index
    # if clipped[-1]:
    #     # If last spike is cut off by the end of the window, trough is undefined
    #     trough_indexes[-1] = np.nan
    # else:
    #     trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method
    
    # # If peak is the same point as the trough, drop that point
    # trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]
    
    fast_AHP_indexes = np.array(fast_AHP_indexes)

    return fast_AHP_indexes

def find_slow_trough_indexes(v, t, spike_indexes, fast_trough_indexes, clipped=None, end=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find indexes of minimum voltage (trough) between spikes.
    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    Returns
    -------
    slow_trough_indexes : numpy array of threshold indexes
    """
    
    if not spike_indexes.size or not fast_trough_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)

    slow_trough_indexes = np.zeros_like(spike_indexes, dtype=float)
    slow_trough_indexes[:-1] = [v[fast_trough:spk].argmin() + fast_trough for fast_trough, spk
                           in zip(fast_trough_indexes[:-1], spike_indexes[1:])]
    
    if clipped[-1]:
        # If last spike is cut off by the end of the window, trough is undefined
        slow_trough_indexes[-1] = np.nan
    else:
        if fast_trough_indexes[-1] == end_index:
            slow_trough_indexes[-1]=fast_trough_indexes[-1]
            
        else:
            
            slow_trough_indexes[-1] = v[fast_trough_indexes[-1]:end_index].argmin() + fast_trough_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method
    slow_trough_indexes=np.array(slow_trough_indexes)

    return slow_trough_indexes

def find_trough_indexes(v, t, spike_indexes, peak_indexes, clipped=None, end=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find indexes of minimum voltage (trough) between spikes.
    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    Returns
    -------
    trough_indexes : numpy array of threshold indexes
    """
    
    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)

    trough_indexes = np.zeros_like(spike_indexes, dtype=float)
    trough_indexes[:-1] = [int(v[peak:spk].argmin() + peak) for peak, spk
                           in zip(peak_indexes[:-1], spike_indexes[1:])]
    
    if clipped[-1]:
        # If last spike is cut off by the end of the window, trough is undefined
        trough_indexes[-1] = np.nan
    else:
        
        trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]
        

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method

    # If peak is the same point as the trough, drop that point
    
    trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]
    trough_indexes = np.array(trough_indexes)

    return trough_indexes

def find_downstroke_indexes(v, t, peak_indexes, trough_indexes, clipped=None, filter=5., dvdt=None):
    """
    From AllenSDK.EPHYS.EPHYS_FEATURES Module
    Find indexes of minimum voltage derivative between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of threshold indexes
    clipped: boolean array - False if spike not clipped by edge of window
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    downstroke_indexes : numpy array of downstroke indexes
    """

    if not trough_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = ordifunc.get_derivative(v, t)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes, dtype=bool)

    if len(peak_indexes) < len(trough_indexes):
        raise ValueError("Cannot have more troughs than peaks")

    
    valid_peak_indexes = peak_indexes[~clipped].astype(int)
    
    valid_trough_indexes = trough_indexes[~clipped].astype(int)

    downstroke_indexes = np.zeros_like(peak_indexes) * np.nan


    downstroke_index_values = [np.argmin(dvdt[peak:trough]) + peak for peak, trough
                         in zip(valid_peak_indexes, valid_trough_indexes)]
    
    downstroke_indexes[~clipped] = downstroke_index_values
    
    return downstroke_indexes

def find_fast_trough_adp_slow_trough(v, t, spike_indexes, peak_indexes, downstroke_indexes, clipped=None, end=None, filter=5.,
                           heavy_filter=1., downstroke_frac=0.01, adp_thresh=1.5, tol=1.,
                           flat_interval=0.002, adp_max_delta_t=0.005, adp_max_delta_v=10., dvdt=None):
    """
    Comes from analyze_trough_details in IPFX
    Analyze trough to determine if an ADP exists and whether the reset is a 'detour' or 'direct'

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike threshold indexes
    peak_indexes : numpy array of spike peak indexes
    downstroke_indexes : numpy array of spike downstroke indexes
    end : end of time window (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    heavy_filter : lower cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    downstroke_frac : fraction of spike downstroke to define spike fast trough (optional, default 0.01)
    adp_thresh: minimum dV/dt in V/s to exceed to be considered to have an ADP (optional, default 1.5)
    tol : tolerance for evaluating whether Vm drops appreciably further after end of spike (default 1.0 mV)
    flat_interval: if the trace is flat for this duration, stop looking for an ADP (default 0.002 s)
    adp_max_delta_t: max possible ADP delta t (default 0.005 s)
    adp_max_delta_v: max possible ADP delta v (default 10 mV)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    isi_types : numpy array of isi reset types (direct or detour)
    fast_trough_indexes : numpy array of indexes at the start of the trough (i.e. end of the spike)
    adp_indexes : numpy array of adp indexes (np.nan if there was no ADP in that ISI
    slow_trough_indexes : numpy array of indexes at the minimum of the slow phase of the trough
                          (if there wasn't just a fast phase)
    """

    if end is None:
        end = t[-1]
    end_index = ordifunc.find_time_index(t, end)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes)

    # Can't evaluate for spikes that are clipped by the window
    

    valid_spike_indexes = spike_indexes[~clipped]
    

    if dvdt is None:
        dvdt = ordifunc.data_treat.get_derivative(v, t)


    dvdt_hvy = ordifunc.get_derivative(ordifunc.filter_trace(v,t,filter=heavy_filter,do_plot=False),t)

    # Writing as for loop - see if I can vectorize any later
    fast_trough_indexes = []
    adp_indexes = []
    slow_trough_indexes = []
    isi_types = []

    update_clipped = []

    for dwnstk, next_spk in zip(downstroke_indexes,np.append(valid_spike_indexes[1:], end_index)):
        dwnstk=int(dwnstk)
        target = downstroke_frac * dvdt[dwnstk]
        
        terminated_points = np.flatnonzero(dvdt[dwnstk:next_spk] >= target)
        #terminated points corresponds  between spike downstroke  and next spike threshold to where dvdt gets higher than 1% of dvdt at spike dowstroke 
        # After the downsteroke the dvdt is neg, so 1% of dvdt at downstroke corresponds to a flatening of the membrane voltage
        if terminated_points.size:
            terminated = terminated_points[0] + dwnstk
            update_clipped.append(False)
        else:
            logging.debug("Could not identify fast trough - marking spike as clipped")
            isi_types.append(np.nan)
            fast_trough_indexes.append(np.nan)
            adp_indexes.append(np.nan)
            slow_trough_indexes.append(np.nan)
            update_clipped.append(True)
            continue
        
    
        # Could there be an ADP?
        adp_index = np.nan
        dv_over_thresh = np.flatnonzero(dvdt_hvy[terminated:next_spk] >= adp_thresh) # did the dvdt went higher than adp threshold after the fist index where dvdt went higher thant 0.01*dvdt at downstroke
        if dv_over_thresh.size:
            cross = dv_over_thresh[0] + terminated #cross = first index after downstroke that crosses the derivative threshold to look for ADP

            # only want to look for ADP before things get pretty flat
            # otherwise, could just pick up random transients long after the spike
            if t[cross] - t[terminated] < flat_interval:
                # Going back up fast, but could just be going into another spike
                # so need to check for a reversal (zero-crossing) in dV/dt
                zero_return_vals = np.flatnonzero(dvdt_hvy[cross:next_spk] <= 0) #zero_return_vals = where dvdt is negative between cross and next spike
                if zero_return_vals.size:
                    putative_adp_index = zero_return_vals[0] + cross #putative_adp_index = first index of dvdt positive after cross
                    min_index = v[putative_adp_index:next_spk].argmin() + putative_adp_index
                    if (v[putative_adp_index] - v[min_index] >= tol and # voltage difference between min and first index of dvdt positive after cross must be higher than tol
                            v[putative_adp_index] - v[terminated] <= adp_max_delta_v and # voltage difference between fats trough and first index of dvdt positive after cross, must be lower than max adp tolerated voltage
                            t[putative_adp_index] - t[terminated] <= adp_max_delta_t):
                        adp_index = putative_adp_index
                        slow_phase_min_index = min_index
                        isi_type = "detour"

        if np.isnan(adp_index):
            v_term = v[terminated]
            min_index = v[terminated:next_spk].argmin() + terminated
            if v_term - v[min_index] >= tol:
                # dropped further after end of spike -> detour reset
                isi_type = "detour"
                slow_phase_min_index = min_index
            else:
                isi_type = "direct"

        isi_types.append(isi_type)
        fast_trough_indexes.append(terminated)
        adp_indexes.append(adp_index)
        if isi_type == "detour":
            slow_trough_indexes.append(slow_phase_min_index)
        else:
            slow_trough_indexes.append(np.nan)
    
    clipped[~clipped] = update_clipped
    
    fast_trough_indexes = np.array(fast_trough_indexes)
    #fast_trough_indexes = fast_trough_indexes.astype(int)
    
    adp_indexes = np.array(adp_indexes)
    #adp_indexes = adp_indexes.astype(int)
    
    slow_trough_indexes = np.array(slow_trough_indexes)
    # slow_trough_indexes = slow_trough_indexes.astype(int)
    
    return fast_trough_indexes, adp_indexes, slow_trough_indexes, clipped
    


def plot_trace_with_spike_features(Full_TVC_table, Full_SF_table,sweep_info_table, sweep, BE_corrected,  superimpose_BE):
    '''
    Plot a two-panel figure showing membrane potential and input current over time, with optional features overlayed.

    This function generates a Plotly figure displaying:
    1. The membrane potential trace over time, optionally corrected for bridge error (BE).
    2. The input current over time, with features from the spike feature table superimposed.

    Parameters
    ----------
    Full_TVC_table : pandas.DataFrame
        A DataFrame containing time-series data for traces of membrane potential and input current.
        Each row corresponds to a sweep, and 'TVC' column contains the trace data as nested DataFrames.
    Full_SF_table : pandas.DataFrame
        A DataFrame containing spike feature data for different sweeps. The 'SF' column contains nested
        DataFrames with spike features (e.g., threshold, peak, trough).
    sweep_info_table : pandas.DataFrame
        A DataFrame containing metadata for each sweep, such as bridge error values in "Bridge_Error_GOhms".
    sweep : str or int
        Identifier for the specific sweep to plot.
    BE_corrected : bool
        If True, correct the membrane potential for bridge error using the formula:
        `Membrane_potential_mV = Membrane_potential_mV - (Bridge_Error * Input_current_pA)`.
    superimpose_BE : bool
        If True, and `BE_corrected` is enabled, superimpose the uncorrected membrane potential trace
        on the corrected trace for comparison.

    Returns
    -------
    None
        Displays an interactive Plotly figure with the following:
        - Top panel: Membrane potential over time, with optional feature markers.
        - Bottom panel: Input current over time.

    '''
    
    TVC = Full_TVC_table.loc[sweep, 'TVC'].copy()
    SF_table = Full_SF_table.loc[sweep,'SF'].copy()
    BE = sweep_info_table.loc[sweep, "Bridge_Error_GOhms"]
        
        
    
    color_dict = {'Input_current_pA': 'black',
                  'Membrane potential': 'black',
                  'Membrane potential BE Corrected': "black",
                  "Membrane potential original" : "red",
                  'Threshold': "#a2c5fc",
                  "Upstroke": "#0562f5",
                  "Peak": "#2f0387",
                  "Downstroke": "#9f02e8",
                  "Fast_Trough": "#c248fa",
                  "fAHP": "#d991fa",
                  "Trough": '#fc3873',
                  "Slow_Trough": "#96022e"
                  }
    
    
   
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Membrane Potential plot", "Input Current plot"))
    
    if BE_corrected == True:
        if not np.isnan(BE):
            TVC.loc[:,'Membrane_potential_mV'] = TVC.loc[:,'Membrane_potential_mV']-BE*TVC.loc[:,'Input_current_pA']
        fig.add_trace(go.Scatter(x=TVC['Time_s'], y=TVC['Membrane_potential_mV'], mode='lines', name='Membrane potential BE Corrected', line=dict(color=color_dict['Membrane potential'])), row=1, col=1)
        if superimpose_BE == True:
            TVC_second = Full_TVC_table.loc[sweep, 'TVC'].copy()
            fig.add_trace(go.Scatter(x=TVC_second['Time_s'], y=TVC_second['Membrane_potential_mV'], mode='lines', name='Membrane potential original', line=dict(color=color_dict['Membrane potential original'])), row=1, col=1)
    
    
    else: 
        fig.add_trace(go.Scatter(x=TVC['Time_s'], y=TVC['Membrane_potential_mV'], mode='lines', name='Membrane potential ', line=dict(color=color_dict['Membrane potential'])), row=1, col=1)
    # Plot for Membrane_potential_mV vs Time_s from SF_table if not empty
    if not SF_table.empty:
        for feature in SF_table['Feature'].unique():
            if feature in ["Spike_heigth", "Spike_width_at_half_heigth"]:
                continue
            subset = SF_table[SF_table['Feature'] == feature]
            
            fig.add_trace(go.Scatter(x=subset['Time_s'], y=subset['Membrane_potential_mV'], mode='markers', name=feature, marker=dict(color=color_dict[feature])), row=1, col=1)
    
    # Plot for Input_current_pA vs Time_s from TVC
    fig.add_trace(go.Scatter(x=TVC['Time_s'], y=TVC['Input_current_pA'], mode='lines', name='Input current', line=dict(color=color_dict['Input_current_pA'])), row=2, col=1)
    
    # Update layout
    
    
    # fig.update_layout(height=800, showlegend=True, title_text="TVC and SF Plots", )
    
    fig.update_layout(
    height=800,
    showlegend=True,
    title_text="TVC and SF Plots",
    hovermode='x unified',  # Use 'x unified' to create a unified hover mode
    )
    fig.update_xaxes(title_text="Time_s", row=2, col=1)
    fig.update_yaxes(title_text="Membrane_potential_mV", row=1, col=1)
    fig.update_yaxes(title_text="Input_current_pA", row=2, col=1)
    
     
    # Show plot
    fig.show()
    

###Functions used for clustering
def get_cell_spikes_traces(Full_TVC_table, Full_SF_table):
    '''
    Extract spike-centered traces from time-series voltage data.

    This function processes time-series data to extract and align spike-centered voltage traces 
    (membrane potential and its first derivative) for each detected spike in the input data. 
    It constructs a table containing these traces, centered on the peak of each spike, along with 
    metadata such as the sweep identifier and spike index.

    Parameters
    ----------
    Full_TVC_table : pandas.DataFrame
        A DataFrame containing time-series voltage and current data for multiple sweeps.
        Each row represents a sweep, and the DataFrame includes at least the following columns:
        - "Sweep": Identifier for each sweep.
        - Nested "TVC" DataFrame containing:
            - "Membrane_potential_mV": Voltage trace.
            - "Potential_first_time_derivative_mV/s": First derivative of the voltage.
            - "Time_s": Time stamps for the traces.
    Full_SF_table : pandas.DataFrame
        A DataFrame containing spike feature data for multiple sweeps.
        Each row represents a sweep, and the DataFrame includes at least the following columns:
        - "Sweep": Identifier for each sweep.
        - Nested "SF" DataFrame containing:
            - "Feature": Type of feature detected (e.g., "Peak").
            - "Time_s": Time stamps of the detected feature (e.g., spike peaks).

    Returns
    -------
    spike_trace_table : pandas.DataFrame
        A DataFrame containing spike-centered traces and metadata. Includes the following columns:
        - "Membrane_potential_mV": Voltage trace values.
        - "Potential_first_time_derivative_mV/s": First derivative of voltage values.
        - "Time_s": Time relative to the spike peak (centered on 0).
        - "Sweep": Sweep identifier for each spike trace.
        - "Spike_index": Unique index for each spike.

    Notes
    -----
    - The function extracts a 3 ms window of data around each detected spike (1 ms before and 2 ms after the spike peak).
    - Spike-centered traces are aligned such that the spike peak occurs at `Time_s = 0`.
    - Spikes are identified based on the "Peak" feature in `Full_SF_table`.
    - If no spikes are detected for a sweep, that sweep is skipped.

    
    '''
    
    sweep_list = Full_TVC_table.loc[:,'Sweep']
    spike_trace_table = pd.DataFrame(columns=["Membrane_potential_mV",'Potential_first_time_derivative_mV/s',"Time_s","Sweep", 'Spike_index'])
    i=0
    for current_sweep in sweep_list:
        current_SF_table = Full_SF_table.loc[current_sweep,"SF"]
        
        if current_SF_table.shape[0]==0:
            continue
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        current_TVC_table = ordifunc.get_sweep_TVC_table(Full_TVC_table, current_sweep)
        peak_table = current_SF_table.loc[current_SF_table['Feature']=='Peak', :]
        peak_table = peak_table.sort_values(by=['Time_s'])
        peak_table = peak_table.reset_index()
        
        for line in range(peak_table.shape[0]):
            peak_time = peak_table.loc[line,'Time_s']
            window_st = peak_time - .001
            window_end = peak_time + .002
            
            current_spike_window = current_TVC_table.loc[(current_TVC_table['Time_s'] >= window_st) & (current_TVC_table['Time_s'] <= window_end ), ['Membrane_potential_mV','Potential_first_time_derivative_mV/s', 'Time_s']]
            current_spike_window.loc[:,'Time_s'] -= peak_time # center on peak time
            current_spike_window['Sweep'] = str(current_sweep)
            current_spike_window['Spike_index'] = str(i+1)
            i+=1
            spike_trace_table = pd.concat([spike_trace_table, current_spike_window],axis=0, ignore_index=True)
            
    return spike_trace_table


def get_first_spike_features(arg_list):
    '''
    
    Extract features of the first spike for a given cell and its sweeps.
    
        This function identifies and extracts the features of the first spike 
        from electrophysiological recordings for a specified cell. If no spikes 
        are detected in any sweeps, the function returns a placeholder row 
        with NaN values.
    
        Parameters
        ----------
        arg_list : list
            A list containing two elements:
            - cell_id : str
                Unique identifier for the cell.
            - database_config_line : dict
                Configuration details for accessing the database, used by `ordifunc.read_cell_file_h5`.
    
        Returns
        -------
        new_spike_line : pandas.DataFrame
            A single-row DataFrame containing the first spike's features for the cell.
            Columns include:
            - 'Cell_id' : str
                Identifier of the cell.
            - 'Sweep' : str
                Identifier of the sweep containing the first spike.
            - 'Stim_amp_pA' : float
                Stimulation amplitude in picoamperes for the sweep.
            - 'Spike_threshold' : float
                Membrane potential (mV) at which the spike initiates.
            - 'Spike_heigth' : float
                Spike height (mV).
            - 'Spike_Upstroke' : float
                Maximum rate of rise (first derivative, mV/s) during the spike upstroke.
            - 'Spike_Downstroke' : float
                Maximum rate of fall (first derivative, mV/s) during the spike downstroke.
            - 'Spike_peak' : float
                Membrane potential (mV) at the spike's peak.
            - 'Spike_trough' : float
                Membrane potential (mV) at the trough following the spike.
            - 'Spike_width' : float
                Spike width at half-height (s).
    
        Notes
        -----
        - The function reads data using `ordifunc.read_cell_file_h5`, assuming the presence of `Sweep_info_table` 
          and `Full_SF_table` in the returned cell dictionary.
        - It processes sweeps in ascending order of stimulation amplitude (`Stim_amp_pA`) and stops after 
          extracting the first spike's features.
        - If a sweep contains no spikes, it is skipped.
        - If no spikes are detected across all sweeps, placeholder values are returned with NaNs.
        - Errors during execution (e.g., missing data or invalid input) result in a default placeholder row.


    '''
    cell_id, database_config_line = arg_list
    try:
        cell_dict = ordifunc.read_cell_file_h5(cell_id,database_config_line,selection=['All'])
        sweep_info_table = cell_dict['Sweep_info_table']
        Full_SF_table = cell_dict['Full_SF_table']
        sweep_info_table = sweep_info_table.sort_values(by=['Stim_amp_pA'])
        sweep_list = np.array(sweep_info_table.loc[:,"Sweep"])
        for sweep in sweep_list:
            sub_SF_table = Full_SF_table.loc[sweep,"SF"]
            if sub_SF_table.shape[0]!=0:
                first_spike_features_table = sub_SF_table.loc[sub_SF_table['Spike_index']==np.nanmin(sub_SF_table['Spike_index']),:]
                Threshold = first_spike_features_table.loc[first_spike_features_table['Feature']=='Threshold',"Membrane_potential_mV"].values[0]
                Heigth = first_spike_features_table.loc[first_spike_features_table['Feature']=='Spike_heigth',"Membrane_potential_mV"].values[0]
                Upstroke = first_spike_features_table.loc[first_spike_features_table['Feature']=='Upstroke',"Potential_first_time_derivative_mV/s"].values[0]
                Downstroke = first_spike_features_table.loc[first_spike_features_table['Feature']=='Downstroke',"Potential_first_time_derivative_mV/s"].values[0]
                Peak = first_spike_features_table.loc[first_spike_features_table['Feature']=='Peak',"Membrane_potential_mV"].values[0]
                Trough = first_spike_features_table.loc[first_spike_features_table['Feature']=='Trough',"Membrane_potential_mV"].values[0]
                Width = first_spike_features_table.loc[first_spike_features_table['Feature']=='Spike_width_at_half_heigth',"Time_s"].values[0]
                
                Stim_amp_pA = sweep_info_table.loc[sweep, 'Stim_amp_pA']
                
                new_spike_line = pd.DataFrame([cell_id, sweep, Stim_amp_pA, Threshold, Heigth, Upstroke, Downstroke, Peak, Trough, Width ]).T
                new_spike_line.columns = ['Cell_id','Sweep','Stim_amp_pA','Spike_threshold','Spike_heigth', "Spike_Upstroke", "Spike_Downstroke", "Spike_peak", 'Spike_trough', 'Spike_width']
                break
            else:
                new_spike_line = pd.DataFrame([cell_id, sweep, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ]).T
                new_spike_line.columns = ['Cell_id','Sweep','Stim_amp_pA','Spike_threshold','Spike_heigth', "Spike_Upstroke", "Spike_Downstroke", "Spike_peak", 'Spike_trough', 'Spike_width']
        return new_spike_line
    except:
        new_spike_line = pd.DataFrame([cell_id, '--', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ]).T
        new_spike_line.columns = ['Cell_id','Sweep','Stim_amp_pA','Spike_threshold','Spike_heigth', "Spike_Upstroke", "Spike_Downstroke", "Spike_peak", 'Spike_trough', 'Spike_width']
        return new_spike_line




