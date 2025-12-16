#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:22:09 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import plotnine as p9
import scipy
from lmfit.models import Model, QuadraticModel, ExponentialModel, ConstantModel
from lmfit import Parameters
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import importlib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import traceback
import warnings

import Ordinary_functions as ordifunc
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import matplotlib.pyplot as plt


class Single_Expo_Fit_Error(Exception):
    """Exception raised for errors in the fitting process."""
    pass

# LG cell_Full_TVC_table->cell_TVC_table (here and below)
def sweep_analysis_processing(cell_TVC_table, cell_stim_time_table):
    '''
    Create cell_sweep_info_table using parallel processing.
    For each sweep contained in cell_TVC_table, extract different recording information (Bridge Error, sampling frequency),
    stimulus and voltage trace information (holding current, resting voltage...),
    and trace-computed linear properties if possible (membrane time constant, input resistance)

    Parameters
    ----------
    cell_TVC_table : pd.DataFrame
        2 columns DataFrame, cotaining in column 'Sweep' the sweep_id and in the column 'TVC' the corresponding 3 columns DataFrame containing Time, Current and Potential Traces
        
    cell_stim_time_table : pd.DataFrame
        DataFrame, containing foir each Sweep the corresponding stimulus start and end times.
     

    Returns
    -------
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    '''
    cell_sweep_info_table = pd.DataFrame(columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_SS_pA', 'Holding_current_pA','Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz'])
    cell_TVC_table.index.name = 'Index'
    cell_TVC_table = cell_TVC_table.sort_values(by=['Sweep'])
    cell_stim_time_table.index.name = 'Index'
    cell_stim_time_table = cell_stim_time_table.sort_values(by=['Sweep'])
    sweep_list = np.array(cell_TVC_table['Sweep'], dtype=str)
    
    cell_sweep_info_table = get_sweep_info_loop(cell_TVC_table, cell_stim_time_table)

    ### create table of trace data
    extrapolated_BE = pd.DataFrame(columns=['Sweep', 'Bridge_Error_GOhms'])
    
    cell_sweep_info_table['Bridge_Error_GOhms']=removeOutliers(
        np.array(cell_sweep_info_table['Bridge_Error_GOhms']), 3)
    
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'
    
    ### For sweeps whose Brige Error couldn't be estimated, extrapolate from neighboring traces (protocol_wise)
    for current_Protocol in cell_sweep_info_table['Protocol_id'].unique():

        reduced_cell_sweep_info_table = cell_sweep_info_table[
            cell_sweep_info_table['Protocol_id'] == current_Protocol]
        reduced_cell_sweep_info_table=reduced_cell_sweep_info_table.astype({"Trace_id":"int"})
        reduced_cell_sweep_info_table=reduced_cell_sweep_info_table.sort_values(by=['Trace_id'])
        BE_array = np.array(
            reduced_cell_sweep_info_table['Bridge_Error_GOhms'])
        sweep_array = np.array(reduced_cell_sweep_info_table['Sweep'])

        nan_ind_BE = np.isnan(BE_array)

        x = np.arange(len(BE_array))
        if False in nan_ind_BE:

            BE_array[nan_ind_BE] = np.interp(
                x[nan_ind_BE], x[~nan_ind_BE], BE_array[~nan_ind_BE])

        extrapolated_BE_Series = pd.Series(BE_array)
        sweep_array = pd.Series(sweep_array)
        extrapolated_BE_Series = pd.DataFrame(
            pd.concat([sweep_array, extrapolated_BE_Series], axis=1))
        extrapolated_BE_Series.columns = ['Sweep', 'Bridge_Error_GOhms']
        extrapolated_BE = pd.concat(
            [extrapolated_BE,extrapolated_BE_Series], ignore_index=True)

    cell_sweep_info_table.pop('Bridge_Error_GOhms')

    cell_sweep_info_table = cell_sweep_info_table.merge(
        extrapolated_BE, how='inner', on='Sweep')

    ### Create Linear properties table
    
    TVC_list = []
    
    for x in sweep_list:
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        sweep_TVC = ordifunc.get_sweep_TVC_table(cell_TVC_table, x, do_filter=True, filter=5., do_plot=False)
        TVC_list.append(sweep_TVC)
    
    stim_start_time_list = list(cell_stim_time_table.loc[:,'Stim_start_s'])
    stim_end_time_list = list(cell_stim_time_table.loc[:,'Stim_end_s'])
    sweep_info_zip = zip(TVC_list,sweep_list,stim_start_time_list,stim_end_time_list)
    sweep_info_list= list(sweep_info_zip)
    
    Linear_table=pd.DataFrame(columns=['Sweep',"Time_constant_ms", 'Input_Resistance_GOhms', 'Holding_potential_mV','SS_potential_mV','Resting_potential_mV'])
    cell_sweep_info_table = cell_sweep_info_table.sort_values(by=['Sweep'])
    stim_start_time_list = list(cell_sweep_info_table.loc[:,'Stim_start_s'])
    stim_end_time_list = list(cell_sweep_info_table.loc[:,'Stim_end_s'])
    BE_extrapolated_list = list(cell_sweep_info_table.loc[:,'Bridge_Error_extrapolated'])
    BE_list = list(cell_sweep_info_table.loc[:,'Bridge_Error_GOhms'])
    stim_amp_list = list(cell_sweep_info_table.loc[:,'Stim_amp_pA'])
    
    sweep_info_zip = zip(TVC_list,sweep_list,stim_start_time_list,stim_end_time_list, BE_extrapolated_list, BE_list, stim_amp_list)
    sweep_info_list= list(sweep_info_zip)
    
    for x in sweep_info_list:
        result = get_sweep_linear_properties(x)
        Linear_table = pd.concat([
            Linear_table,result], ignore_index=True)

    cell_sweep_info_table = cell_sweep_info_table.merge(
        Linear_table, how='inner', on='Sweep')

# LG
    convert_dict = {'Sweep': str,
                    'Protocol_id': str,
                    'Trace_id': int,
                    'Stim_amp_pA': float,
                    'Holding_current_pA':float,
                    'Stim_SS_pA':float,
                    'Stim_start_s': float,
                    'Stim_end_s': float,
                    'Bridge_Error_GOhms': float,
                    'Bridge_Error_extrapolated': bool,
                    'Time_constant_ms': float,
                    'Input_Resistance_GOhms': float,
                    'Holding_potential_mV':float,
                    "SS_potential_mV": float,
                    "Resting_potential_mV":float,
                    'Sampling_Rate_Hz': float}
    cell_sweep_info_table = cell_sweep_info_table.loc[:, convert_dict.keys()]
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'
    cell_sweep_info_table = cell_sweep_info_table.astype(convert_dict)
    
    ### Remove extremes outliers of Time_constant and Input_Resistance (Q1/Q3 ± 3IQR)
    remove_outier_columns = ['Time_constant_ms', 'Input_Resistance_GOhms']

    for current_column in remove_outier_columns:
        cell_sweep_info_table.loc[:, current_column] = removeOutliers(
            np.array(cell_sweep_info_table.loc[:, current_column]), 3)

    
    cell_sweep_info_table.index = cell_sweep_info_table.loc[:, 'Sweep']
    cell_sweep_info_table.index = cell_sweep_info_table.index.astype(str)
    cell_sweep_info_table.index.name = 'Index'
   
    return cell_sweep_info_table

def get_sweep_info_loop(cell_TVC_table, cell_stim_time_table):
    '''
    Function to be used in parallel to extract or compute different sweep related information
    
    Parameters
    ----------
    sweep_info_list : List
        List of parameters required by the function.

    Returns
    -------
    output_line : pd.DataFrame
        One Row DataFrame containing different sweep related information.

    '''
    # LG
    # cell_sweep_info_table = pd.DataFrame(columns=['Sweep', 'Protocol_id', 'Trace_id', 'Stim_SS_pA', 'Holding_current_pA','Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz'])

    for sweep in cell_TVC_table.loc[:,'Sweep']:

        stim_start_time = cell_stim_time_table.loc[sweep,'Stim_start_s']
        stim_end_time = cell_stim_time_table.loc[sweep,'Stim_end_s']
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        filtered_sweep_TVC = ordifunc.get_sweep_TVC_table(cell_TVC_table, sweep, do_filter=True, filter=5., do_plot=False)
        unfiltered_sweep_TVC = ordifunc.get_sweep_TVC_table(cell_TVC_table, sweep, do_filter=False, filter=5., do_plot=False)

        if len(str(sweep).split("_")) == 1:
            Protocol_id = str(1)
            Trace_id = sweep
            
        elif len(str(sweep).split("_")) == 2 :
            Protocol_id, Trace_id = str(sweep).split("_")
            
        else :
            Protocol_id, Trace_id = str(sweep).split("_")[-2:]
        
        Protocol_id=str(Protocol_id)
            
        Holding_current,SS_current = fit_stimulus_trace(
            filtered_sweep_TVC, stim_start_time, stim_end_time,do_plot=False)[:2]

        if np.abs((Holding_current-SS_current))>=20.:
            Bridge_Error = estimate_bridge_error(unfiltered_sweep_TVC, SS_current, stim_start_time, stim_end_time, do_plot=False)[0]
        else:
            Bridge_Error=np.nan
        
        if np.isnan(Bridge_Error):
            BE_extrapolated = True
        else:
            BE_extrapolated = False
            
        time_array = np.array(unfiltered_sweep_TVC.loc[:,"Time_s"])
        sampling_rate = 1/(time_array[1]-time_array[0])
    
        stim_amp = SS_current - Holding_current

        output_line = pd.DataFrame(
            [str(sweep), str(Protocol_id), Trace_id,SS_current,Holding_current, stim_amp, stim_start_time, stim_end_time, Bridge_Error, BE_extrapolated, sampling_rate]).T
        output_line.columns=[
            'Sweep', 'Protocol_id', 'Trace_id', 'Stim_SS_pA', 'Holding_current_pA','Stim_amp_pA', 'Stim_start_s', 'Stim_end_s', 'Bridge_Error_GOhms', 'Bridge_Error_extrapolated', 'Sampling_Rate_Hz']
        #  LG
        cell_sweep_info_table = pd.DataFrame(columns=output_line.columns)
        cell_sweep_info_table = pd.concat([cell_sweep_info_table, output_line], ignore_index = True)
    
    return cell_sweep_info_table

def get_sweep_linear_properties(sweep_info_list):
    '''
    Function to be used in parallel to compute different sweep based linear properties 
    
    Parameters
    ----------
    sweep_info_list : List
        List of parameters required by the function.

    Returns
    -------
    sweep_line : pd.DataFrame
        One Row DataFrame containing different sweep based linear properties 


    '''
    
    
    original_TVC_table, sweep, stim_start, stim_end, BE_extrapolated, BE, stim_amp = sweep_info_list

    TVC_table = original_TVC_table.copy()
    #rely on BE corrected trace
    if not np.isnan(BE): 
        # in a case where BE could not be estimated for any traces in a given protocol, BE would be np.nan
        # In that case, do not compute BE corrected trace
        TVC_table.loc[:,'Membrane_potential_mV'] = TVC_table.loc[:,'Membrane_potential_mV']-BE*TVC_table.loc[:,'Input_current_pA']
    
    
    
    
    best_A,best_tau,SS_potential,RMSE = fit_membrane_time_cst_new(TVC_table,stim_end+0.002,
                                                                                     (stim_end+0.060),
                                                                                     do_plot=False)
    
    if RMSE/best_A > 0.1:
        best_tau = np.nan
    
    #Check stability of traces
    
    if np.isnan(best_tau):
        transient_time = .030 
    else:
        transient_time = 3*best_tau
    
   
    R_in, resting_potential, holding_potential, SS_potential, R2 = estimate_input_resistance_and_resting_potential(TVC_table, 
                                                                              stim_start, 
                                                                              stim_end, 
                                                                              best_tau, 
                                                                              do_plot = False)
    
    Do_linear_analysis = check_criteria_linear_analysis(TVC_table,  stim_start, stim_end, transient_time, R2)[0]

    if Do_linear_analysis == False:
    
        
        SS_potential = np.nan
        resting_potential = np.nan
        holding_potential = np.nan

        R_in = np.nan
        time_cst = np.nan
    
    else:
        
            
        Holding_current,SS_current = fit_stimulus_trace(
            TVC_table, stim_start, stim_end, do_plot=False)[:2]
        
        
        time_cst=best_tau*1e3 #convert s to ms

        
        
            
    sweep_line=pd.DataFrame([str(sweep),time_cst,R_in,holding_potential,SS_potential,resting_potential]).T        
    sweep_line.columns=['Sweep',"Time_constant_ms", 'Input_Resistance_GOhms', 'Holding_potential_mV','SS_potential_mV','Resting_potential_mV']
    
    return sweep_line

def check_criteria_linear_analysis(TVC_table, stim_start, stim_end, transient_time, R2):
    """
    Check the traces fill some criteria to accept linear analysis

    

    """
    
    voltage_spike_table = sp_an.identify_spike(np.array(TVC_table.Membrane_potential_mV),
                                    np.array(TVC_table.Time_s),
                                    np.array(TVC_table.Input_current_pA),
                                    stim_start,
                                    stim_end,
                                    do_plot=False)
    
    
    OFF_pre_stim_table = TVC_table.loc[(TVC_table['Time_s'] <= stim_start - 0.005),:]
    OFF_post_stim_table = TVC_table.loc[(TVC_table['Time_s'] >= stim_end + transient_time),:]
    
    OFF_table = pd.concat([OFF_pre_stim_table, OFF_post_stim_table], ignore_index = True)
    
    ON_table = TVC_table.loc[(TVC_table['Time_s'] >= stim_start + transient_time) & (TVC_table['Time_s'] <= stim_end - 0.005),:]
    
    I_bar_OFF = np.nanmean(OFF_table.loc[:,'Input_current_pA'])
    I_std_OFF = np.nanstd(OFF_table.loc[:,'Input_current_pA'])
    
    I_bar_ON = np.nanmean(ON_table.loc[:,'Input_current_pA'])
    I_std_ON = np.nanstd(ON_table.loc[:,'Input_current_pA'])
    
    V_bar_OFF = np.nanmean(OFF_table.loc[:,'Membrane_potential_mV'])
    V_std_OFF = np.nanstd(OFF_table.loc[:,'Membrane_potential_mV'])
    
    V_bar_ON = np.nanmean(ON_table.loc[:,'Membrane_potential_mV'])
    V_std_ON = np.nanstd(ON_table.loc[:,'Membrane_potential_mV'])
    
    condition_line = pd.DataFrame(["No spike",
                                   'max(V_on_std, V_off_std)/abs(V_on_mean-V_off_mean) ≤ 0.2',
                                   'max(I_on_std, I_off_std)/abs(I_on_mean-I_off_mean) ≤ 0.3',
                                   'abs(I_on_mean - I_off_mean) ≥ 40 pA',
                                   "abs(I_off_mean) ≤ 50 pA",
                                   "IR_R^2 ≥ 0.8"]).T
    
    
    measure_line = pd.DataFrame([str(len(voltage_spike_table['Peak'])),
                                 str(np.round((np.nanmax([V_std_OFF, V_std_ON]) / abs((V_bar_OFF - V_bar_ON))),2)),
                                 str(np.round(np.nanmax([I_std_OFF, I_std_ON]) / abs((I_bar_OFF - I_bar_ON)),2)),
                                 str(np.round(abs((I_bar_OFF - I_bar_ON)),2)),
                                 str(np.round(abs(I_bar_OFF),2)),
                                 str(np.round(R2,2))
                                 ]).T
    
    condition_respected_line = pd.DataFrame([len(voltage_spike_table['Peak']) ==0,
                                 (np.nanmax([V_std_OFF, V_std_ON]) / abs((V_bar_OFF - V_bar_ON))) <= 0.2,
                                 (np.nanmax([I_std_OFF, I_std_ON]) / abs((I_bar_OFF - I_bar_ON))) <= 0.3,
                                 abs((I_bar_OFF - I_bar_ON)) >= 40.0,
                                 abs(I_bar_OFF) <= 50.0,
                                 R2 >= 0.8]).T
    
    condition_table = pd.concat([measure_line, condition_respected_line])
    condition_table.columns = condition_line.iloc[0]
    condition_table.index = ["Measure", "Condition Respected"]
    
    
    # Check if all conditions are True
    all_conditions_met = condition_respected_line.all(axis=1).iloc[0]
    return all_conditions_met, condition_table
    
    
    
    
    
    

def fit_membrane_time_cst_new (original_TVC_table,start_time,end_time,do_plot=False):
    '''
    Fit decaying time constant model to membrane voltage trace

    Parameters
    ----------
    original_TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    start_time : Float
        Start time of the window to consider.
        
    end_time : Float
        End time of the window to consider.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau, best_C : Float
        Fitting results
        
    membrane_resting_voltage : Float
        
    NRMSE : Float
        Godness of fit.

    '''
    try:
        TVC_table=original_TVC_table.copy()
        
        sub_TVC_table=TVC_table[TVC_table['Time_s']<=(end_time)]
        sub_TVC_table=sub_TVC_table[sub_TVC_table['Time_s']>=(start_time)]
        

        normalized_time_sub_TVC_table = sub_TVC_table.copy()
        normalize_time = np.nanmin(normalized_time_sub_TVC_table.loc[:,"Time_s"])
        normalized_time_sub_TVC_table.loc[:,"Time_s"] -= normalize_time
        normalized_time_sub_TVC_table = normalized_time_sub_TVC_table.reset_index(drop=True)
        
        x_data=np.array(normalized_time_sub_TVC_table.loc[:,'Time_s'])
        y_data=np.array(normalized_time_sub_TVC_table.loc[:,"Membrane_potential_mV"])
        

        normalized_start_time = x_data[0]
        normalized_end_time = x_data[-1]
        
        normalized_start_time = start_time-np.nanmin(sub_TVC_table.loc[:,"Time_s"])
        normalized_end_time = end_time-np.nanmin(sub_TVC_table.loc[:,"Time_s"])
        
        
        start_idx = np.argmin(abs(x_data - normalized_start_time))
        end_idx = np.argmin(abs(x_data - normalized_end_time))
        
        

        mid_idx=int((end_idx+start_idx)/2)
        initial_membrane_SS=np.median(y_data[mid_idx:end_idx])
        
        x_0=normalized_start_time
        y_0=y_data[0]




        
        initial_A = y_0-y_data[-1]
        
        
        
        if (y_data[mid_idx]-initial_membrane_SS)/(initial_A) <=0:
            membrane_delta = initial_membrane_SS-y_0
        
            
            initial_voltage_time_cst = y_0+(2/3)*membrane_delta

            initial_voltage_time_cst_idx = np.argmin(abs(y_data[start_idx:end_idx] - initial_voltage_time_cst))+start_idx
            initial_time_cst = x_data[initial_voltage_time_cst_idx]
            
        else:
            initial_time_cst = -x_data[mid_idx]/(np.log((y_data[mid_idx]-initial_membrane_SS)/(initial_A)))
        

       
            
        membrane_time_cst_model=Model(time_cst_model)
       
        membrane_time_cst_model_pars=Parameters()
        
        membrane_time_cst_model_pars.add('A',value=initial_A)
        membrane_time_cst_model_pars.add('tau',value=initial_time_cst)
        membrane_time_cst_model_pars.add('C',value=initial_membrane_SS)
        
        membrane_time_cst_out=membrane_time_cst_model.fit(y_data, membrane_time_cst_model_pars, x=x_data)        

        best_A=membrane_time_cst_out.best_values['A']
        best_tau=membrane_time_cst_out.best_values['tau']
        best_C=membrane_time_cst_out.best_values['C']
        
        parameters_table = fir_an.get_parameters_table(membrane_time_cst_model_pars, membrane_time_cst_out)

        simulation=time_cst_model(x_data,best_A,best_tau,best_C)
        sim_table=pd.DataFrame(np.column_stack((x_data,simulation)),columns=["Time_s","Membrane_potential_mV"])
        
        RMSE = root_mean_squared_error(y_data, simulation)
        


        
        
        if do_plot==True:
            sim_table.loc[:,'Time_s']+=normalize_time
            
            plot_dict = {"TVC_table" : TVC_table,
                         "Sim_table" : sim_table,
                         "A":best_A,
                         "tau":best_tau,
                         "C" : best_C,
                        "RMSE":RMSE, 
                        "start_time":start_time,
                        "end_time":end_time}
            
            
    
            #print(my_plot)
            return plot_dict
        
        if best_tau > .080 or best_tau < .0029:
            raise Tau_Outside_Range_Error(f"Resulting tau is lower than 2.9 ms or higher than 80 ms.")
            
        
        
        return best_A,best_tau,best_C,RMSE
    except (TypeError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        RMSE=np.nan
        return best_A,best_tau,best_C,RMSE
    
    except Tau_Outside_Range_Error as e:
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        RMSE=np.nan
        return best_A,best_tau,best_C,RMSE
    
    # except Error_too_High as e:
    #     best_A=np.nan
    #     best_tau=np.nan
    #     best_C=np.nan
    #     RMSE=np.nan
    #     return best_A,best_tau,best_C,RMSE
        
    except(ValueError):
        best_A=np.nan
        best_tau=np.nan
        best_C=np.nan
        RMSE=np.nan
        return best_A,best_tau,best_C,RMSE
    



def estimate_input_resistance_and_resting_potential(original_TVC_table, stim_start_time, stim_end_time, membrane_time_constant, do_plot = False):
    """
    Input resistance is estimated by the slope of the linear fit of membrane potential and input current when the stimulus is off(before stim start time and after stim end time) and on (between stim start and stim end)
    To remove any transient membrane potential, a time wondow of 5ms is removed after stim start and stim end times
    The resting potential is computed as the intercept of the fit

    """
    
    TVC_table = original_TVC_table.copy()
    if np.isnan(membrane_time_constant):
        transient_time = .030 
    else:
        transient_time = 3*membrane_time_constant
    
    
    
    Pre_stim_start_table = TVC_table.loc[(TVC_table['Time_s'] <= stim_start_time-0.005),:]
    During_stim_table = TVC_table.loc[(TVC_table['Time_s'] >= stim_start_time+transient_time)&(TVC_table['Time_s'] <= stim_end_time-0.005),:]
    Post_stim_end_table =  TVC_table.loc[(TVC_table['Time_s'] >= stim_end_time+transient_time),:]
    
    Fitted_TCV_table = pd.concat([Pre_stim_start_table, During_stim_table, Post_stim_end_table], ignore_index = True)
    
    IR, V_rest = fir_an.linear_fit(np.array(Fitted_TCV_table.loc[:,'Input_current_pA']),
                                                       np.array(Fitted_TCV_table.loc[:,'Membrane_potential_mV']))
    
    pred_voltage = IR*np.array(Fitted_TCV_table.loc[:,'Input_current_pA']) + V_rest
    R2 = r2_score(np.array(Fitted_TCV_table.loc[:,'Membrane_potential_mV']), pred_voltage)
    
    
    holding_potential = np.nanmedian(Pre_stim_start_table.loc[:,'Membrane_potential_mV'])
    SS_potential = np.nanmedian(During_stim_table.loc[:,'Membrane_potential_mV'])
    
    Holding_current,SS_current = fit_stimulus_trace(
        TVC_table, stim_start_time, stim_end_time, do_plot=False)[:2]
     
    
    current_step = SS_current - Holding_current
    
    if do_plot == True:
        
        plot_dict = {"Full_TVC_table":TVC_table,
                         "Pre_stim_start_table" : Pre_stim_start_table,
                         "During_stim_table" : During_stim_table,
                         "Post_stim_end_table" : Post_stim_end_table,
                         "IR" : IR ,
                         "V_rest" : V_rest, 
                         "R2" : R2, 
                         "current_step" : current_step}
        return plot_dict
    
    if  np.abs(current_step) < 40.0 or R2 < 0.8:
        #To be accepted, the current step must be higher than 40pA, and the R2 of the fit must be higher than 0.8
        IR = np.nan
        V_rest = np.nan
    
    
    return IR, V_rest, holding_potential, SS_potential, R2
    
    

    

def removeOutliers(x, outlierConstant=3):
    '''
    Remove from an array outliers defined by values x<Q1-nIQR or x>Q3+nIQR
    with x a value in array and n the outlierConstant

    Parameters
    ----------
    x : np.array or list
        Values from whihc remove outliers.
        
    outlierConstant : int, optional
        The number of InterQuartile Range to define outleirs. The default is 3.

    Returns
    -------
    np.array
        x array without outliers.

    '''
    a = np.array(x)

    
    upper_quartile = np.nanpercentile(a, 75)
    lower_quartile = np.nanpercentile(a, 25)

    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
        else:
            resultList.append(np.nan)

    return np.array(resultList)


def fit_stimulus_trace(TVC_table_original,stim_start,stim_end,do_plot=False):
    '''
    Fit double Heaviside function to the time-varying trace representing the input current.
    The Trace is low-pass filtered at 1kHz, and the fit is weighted to the inverse of the time derivative of the trace

    Parameters
    ----------
    TVC_table_original : Tpd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stim_start : Float
        Stimulus start time.
        
    stim_end : Float
        Stimulus end time.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_baseline ,best_stim_amp, best_stim_start, best_stim_end : Float
        Fitting results.
        
    NRMSE_double_Heaviside : Float
        Godness of fit.

    '''
    
    TVC_table=TVC_table_original.copy()
    stim_table=TVC_table.loc[:,['Time_s','Input_current_pA']].copy()
    stim_table = stim_table.reset_index(drop=True)
    x_data=stim_table.loc[:,"Time_s"]
    index_stim_start=next(x for x, val in enumerate(x_data[0:]) if val >= stim_start )
    index_stim_end=next(x for x, val in enumerate(x_data[0:]) if val >= (stim_end) )
    
    

    stim_table['Input_current_pA']=np.array(ordifunc.filter_trace(stim_table['Input_current_pA'],
                                                                        stim_table['Time_s'],
                                                                        filter=1,
                                                                        do_plot=False))
    
    first_current_derivative=ordifunc.get_derivative(np.array(stim_table["Input_current_pA"]),np.array(stim_table["Time_s"]))
    
    
    stim_table['Filtered_Stimulus_trace_derivative_pA/ms']=np.array(first_current_derivative)
    
    
    
    before_stim_start_table=stim_table[stim_table['Time_s']<stim_start-.01]
    baseline_current=np.mean(before_stim_start_table['Input_current_pA'])
    
    during_stimulus_table=stim_table[stim_table['Time_s']<stim_end]
    during_stimulus_table=during_stimulus_table[during_stimulus_table['Time_s']>stim_start]
    estimate_stim_amp=np.median(during_stimulus_table.loc[:,'Input_current_pA'])
    
    
    
    y_data=stim_table.loc[:,"Input_current_pA"]
    stim_table['weight']=np.abs(1/stim_table['Filtered_Stimulus_trace_derivative_pA/ms'])**2
    

    
    
    weight=stim_table.loc[:,"weight"]
    
    
    if np.isinf(weight).sum!=0:
        
        if np.isinf(weight).sum() >= (len(weight)/2): # if inf values represent more than half the values

            weight=np.ones(len(weight))


        else: # otherwise replace inf values by the maximum value non inf

            max_weight_without_inf=np.nanmax(weight[weight != np.inf])
            weight.replace([np.inf], max_weight_without_inf, inplace=True)
    
    weight/=np.nanmax(weight) # normalize the weigth to the maximum weight 
    
    stim_table.loc[:,"weight"] = weight
    double_step_model=Model(Double_Heaviside_function)
    double_step_model_parameters=Parameters()
    double_step_model_parameters.add('stim_start',value=stim_start,vary=False)
    double_step_model_parameters.add('stim_end',value=stim_end,vary=False)
    double_step_model_parameters.add('baseline',value=baseline_current)
    double_step_model_parameters.add('stim_amplitude',value=estimate_stim_amp)

    double_step_out=double_step_model.fit(y_data,double_step_model_parameters,x=x_data,weights=weight)
    
    fit_results = fir_an.get_parameters_table(double_step_model_parameters, double_step_out)

    
    best_baseline=double_step_out.best_values['baseline']
    best_stim_amp=double_step_out.best_values['stim_amplitude']
    best_stim_start=double_step_out.best_values['stim_start']
    best_stim_end=double_step_out.best_values['stim_end']
    try:
        NRMSE_double_Heaviside=root_mean_squared_error(y_data.iloc[index_stim_start:index_stim_end], Double_Heaviside_function(x_data, best_stim_start, best_stim_end, best_baseline, best_stim_amp)[index_stim_start:index_stim_end])/(best_stim_amp)
    except:
        NRMSE_double_Heaviside = np.nan
    if do_plot:
        computed_y_data=pd.Series(Double_Heaviside_function(x_data, best_stim_start,best_stim_end, best_baseline, best_stim_amp))
        model_table=pd.DataFrame({'Time_s' :x_data,
                                  'Input_current_pA': computed_y_data})#np.column_stack((x_data,computed_y_data)),columns=['Time_s ',"Stim_amp_pA"])
        
        
        TVC_table['Legend']='Original_Data'
        
        stim_table['Legend']='Filtered_fitted_Data'
        model_table['Legend']='Fit'
        

        data_table = pd.concat([TVC_table,stim_table],ignore_index=True)
        data_table = pd.concat([data_table,model_table],ignore_index = True)
        
        my_plot = p9.ggplot()
        my_plot += p9.geom_line(TVC_table, p9.aes(x='Time_s',y='Input_current_pA'),colour = 'black')
        my_plot += p9.geom_line(stim_table,p9.aes(x='Time_s',y='Input_current_pA'),colour = 'red')
        #my_plot += p9.geom_line(stim_table,p9.aes(x='Time_s',y='weight'),colour = 'green')
        
        my_plot += p9.geom_line(model_table,p9.aes(x='Time_s',y='Input_current_pA'),colour='blue')
        my_plot += p9.xlab(str("Time_s"))
        my_plot += p9.xlim((stim_start-0.05), (stim_end+0.05))
        
        second_plot = p9.ggplot()
        second_plot += p9.geom_line(stim_table,p9.aes(x='Time_s',y='weight'),colour = 'green')
        second_plot += p9.xlab(str("Time_s"))
        second_plot += p9.xlim((stim_start-0.05), (stim_end+0.05))
        print(second_plot)
        
        

        print(my_plot)
        return stim_table
        
    return best_baseline,best_stim_amp,best_stim_start,best_stim_end,NRMSE_double_Heaviside


def estimate_bridge_error(original_TVC_table,stim_amplitude,stim_start_time,stim_end_time,do_plot=False):
    '''
    A posteriori estimation of bridge error, by estimating 'very fast' membrane voltage transient around stimulus start and end

    Parameters
    ----------
    original_TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
        
    stim_amplitude : float
        Value of Stimulus amplitude (between stimulus start and end).
        
    stim_start_time : float
        Stimulus start time.
        
    stim_end_time : float
        Stimulus end time.
        
    do_plot : TYPE, optional
        If True, returns Bridge Error Plots. The default is False.

    Returns
    -------
    if do_plot == True: return plots
    
    if do_plot == False: return Bridge Error in GOhms

    '''
    
    try:
        TVC_table=original_TVC_table.reset_index(drop=True).copy()
        start_time_index = np.argmin(abs(np.array(TVC_table['Time_s']) - stim_start_time))
        
        stimulus_baseline=np.mean(TVC_table.loc[:(start_time_index-1),'Input_current_pA'])
        
        Five_kHz_LP_filtered_current_trace = np.array(ordifunc.filter_trace(np.array(TVC_table.loc[:,'Input_current_pA']),
                                                                            np.array(TVC_table.loc[:,'Time_s']),
                                                                            filter=5,
                                                                            filter_order=4,
                                                                            zero_phase=True,
                                                                            do_plot=False
                                                                            ))
        
        
        current_trace_derivative_5kHz = np.array(ordifunc.get_derivative(Five_kHz_LP_filtered_current_trace,
                                                                          np.array(TVC_table.loc[:,'Time_s'])))
        

        TVC_table.loc[:,"I_dot_five_kHz"] = current_trace_derivative_5kHz


        Five_kHz_LP_filtered_voltage_trace = np.array(ordifunc.filter_trace(np.array(TVC_table.loc[:,'Membrane_potential_mV']),
                                                                            np.array(TVC_table.loc[:,'Time_s']),
                                                                            filter=5,
                                                                            filter_order=4,
                                                                            zero_phase=True,
                                                                            do_plot=False
                                                                            ))
        voltage_trace_derivative_5kHz = np.array(ordifunc.get_derivative(Five_kHz_LP_filtered_voltage_trace,
                                                                          np.array(TVC_table.loc[:,'Time_s'])))
        voltage_trace_second_derivative_5kHz = np.array(ordifunc.get_derivative(voltage_trace_derivative_5kHz,
                                                                          np.array(TVC_table.loc[:,'Time_s'])))
    
        TVC_table.loc[:,"V_dot_five_kHz"] = voltage_trace_derivative_5kHz
        TVC_table.loc[:,"V_double_dot_five_kHz"] = voltage_trace_second_derivative_5kHz
        
        One_kHz_LP_filtered_voltage_trace = np.array(ordifunc.filter_trace(np.array(TVC_table.loc[:,'Membrane_potential_mV']),
                                                                            np.array(TVC_table.loc[:,'Time_s']),
                                                                            filter=1,
                                                                            filter_order=4,
                                                                            zero_phase = True,
                                                                            do_plot=False
                                                                            ))
        voltage_trace_derivative_1kHz = np.array(ordifunc.get_derivative(One_kHz_LP_filtered_voltage_trace,
                                                                          np.array(TVC_table.loc[:,'Time_s'])))

        TVC_table.loc[:,"V_dot_one_kHz"] = voltage_trace_derivative_1kHz
        
        # Determine actual stimulus transition time and current step
        # Stimulus transition time = maximum(minimum) of first time current trace derivative in a ±4ms time window around stim_end_time for a positive (negative) current step
        
        stimulus_end_table = TVC_table.loc[(TVC_table["Time_s"]<=(stim_end_time+.004))&(TVC_table["Time_s"]>=(stim_end_time-.004)),:]
        stimulus_end_table = stimulus_end_table.reset_index(drop=True)
        
        if stim_amplitude <= stimulus_baseline: # negative current_step
            maximum_current_derivative_index = stimulus_end_table['I_dot_five_kHz'].idxmax()
            actual_transition_time = stimulus_end_table.loc[maximum_current_derivative_index, 'Time_s']
        
        elif stim_amplitude > stimulus_baseline:# Positive current_step
            minimum_current_derivative_index = stimulus_end_table['I_dot_five_kHz'].idxmin()
            actual_transition_time = stimulus_end_table.loc[minimum_current_derivative_index, 'Time_s']
        
        #Fit a sigmoid to current trace
        
        linear_slope, linear_intercept = fir_an.linear_fit(np.array(stimulus_end_table.loc[:,'Time_s']),
                                                           np.array(stimulus_end_table.loc[:,'Input_current_pA']))
        
        
        
        current_trace_table = TVC_table.loc[(TVC_table["Time_s"]<=(stim_end_time+.01))&(TVC_table["Time_s"]>=(stim_end_time-.01)),:]
        min_time_current = np.nanmin(current_trace_table.loc[:,'Time_s'])
        max_time_current = np.nanmax(current_trace_table.loc[:,'Time_s'])
       
        pre_T_current = np.median(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(actual_transition_time-0.001))&(current_trace_table["Time_s"]>=(actual_transition_time-0.002)),"Input_current_pA"]))
        post_T_current = np.median(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(actual_transition_time+0.002))&(current_trace_table["Time_s"]>=(actual_transition_time+0.001)),"Input_current_pA"]))
        
        
        
        delta_I = post_T_current - pre_T_current
    
        
        ##Are there fluctuations?
        #Compute std of voltage trace in the first half of the trace before the stimulus start
        std_v_double_dot = np.nanstd(np.array(TVC_table.loc[(TVC_table['Time_s']>=0.)&(TVC_table['Time_s']<=(stim_start_time/2)),'V_double_dot_five_kHz']))
        alpha_FT = 6*std_v_double_dot


        Fast_ringing_table = TVC_table.loc[(TVC_table["Time_s"]<=(actual_transition_time+.005))&(TVC_table["Time_s"]>=actual_transition_time),:]

        filtered_Fast_ringing_table = Fast_ringing_table[abs(Fast_ringing_table['V_double_dot_five_kHz']) > alpha_FT]

        if not filtered_Fast_ringing_table.empty: # A fast transient is detected
            T_FT = np.nanmax(filtered_Fast_ringing_table.loc[:,'Time_s'])
            T_ref_cell = T_FT

        else:
            T_FT = np.nan
            T_ref_cell = actual_transition_time
            
        # get T_Start_fit
        Post_T_ref_table = TVC_table.loc[TVC_table['Time_s'] >= T_ref_cell,:]
        Post_T_ref_table = Post_T_ref_table.reset_index(drop=True)


        if stim_amplitude <= stimulus_baseline: # negative current_step and Positive curent transition (as we are at stim end)
            filtered_Post_T_ref_table = Post_T_ref_table.loc[Post_T_ref_table['V_dot_one_kHz'] > 0,:]

            if not filtered_Post_T_ref_table.empty:
                first_positive_time = np.nanmin(filtered_Post_T_ref_table.loc[:,'Time_s'])
                delta_t_ref_first_positive = first_positive_time - T_ref_cell
            else:
                delta_t_ref_first_positive = np.nan
        
        elif stim_amplitude > stimulus_baseline:# Positive current_step and Negative curent transition (as we are at stim end)
            filtered_Post_T_ref_table = Post_T_ref_table.loc[Post_T_ref_table['V_dot_one_kHz'] < 0,:]
            
            if not filtered_Post_T_ref_table.empty:
                
                first_positive_time = np.nanmin(filtered_Post_T_ref_table.loc[:,'Time_s'])
                delta_t_ref_first_positive = first_positive_time - T_ref_cell
            else:
                delta_t_ref_first_positive = np.nan

        T_start_fit = T_ref_cell + np.nanmax(np.array([2*delta_t_ref_first_positive, 0.0003]))

        
        
        # Fit double exponential
        
        best_single_A,best_single_tau, best_single_C,RMSE_single_expo = np.nan, np.nan, np.nan, np.nan
        
        Exponential_TVC_table = TVC_table.loc[(TVC_table['Time_s'] >= T_start_fit)&(TVC_table['Time_s'] <= T_start_fit+0.004),:]
        
       
        
        
        
    
        best_single_A,best_single_tau, best_single_C, RMSE_single_expo, V_1exp_fit = fit_single_exponential_BE(Exponential_TVC_table,False)
       
        
        extended_time_trace = np.array(TVC_table.loc[(TVC_table['Time_s'] >= actual_transition_time)&(TVC_table['Time_s'] <= T_start_fit+0.005),'Time_s'])
        extended_time_trace_shifted = extended_time_trace-np.nanmin(np.array(Exponential_TVC_table.loc[:,"Time_s"]))
        estimated_membrane_potential = time_cst_model(extended_time_trace_shifted,
                                                                            best_single_A, best_single_tau, best_single_C)
        
        V_fit_table = pd.DataFrame({'Time_s':np.array(TVC_table.loc[(TVC_table['Time_s'] >= actual_transition_time)&(TVC_table['Time_s'] <= T_start_fit+0.005),'Time_s']),
                                       'Membrane_potential_mV' : np.array(estimated_membrane_potential)})

        V_table_to_fit = V_1exp_fit.loc[V_1exp_fit['Data'] == 'Original_Data',:]
        
        shift_transition_time = actual_transition_time-np.nanmin(Exponential_TVC_table.loc[:,'Time_s'])
        V_post_transition_time = time_cst_model(shift_transition_time, best_single_A, best_single_tau, best_single_C)
            
        #Determine V_pre and V_post
        # LG change - Estimate V_pre from 0.5kHz LPF of voltage for (T*-5 <= time <= T*),  not entire trace
        # Keep 0.5kHz LPF of entire trace for plotting only?
        TVC_table_subset = TVC_table.loc[(TVC_table["Time_s"]>=(actual_transition_time-0.005)) & (TVC_table["Time_s"]<=(actual_transition_time)), :].copy()
    
        Zero_5_kHz_LP_filtered_voltage_trace_subset = np.array(ordifunc.filter_trace(np.array(TVC_table_subset.loc[:,'Membrane_potential_mV']),
                                                                            np.array(TVC_table_subset.loc[:,'Time_s']),
                                                                            filter=.5,
                                                                            filter_order=4,
                                                                            zero_phase = True,
                                                                            do_plot=False
                                                                            ))


        
        TVC_table_subset.loc[:,"Membrane_potential_0_5_LPF"] = Zero_5_kHz_LP_filtered_voltage_trace_subset

        TVC_table = pd.merge(TVC_table, TVC_table_subset.loc[:,['Time_s','Membrane_potential_0_5_LPF']], on="Time_s", how='outer')


        V_pre_transition_time_old = TVC_table_subset.loc[TVC_table_subset['Time_s']==actual_transition_time, "Membrane_potential_0_5_LPF"].values[0]

        Time_s=np.array(TVC_table.loc[:,'Time_s'])
        Membrane_potential_mV=np.array(TVC_table.loc[:,'Membrane_potential_mV'])

        pre_T_start_time=(actual_transition_time-0.005)
        # pre_T_time, pre_T_V=ordifunc.time_slice_of_trace(Time_s, Membrane_potential_mV,
        #                                                  start_time=pre_T_start_time, end_time=actual_transition_time)
        V_pre_T_filtered=ordifunc.filter_trace(Membrane_potential_mV, Time_s,
                                               filter=0.5, filter_order = 2, zero_phase = False,do_plot=False,
                                               start_time_sec=pre_T_start_time,
                                               end_time_sec=actual_transition_time)
        nan_padded_V_pre_T_filtered=ordifunc.frame_with_nans(Time_s, pre_T_start_time,
                                                             V_pre_T_filtered)
        # print(f'{len(Time_s)=}')
        # print(f'{len(V_pre_T_filtered)=}')
        # print(f'{len(nan_padded_V_pre_T_filtered)=}')
        # print(f'{len(TVC_table_subset["Time_s"])=}')
        # print(f'{len(TVC_table_subset["Time_s"])=}')

        # TVC_table_subset.loc[:,"V_pre_T_0_5_LPF"] = nan_padded_V_pre_T_filtered
        TVC_table["V_pre_T_0_5_LPF"] = nan_padded_V_pre_T_filtered
        # TVC_table_subset.loc[:,"V_pre_Time_s"] = pre_T_time

        # TVC_table = pd.merge(TVC_table,
        #                      TVC_table_subset.loc[:,['Time_s','V_pre_T_0_5_LPF']],
        #                      on="Time_s", how='outer')

        V_pre_transition_time=V_pre_T_filtered[-1]
        # print(f'  {V_pre_transition_time_old=} {V_pre_transition_time=}')
        delta_V = V_post_transition_time - V_pre_transition_time
        
        Bridge_Error = delta_V/delta_I
    
        
        BE_accepted = True

        # Check if Bridge Error is accepted
        ## Avoid Delayed Response
        Filtered_TVC_V_dot_one_kHz =  TVC_table.loc[(TVC_table['Time_s'] >= T_ref_cell-0.025) & (TVC_table['Time_s'] <= T_ref_cell+0.025),:]
        max_abs_V_dot_index = Filtered_TVC_V_dot_one_kHz['V_dot_one_kHz'].abs().idxmax()
        # Get the corresponding time value
        time_at_max_abs_value = Filtered_TVC_V_dot_one_kHz.loc[max_abs_V_dot_index, 'Time_s']

        BE_accepted, test_table = accept_or_reject_BE(TVC_table, time_at_max_abs_value, T_ref_cell, actual_transition_time, delta_t_ref_first_positive, RMSE_single_expo, best_single_A)
        
        if BE_accepted == False:
            Bridge_Error = np.nan
        
        if do_plot:
            
            dict_plot = {'TVC_table':TVC_table,
                        "Transition_time" : actual_transition_time,
                        "min_time_current":min_time_current,
                        "max_time_current":max_time_current,
                        "T_FT" : T_FT,
                        "alpha_FT":alpha_FT,
                        "T_ref_cell":T_ref_cell,
                        "delta_t_ref_first_positive":delta_t_ref_first_positive,
                        "Membrane_potential_mV" : {"V_fit_table" : V_fit_table,
                                                   "V_table_to_fit" : V_table_to_fit,
                                                    "Voltage_Pre_transition" : V_pre_transition_time,
                                                    "Voltage_Post_transition" :V_post_transition_time},
                         "Input_current_pA" : {"pre_T_current":pre_T_current,
                                               "post_T_current" : post_T_current,
                                               "current_trace_table" : current_trace_table}}
                
            
            return dict_plot
        
        
        return Bridge_Error, test_table
    except RuntimeError:
        error= traceback.format_exc()
        Bridge_Error = np.nan
        test_table = pd.DataFrame()
        
        return  Bridge_Error, test_table
    
    except Single_Expo_Fit_Error as e:
            # Handle the custom exception
        obs = str(e)  # Capture the custom error message
        Bridge_Error = np.nan
        test_table = pd.DataFrame()
        
        return  Bridge_Error, test_table
    
def accept_or_reject_BE(TVC_table, time_at_max_abs_value, T_ref_cell,Transition_time, delta_t_ref_first_positive, RMSE_single_expo, best_single_A):
    """
    Evaluate several conditions to accept the a posteriori Bridge Error estimate based on several intermediate measurement of the procedure

    Parameters
    ----------
    TVC_table : pd.DataFrame
        Contains the Time, voltage, Current and voltage 1st and 2nd derivatives arranged in columns.
    time_at_max_abs_value : float

    T_ref_cell : float

    Transition_time : float
        
    delta_t_ref_first_positive : float

    exponential_fit : float

    RMSE_double_expo : float

    best_first_A : float

    best_second_A : float

    RMSE_single_expo : float

    best_single_A : float


    Returns
    -------
    BE_accepted : Bool
        Wether accept the Bridge Error estimate or not based on the different conditions.
    test_table : pd.DataFrame
        Table describing if the procedure passed the different criteria for validation.

    """
    
    BE_accepted = True  # Start with the assumption that the condition is accepted
    obs = '--'

    # Initialize a list to store test results
    test_results = {
        'Condition': [],
        'Met': [],
        'Details': []
    }

    # Condition 1: Check for delayed response
    if time_at_max_abs_value > T_ref_cell + 0.002:
        BE_accepted = False
        obs = f"time_at_max_abs_value = {time_at_max_abs_value}, T_Ref + 0.002={T_ref_cell+0.002}"
        test_results['Condition'].append('time_at_max_abs_value <=T_ref_cell+0.002')
        test_results['Met'].append(False)
        test_results['Details'].append(obs)
    else:
        obs = f"time_at_max_abs_value = {time_at_max_abs_value}, T_ref_cell + 0.002={T_ref_cell+0.002}"
        test_results['Condition'].append('time_at_max_abs_value <= T_ref_cell+0.002')
        test_results['Met'].append(True)
        test_results['Details'].append(obs)

    # Condition 2: Avoid long biphasic phase
    if delta_t_ref_first_positive > 1:
        BE_accepted = False
        obs = f"delta_t_ref_first_positive = {delta_t_ref_first_positive}"
        test_results['Condition'].append('Long biphasic phase, delta_t_ref_first_positive ≤ 1 ')
        test_results['Met'].append(False)
        test_results['Details'].append(obs)
    else:
        obs = f"delta_t_ref_first_positive = {delta_t_ref_first_positive} "
        test_results['Condition'].append('Long biphasic phase,delta_t_ref_first_positive  ≤ 1 ')
        test_results['Met'].append(True)
        test_results['Details'].append(obs)

    # Condition 3: Reliable estimate of V_pre_transition_time
    sub_table = TVC_table.loc[(TVC_table['Time_s'] >= Transition_time-0.005) & (TVC_table['Time_s'] <= Transition_time),:]
    sub_membrane_trace = np.array(sub_table.loc[:,'Membrane_potential_mV'])

    std_V = np.nanstd(sub_membrane_trace)
    if std_V > 2.0:
        BE_accepted = False
        obs = f"Reliability of V_pre_transition_time estimate, std_V = {std_V}"
        test_results['Condition'].append('std_V ≤ 2')
        test_results['Met'].append(False)
        test_results['Details'].append(obs)
    else:
        obs = f"Reliability of V_pre_transition_time estimate, std_V = {std_V}"
        test_results['Condition'].append('std_V ≤ 2')
        test_results['Met'].append(True)
        test_results['Details'].append(obs)

   

    # Condition 5: Error limit for 1 exponential fit
    
    error_ratio = RMSE_single_expo / best_single_A
    if error_ratio > 0.12:
        BE_accepted = False
        obs = f"Error for 1 exponential fit, RMSE_single_expo / best_single_A = {error_ratio}"
        test_results['Condition'].append('RMSE_single_expo / best_single_A ≤ 0.12')
        test_results['Met'].append(False)
        test_results['Details'].append(obs)
    else:
        obs = f"Error for 1 exponential fit, RMSE_single_expo / best_single_A = {error_ratio}"
        test_results['Condition'].append('RMSE_single_expo / best_single_A ≤ 0.12')
        test_results['Met'].append(True)
        test_results['Details'].append(obs)

    voltage_spike_table = sp_an.identify_spike(np.array(TVC_table.Membrane_potential_mV),
                                    np.array(TVC_table.Time_s),
                                    np.array(TVC_table.Input_current_pA),
                                    T_ref_cell,
                                    np.nanmax(np.array(TVC_table.Time_s)),do_plot=False)

    if len(voltage_spike_table['Peak']) != 0:
        BE_accepted = False
        obs = f"Presence of {len(voltage_spike_table['Peak'])} spikes in a ± 25ms window around T_ref"
        test_results['Condition'].append('No rebound spike must be present')
        test_results['Met'].append(False)
        test_results['Details'].append(obs)
    else:
        obs = f"Presence of {len(voltage_spike_table['Peak'])} spikes in a ± 25ms window around T_ref"
        test_results['Condition'].append('No rebound spike must be present')
        test_results['Met'].append(True)
        test_results['Details'].append(obs)

    # Create the test_table DataFrame
    test_table = pd.DataFrame(test_results)

    return BE_accepted, test_table
        
    

def plot_BE(dict_plot):
    '''
    Generate interactive plotly graph descrbing the Bridge Error analysis, with the different intermediate measurements

    Parameters
    ----------
    dict_plot : dict
        dictionnary containing the different elements from the analysis. 
        Produced by function estimate_bridge_error, when do_plot = True.

    

    '''
    
    TVC_table = dict_plot['TVC_table']

    min_time_current = dict_plot['min_time_current']
    max_time_current = dict_plot['max_time_current']
        # Transition_time
    actual_transition_time = dict_plot["Transition_time"]
    alpha_FT = dict_plot['alpha_FT']
    T_FT = dict_plot['T_FT']
    delta_t_ref_first_positive = dict_plot["delta_t_ref_first_positive"]
    T_ref_cell = dict_plot['T_ref_cell']
    
    
    # Membrane_potential_mV
    V_fit_table = dict_plot["Membrane_potential_mV"]["V_fit_table"]
    V_table_to_fit = dict_plot['Membrane_potential_mV']['V_table_to_fit']
    V_pre_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Pre_transition"]
    V_post_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Post_transition"]
    
    # Input_current_pA
    pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
    post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
    current_trace_table = dict_plot["Input_current_pA"]["current_trace_table"]
    
    # Create subplots
    fig = make_subplots(rows=5, cols=1,  shared_xaxes=True, subplot_titles=("Membrane Potential plot", "Input Current plot", "Membrane potential first derivative 1kHz LPF","Membrane potential second derivative 5kHz LPF","Input current derivative 5kHz LPF"), vertical_spacing=0.03)
    
    # Membrane potential plot
    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_mV'], mode='lines', name='Cell_trace', line=dict(color='black', width =1 )), row=1, col=1)
    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Membrane_potential_0_5_LPF'], mode='lines', name="Membrane_potential_0_5_LPF", line=dict(color='#680396')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_pre_transition_time], mode='markers', name='Voltage_Pre_transition', marker=dict(color='#fa2df0', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[actual_transition_time], y=[V_post_transition_time], mode='markers', name='Voltage_Post_transition', marker=dict(color='#9e2102', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=V_table_to_fit['Time_s'], y=V_table_to_fit['Membrane_potential_mV'], mode='lines', line=dict(color='#c96a04'), name='Fitted membrane potential'), row=1, col=1)
    fig.add_trace(go.Scatter(x=V_fit_table['Time_s'], y=V_fit_table['Membrane_potential_mV'], mode='lines', line=dict(color='orange'), name='Post_transition_voltage_fit'), row=1, col=1)
    
    
    
    # Input current plot
    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['Input_current_pA'], mode='lines', name='Input_current_trace', line=dict(color='black', width =1 )), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(min_time_current+0.005))&(current_trace_table["Time_s"]>=(min_time_current)),"Time_s"]), 
        y=[pre_T_current]*len(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(min_time_current+0.005))&(current_trace_table["Time_s"]>=(min_time_current)),"Time_s"])), 
        mode='lines', name="Pre transition fit Median", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(max_time_current))&(current_trace_table["Time_s"]>=(max_time_current-0.005)),"Time_s"]), 
        y=[post_T_current]*len(np.array(current_trace_table.loc[(current_trace_table["Time_s"]<=(max_time_current))&(current_trace_table["Time_s"]>=(max_time_current-0.005)),"Time_s"])), 
        mode='lines', name="Post transition fit Median", line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[actual_transition_time], y=[pre_T_current], mode='markers', name='pre_Transition_current', marker=dict(color='red', size = 8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[actual_transition_time], y=[post_T_current], mode='markers', name='post_Transition_current', marker=dict(color='blue', size=8)), row=2, col=1)
    
    # V_dot_one_kHz
    first_sign_change = delta_t_ref_first_positive+T_ref_cell
    min_V_dot = np.nanmin(TVC_table['V_dot_one_kHz'])
    max_V_dot = np.nanmax(TVC_table['V_dot_one_kHz'])

    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['V_dot_one_kHz'], mode='lines', name='V_dot_one_kHz', line=dict(color='black', width = 1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[first_sign_change]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                             mode='lines', name=f'first_sign_change = {round(first_sign_change,4)}s <br> delta_t_ref_first_positive = {round(delta_t_ref_first_positive,4)}s ', line=dict(color='blueviolet', dash='dash')), row=3, col=1)
    # Add dashed vertical line at time = T_star
    fig.add_trace(go.Scatter(x=[T_ref_cell]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                             mode='lines', name=f'T_ref_cell = {round(T_ref_cell,4)}s', line=dict(color='red', dash='dash')), row=3, col=1)
    T_start_fit = np.nanmin(V_table_to_fit['Time_s'])
    fig.add_trace(go.Scatter(x=[T_start_fit]*len(np.arange(min_V_dot,max_V_dot,1)), y=np.arange(min_V_dot,max_V_dot,1),
                             mode='lines', name=f'T_start_fit = {round(T_start_fit,4)}s', line=dict(color='#c96a04', dash='dash')), row=3, col=1)
    
    
    # V_double_dot_5KHz plot
    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['V_double_dot_five_kHz'], mode='lines', name='V_double_dot_five_kHz', line=dict(color='black', width = 1)), row=4, col=1)
    # Add dashed vertical line at time = T_star
    Fast_ring_time = np.array(TVC_table.loc[(TVC_table["Time_s"]<=(actual_transition_time+.005))&(TVC_table["Time_s"]>=actual_transition_time),'Time_s'])
    max_V_double_dot_five_kHz = np.nanmax(TVC_table['V_double_dot_five_kHz'])
    min_V_double_dot_five_kHz = np.nanmin(TVC_table['V_double_dot_five_kHz'])
    fig.add_trace(go.Scatter(x=Fast_ring_time, y=[alpha_FT]*len(Fast_ring_time),
                             mode='lines', name=f'alpha_FT = {round(alpha_FT,4)} mV/s/s', line=dict(color='darkred', dash='dash')), row=4, col=1)
    fig.add_trace(go.Scatter(x=Fast_ring_time, y=[-alpha_FT]*len(Fast_ring_time),
                             mode='lines', name=f'-alpha_FT = {round(-alpha_FT,4)} mV/s/s', line=dict(color='darkred', dash='dash')), row=4, col=1)
    fig.add_trace(go.Scatter(x=[T_FT]*len(np.arange(min_V_double_dot_five_kHz, max_V_double_dot_five_kHz, 1.)), y=np.arange(min_V_double_dot_five_kHz, max_V_double_dot_five_kHz, 1.),
                             mode='lines', name=f'T_FT = {round(T_FT,4)}s', line=dict(color='darkred', dash='dash')), row=4, col=1)

    # I_dot_5_kHz plot
    fig.add_trace(go.Scatter(x=TVC_table['Time_s'], y=TVC_table['I_dot_five_kHz'], mode='lines', name='I_dot_5_kHz', line=dict(color='black', width = 1)), row=5, col=1)
    
    # Add dashed vertical line at time = T_star
    fig.add_trace(go.Scatter(x=[actual_transition_time, actual_transition_time], y=[TVC_table['I_dot_five_kHz'].min(), TVC_table['I_dot_five_kHz'].max()],
                             mode='lines', name=f'T_star = {round(actual_transition_time,4)}s', line=dict(color='red', dash='dash')), row=5, col=1)
    

    # Update layout
    fig.update_layout(

        height=1200,
        showlegend=True,
        
    )
    
    # Update x and y axes

    fig.update_yaxes(title_text="mV", row=1, col=1)
    fig.update_xaxes(title_text="Time s", row=5, col=1)
    fig.update_yaxes(title_text="pA", row=2, col=1)
    fig.update_yaxes(title_text="mV/ms", row=3, col=1)
    fig.update_yaxes(title_text="mV/ms/ms", row=4, col=1)
    fig.update_yaxes(title_text="pA/ms", row=5, col=1)
    
    fig.show()

    

def fit_single_exponential_BE(original_fit_table, do_plot=False):
    '''
    Fit an exponential curve to to membrane trace at either the stimulus start or the stimulus end, during the estimation of the Bridge Error

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    stim_start : Boolean, optional
        Wether to fit at the stimulus start or end time. The default is True.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau : Float
        Fit result.
    RMSE_expo : Float
        Godness of fit.

    '''
    
    try:
        fit_table=original_fit_table.copy()
        fit_table=fit_table.reset_index(drop=True)
        #Shift time_value so that it starts at 0, easier to fit
        Time_shift = np.nanmin(fit_table.loc[:,'Time_s'])
        fit_table.loc[:,'Time_s']-=Time_shift

        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        
        membrane_start_voltage = y_data[0]
        membrane_end_voltage = y_data[-1]
        
        membrane_delta=membrane_end_voltage - membrane_start_voltage

        
        membrane_voltage_2_3 = membrane_start_voltage + membrane_delta*2/3
        fit_table['abs_diff'] = abs(fit_table['Membrane_potential_mV'] - membrane_voltage_2_3)
        min_diff_index = fit_table['abs_diff'].idxmin()
        x_2_3 = fit_table.loc[min_diff_index, 'Time_s']
        
       
        init_A = (y_data[0]-y_data[-1])/(np.exp(0/(x_2_3-x_data[0])))
        
        initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
        initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
        initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
        
        

        single_exponential_model = Model(time_cst_model)
        
        single_exponential_parameters=Parameters()
      
        single_exponential_parameters.add("A",init_A)
        single_exponential_parameters.add('tau',min=0.0005, value = initial_time_cst)
        
        single_exponential_parameters.add("C", membrane_end_voltage)

        result = single_exponential_model.fit(y_data, single_exponential_parameters, x=x_data)

        best_first_A=result.best_values['A']
        best_first_tau=result.best_values['tau']
        
        best_C = result.best_values['C']
        
        
        
        pred = time_cst_model(x_data, best_first_A, best_first_tau, best_C)
        
        RMSE_double_expo = root_mean_squared_error(y_data, pred)

        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Single_Expo_Data'
        fit_table.loc[:,'Time_s']+=Time_shift
        simulation_table.loc[:,'Time_s']+=Time_shift
        
        fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            print(my_plot)
        
        return best_first_A,best_first_tau, best_C, RMSE_double_expo, fit_table
    
    except (ValueError):
        best_first_A=np.nan
        best_first_tau=np.nan
        
        best_C = np.nan
        RMSE_double_expo=np.nan
        fit_table = pd.DataFrame(columns=['Time_s', 'Membrane_potential_mV'])
        return best_first_A,best_first_tau, best_C, RMSE_double_expo, fit_table
    

def double_exponential_decay_function(x, first_A, first_tau, second_A, second_tau, C):
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

    
    return  first_A*np.exp(-(x)/first_tau) + second_A*np.exp(-(x)/second_tau) + C

def fit_double_exponential_BE(original_fit_table, do_plot=False):
    '''
    Fit an exponential curve to to membrane trace at either the stimulus start or the stimulus end, during the estimation of the Bridge Error

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    stim_start : Boolean, optional
        Wether to fit at the stimulus start or end time. The default is True.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau : Float
        Fit result.
    RMSE_expo : Float
        Godness of fit.

    '''
    
    try:
        fit_table=original_fit_table.copy()
        #Shift time_value so that it starts at 0, easier to fit
        Time_shift = np.nanmin(fit_table.loc[:,'Time_s'])
        fit_table.loc[:,'Time_s']-=Time_shift
        
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        
        
        
        membrane_start_voltage = y_data[0]
        membrane_end_voltage = y_data[-1]
        
        membrane_delta=membrane_end_voltage - membrane_start_voltage

        
        membrane_voltage_2_3 = membrane_start_voltage + membrane_delta*2/3
        fit_table['abs_diff'] = abs(fit_table['Membrane_potential_mV'] - membrane_voltage_2_3)
        min_diff_index = fit_table['abs_diff'].idxmin()
        x_2_3 = fit_table.loc[min_diff_index, 'Time_s']
        
       
        init_A = (y_data[0]-y_data[-1])/(np.exp(0/(x_2_3-x_data[0])))
        

        initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
        initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
        initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
        
        
        
        
        double_exponential_model = Model(double_exponential_decay_function)
        
        double_exponential_parameters=Parameters()

    
        double_exponential_parameters.add("first_A", value= init_A)
        double_exponential_parameters.add('first_tau',min=0.0005, value = initial_time_cst)
        double_exponential_parameters.add("second_A", value= init_A)
        double_exponential_parameters.add('second_tau',min=0.0005, value = initial_time_cst)
        double_exponential_parameters.add("C", value = y_data[-1])
        result = double_exponential_model.fit(y_data, double_exponential_parameters, x=x_data)
    
        best_first_A=result.best_values['first_A']
        best_first_tau=result.best_values['first_tau']
        best_second_A=result.best_values['second_A']
        best_second_tau=result.best_values['second_tau']
        best_C = result.best_values['C']
        
        
        
        pred = double_exponential_decay_function(x_data,best_first_A,best_first_tau,best_second_A, best_second_tau, best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE_double_expo = np.sqrt(sum_squared_error / y_data.size)
        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Double_Expo_Data'
        fit_table.loc[:,'Time_s']+=Time_shift
        simulation_table.loc[:,'Time_s']+=Time_shift
        fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            print(my_plot)
        
        return best_first_A,best_first_tau,best_second_A, best_second_tau, best_C, RMSE_double_expo, fit_table
    
    except (ValueError):
        best_first_A=np.nan
        best_first_tau=np.nan
        best_second_A=np.nan
        best_second_tau=np.nan
        best_C = np.nan
        RMSE_double_expo=np.nan
        return best_first_A,best_first_tau,best_second_A, best_second_tau, best_C, RMSE_double_expo, fit_table
  
class Tau_Too_Long_Error(Exception):
    """Exception raised when resulting tau is longer than fitted time window."""
    pass
class Tau_Outside_Range_Error(Exception):
    """Exception raised when resulting tau lower than 2.9 ms or higher than 80 ms."""
    pass
class Error_too_High(Exception):
    """Exception raised when A/RMSE for time constant fit is higher than 0.1 ."""
    pass

 

    
def Double_Heaviside_function(x, stim_start, stim_end,baseline,stim_amplitude):
    """Heaviside step function."""
    
    if stim_end<=min(x):
        o=np.empty(x.size);o.fill(stim_amplitude)
        return o
    
    elif stim_start>=max(x):
        o=np.empty(x.size);o.fill(baseline)
        return o
    
    else:
        o=np.empty(x.size);o.fill(baseline)
        
        
        start_index = max(np.where( x < stim_start)[0])

        end_index=max(np.where( x < stim_end)[0])
        
        o[start_index:end_index] = stim_amplitude
    
        return o

def fit_second_order_poly(original_fit_table,do_plot=False):
    '''
    Fit 2nd order polynomial to time varying signal

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    a, b, c : Float
        Fitting results.
    RMSE_poly : Float
        Godness of fit.

    '''
    try:
        fit_table=original_fit_table.copy()
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        
        poly_model=QuadraticModel()
        pars = poly_model.guess(y_data, x=x_data)
        out = poly_model.fit(y_data, pars, x=x_data)
        
        a=out.best_values["a"]
        b=out.best_values["b"]
        c=out.best_values["c"]
        
        pred=a*((x_data)**2)+b*x_data+c
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE_poly = np.sqrt(sum_squared_error / y_data.size)
        
        
        
        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Poly_Data'
       # fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            my_plot+=p9.geom_line(simulation_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))
            print(my_plot)
        return a,b,c,RMSE_poly
    
    except (ValueError):
        a = np.nan
        b = np.nan
        c = np.nan
        RMSE_poly = np.nan
        return  a,b,c,RMSE_poly
    
def fit_exponential_BE(original_fit_table,stim_start=True,do_plot=False):
    '''
    Fit an exponential curve to to membrane trace at either the stimulus start or the stimulus end, during the estimation of the Bridge Error

    Parameters
    ----------
    original_fit_table : pd.DataFrame
        Contains the Time, voltage, Current traces arranged in columns.
        
    stim_start : Boolean, optional
        Wether to fit at the stimulus start or end time. The default is True.
        
    do_plot : Boolean, optional
        Do plot. The default is False.

    Returns
    -------
    best_A, best_tau : Float
        Fit result.
    RMSE_expo : Float
        Godness of fit.

    '''
    
    try:
        fit_table=original_fit_table.copy()
        x_data=np.array(fit_table.loc[:,'Time_s'])
        y_data=np.array(fit_table.loc[:,"Membrane_potential_mV"])
        expo_model=ExponentialModel()
        
        
        membrane_start_voltage = y_data[0]
        membrane_end_voltage = y_data[-1]
        
        membrane_delta=membrane_end_voltage - membrane_start_voltage
    
        
        if stim_start == True:
            
            exp_offset = y_data[-1]
            initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
            initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
            initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
            initial_A=(membrane_start_voltage-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            

                
        elif stim_start == False:

            
            
            exp_offset = y_data[-1]
            initial_voltage_time_cst=membrane_end_voltage+(1-(1/np.exp(1)))*membrane_delta
            initial_voltage_time_cst_idx=np.argmin(abs(y_data - initial_voltage_time_cst))
            initial_time_cst=x_data[initial_voltage_time_cst_idx]-x_data[0]
            
            initial_time_cst= -initial_time_cst
            initial_A=(membrane_start_voltage-exp_offset)/np.exp(-x_data[0]/(initial_time_cst))
            
        
        
    
    
        expo_model = Model(time_cst_model)
        expo_model_pars=Parameters()
        expo_model_pars.add('A',value=initial_A)
        expo_model_pars.add('tau',value=initial_time_cst)
        expo_model_pars.add('C',value=exp_offset)
        
        expo_out=expo_model.fit(y_data, expo_model_pars, x=x_data)        
        
        
        best_A=expo_out.best_values['A']
        best_tau=expo_out.best_values['tau']
        best_C=expo_out.best_values['C']
        
        
        
        pred = time_cst_model(x_data,best_A,best_tau,best_C)
        squared_error = np.square((y_data - pred))
        sum_squared_error = np.sum(squared_error)
        RMSE_expo = np.sqrt(sum_squared_error / y_data.size)
        fit_table['Data']='Original_Data'
        simulation_table=pd.DataFrame(np.column_stack((x_data,pred)),columns=["Time_s","Membrane_potential_mV"])
        simulation_table['Data']='Fit_Expo_Data'
        fit_table=pd.concat([fit_table, simulation_table], axis=0)
        if do_plot:
            
            my_plot=p9.ggplot(fit_table,p9.aes(x="Time_s",y="Membrane_potential_mV",color='Data'))+p9.geom_line()
            print(my_plot)
        
        return best_A,best_tau,RMSE_expo
    
    except (ValueError):
        best_A=np.nan
        best_tau=np.nan
        RMSE_expo=np.nan
        return best_A,best_tau,RMSE_expo
    
def time_cst_model(x,A,tau,C):

    y=A*np.exp(-(x)/tau)+C
    return y
    
    
def create_cell_sweep_QC_table_new_version(cell_sweep_info_table):
    '''
    Apply a series of Quality Criteria to each sweep, and indicates which sweep pass the QC

    Parameters
    ----------
    cell_sweep_info_table : pd.DataFrame
        DataFrame containing the information about the different traces for each sweep (one row per sweep).

    Returns
    -------
    sweep_QC_table : pd.DataFrame
        DataFrame specifying for each Sweep wether it has passed Quality Criteria. 
        The sweep for which 'Passed_QC' id False will not be used for I/O or Adaptation fit

    '''


    sweep_QC_table = pd.DataFrame(columns=["Sweep", "Passed_QC"])
    QC_list=['Passed_QC']
    sweep_id_list = cell_sweep_info_table.loc[:, 'Sweep']
    

    
    
    for sweep in sweep_id_list:

        
        sweep_QC = True


        sweep_line = pd.DataFrame([str(sweep),sweep_QC]).T
        
        sweep_line.columns=["Sweep","Passed_QC"]
        sweep_QC_table=pd.concat([sweep_QC_table,sweep_line],ignore_index=True)


    for line in sweep_QC_table.index:
        sweep_QC_table.loc[line,'Passed_QC']=sweep_QC_table.loc[line,QC_list].product()
    
    sweep_QC_table.index = sweep_QC_table['Sweep']
    sweep_QC_table.index = sweep_QC_table.index.astype(str)
    sweep_QC_table.index.name = 'Index'

    return sweep_QC_table

# LG Following renaming of this table to cell_TVC_table in Analysis_pipeline.py, is it correct
# to do that here?
# import warnings
# cell_TVC_table = None
"""
cell_TVC_table stores the Time–Voltage–Current (TVC) sweep tables
for the currently loaded cell.

Expected columns of sweep tables (CHECK):
- Time_s
- Membrane_potential_mV
- Injected_current_pA
- (optional derived signals)

Lifecycle:
- Initialized as None
- Set once a cell is loaded
- Read by analysis and plotting functions
"""
def get_cell_TVC_table(
        module, full_path_to_python_script=None,
        db_function_name=None, db_original_file_directory=None,
        cell_id=None, cell_sweep_list=[], db_cell_sweep_file='',
        stimulus_time_provided=False, db_stimulus_duration=0):
    '''
    FIX ----------
    module : TYPE
    full_path_to_python_script : path to python script containing db_function_name.
    db_function_name
    db_original_file_directory : Path to original cell files.
    cell_id
    cell_sweep_list : List of sweep ids, default is [].
    db_cell_sweep_file : Cell sweep csv table, default is ''.
    stimulus_time_provided : Default is False.
    db_stimulus_duration : Stimulus duration.

    Returns cell_TVC_table
    '''
    """
    Organize the access to raw traces for a given cel, given :
    module, #name of python script
                  full_path_to_python_script, #path to python script
                  current_db["db_function_name"], #Database name

    """
        # cell_TVC_table, cell_stim_time_table = sw_an.get_TVC_table(args_list)
    # module,full_path_to_python_script,db_function_name,db_original_file_directory,cell_id,sweep_list,db_cell_sweep_file,stimulus_time_provided, stimulus_duration  = arg_list
    with warnings.catch_warnings(record=True) as warning_TVC:
        cell_TVC_table = pd.DataFrame(columns=['Sweep','TVC'])
        cell_stim_time_table = pd.DataFrame(columns=['Sweep','Stim_start_s', 'Stim_end_s'])
    cell_TVC_table_list = []
    Stim_time_list = []
    spec=importlib.util.spec_from_file_location(module,full_path_to_python_script)
    DB_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DB_module)
    DB_get_trace_function = getattr(DB_module,db_function_name)
    if stimulus_time_provided == True:
        time_trace_list, voltage_trace_list, current_trace_list,stimulus_start_list,stimulus_end_list = DB_get_trace_function(db_original_file_directory,
                                                  db_cell_sweep_file)
    else :
         time_trace_list, voltage_trace_list, current_trace_list = DB_get_trace_function(db_original_file_directory,
                                                                                         cell_id,
                                                                                         cell_sweep_list,
                                                                                         db_cell_sweep_file)
    for i in range(len(time_trace_list)):
        sweep_TVC = ordifunc.create_TVC(time_trace=time_trace_list[i],
                                              voltage_trace=voltage_trace_list[i],
                                              current_trace= current_trace_list[i])
        sweep_TVC_line = pd.DataFrame([str(cell_sweep_list[i]), sweep_TVC]).T
        sweep_TVC_line.columns = ['Sweep', "TVC"]
        cell_TVC_table_list.append(sweep_TVC_line)
        if stimulus_time_provided:
            stimulus_start=stimulus_start_list[i]
            stimulus_end=stimulus_end_list[i]
        else:
            stimulus_start,stimulus_end = ordifunc.estimate_trace_stim_limits(
                sweep_TVC, db_stimulus_duration, do_plot=False)
        stim_time_line = pd.DataFrame([str(cell_sweep_list[i]), stimulus_start, stimulus_end]).T
        stim_time_line.columns = ['Sweep','Stim_start_s', 'Stim_end_s']
        Stim_time_list.append(stim_time_line)

    cell_TVC_table = pd.concat(cell_TVC_table_list, ignore_index = True)
    cell_stim_time_table = pd.concat(Stim_time_list, ignore_index = True)
    # return cell_TVC_table, cell_stim_time_table
    cell_TVC_table.index = cell_TVC_table.loc[:,"Sweep"]
    cell_TVC_table.index = cell_TVC_table.index.astype(str)
    
    cell_stim_time_table.index = cell_stim_time_table.loc[:,"Sweep"]
    cell_stim_time_table.index = cell_stim_time_table.index.astype(str)
    cell_stim_time_table=cell_stim_time_table.astype({'Stim_start_s':float, 'Stim_end_s':float})

    return cell_TVC_table, cell_stim_time_table, warning_TVC

    
def get_max_frequency_parallel(arg_list):

    cell_id, config_line_db = arg_list
    try:
        cell_dict = ordifunc.read_cell_file_h5(str(cell_id),config_line_db,selection=['All'])
        Full_SF_table = cell_dict['Full_SF_table']
        cell_sweep_info_table = cell_dict['Sweep_info_table']
        sweep_QC_table = cell_dict['Sweep_QC_table']
        stim_freq_table = fir_an.get_stim_freq_table(
            Full_SF_table.copy(), cell_sweep_info_table.copy(),sweep_QC_table.copy(), .5,'Time_based')
        if stim_freq_table.shape[0]==0:
            max_frequency = np.nan
            
        else:
            # 1. Sort the dataframe by Stim_amp_pA in increasing order
            stim_freq_table = stim_freq_table.sort_values(by="Stim_amp_pA").reset_index(drop=True)
            
            # 2. Compute the frequency steps
            stim_freq_table['Frequency_step'] = stim_freq_table['Frequency_Hz'].diff()
        
            # 3. Sort the frequency steps in descending order to get the top two
            stim_freq_table_steps = stim_freq_table.sort_values(by='Frequency_step', ascending=False)
        
            # 4. Find the maximum and second highest frequency steps and the corresponding stimuli
            max_frequency_step = stim_freq_table_steps.iloc[0]['Frequency_step']
            stim_for_max_step = stim_freq_table_steps.iloc[0]['Stim_amp_pA']
        
            # 5. Find the maximum and second highest frequency steps and the corresponding stimuli
            second_max_frequency_step = stim_freq_table_steps.iloc[1]['Frequency_step']
            stim_second_max_frequency_step = stim_freq_table_steps.iloc[1]['Stim_amp_pA']
        
            # 6. Find the stimulus corresponding to the maximum observed frequency
            max_frequency = stim_freq_table['Frequency_Hz'].max()
            stim_for_max_frequency = stim_freq_table.loc[stim_freq_table['Frequency_Hz'].idxmax(), 'Stim_amp_pA']
    
            
            max_freq_tep_ratio = max_frequency_step/second_max_frequency_step
            
        cell_df = pd.DataFrame([str(cell_id), max_frequency, stim_for_max_frequency, max_frequency_step, stim_for_max_step, second_max_frequency_step,stim_second_max_frequency_step, max_freq_tep_ratio ]).T
        cell_df.columns = ['Cell_id','Maximum_Frequency_Hz', "Maximum_Frequency_Stimulus_pA", "Maximum_Frequency_step_Hz", "Stimulus_for_Maximum_freq_Step_pA", "Second_Maximum_Frequency_Step_Hz","Stimulus_for_Second_Maximum_freq_Step_pA", "Maximum_frequency_Step_ratio"]
        return cell_df
    except RuntimeError:
        cell_df = pd.DataFrame([str(cell_id), np.nan, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan]).T
        cell_df.columns = ['Cell_id','Maximum_Frequency_Hz', "Maximum_Frequency_Stimulus_pA", "Maximum_Frequency_step_Hz", "Stimulus_for_Maximum_freq_Step_pA", "Second_Maximum_Frequency_Step_Hz","Stimulus_for_Second_Maximum_freq_Step_pA", "Maximum_frequency_Step_ratio"]
        return cell_df
    

