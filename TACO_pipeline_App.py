#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:19:51 2023

@author: julienballbe
"""

import pandas as pd
import numpy as np
import os
import importlib
import json
import ast
from shiny import App, Inputs, Outputs, Session, render, ui,reactive
import math
import time
import concurrent
import random
import traceback
import Analysis_pipeline as analysis_pipeline
import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import Ordinary_functions as ordifunc
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
import plotly.express as px
import pickle
from sklearn.metrics import r2_score 
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
from sklearn.preprocessing import minmax_scale 
import TACO_pipeline_plots as TACO_plots



# # Load the external script dynamically
# script_path = "Analysis_pipeline.py"
# spec = importlib.util.spec_from_file_location("Analysis_pipeline", script_path)
# script_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(script_module)


# In fast mode, throttle interval in ms.
FAST_INTERACT_INTERVAL = 60

app_ui = ui.page_fluid(
    ui.h1("Welcome to TACO pipeline"),
    ui.page_navbar(
        
        ui.nav_panel(
            "Create JSON configuration file",
            
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h3("Create JSON configuration file"),
                    ui.output_ui("add_database"),
                    ui.input_action_button("save_json", "Save JSON"),
                    ui.output_ui("save_json_status"),
                    
                    
                ),
                #ui.panel_main(
                    #ui.layout_column_wrap(
                        #2,  # Two columns layout
                        # First column
                       # ui.panel_main(
                            ui.input_text(
                                "json_file_path",
                                "Enter File Path for Saving JSON file (ending in .json):",
                                placeholder="Enter full path to save the JSON file, e.g., /path/to/folder/file.json",
                            ),
                            ui.input_action_button("check_json_file_path", "Check File Path"),
                            ui.br(),
                            ui.output_ui("status_json_file_path"),
                        #),
                        # Second column
                        #ui.panel_main(
                            ui.input_text(
                                "Saving_folder_path",
                                "Enter Saving Directory for Analysis:",
                                placeholder="Enter path to saving folder, e.g., /path/to/folder/",
                            ),
                            ui.input_action_button("check_saving_folder_path", "Check Saving Folder Path"),
                            ui.br(),
                            ui.output_ui("status_saving_folder_path"),
                        #),
                        # Second column
                        #ui.panel_main(
                            ui.input_text(
                                "Path_to_QC_file",
                                "Enter Path to Quality Criteria File:",
                                placeholder="Enter path to QC file, e.g., /path/to/folder/file.py",
                            ),
                            ui.input_action_button("check_QC_file_path", "Check QC file path"),
                            ui.br(),
                            ui.output_ui("status_QC_file"),
                        #),
                        

                    #),
                    ui.hr(),
                    #ui.layout_column_wrap(2,
                        # Third column
                        #ui.panel_main(
                            ui.row(
                                # Row containing the buttons and input field
                                
                                ui.input_text(
                                    "database_name",
                                    "Enter Database Name:",
                                    placeholder="Enter the name of the database, e.g., MyDatabase",
                                ),
                            ),
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "original_file_path",
                                              "Path to Original File:",
                                              placeholder="Enter full path to the original file, e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4, 
                                          ui.input_action_button("Check_original_file_path", "Check original file path"),),
                                ui.column(4,
                                          ui.output_ui("status_original_file_path"),),),
                                # Row containing the 'original_file_path' input and 'Check original file path' button
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "database_python_script_path",
                                              "Path to database python script:",
                                              placeholder="e.g., /path/to/original/script.py",
                                          )),
                                ui.column(4,
                                          ui.input_action_button("check_database_python_script_path", "Check database python script")),
                                ui.column(4,
                                          ui.output_ui("status_database_script_file")),
                                ui.column(4,
                                          ui.output_ui('import_database_script_file'))
                                ),
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "population_class_file",
                                              "Path to database population class csv file:",
                                              placeholder="e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4,
                                          ui.input_action_button("check_population_class_file_path", "Check database population class csv file")),
                                ui.column(4,
                                          ui.output_ui("status_population_class_file"))),
                            
                            
                            ui.row(
                                ui.column(4,
                                          ui.input_text(
                                              "cell_sweep_table_file",
                                              "Path to database cell sweep csv file:",
                                              placeholder="e.g., /path/to/original/file.csv",
                                          ),),
                                ui.column(4,
                                          ui.input_action_button("check_cell_sweep_table_file_path", "Check database cell sweep csv file")),
                                ui.column(4,
                                          ui.output_ui("status_cell_sweep_table_file"))),
                            
                            ui.row(
                                   ui.column(4,
                                             ui.input_checkbox("stimulus_time_provided", "Are stimulus start and end times provided?")),
                                   ui.column(4,
                                             ui.input_numeric("stimulus_duration", "Indicate stimulus duration in s", 0.5))
                           
                                
                                
                                
                                
                            ),
                            
                            ui.input_action_button("Add_Database", "Add Database"),
                            ui.br(),
                            
                            
                        #),
                    #)
                #),
            ),
            ),
        
        ui.nav_panel(
            "Run Analysis",
            ui.input_numeric("n_CPU", "number of CPU cores to use", value=int(os.cpu_count()/2)),
            ui.input_file("json_file_input", "Upload Json configuration File:"),
            ui.input_selectize("select_analysis", "Choose Analysis to perform:", {"All":"All" ,
                                                                               "Metadata":"Metadata",
                                                                               "Sweep analysis": "Sweep analysis",
                                                                               "Spike analysis":"Spike analysis",
                                                                               "Firing analysis":"Firing analysis"
                                                                               }, multiple=True, selected="All"),
            ui.input_select("overwrite_existing_files", "Overwrite existing files?", ['Yes','No']),
            
            ui.input_action_button("run_analysis", "Run Analysis"),
            ui.output_text_verbatim("script_output"),
            ui.input_text(
                "summary_folder_path",
                "Enter Folder Path for Saving Summary tables file :",
                placeholder="Enter full path to save the summary tables e.g., /path/to/folder/",
            ),
            ui.input_action_button("check_saving_summary_folder_path", "Check Folder Path"),
            
            ui.output_ui("summary_folder_path_status"),
            ui.input_action_button("summarize_analysis", "Summarize Analysis"),
            ui.output_text_verbatim("summarize_analysis_output")),
        
        
        
        ui.nav_panel(
            "Cell Visualization",
            
            ui.navset_card_pill(
                # Encapsulate existing tabs into the "Cell Visualization" nav_pill
                ui.nav_panel(
                    "Select Cell",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.input_file("JSON_config_file", 'Choose JSON configuration files'),
                            ui.input_selectize("Cell_file_select", "Select Cell to Analyse", choices=[]),
                            ui.input_action_button('Update_cell_btn', 'Update Cell'),
                        ),
                        #ui.panel_main(
                        ui.layout_column_wrap(
                            ui.card(
                                ui.card_header("Analysis JSON file"),
                                ui.output_data_frame('config_json_table')
                                ),
                            ui.card(
                                ui.card_header("Cell file processing"),
                                ui.output_ui("processing_status_output")
                                )
                            ),
                            
                            
                        #),
                    ),
                ),
                ui.nav_panel(
                    "Cell Information",
                    ui.navset_card_pill(
                        ui.nav_panel(
                            "Cell summary",
                            ui.layout_column_wrap(
                                   
                                   
                                ui.card(
                                        ui.card_header(
                                                        ui.tags.span(
                                                            "Cell metadata",
                                                            style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                        )
                                                    ),
                                        ui.output_ui("Cell_Metadata_table_button_ui"),
                                        ui.output_data_frame("Cell_Metadata_table"),
                                    ),
                                
                                ui.card(
                                        ui.card_header(
                                                        ui.tags.span(
                                                            "Linear properties",
                                                            style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                        )
                                                    ),
                                        ui.output_ui("Cell_linear_prop_table_button_ui"),
                                        ui.output_data_frame("Cell_linear_properties_table"),
                                    ),
                                
                                ui.card(
                                        ui.card_header(
                                                        ui.tags.span(
                                                            "Firing properties",
                                                            style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                        )
                                                    ),
                                        ui.output_ui("Cell_firing_prop_table_button_ui"),
                                        ui.output_data_frame("Cell_Firing_properties_table"),
                                    ),
                                
                                ui.card(
                                        ui.card_header(
                                                        ui.tags.span(
                                                            "Adaptation properties",
                                                            style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                        )
                                                    ),
                                        ui.output_ui("Cell_Adaptation_prop_table_button_ui"),
                                        ui.output_data_frame("Cell_Adaptation_table"),
                                    ),
                                   
                                
                            fill = False),
                        ),
                        
                            
                            
                        ui.nav_panel(
                            "Processing report",
                            ui.output_data_frame('cell_processing_time_table')
                        ),
                    ),
                    
                ),
                ui.nav_panel(
                    "Sweep Analysis",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.h4("Cell_id"),
                            ui.output_text_verbatim('display_cell_id_sweep_analysis'),
                            ui.input_selectize("Sweep_selected", "Select Sweep to Analyse", choices=[]),
                            ui.input_checkbox("BE_correction", "Correct for Bridge Error"),
                            ui.input_checkbox("Superimpose_BE_Correction", "Superimpose BE Correction"),
                            
                        ),

                            ui.navset_card_pill(
                                ui.nav_panel(
                                    "Sweep information",
                                    ui.card(
                                            ui.card_header(
                                                            ui.tags.span(
                                                                "Sweep properties",
                                                                style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                            )
                                                        ),
                                            ui.layout_columns(
                                                ui.output_ui("sweep_info_table_button_ui"),
                                                ),
                                            ui.output_data_frame('Sweep_info_table')
                                        ),
                                    ui.card(
                                            ui.card_header(
                                                            ui.tags.span(
                                                                "Quality Criteria evaluation",
                                                                style="font-weight: bold; font-size: 20px; text-align: center; display: block;"
                                                            )
                                                        ),
                                            ui.layout_columns(
                                                ui.output_ui("sweep_QC_table_button_ui"),
                                                ),
                                            ui.output_data_frame('Sweep_QC_table')
                                        ),
                                    
                                ),
                                
                                ui.nav_panel(
                                    'Spike Analysis',
                                    
                                    
                                    ui.card(ui.card_header("Time-Voltage-Current (TVC) plot with spikes features"),
                                            ui.output_ui("TVC_plot_button_ui"),
                                            output_widget("Sweep_spike_plotly"),
                                            
                                            full_screen = True
                                        ),
                                    ui.card(
                                        ui.card_header("Spikes features"),
                                        ui.layout_columns(
                                            ui.input_action_button("trigger_spike_table_saving", "💾 Save Spike features table")
                                            ),
                                        ui.output_data_frame('Sweep_spike_features_table'),
                                        full_screen = True
                                        ),
                                    
                                    
                                ),
                                ui.nav_panel(
                                    "Spike phase plot",
                                    ui.row(
                                        ui.column(6, output_widget("Spike_superposition_plot")),
                                        ui.column(6, output_widget("Spike_phase_plane_plot")),
                                    ),
                                ),
                                
                                ui.nav_panel(
                                    'Bridge Error analysis',
                                    
                                    
                                    ui.card(ui.card_header("Bridge Error fit"),
                                            #ui.input_action_button("trigger_BE_plot_saving", "📊 Save BE plot"),
                                            ui.output_ui("BE_button_ui"),  # <-- conditional UI here
                                            output_widget("BE_plotly"),
                                    ),
                                    
                                    ui.layout_columns(
                                        ui.card(ui.card_header("Bridge Error fit values"),
                                                ui.input_action_button("trigger_BE_fit_table_saving", "💾 Save BE fit table"),
                                                ui.output_data_frame('BE_fit_values_table')
                                        ),
                                        ui.card(ui.card_header("Bridge Error conditions table"),
                                                ui.input_action_button("trigger_BE_conditions_table_saving", "💾 Save BE conditions table"),
                                                ui.output_data_frame('BE_conditions_table'),
                                        ),
                                                      ),
                                    
                                    #ui.output_text_verbatim("BE_value"),
                                    
                                    
                                ),
                                ui.nav_panel(
                                    'Linear properties analysis',
                                    
                                    ui.row(
                                        ui.card(
                                            ui.card_header("Linear analysis results"),
                                            ui.layout_columns(
                                                ui.output_ui("Linear_analysis_button_ui"),
                                                
                                                ),
                                            ui.layout_columns(
                                                ui.input_action_button("trigger_linear_properties_table_saving", "💾 Save Linear properties table"),
                                                ),
                                            ui.output_data_frame("sweep_linear_properties_table"),
                                            full_screen=True,
                                            )
                                        ),
                                    ui.row(
                                        ui.column(6,
                                            ui.card(
                                                ui.card_header("IR analysis"),
                                                ui.layout_columns(
                                                    ui.input_action_button("trigger_IR_table_saving", "💾 Save Input Resistance conditions table"),
                                                    ui.input_action_button("trigger_IR_plot_saving", "📊 Save Input Resistance fit data")
                                                    ),
                                                ui.output_data_frame("IR_validation_condition_table"),
                                                output_widget("cell_Input_Resitance_analysis"),
                                                full_screen=True,
                                            )
                                        ),
                                
                                        ui.column(6,
                                            ui.card(
                                                ui.card_header("TC analysis"),
                                                ui.layout_columns(
                                                    ui.input_action_button("trigger_TC_analysis_table_saving", "💾 Save TC analysis table"),
                                                    ui.input_action_button("trigger_TC_analysis_plot_saving", "📊 Save TC analysis data")
                                                    ),
                                                ui.output_data_frame("TC_fit_table"),
                                                output_widget("cell_Time_cst_analysis"),
                                                full_screen=True,
                                            )
                                        ),
                                        ),
                                    
                                    
                                ),
                            ),

                    ),
                ),
                ui.nav_panel(
                    "Firing Analysis",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.h4("Cell_id"),
                            ui.output_text_verbatim('display_cell_id_firing_analysis'),
                            ui.input_select(
                            'Firing_response_type',
                            "Select response type",
                            choices=['Time_based', 'Index_based', 'Interval_based']
                            
                        ),
                            ),
                        #ui.panel_main(
                            ui.navset_card_pill(
                                
                                
                                ui.nav_panel("I/O",

                                            # --- Main IO plot at the top ---
                                            ui.card(
                                                ui.card_header("I/O Plot"),
                                                #ui.input_action_button("trigger_IO_plot_saving", "📊 Save IO fit plot data"),
                                                ui.output_ui("I_O_button_ui"),
                                                output_widget("IO_plotly"),
                                                full_screen=True,
                                            ),
                                        
                                            # --- Features in a single row ---
                                            ui.row(
                                        
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Gain"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Gain_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.input_action_button("trigger_gain_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_gain_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Gain_regression_plot"),
                                                        ui.output_data_frame("Gain_regression_table"),

                                                    )
                                                ),
                                        
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Threshold"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Threshold_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            ui.input_action_button("trigger_threshold_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_threshold_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Threshold_regression_plot"),
                                                        ui.output_data_frame("Threshold_regression_table"),

                                                    )
                                                ),
                                        
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Saturation Frequency"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Sat_Freq_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            ui.input_action_button("trigger_sat_freq_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_sat_freq_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Saturation_Frequency_regression_plot"),
                                                        ui.output_data_frame("Saturation_Frequency_regression_table"),

                                                    )
                                                )),
                                            ui.row(
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Saturation Stimulus"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Sat_Stim_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            ui.input_action_button("trigger_sat_stim_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_sat_stim_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Saturation_Stimulus_regression_plot"),
                                                        ui.output_data_frame("Saturation_Stimulus_regression_table"),

                                                    )
                                                ),
                                        
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Response Failure Frequency"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Response_Fail_Freq_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            ui.input_action_button("trigger_Response_fail_freq_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_Response_fail_freq_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Response_Failure_Frequency_regression_plot"),
                                                        ui.output_data_frame("Response_Fail_Frequency_regression_table"),
                                                     
                                                    )
                                                ),
                                        
                                                ui.column(4,
                                                    ui.card(
                                                        ui.card_header("Response Failure Stimulus"),
                                                        ui.layout_column_wrap(
                                                            
                                                            ui.output_ui('Response_Fail_Stim_error_button_ui')
                                                            ),
                                                        ui.layout_column_wrap(
                                                            ui.input_action_button("trigger_Response_fail_stim_regression_plot_saving", "📊 Save plot"),
                                                            ui.input_action_button("trigger_Response_fail_stim_regression_table_saving", "💾 Save table"),
                                                            
                                                            ),
                                                        output_widget("Response_Failure_Stimulus_regression_plot"),
                                                        ui.output_data_frame("Response_Fail_Stimulus_regression_table"),

                                                    )
                                                ),
                                            )
                                            
                                ),
                                
                                
                                ui.nav_panel(
                                    "IO detailed fit",
                                    ui.row(
                                        
                                        ui.input_select('Firing_output_duration_new', 'Select output duration for fitting explanation', choices=[]),
                                    ),
                                    ui.layout_column_wrap(
                                        ui.card(
                                            ui.card_header("Detailed IO fit"),
                                            ui.layout_columns(
                                                ui.output_ui("I_O_detailed_button_ui"),
                                                ),
                                            ui.input_action_button("trigger_IO_detailed_plot_dict_saving", "📊 Save IO fit plot data"),
                                            output_widget("IO_analysis_plotly"),
                                            full_screen=True,
                                        ),
                                        ui.card(
                                            ui.card_header("Stimulus-Frequency table"),
                                            ui.input_action_button("trigger_stim_freq_table_saving", "💾 Save Stimulus-Frequency table"),
                                            ui.output_data_frame("IO_fit_stim_freq_table"),
                                            full_screen=True,
                                        ),

                                        ),
                                    ui.layout_column_wrap(
                                        ui.card(
                                            ui.card_header('IO fit conditions'),
                                            ui.input_action_button("trigger_IO_fit_conditions_table_saving", "💾 Save IO fit conditions table"),
                                            ui.output_data_frame("IO_fit_pruning_obs_table"),
                                            full_screen = True))
                                    
                                    
                                    
                                ),
                                ui.nav_panel(
                                    "Adaptation",
                                    ui.layout_column_wrap(
                                        ui.input_select('Adaptation_feature_to_display', "Select feature ", choices=[]),
                                        
                                        
                                        ),
                                    
                                    ui.layout_column_wrap(ui.card(
                                        ui.card_header("Detailed Adaptation fit"),
                                        ui.layout_column_wrap(
                                            
                                            ui.output_ui('Adaptation_detailed_button_ui')
                                            ),
                                        ui.input_action_button("trigger_adaptation_plot_dict_saving", "📊 Save adaptation plot data"),
                                        output_widget("Adaptation_plotly"),
                                        full_screen=True,
                                    ),
                                    ui.card(
                                        ui.card_header("Adaptation table"),
                                        ui.input_action_button("trigger_adaptation_table_saving", "💾 Save Adaptation table"),
                                        ui.output_data_frame("Adaptation_table"),
                                        full_screen=True,
                                    )
                                    ),
                                    
                                    
                                    
                                ),
                            ),
                        #),
                    ),
                ),
                ui.nav_panel(
                    "In progress",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.input_selectize("Sweep_selected_in_progress", "Select Sweep to Analyse", choices=[]),
                            ui.input_selectize(
                                "x_Spike_feature",
                                "Select spike feature X axis",
                                choices=['Downstroke', 'fAHP', 'Fast_Trough', 'Peak', 'Spike_heigth', 'Spike_width_at_half_heigth', 'Threshold', 'Trough', 'Upstroke']
                            ),
                            ui.input_selectize(
                                "y_Spike_feature",
                                "Select spike feature Y axis",
                                choices=["Peak", 'Downstroke', 'fAHP', 'Fast_Trough', 'Spike_heigth', 'Spike_width_at_half_heigth', 'Threshold', 'Trough', 'Upstroke']
                            ),
                            ui.input_selectize("y_Spike_feature_index", "Which spike index for Y axis", choices=['N', "N-1", "N+1"]),
                            ui.input_checkbox("BE_correction_in_process", "Correct for Bridge Error"),
                            width=2
                        ),
                       # ui.panel_main(
                            ui.navset_card_pill(
                                ui.nav_panel(
                                    "Spike feature correlation",
                                    output_widget("Spike_features_index_correlation"),
                                ),
                            ),
                            width=10
                        #),
                    ),
                ),
            ),
        ),
    ),
    fill=True
)

# LG Need to rewrite clearly
def import_json_config_files(json_config_files, arg_from_ui=True):
    i=0
    for file in json_config_files:
        if isinstance(file, dict):
            file_path = file['datapath']
        else:
            file_path = file

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if i == 0:
            config_df = ordifunc.open_json_config_file(file_path)
            i+=1
            
        else:
            new_table = ordifunc.open_json_config_file(file_path)
            config_df=pd.concat([config_df, new_table],ignore_index = True)
            
    return config_df

def estimate_processing_time(start_time, total_iteration,i):
    process_time = time.time()-start_time
    hours = math.floor(process_time / 3600)
    minutes = math.floor((process_time % 3600) / 60)
    seconds = int(process_time % 60)
    formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    
    estimated_total_time = process_time*total_iteration/i
    estimated_remaining_time = estimated_total_time-process_time
    remaining_hours = math.floor(estimated_remaining_time / 3600)
    remaining_minutes = math.floor((estimated_remaining_time % 3600) / 60)
    remaining_seconds = int(estimated_remaining_time % 60)
    remaining_formatted_time = f"{remaining_hours}:{remaining_minutes:02d}:{remaining_seconds:02d}"
    
    return formatted_time, remaining_formatted_time

# Function to highlight cells
def highlight_late(s):
    
    return ['background-color: green' if s_ == "True" else 'background-color: red' if s_ =="False" else 'background-color: white' for s_ in s]

def highlight_rmse_a(s):

    if s.name == "RMSE/A":
        return ['background-color: green' if s.iloc[0] < 0.1 else 'background-color: red']
    return ['']


def highlight_condition(s):
    return ['background-color: green' if v else 'background-color: red' for v in s]


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


    
def get_firing_analysis(cell_dict,response_type):
    """
    Organize the firing analysis details and results of a cell for a given response type.
    Produces the corresponding plotly plot

    """
    
    
    Sweep_info_table = cell_dict['Sweep_info_table']
    Full_SF_table = cell_dict["Full_SF_table"]
    
    sweep_QC_table = cell_dict["Sweep_QC_table"]
    cell_fit_table = cell_dict["Cell_fit_table"]
    cell_feature_table = cell_dict['Cell_feature_table']

    response_type=response_type.replace(' ','_')
    
    sub_cell_fit_table = cell_fit_table.loc[cell_fit_table['Response_type'] == response_type,:]
    sub_cell_feature_table = cell_feature_table.loc[cell_feature_table['Response_type'] == response_type,:]
    output_response_list = list(sub_cell_fit_table.loc[:,'Output_Duration'])
    
    stim_freq_table_list=[]
    model_table_list = []
    IO_table_list = []
    Saturation_table_list = []

    for current_response_duration in output_response_list:
        
        if response_type == 'Time_based':
            response = str(str(int(current_response_duration*1e3))+'ms')
        else:
            
            response = str(response_type.replace('_based',' ') + str(int(current_response_duration)))
            current_response_duration = int(current_response_duration)
            
        current_stim_freq_table = fir_an.get_stim_freq_table(
            Full_SF_table.copy(), Sweep_info_table.copy(),sweep_QC_table.copy(), current_response_duration,response_type)
        
        
        current_stim_freq_table['Response'] = response
        stim_freq_table_list.append(current_stim_freq_table)
        
        model = sub_cell_fit_table.loc[sub_cell_fit_table['Output_Duration'] == current_response_duration,'I_O_obs'].values[0]
        if model == 'Hill' or model == 'Hill-Sigmoid':
            
            model_table = fir_an.fit_IO_relationship(current_stim_freq_table, "--")[1]
            
            min_stim = np.nanmin(current_stim_freq_table.loc[:,'Stim_amp_pA'])
            max_stim = np.nanmax(current_stim_freq_table.loc[:,'Stim_amp_pA'])
            stim_array = np.arange(min_stim,max_stim,.1)
            
            model_table['Response'] = response
            Gain = sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Gain'].values[0]
            
            plot_list = fir_an.get_IO_features(current_stim_freq_table, response_type, current_response_duration, True,True)
            IO_plot = plot_list["3-IO_fit"]
            Intercept = IO_plot['intercept']
            
            IO_freq = stim_array*Gain+Intercept
            
            IO_table = pd.DataFrame({'Stim_amp_pA' : stim_array,
                                        "Frequency_Hz" : IO_freq})
            IO_table['Response'] = response
            
            
            if not np.isnan(sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]):
                Saturation_stimulus =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Stimulus'].values[0]
                Saturation_freq =sub_cell_feature_table.loc[sub_cell_feature_table['Output_Duration'] == current_response_duration,'Saturation_Frequency'].values[0]
                Saturation_table = pd.DataFrame({'Stim_amp_pA' : [Saturation_stimulus],
                                            "Frequency_Hz" : [Saturation_freq],
                                            'Response' : [response]})
                Saturation_table_list.append(Saturation_table)
            else:
                Saturation_freq = np.nan
                
            IO_table=IO_table.loc[(IO_table['Frequency_Hz']>=0.)&(IO_table['Frequency_Hz']<=np.nanmin([Saturation_freq, np.nanmax(current_stim_freq_table['Frequency_Hz'])])),:]
            
            
            model_table_list.append(model_table)
            IO_table_list.append(IO_table)
    
    
    
    Full_stim_freq_table = pd.concat([*stim_freq_table_list],axis=0,ignore_index=True)
    Full_model_table = pd.concat([*model_table_list],axis=0,ignore_index=True)
    Full_IO_table = pd.concat([*IO_table_list],axis=0,ignore_index=True)
    Full_IO_table['Response'] = Full_IO_table['Response'].astype('category')
    
    
    
    
    if len(Saturation_table_list)!=0:
        Full_Saturation_table = pd.concat([*Saturation_table_list],axis=0,ignore_index=True)
    else:
        Full_Saturation_table=pd.DataFrame()
    
    
    Full_IO_dict = {'Full_stim_freq_table': Full_stim_freq_table, 
                    'Full_model_table' : Full_model_table,
                    'Full_IO_table' : Full_IO_table,
                    'Full_Saturation_table' : Full_Saturation_table}
    
    
    Full_IO_plot_fig = TACO_plots.plot_full_IO_fit(Full_IO_dict)
    

    return Full_IO_plot_fig, Full_IO_dict
    
def get_regression_plot(feature_table, feature ,firing_response_type, unit_dict):
    
    #cell_feature_table  = cell_dict['Cell_feature_table']
    #gain_table = cell_feature_table.loc[cell_feature_table['Response_type'] == input.Firing_response_type(),:]
    
    sub_feature_table = feature_table.dropna(subset=[feature])
    current_unit = unit_dict[feature]
    if sub_feature_table.shape[0]>1:
        slope, intercept = fir_an.linear_fit(np.array(sub_feature_table.loc[:,'Output_Duration']),
                                             np.array(sub_feature_table.loc[:,feature]))
        
        linear_fit_y = slope * np.array(sub_feature_table.loc[:,'Output_Duration']) + intercept
        r2 = r2_score(sub_feature_table.loc[:,feature], linear_fit_y)
        
        table = pd.DataFrame([slope, intercept, r2]).T
        table.columns = ['Slope', 'Intercept', 'R2']
        table = table.T
        table = table.reset_index(drop=False)
        table.columns = [" feature", "Value"]
        table.loc[:,"Value"] = np.round(table.loc[:,"Value"], 2)
        
        if firing_response_type == 'Time_based':
            extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.001)
            x_legend = "Response duration ms"
        else:
            extended_x_data = np.arange(0,np.nanmax(np.array(feature_table.loc[:,'Output_Duration'])),.1)
            if firing_response_type == "Index_based":
                x_legend = "Response duration spike index"
            else:
                x_legend = "Response duration spike interval"
       
        original_data = feature_table.loc[:,['Output_Duration', feature]]
        
        linear_fit_y = slope * extended_x_data + intercept
        linear_fit_table = pd.DataFrame({'Output_Duration' : extended_x_data,
                                         f"{feature}" : linear_fit_y})
        
        regression_plot_dict = {'Regression_feature_table' : original_data,
                                'Linear_fit_table' : linear_fit_table}
        
        fig = TACO_plots.plot_regression_plots(regression_plot_dict, feature, x_legend, current_unit)
        
    
        return fig, table, regression_plot_dict
    
    else:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"Not enough {feature} values to make regression",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="red", size=16)
        )
        table = pd.DataFrame(columns = [feature, 'Not enough values'])
        regression_plot_dict = dict()
        return empty_fig, table, regression_plot_dict
        
def get_sweep_spike_plot_table(cell_dict, current_sweep, BE_correction,superimpose_BE_correction, color_dict):
    
    Full_TVC_table = cell_dict['Full_TVC_table']
    Full_SF_table = cell_dict['Full_SF_table']
    Full_SF_dict_table = cell_dict['Full_SF_dict_table']
    Sweep_info_table = cell_dict["Sweep_info_table"]
    #current_sweep = input.Sweep_selected()
    BE = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Bridge_Error_GOhms'].values[0]
    
    
    TVC_plot_dict = TACO_plots.get_TVC_spike_features_data(Full_TVC_table, 
                                                           Full_SF_dict_table, 
                                                           Sweep_info_table, 
                                                           current_sweep, 
                                                           BE, 
                                                           BE_correction, 
                                                           superimpose_BE_correction, 
                                                           color_dict)
    
    current_TVC_table = TVC_plot_dict['current_TVC_table']
    current_sweep_SF_table = TVC_plot_dict['current_sweep_SF_table']
    
    
    fig = TACO_plots.plot_TVC_spike_features_plotly(TVC_plot_dict, return_plot = True)
    
    return fig, current_sweep_SF_table, current_TVC_table, TVC_plot_dict

def get_current_sweep_BE_analysis(cell_dict, current_sweep):
    

    Full_TVC_table = cell_dict['Full_TVC_table']
    sweep_info_table = cell_dict['Sweep_info_table']
    
    
    if np.isnan(sweep_info_table.loc[sweep_info_table['Sweep'] == current_sweep,"Bridge_Error_GOhms"].values[0]) == False:
        TVC_table = Full_TVC_table.loc[current_sweep,"TVC"]
        
        stim_amp = sweep_info_table.loc[sweep_info_table['Sweep'] == current_sweep,"Stim_SS_pA"].values[0]
        Stim_start_s = sweep_info_table.loc[sweep_info_table['Sweep'] == current_sweep,"Stim_start_s"].values[0]
        Stim_end_s = sweep_info_table.loc[sweep_info_table['Sweep'] == current_sweep,"Stim_end_s"].values[0]
        dict_plot = sw_an.estimate_bridge_error(TVC_table, stim_amp, Stim_start_s, Stim_end_s,do_plot=True)
        BE_val, condition_table = sw_an.estimate_bridge_error(TVC_table, stim_amp, Stim_start_s, Stim_end_s,do_plot=False)
        
        plotly_BE_fig = TACO_plots.plot_BE_TACO_choice(dict_plot, "plotly", do_fit = True)
        
        
        
        actual_transition_time = dict_plot["Transition_time"]


        
        alpha_FT = dict_plot["alpha_FT"]
        T_FT = dict_plot["T_FT"]
        delta_t_ref_first_positive = dict_plot["delta_t_ref_first_positive"]
        T_ref_cell = dict_plot["T_ref_cell"]

        pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
        post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
        
        
        BE_fit_values_dict={"Bridge Error (GΩ)" : np.round(BE_val,4),
                            "Transition time" : np.round(actual_transition_time,4),
                            "⍺ FT" : np.round(alpha_FT, 4), 
                            "T_FT" : np.round(T_FT, 4),
                            "Δ t_ref_first_positive" : np.round(delta_t_ref_first_positive,4),
                            "T_ref_cell" : np.round(T_ref_cell,4),
                            "Pre_T_current" : np.round(pre_T_current,4),
                            "Post_T_current" : np.round(post_T_current, 4),}
                            # "first sign change" : np.round(first_sign_change,4),
                            # "T_start_fit" : np.round(T_start_fit, 4)}
        
        BE_fit_values_table = pd.DataFrame(list(BE_fit_values_dict.items()), 
                                           columns=["Parameter", "Value"])

        
        
        return plotly_BE_fig, BE_val, condition_table, BE_fit_values_table, dict_plot
    

def get_sweep_linear_properties_analysis(cell_dict, current_sweep):
    
    #Get sweep based linear properties and plots dicts
    #Same code as in sw_an.get_sweep_linear_properties
    
        
        
    Full_TVC_table = cell_dict['Full_TVC_table']
    Sweep_info_table = cell_dict['Sweep_info_table']
    Stim_amp_pA = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_amp_pA'].values[0]
    time_cst = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Time_constant_ms'].values[0]
    R_in = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Input_Resistance_GOhms'].values[0]
    
    stim_start = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_start_s'].values[0]
    stim_end = Sweep_info_table.loc[Sweep_info_table['Sweep'] == current_sweep,'Stim_end_s'].values[0]
    # LG get_filtered_TVC_table -> get_sweep_TVC_table
    current_TVC_table= ordifunc.get_sweep_TVC_table(Full_TVC_table,current_sweep,do_filter=True,filter=5.,do_plot=False)

    BE = Sweep_info_table.loc[current_sweep, "Bridge_Error_GOhms"]
    TVC_table = current_TVC_table.copy()
    #rely on BE corrected trace
    TVC_table.loc[:,'Membrane_potential_mV'] = TVC_table.loc[:,'Membrane_potential_mV']-BE*TVC_table.loc[:,'Input_current_pA']
    

    TC_plot_dict = sw_an.fit_membrane_time_cst_new(TVC_table,stim_end+0.002,
                                                      (stim_end+0.060),
                                                      do_plot=True)
    best_A,best_tau,SS_potential,RMSE = sw_an.fit_membrane_time_cst_new(TVC_table,stim_end+0.002,
                                                                                     (stim_end+0.060),
                                                                                     do_plot=False)
    
    if np.isnan(best_tau):
        transient_time = .030 
    else:
        transient_time = 3*best_tau
    
    R_in, resting_potential, holding_potential, SS_potential, R2 = sw_an.estimate_input_resistance_and_resting_potential(TVC_table, 
                                                                              stim_start, 
                                                                              stim_end, 
                                                                              best_tau, 
                                                                              do_plot = False)
    
    R_in_plot_dict = sw_an.estimate_input_resistance_and_resting_potential(TVC_table, 
                                                       stim_start, 
                                                       stim_end, 
                                                       best_tau, 
                                                       do_plot = True)
    
    Do_linear_analysis, Do_linear_analysis_table = sw_an.check_criteria_linear_analysis(TVC_table,  
                                                                                        stim_start, 
                                                                                        stim_end, 
                                                                                        transient_time, 
                                                                                        R2)
    if Do_linear_analysis == False:
    
        
        SS_potential = np.nan
        resting_potential = np.nan
        holding_potential = np.nan

        R_in = np.nan
        time_cst = np.nan
    
    linear_fit_table=pd.DataFrame([best_A,
                                  best_tau,
                                  SS_potential,
                                  RMSE,
                                  RMSE/best_A])
    linear_fit_table.index=['A','tau', 'C','RMSE', "RMSE/A"]
    
    
    properties_table=pd.DataFrame([BE,
                                  Stim_amp_pA,
                                  holding_potential,
                                  resting_potential,
                                  R_in,
                                  time_cst
                                  ])
    
    properties_table.index= ['Bridge Error', "Stimulus amplitude pA",'Holding potential mV', 'Resting Potential mV', 'Input Resistance GOhms', 'Time constant ms']
    
    
    TC_plotly = TACO_plots.plot_fit_membrane_time_cst_plotly(TC_plot_dict, return_plot=True)
    
    
    R_in_plot = TACO_plots.plot_estimate_input_resistance_and_resting_potential_plotly(R_in_plot_dict, 
                                                                          sampling_step=1, 
                                                                          return_plot=True)
    
    return Do_linear_analysis_table, TC_plotly,TC_plot_dict, R_in_plot, R_in_plot_dict,  properties_table, linear_fit_table


def replace_bools_with_emojis(df, true_emoji="✅", false_emoji="❌"):
    """
    Replace all boolean values in a DataFrame with emojis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    true_emoji : str, optional
        Emoji to replace True values. Default is "✅".
    false_emoji : str, optional
        Emoji to replace False values. Default is "❌".

    Returns
    -------
    pd.DataFrame
        A new DataFrame with boolean values replaced by emojis.
    """
    df_replaced = df.copy()
    bool_mask = df_replaced.applymap(lambda x: isinstance(x, (bool, pd.BooleanDtype().type)))
    df_replaced = df_replaced.where(~bool_mask, df_replaced.replace({True: true_emoji, False: false_emoji}))
    return df_replaced
       
def get_adaptation_analysis(cell_dict, Adaptation_feature_to_display):

    
    Adaptation_table = cell_dict['Cell_Adaptation']
    Full_SF_table = cell_dict['Full_SF_table']
    sweep_info_table = cell_dict['Sweep_info_table']
    Sweep_QC_table = cell_dict['Sweep_QC_table']
    
    current_adaptation_measure = Adaptation_table.loc[Adaptation_table['Feature'] == Adaptation_feature_to_display, "Measure" ].values[0]
    

    
    interval_based_feature = fir_an.collect_interval_based_features_test(Full_SF_table, sweep_info_table, Sweep_QC_table, 0.5, Adaptation_feature_to_display, current_adaptation_measure)
            
    plot_dict = fir_an.fit_adaptation_test(interval_based_feature, do_plot=True)
    
    adaptation_plotly = TACO_plots.plot_Adaptation_fit_plot_choice(plot_dict, 
                                                                   plot_type = "plotly")
    
    
    Adaptation_table = Adaptation_table.apply(lambda col: col.round(3) if col.dtype == "float" else col)
    
    return adaptation_plotly, plot_dict, Adaptation_table




def get_detailed_IO_fit(cell_dict, Firing_response_type, response_duration):
    
    
    Full_SF_table = cell_dict['Full_SF_table']
    cell_sweep_info_table = cell_dict['Sweep_info_table']
    sweep_QC_table = cell_dict['Sweep_QC_table']
    
    stim_freq_table = fir_an.get_stim_freq_table(
        Full_SF_table.copy(), 
        cell_sweep_info_table.copy(),
        sweep_QC_table.copy(), 
        float(response_duration),
        str(Firing_response_type))
    
    pruning_obs, do_fit, condition_table = fir_an.data_pruning_I_O(stim_freq_table,cell_sweep_info_table)
    
    # --- Default style ---
    default_style = {
        "fontsize": 14,
        "tick_font_size": 12,
        "line_width": 2,
        "marker_size": 10,
        "passed_color": "#0072B2",
        "failed_color": "#EE6677",
        "trimmed_color": "#00694d",
        "initial_color": "#56B4E9",
        "final_color": "#D55E00",
        "model_color": "#E69F00",
        "gain_color": "#d52b00",
        "linear_color": "#000000",
        "threshold_color": "#000000",
        "saturation_color": "#D55E00",
        "failure_color": "#D55E00"
    }
    
    s = default_style
    
    if do_fit == True:
        IO_plot_dict = fir_an.get_IO_features(stim_freq_table, str(Firing_response_type), float(response_duration), do_plot = True)
        
        
        fig_plotly = TACO_plots.plot_IO_detailed_fit_plot_choice(IO_plot_dict, 
                                                                 plot_type ="plotly",
                                                                 do_fit = True)
        
    else:
        IO_plot_dict = {"Stim_Freq_table" : stim_freq_table}
        fig_plotly = TACO_plots.plot_IO_detailed_fit_plot_choice(None, 
                                                                 stim_freq_table,
                                                                 "plotly",
                                                                 do_fit = False)
        
    
    return fig_plotly, IO_plot_dict, stim_freq_table, pruning_obs, condition_table
    
def error_message_modal(msg: str):
    return ui.modal(
        ui.h4("Error Traceback"),
        ui.pre(msg),                       # preserves traceback formatting
        title="An error occurred",
        easy_close=True,
        footer=ui.modal_button("Close"),
        size="xl",  # ← HERE
    )


def server(input: Inputs, output: Outputs, session: Session):
    

    
    print("Hello World!")
    unit_dict = {"Gain" : "Hz/pA",
                 "Threshold" : "pA",
                 "Saturation_Frequency" :  "Hz",
                 "Saturation_Stimulus" : "pA",
                 "Response_Fail_Frequency" : "Hz",
                 "Response_Fail_Stimulus" : "pA"}
    
    rename_dict = {"Gain" : "Gain (Hz/pA)",
                 "Threshold" : "Threshold (pA)",
                 "Saturation_Frequency" :  "Saturation Frequency (Hz)",
                 "Saturation_Stimulus" : "Saturation Stimulus (pA)",
                 "Response_Fail_Frequency" : "Response Fail Frequency (Hz)",
                 "Response_Fail_Stimulus" : "Response Fail Stimulus (pA)"}
    
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
                  "Slow_Trough": "#96022e",
                  "ADP":'#fc0366'
                  }
    
    saving_dict = reactive.Value({
        "TVC_table": {"trigger": 0, "object": None, "format": "csv"},
        "Spike_feature_table": {"trigger": 0, "object": None, "format": "csv"},
    })
    
    error_dict = reactive.Value({})
    
    saving_parameters_dict = reactive.Value({"saving_path" : None,
                                             "object_to_save" : None,
                                             "saving_format" : None})
    
    cell_upload_error_message = reactive.Value(None)
    cell_id_reactive_var = reactive.Value(None)
    
    
    metadata_table_react_var = reactive.Value(None)
    is_metadata_table_error_react_var = reactive.Value(None)
    metadata_table_error_react_var = reactive.Value(None)
    
    Linear_properties_table_react_var = reactive.Value(None)
    is_linear_prop_table_error_react_var = reactive.Value(None)
    linear_prop_table_error_react_var = reactive.Value(None)
    
    firing_properties_table_react_var = reactive.Value(None)
    is_firing_prop_table_error_react_var = reactive.Value(None)
    firing_prop_table_error_react_var = reactive.Value(None)
    
    Adaptation_properties_table_react_var = reactive.Value(None)
    is_adaptation_prop_table_error_react_var = reactive.Value(None)
    adaptation_prop_table_error_react_var = reactive.Value(None)
    
    current_sweep_TVC_table_react_var = reactive.Value(None)
    sweep_spike_feature_plot_dict_react_var = reactive.Value(None)
    is_TVC_spikes_feature_error_react_var = reactive.Value(None)
    TVC_spikes_feature_error_react_var = reactive.Value(None)
    
    current_sweep_info_table_react_var = reactive.Value(None)
    is_sweep_table_error_react_var = reactive.Value(None)
    Sweep_info_table_error_react_var = reactive.Value(None)
    
    current_sweep_QC_table_react_var = reactive.Value(None)
    is_sweep_QC_table_error_react_var = reactive.Value(None)
    Sweep_QC_table_error_react_var = reactive.Value(None)
    
    
    Full_IO_figure_react_var = reactive.Value(None)
    Full_IO_dict_figure_react_var = reactive.Value(None)
    is_IO_error_react_var = reactive.Value(None)
    IO_error_message = reactive.Value(None)
    
    # ------------------- Gain -------------------
    Gain_regression_plotly_fig_react_var = reactive.Value(None)
    Gain_regression_table_react_var = reactive.Value(None)
    Gain_regression_plot_dict_fig_react_var = reactive.Value(None)
    is_Gain_regression_error_react_var = reactive.Value(None)
    Gain_regression_error_message_react_var = reactive.Value(None)
    
    # ------------------- Threshold -------------------
    Threshold_regression_plotly_fig_react_var = reactive.Value(None)
    Threshold_regression_table_react_var = reactive.Value(None)
    Threshold_regression_plot_dict_fig_react_var = reactive.Value(None)
    is_Threshold_regression_error_react_var = reactive.Value(None)
    Threshold_regression_error_message_react_var = reactive.Value(None)
    
    # ------------------- Saturation Frequency -------------------
    Saturation_Frequency_regression_plotly_fig_react_var = reactive.Value(None)
    Saturation_Frequency_regression_table_react_var = reactive.Value(None)
    Saturation_Frequency_regression_plot_dict_fig_react_var = reactive.Value(None)
    is_Saturation_Frequency_regression_error_react_var = reactive.Value(None)
    Saturation_Frequency_regression_error_message_react_var = reactive.Value(None)
    
    # ------------------- Saturation Stimulus -------------------
    Saturation_Stimulus_regression_plotly_fig_react_var = reactive.Value(None)
    Saturation_Stimulus_regression_table_react_var = reactive.Value(None)
    Saturation_Stimulus_regression_plot_dict_fig_react_var = reactive.Value(None)
    is_Saturation_Stimulus_regression_error_react_var = reactive.Value(None)
    Saturation_Stimulus_regression_error_message_react_var = reactive.Value(None)
    
    # ------------------- Response Failure Frequency -------------------
    Response_Failure_Frequency_regression_plotly_fig_react_var = reactive.Value(None)
    Response_Failure_Frequency_regression_table_react_var = reactive.Value(None)
    Response_Failure_Frequency_regression_plot_dict_react_var = reactive.Value(None)
    is_Response_Failure_Frequency_regression_error_react_var = reactive.Value(None)
    Response_Failure_Frequency_regression_error_message_react_var = reactive.Value(None)
    
    # ------------------- Response Failure Stimulus -------------------
    Response_Failure_Stimulus_regression_plotly_fig_react_var = reactive.Value(None)
    Response_Failure_Stimulus_regression_table_react_var = reactive.Value(None)
    Response_Failure_Stimulus_regression_plot_dict_react_var = reactive.Value(None)
    is_Response_Failure_Stimulus_regression_error_react_var = reactive.Value(None)
    Response_Failure_Stimulus_regression_error_message_react_var = reactive.Value(None)

    
    IO_fit_plotly_react_var = reactive.Value(None)
    IO_fit_plot_dict_react_var = reactive.Value(None)
    is_IO_fit_detailed_error_react_var = reactive.Value(None)
    IO_fit_detailed_error_message_react_var = reactive.Value(None)
    
    stim_freq_table_react_var = reactive.Value(None)
    pruning_obs_react_var = reactive.Value(None)
    IO_condition_table_react_var = reactive.Value(None)
    
    sweep_spike_feature_plot_react_var = reactive.Value(None)
    sweep_spike_feature_table_react_var = reactive.Value(None)
    
    is_BE_error_react_var = reactive.Value(None)
    BE_error_message = reactive.Value(None)
    BE_plot_react_var = reactive.Value(None)
    BE_table_react_var = reactive.Value(None)
    BE_value_react_var = reactive.Value(None)
    BE_fit_values_dict_react_var = reactive.Value(None)
    BE_plot_dict_react_var = reactive.Value(None)
    
    current_sweep_linear_analysis_conditions_table_react_var = reactive.Value(None)
    is_current_sweep_linear_analysis_error_react_var = reactive.Value(None)
    current_sweep_linear_analysis_error_message_react_var = reactive.Value(None)
    current_sweep_time_cst_plot_react_var = reactive.Value(None)
    current_sweep_TC_plot_dict_react_var = reactive.Value(None)
    current_sweep_IR_plot_react_var = reactive.Value(None)
    current_sweep_IR_plot_dict_react_var = reactive.Value(None)
    current_sweep_properties_table_react_vat = reactive.Value(None)
    current_sweep_time_cst_table_react_var = reactive.Value(None)
    
    Adaptation_figure_plotly_react_var = reactive.Value(None)
    Adaptation_figure_plot_dict_react_var = reactive.Value(None)
    Adaptation_table_react_var = reactive.Value(None)
    is_Adaptation_error_react_var = reactive.Value(None)
    Adaptation_error_message_react_var = reactive.Value(None)
    
    database_list = reactive.Value([])
    db_done = reactive.Value([])
    cell_file_correctly_opened = reactive.Value(False)
    variables_updated = reactive.Value(False)

    current_status = reactive.Value({})
    
    def set_status(task_name: str, message: str):
        """Update or add a status message for a specific analysis task."""
        status = current_status.get().copy()
        status[task_name] = message
        current_status.set(status)
        
    def get_error_message(tb, e):
        # Try to find the last frame from *your* script
        user_frame = next(
            (f for f in reversed(tb) if "TACO_pipeline_App.py" in f.filename),
            tb[-1]
        )

        # Compose short custom message
        short_msg = (f"{type(e).__name__} at line {user_frame.lineno} \n"
                     f"in {user_frame.name}: {str(e)}")

        # Optional: include the code line itself
        if user_frame.line:
            short_msg += f"\n→ {user_frame.line.strip()}"
            
        return short_msg


    # Check for the first directory
    @output
    @render.ui
    @reactive.event(input.check_json_file_path)
    def status_json_file_path():
        """
        Determines the status of a JSON file path provided by the user.
    
        Returns:
            - A UI message indicating whether the directory exists,
              whether the file has a `.json` extension, and whether it already exists.
        """
        full_path = input.json_file_path()  # Get the file path input from the user
    
        if not full_path:  # Check if the input is empty or None
            return ui.HTML("<b>No path provided for Path 1.</b>")  # Return an error message
    
        dir_path, file_name = os.path.split(full_path)  # Extract directory and file name
    
        if os.path.isdir(dir_path):  # Check if the directory exists
            if file_name.endswith(".json"):  # Check if the file has a .json extension
                if os.path.isfile(full_path):  # Check if the file already exists
                    return ui.HTML(
                        f"""<span style='color:green;'><b>Directory exists:</b> `{dir_path}`</span><br>
                        <span style='color:red;'><b>File </b> {file_name} already exists </span>"""
                    )
                else:  # If the file does not exist, confirm it will be saved under the given name
                    return ui.HTML(
                        f"""<span style='color:green;'><b>Directory exists:</b> {dir_path}</span><br>
                        <b>File will be saved as:</b> {file_name}"""
                    )
            else:  # If the file does not have a .json extension, return a warning
                return ui.HTML(
                    f"""<span style='color:green;'><b>Directory exists:</b> {dir_path}</span><br>
                    <span style='color:red;'><b>Warning:</b> {file_name} does not have a .json extension.</span>"""
                )
        else:  # If the directory does not exist, return an error message
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {dir_path}</span>"""
            )
        
        
    @output
    @render.ui
    def processing_status_output():
        status_dict = current_status.get()
        if not status_dict:
            return ui.p("No process running yet.")
        messages = [f"{v}" for v in status_dict.values()]
        return ui.HTML("<br>".join(messages))


    @output
    @render.ui
    @reactive.event(input.check_saving_folder_path)
    def status_saving_folder_path():
        """
        Determines the status of a saving folder path provided by the user.
    
        Returns:
            - A UI message indicating whether the directory exists or not.
        """
        full_path = input.Saving_folder_path()  # Get the folder path input from the user
    
        if not full_path:  # Check if the input is empty or None
            return ui.HTML("<b>No path provided for Path 2.</b>")  # Return an error message
    
        if os.path.isdir(full_path):  # Check if the directory exists
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory exists:</b> {full_path}</span><br>"""
            )
        else:  # If the directory does not exist, return an error message
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {full_path}</span>"""
            )
        

    @output
    @render.ui
    @reactive.event(input.check_QC_file_path)
    def status_QC_file():
         """
         Determines the status of a Quality Criteria (QC) file provided by the user.
     
         Returns:
             - A UI message indicating whether the file exists, 
               whether it has a `.py` extension, or whether it is missing.
         """
         full_path = input.Path_to_QC_file()  # Get the QC file path input from the user
     
         if not full_path:  # Check if no path is provided
             return ui.HTML("<b>No path provided for Quality Criteria file.</b>")  # Return an error message
     
         dir_path, file_name = os.path.split(full_path)  # Extract directory and file name
     
         if os.path.isfile(full_path):  # Check if the file exists
             if file_name.endswith(".py"):  # Check if the file has a .py extension
                 return ui.HTML(
                     f"""<span style='color:green;'><b>Quality Criteria file:</b> `{full_path}` exists </span>"""
                 )
             else:  # If the file does not have a .py extension, return a warning
                 return ui.HTML(
                     f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a Python file.</span>"""
                 )
         else:  # If the file does not exist, return an error message
             return ui.HTML(
                 f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
             )


    
    @output
    @render.ui
    @reactive.event(input.Check_original_file_path)
    def status_original_file_path():
        """
        Determines the status of the original file path provided by the user.
    
        Returns:
            - A UI message indicating whether the directory exists or not.
        """
        original_file_path = input.original_file_path()  # Get the original file path from the user
    
        if not original_file_path:  # Check if the input is empty or None
            return ui.HTML("<b>No path provided for original file path.</b>")  # Return an error message
    
        if os.path.isdir(original_file_path):  # Check if the directory exists
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory exists:</b> `{original_file_path}`</span>"""
            )
        else:  # If the directory does not exist, return an error message
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory does not exist:</b> {original_file_path}</span>"""
            )
        
    @output
    @render.ui
    @reactive.event(input.check_database_python_script_path)
    def status_database_script_file():
        """
        Determines the status of the database Python script file provided by the user.
    
        Returns:
            - A UI message indicating whether the file exists or not.
        """
        database_script_file = input.database_python_script_path()  # Get the database script path from the user
    
        if not database_script_file:  # Check if no path is provided
            return ui.HTML("<b>No path provided for database script file.</b>")  # Return an error message
    
        if os.path.isfile(database_script_file):  # Check if the file exists
            return ui.HTML(
                f"""<span style='color:green;'><b>File: </b> `{database_script_file}` exists</span><br>"""
            )
        else:  # If the file does not exist, return an error message
            return ui.HTML(
                f"""<span style='color:red;'><b>File:</b> {database_script_file} does not exist</span>"""
            )
    
    @output
    @render.ui
    @reactive.event(input.check_database_python_script_path)
    def import_database_script_file():
        original_file_path = input.database_python_script_path()
        
        if not original_file_path:
            return ui.HTML("<b>No path provided for original file path.</b>")
    
        if os.path.isfile(original_file_path):
            # When the directory exists, display a success message and a selectInput
            
            with open(original_file_path, "r") as file:
                tree = ast.parse(file.read(), filename=original_file_path)
            
                # Extract function names
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            return ui.input_select(
                "select_database_function", "Choose database's function", 
                choices=functions  # Replace with dynamic options if needed
            )
    
    
    @output
    @render.ui
    @reactive.event(input.check_population_class_file_path)
    def status_population_class_file():
        full_path = input.population_class_file()
        if not full_path:# Check if no path is provided
            return ui.HTML("<b>No path provided for Population class file.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isfile(full_path):
            if file_name.endswith(".csv"): 
                return ui.HTML(
                    f"""<span style='color:green;'><b>Population class file:</b> `{full_path}` exists </span>"""
                )
            else:
                return ui.HTML(
                    f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a csv.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
            )
        
    # Check for the first directory
    @output
    @render.ui
    @reactive.event(input.check_cell_sweep_table_file_path)
    def status_cell_sweep_table_file():
        full_path = input.cell_sweep_table_file()
        if not full_path:
            return ui.HTML("<b>No path provided for Cell Sweep table file.</b>")
        
        dir_path, file_name = os.path.split(full_path)
        if os.path.isfile(full_path):
            if file_name.endswith(".csv"): 
                return ui.HTML(
                    f"""<span style='color:green;'><b>Cell Sweep table file:</b> `{full_path}` exists </span>"""
                )
            else:
                return ui.HTML(
                    f"""<span style='color:red;'><b>Warning:</b> {full_path} is not a csv.</span>
                    """
                )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>File :</b> {full_path} does not exist</span>"""
            )
        
        
    # Add Database button action
    @output
    @render.ui
    @reactive.event(input.Add_Database)
    def add_database():
        
        
        database_name = input.database_name()
        if not database_name : # Check if database's name is provided
            return ui.HTML("<span style='color:red;'><b>Database Name must be provided</b></span>")
    
        original_file_path = input.original_file_path()
        if not original_file_path : #Check if path to original files is provided 
            return ui.HTML("<span style='color:red;'><b>Folder of original files must be provided</b></span>")
        
        if not original_file_path.endswith(os.path.sep):
            original_file_path += os.path.sep
        
        path_to_db_script_folder = input.database_python_script_path()
        if not path_to_db_script_folder : # check if path to database's script is provided
            return ui.HTML("<span style='color:red;'><b>Path to database script must be provided</b></span>")
        
        database_script_path, database_script_file_name = os.path.split(path_to_db_script_folder)
        if not database_script_path.endswith(os.path.sep):
            database_script_path += os.path.sep
        database_function = input.select_database_function()
        if not database_function : #Check if database's function name is provided
            return ui.HTML("<span style='color:red;'><b>Function name for database must be provided</b></span>")
        
        
        database_population_class_file = input.population_class_file()
        if not database_population_class_file : #Check is database's population class table is provided
            return ui.HTML("<span style='color:red;'><b>Population class file must be provided</b></span>")
        
        database_cell_sweep_table_file = input.cell_sweep_table_file()
        if not database_cell_sweep_table_file : #Check if database's cell-Sweep table is provided
            return ui.HTML("<span style='color:red;'><b>Cell Sweep table must be provided</b></span>")
        
        stimulus_provided = input.stimulus_time_provided()
        stimulus_duration = input.stimulus_duration()
        


        # Store in the reactive dictionary
        current_list = database_list()
        current_db_done = db_done()
        for i, d in enumerate(current_list):
            #if a Database's information has already been provided, then delete the previous information
            if d.get("database_name") == database_name:
                del current_list[i]
                break  # Exit loop after removing the first match
                
        #Add new database's information 
        db_list = {
            "database_name" : database_name,
            "original_file_directory" :original_file_path,
            "path_to_db_script_folder" : database_script_path,
            "python_file_name" :database_script_file_name,
            'db_function_name':database_function,
            "db_population_class_file" : database_population_class_file,
            'db_cell_sweep_csv_file':database_cell_sweep_table_file,
            "db_stimulus_duration" : float(stimulus_duration),
            "stimulus_time_provided": stimulus_provided}
        current_list.append(db_list)
        current_db_done.append(database_name)
        
        
        database_list.set(current_list)
        db_done.set(current_db_done)

        # Reset inputs manually by setting them to empty strings
        ui.update_text("database_name", value="")
        ui.update_text("original_file_path", value="")
        ui.update_text('database_python_script_path', value = "")

        ui.update_text("population_class_file", value="")
        ui.update_text("cell_sweep_table_file", value="")
        ui.update_numeric("stimulus_duration", value=0.5)

        # Display the updated dictionary
        return ui.HTML(f"<b>Updated Database Dictionary:</b><br><pre>{json.dumps(current_list, indent=4)}</pre>")
    
        
    # Save JSON file if inputs are valid
    @output
    @render.ui
    @reactive.event(input.save_json)
    def save_json_status():
        
        DB_parameters = database_list()
        json_file_path = input.json_file_path()
        
        if not json_file_path :  #Check that the JSON config file is provided
            return ui.HTML("<span style='color:red;'><b>Saving file for JSON config file must be provided</b></span>")
        
        
        Saving_folder = input.Saving_folder_path()
        
        if not Saving_folder : #Check that the saving folder is indicated
            return ui.HTML("<span style='color:red;'><b>Saving folder for analysis must be provided</b></span>")
        
        Path_to_QC_file = input.Path_to_QC_file()
        if not Path_to_QC_file : # Check that the script containing quality criterias is provided
            return ui.HTML("<span style='color:red;'><b>Quality Criteria Python Script must be provided</b></span>")
        
        if not Saving_folder.endswith(os.path.sep):
            Saving_folder += os.path.sep
        
        configuration_file = {
            
            'path_to_saving_file' :Saving_folder,
            'path_to_QC_file' : Path_to_QC_file,
            'DB_parameters':DB_parameters
            }
        
        # Serializing json
        configuration_file_json = json.dumps(configuration_file, indent=4)
         
        # Writing to config json file
        with open(json_file_path, "w") as outfile:
            outfile.write(configuration_file_json)
        
        return ui.HTML(f"<span style='color:green;'><b>JSON file saved successfully to:</b> {json_file_path}</span>")
        
    
    

    
        
    @output
    @render.text
    @reactive.event(input.run_analysis)
    def script_output():
        """
        This function runs the analysis according to the JSON configuration file, as well as the parameters defined by the users (CPU cores, overwriting of exisiting files,... )
        The function displys the progression of the analysis in the GUI
        
        The function returns the list of cell id for which the analysis failed
        

        """
        # Get inputs
        problem_cell = []

        if not input.json_file_input():
            return "No file uploaded."
        else:

            # LG Change e.g. config_json_file -> config_json_file_df
            # json_config_file=[{'name': 'config_json_file_test_lg.json', 'size': 909,
            #                    'type': 'application/json',
            #                    'datapath': '/tmp/fileupload-mz7oynv3/tmpn421445y/0.json'}]

            config_files = input.json_file_input()
            config_df = import_json_config_files(config_files)
            
            #GEt nb of CPU cores to use
            nb_of_workers_to_use = int(input.n_CPU())
            if nb_of_workers_to_use == 0:
                nb_of_workers_to_use = 1
            
            analysis = input.select_analysis()
            overwrite_files_yes_no = input.overwrite_existing_files()
            if overwrite_files_yes_no == 'Yes':
                overwrite_files = True
            elif overwrite_files_yes_no == 'No':
                overwrite_files = False
            
            path_to_saving_file = config_df.loc[0,'path_to_saving_file']
            path_to_QC_file = config_df.loc[0,'path_to_QC_file']
            
            for config_files_idx in config_df.index:
                current_db = config_df.loc[config_files_idx,:].to_dict()
                database_name = current_db["database_name"]
                db_cell_sweep_file = pd.read_csv(current_db['db_cell_sweep_csv_file'],sep =',',encoding = "unicode_escape")
                cell_id_list = db_cell_sweep_file['Cell_id'].unique()
                # LG Why shuffle?
                random.shuffle(cell_id_list)
                start_time = time.time()
                with ui.Progress(min=0, max=len(cell_id_list)) as progress:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_of_workers_to_use) as executor:
                        problem_cell_list = {executor.submit(
                            analysis_pipeline.cell_processing,
                            cell_id,
                            config_files_idx=config_files_idx,
                            config_files=config_files,
                            analysis=analysis,
                            overwrite_files=overwrite_files): cell_id for cell_id in cell_id_list}
                        i=1
                        for f in concurrent.futures.as_completed(problem_cell_list):
                            cell_id_problem = f.result()
                            if cell_id_problem is not None:
                                problem_cell.append(cell_id_problem)

                            formatted_time, remaining_formatted_time = estimate_processing_time(start_time, len(cell_id_list), i)

                            progress.set(i, message=f"Processing {database_name}:", detail=f" {i}/{len(cell_id_list)};  Time spent: {formatted_time}, Estimated remaining time:{remaining_formatted_time}")
                            i+=1

                print(f'Database {database_name} processed')
            
            print('Done')
            return f'Problem occured with cells : {problem_cell}'
    
    @output
    @render.ui
    @reactive.event(input.check_saving_summary_folder_path)
    def summary_folder_path_status():
        """
        This functions checks if the saving path for the summary tables is correctly provided

        
        """
        
        full_path = input.summary_folder_path()
        if not full_path: 
            return ui.HTML("<b>No path provided for Path 1.</b>")
        
        
        if os.path.isdir(full_path):
            return ui.HTML(
                f"""<span style='color:green;'><b>Directory :</b> `{full_path}` exists</span>"""
            )
        else:
            return ui.HTML(
                f"""<span style='color:red;'><b>Directory :</b> `{full_path}` Doesn't exist</span>"""
            )
            
    
    @output
    @render.text
    @reactive.event(input.summarize_analysis)
    def summarize_analysis_output():
        """
        This function summarizes the analysis performed for the cells indicated in the JSON configuration file
        One summary tbale is saved for each output duration for each response type (both features and fit parameters), as well as for adaptation, and linear values
        A general Full population class table is also created
        

        """
        
        if not input.json_file_input():
            return "No file uploaded."
        else:
            json_config_files = input.json_file_input()

            # LG Change e.g. config_json_file -> config_json_file_df
            # json_config_file is a list of dicts?
            config_json_file = import_json_config_files(json_config_files)
            
        saving_path = input.summary_folder_path()
        if not saving_path.endswith(os.path.sep):
            saving_path += os.path.sep

        # Initialize the different dataframes with unit lines and column names.
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
        
        #Create the Full population class table
        Full_population_class_table = pd.DataFrame()
        
        for line in config_json_file.index:
            current_db_population_class_table = pd.read_csv(config_json_file.loc[line,'db_population_class_file'])
            Full_population_class_table= pd.concat([Full_population_class_table,current_db_population_class_table],ignore_index = True)
        Full_population_class_table=Full_population_class_table.astype({'Cell_id':'str'})
        cell_id_list = Full_population_class_table.loc[:,'Cell_id'].unique()
        
        i=1
        start_time = time.time()
        response_duration_dictionary={
            'Time_based':[.005, .010, .025, .050, .100, .250, .500],
            'Index_based':list(np.arange(2,18)),
            'Interval_based':list(np.arange(1,17))}
        #Gather analysis for all cells in Full population class table
        with ui.Progress(min=0, max=len(cell_id_list)) as progress:
            for cell_id in cell_id_list: # For each cell
                try:
                    #Get the database from which the cell comes from, and the corresponding information
                    current_DB = Full_population_class_table.loc[Full_population_class_table['Cell_id']==cell_id,'Database'].values[0]
                    config_line = config_json_file.loc[config_json_file['database_name']==current_DB,:]
                    
                    #Open the cell analysis file
                    cell_dict = ordifunc.read_cell_file_h5(str(cell_id),config_line,['Sweep analysis','Firing analysis','Processing_report', "Sweep QC"])
                    sweep_info_table = cell_dict['Sweep_info_table']
                    cell_fit_table = cell_dict['Cell_fit_table']
                    cell_feature_table = cell_dict['Cell_feature_table']
                    Processing_df = cell_dict['Processing_table']
                    cell_adaptation_table = cell_dict['Cell_Adaptation']
                    sweep_QC_table = cell_dict['Sweep_QC_table']
                    sweep_info_QC_table = pd.merge(sweep_info_table, sweep_QC_table.loc[:,['Passed_QC', "Sweep"]], on = "Sweep")
                    
                    #Keep only sweeps that passed the QC analysis
                    sub_sweep_info_QC_table = sweep_info_QC_table.loc[sweep_info_QC_table['Passed_QC'] == True,:]
                    sub_sweep_info_QC_table['Protocol_id'] =  sub_sweep_info_QC_table['Protocol_id'].astype(str)
                    
                    # sub_sweep_info_QC_table_SS_potential = sub_sweep_info_QC_table.dropna(subset=['SS_potential_mV'])
                    # SS_potential_mV_list = list(sub_sweep_info_QC_table_SS_potential.loc[:,'SS_potential_mV'])
                    # Stim_amp_pA_list = list(sub_sweep_info_QC_table_SS_potential.loc[:,'Stim_SS_pA'])
                    

                    IR_mean_SD = ordifunc.compute_cell_input_resistance(cell_dict)
                    Cell_IR = IR_mean_SD[0]
                    IR_SD = IR_mean_SD[1]
                    
                    if Cell_IR <=0:
                        Cell_IR = np.nan
                        IR_SD = np.nan
                   
                    Time_cst_mean, Time_cst_SD = ordifunc.compute_cell_time_constant(cell_dict)
                    
                    
                    Resting_potential_mean, Resting_potential_SD = ordifunc.compute_cell_resting_potential(cell_dict)
                    
                    
                    cell_linear_values_line = pd.DataFrame([str(cell_id),Cell_IR,IR_SD,Time_cst_mean,Time_cst_SD,Resting_potential_mean, Resting_potential_SD ]).T
                    cell_linear_values_line.columns = ['Cell_id','Input_Resistance_GOhms','Input_Resistance_GOhms_SD','Time_constant_ms','Time_constant_ms_SD', "Resting_potential_mV", 'Resting_potential_mV_SD']
                    cell_linear_values=pd.concat([cell_linear_values,cell_linear_values_line],ignore_index=True)
                    
                    

                    if cell_fit_table.shape[0]==1:

                        for response_type in response_duration_dictionary.keys():
                            output_duration_list=response_duration_dictionary[response_type]
                            for output_duration in output_duration_list:
                                I_O_obs=cell_fit_table.loc[(cell_fit_table['Response_type']==response_type )& (cell_fit_table['Output_Duration']==output_duration),"I_O_obs"]
                                
                                if len(I_O_obs)!=0: #If analysis has been coreclty performed for this cell
                                    Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = np.array(cell_feature_table.loc[(cell_feature_table['Response_type']==response_type )&
                                                                                                                                               (cell_feature_table['Output_Duration']==output_duration),
                                                                                                                                        ["Gain","Threshold","Saturation_Frequency","Saturation_Stimulus","Response_Fail_Frequency", "Response_Fail_Stimulus"]]).tolist()[0]
                                    I_O_obs=I_O_obs.tolist()[0]
                                    
                                    Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = np.array(cell_fit_table.loc[(cell_fit_table['Response_type']==response_type )&
                                                                                                                                               (cell_fit_table['Output_Duration']==output_duration),
                                                                                                                                        ["Hill_Half_cst", "Hill_amplitude", "Hill_coef", "Hill_x0", "Output_Duration", "Response_type", "Sigmoid_k","Sigmoid_x0"]]).tolist()[0]
                                
                                else: # If analysis failed
                                    I_O_obs="No_I_O_Adapt_computed"
                                    empty_array = np.empty(6)
                                    empty_array[:] = np.nan
                                    Gain,Threshold,Saturation_Frequency,Saturation_Stimulus,Response_Failure_Frequency,Response_Failure_Stimulus = empty_array
                                    
                                    empty_array = np.empty(8)
                                    empty_array[:] = np.nan
                                    Hill_Half_cst, Hill_amplitude, Hill_coef, Hill_x0, Output_Duration, Response_type, Sigmoid_k,Sigmoid_x0 = empty_array
                                
                                    
                                if Gain<=0:
                                    Gain=np.nan

                                

                                #Concatenate analysis results (failed or not)                                    
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
                    cell_adaptation_table_copy['Feature'] = cell_adaptation_table_copy.apply(lambda row: ordifunc.append_last_measure(row['Feature'], row['Measure']), axis=1)
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
                        cell_dict = ordifunc.read_cell_file_h5(str(cell_id),config_line,['Processing_report'])
                        
                        Processing_df = cell_dict['Processing_table']
                        for elt in Processing_df.index:
                            if "Error" in Processing_df.loc[elt,'Warnings_encontered'] :
                                new_line = pd.DataFrame([cell_id, Processing_df.loc[elt,'Warnings_encontered']]).T
                                new_line.columns = problem_df.columns
                                problem_df = pd.concat([problem_df, new_line], ignore_index=True)
                                break
                    except:

                        problem_cell.append(cell_id)
                
                formatted_time, remaining_formatted_time = estimate_processing_time(start_time, len(cell_id_list), i)
                progress.set(i, message="Processing ", detail=f" {i}/{len(cell_id_list)}; Time spent: {formatted_time}, Estimated remaining time:{remaining_formatted_time}")
                i+=1
                
            for response_type in response_duration_dictionary.keys():
                output_duration_list=response_duration_dictionary[response_type]
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
            for col in Full_population_class_table.columns:
                if "Unnamed" in col:
                    Full_population_class_table = Full_population_class_table.drop(columns=[col])
            Full_population_class_table.to_csv(f'{saving_path}Full_Population_Class.csv')
            print("Done")
            return  f'Analysis Summary done. Problem with cells {problem_cell}'

################## Cell Visualization    
#%%General

   
    @reactive.Effect
    @reactive.event(input.JSON_config_file)
    def _(): #Update automatically the list of cell id after the config file has been uploaded
        
        config_table = get_config_json_table()
        saving_folder = config_table.loc[0,"path_to_saving_file"]
        file_list = os.listdir(saving_folder)
        file_df = pd.DataFrame(file_list,columns=['Cell_file'])
        file_df['Cell_id']='--'
        for line in range(file_df.shape[0]):
            cell_id = file_df.loc[line,'Cell_file']
            cell_id = cell_id.replace('Cell_','')
            cell_id = cell_id.replace('.h5','')
            file_df.loc[line,'Cell_id']=cell_id
            
        cell_file_dict = dict(zip(list(file_df.loc[:,'Cell_file']),list(file_df.loc[:,'Cell_id'])))

        ui.update_selectize("Cell_file_select" ,choices=cell_file_dict)
        
    @reactive.Effect
    @reactive.event(input.Update_cell_btn)
    def update_variables_lists():
        
        # Retrieve the cell dictionary from the reactive calc
        cells_dict = get_cell_tables()
        
        if cells_dict is None:
            return  # Safety check
    
        # --- Update sweep lists ---
        Sweep_info_table = cells_dict["Sweep_info_table"]
        sweep_list = list(Sweep_info_table["Sweep"])
        ui.update_selectize("Sweep_selected", choices=sweep_list)
        ui.update_selectize("Sweep_selected_in_progress", choices=sweep_list)
        ui.update_selectize("Sweep_selected_Linear_analysis", choices=sweep_list)
    
        # --- Update output duration lists by response type ---
        cell_fit_table = cells_dict["Cell_fit_table"]
    
        for response_type, select_id in [
            ("Time_based", "Firing_output_duration_Time_based"),
            ("Index_based", "Firing_output_duration_Index_based"),
            ("Interval_based", "Firing_output_duration_Interval_based"),
        ]:
            sub_cell_fit_table = cell_fit_table.loc[
                cell_fit_table["Response_type"] == response_type, :
            ]
            output_durations = list(sub_cell_fit_table["Output_Duration"].unique())
            ui.update_select(select_id, choices=output_durations)
    
        # --- Update adaptation features ---
        Adaptation_table = cells_dict["Cell_Adaptation"]
        Sub_Adaptation_table = Adaptation_table.loc[
            (Adaptation_table["Obs"] == "--")
            & (Adaptation_table["Adaptation_Index"].notna()),
            :,
        ]
        validated_features = list(Sub_Adaptation_table["Feature"])
        ui.update_select("Adaptation_feature_to_display", choices=validated_features)
    
        
        
        # Mark that variable lists were successfully updated
        variables_updated.set(True)
        
    @reactive.Calc
    @reactive.event(input.Update_cell_btn)
    def get_cell_tables():  # function to open cell files
    
        cell_id = input.Cell_file_select()
        cell_file_correctly_opened.set(False)
        current_status.set({})
        # --- Step 1: Show "Opening file..." modal ---
        ui.modal_show(
            ui.modal(
                f"Opening cell {cell_id} file... ⏳",
                title="Please wait",
                easy_close=True,
                footer=None,
            )
        )
    
        try:
            # --- Step 2: Main processing logic ---
            config_json_input = input.JSON_config_file()
            if config_json_input and cell_id:
                config_table = get_config_json_table()
                saving_folder = config_table.loc[0, "path_to_saving_file"]
                cell_file_path = str(saving_folder + str(cell_id))
    
                Metadata_table = ordifunc.read_cell_file_h5(
                    cell_file_path, "--", selection=["Metadata"]
                )
                Database = Metadata_table["Database"].values[0]
    
                current_config_line = config_table[
                    config_table["database_name"] == Database
                ]
                cell_id = cell_id.replace("Cell_", "").replace(".h5", "")
    
                
    
                cell_dict = ordifunc.read_cell_file_h5(
                    cell_id, current_config_line, selection=["All"]
                )
                if cell_id == "120426S1P1":
                    t = 2/0
                cell_upload_error_message.set(None)
            else:
                raise ValueError("Missing JSON config or cell file selection.")
    
            # --- Step 3: Success message ---
            time.sleep(1)
            ui.modal_remove()  # Close the "Opening file..." modal
            ui.modal_show(
                ui.modal(
                    f"Cell {cell_id} ready ✅",
                    title="Success",
                    easy_close=True,
                    footer=ui.modal_button("OK", class_="btn-primary"),
                )
            )
            cell_file_correctly_opened.set(True)
            cell_id_reactive_var.set(cell_id)

            return cell_dict
    
        except Exception as e:
            # --- Step 4: On error, close old modal and show error modal ---
            #tb = traceback.extract_tb(e.__traceback__)
            cell_upload_error_message.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            ui.modal_remove()
            ui.modal_show(
            ui.modal(
                    ui.HTML(
                        f"An error occurred while opening cell <b>{cell_id}</b>:<br>"
                        f"<b>{type(e).__name__}</b>: {e}"
                    ),
                    ui.div(
                        ui.input_action_button(
                            "show_cell_upload_error", "⚠️ Show Error", class_="btn-danger mt-3"
                        )
                    ),
                    title="❌ Error",
                    easy_close=True,
                    footer=ui.modal_button("Close", class_="btn-danger"),
                )
            )
            cell_file_correctly_opened.set(False)
    
            # # Optionally, return None or re-raise depending on your logic
            # return None
    @reactive.Effect
    @reactive.event(input.show_cell_upload_error)
    def cell_uploads_error_triggered():
        d = error_dict.get().copy()
        d["cell_uploads_error"] = {
                            "trigger": 1,
                            "error_message": cell_upload_error_message.get()
                        }
        error_dict.set(d)
        
    @reactive.Calc
    def get_config_json_table():
        config_json_input=input.JSON_config_file()
        i=0
        
        if config_json_input:
            for file in config_json_input:
                file_path = file['datapath']
                if i == 0:
                    config_df = ordifunc.open_json_config_file(file_path)
                    i+=1
                    
                else:
                    new_table = ordifunc.open_json_config_file(file_path)
                    config_df=pd.concat([config_df, new_table],ignore_index = True)
            return config_df
        
        
    @output
    @render.data_frame
    def config_json_table():  # Display the config JSON file
        config_table = get_config_json_table()
        
        if config_table is not None and not config_table.empty:
            # Transpose
            config_table_to_display = config_table.T
            
            # Reset index to turn former column names into a column
            config_table_to_display = config_table_to_display.reset_index()
            
            # Rename columns: first column = former column names, rest = database_name
            config_table_to_display.columns = ["Information"] + list(config_table['database_name'])
            
            return config_table_to_display
        else:
            return pd.DataFrame()
        
    # Watch for any triggered save
    @reactive.Effect
    def watch_saving_dict():
        d = saving_dict.get()

        for key, entry in d.items():
            if entry["trigger"] == 1:
                # Reset trigger immediately to prevent loops
                entry["trigger"] = 0
                d[key] = entry
                saving_dict.set(d)
                
                # Save parameters for later
                s = {
                    "object_to_save": entry["object"],
                    "saving_format": entry["format"],
                    "key": key,
                }
                
                saving_parameters_dict.set(s)
                
                
                if entry['format'] == "csv":
                
                    # Show modal for confirmation
                    ui.modal_show(
                        ui.modal(
                            f"Save {key}?",
                            ui.input_text("save_name", "File name:"),
                            ui.input_action_button("confirm_save", "Confirm"),
                            ui.h5('Table preview'),
                            ui.output_data_frame('show_table_to_save'),
                            easy_close=True,
                        )
                    )
                elif entry['format'] == "pkl":
                    
                    # Show modal for confirmation
                    ui.modal_show(
                        ui.modal(
                            f"Save {key}?",
                            ui.input_text("save_name", "File name:"),
                            ui.input_action_button("confirm_save", "Confirm"),
                            # ui.h5("Plot preview"),
                            # #output_widget('show_plot_to_save'),
                            # ui.output_plot("show_plot_to_save"),
                            easy_close=True,
                        )
                    )

                break
            
    # Handle save confirmation
    
    @reactive.Effect
    @reactive.event(input.confirm_save)
    def perform_save():
        s = saving_parameters_dict.get().copy()
        
        
        s["saving_path"] = input.save_name()
        
        saving_parameters_dict.set(s)
        
        filename = s["saving_path"]
        obj = s["object_to_save"]
        
        fmt = s["saving_format"]
    
        if not filename:
            ui.modal_show(
                ui.modal("⚠️ No filename provided.", easy_close=True),
                session=session
            )
            return
    
        try:
            if fmt == "csv" and isinstance(obj, pd.DataFrame):
                obj.to_csv(f"{filename}.csv", index=False)
                saving_result = f"✅ Saved {filename}.csv"
            elif fmt == "pkl" and isinstance(obj, dict):
                with open(f"{filename}.pkl", 'wb') as f:
                    pickle.dump(obj, f)
                saving_result = f"✅ Saved {filename}.pkl"
            else:
                saving_result = "❌ Unsupported format or object type"
        except Exception as e:
            saving_result = f"⚠️ Error while saving: {e}"
    
        # Reset parameters
        saving_parameters_dict.set({
            "saving_path": None,
            "object_to_save": None,
            "saving_format": None,
        })
    
        # Close the input modal
        ui.modal_remove()
        
        # ✅ Display confirmation modal
        ui.modal_show(
            ui.modal(
                saving_result,
                easy_close=True,
            ),
            session=session
        )
        
        
        
    @output
    @render.data_frame
    def show_table_to_save():
        s = saving_parameters_dict.get().copy()
        table_to_save = s["object_to_save"]
        return table_to_save
    
    @output
    @render.plot
    def show_plot_to_save():#This display modifies the plot. The plot needs to be different than the plot actually saved
        s = saving_parameters_dict.get().copy()
        plot_to_show = s["object_to_show"]
        return plot_to_show
    
    
    
    # Watch for any triggered error button
    @reactive.Effect
    def watch_error_message_dict():
        d = error_dict.get()

        for key, entry in d.items():
            if entry["trigger"] == 1:
                # Reset trigger immediately to prevent loops
                entry["trigger"] = 0
                tb = entry['error_message']
                return ui.modal_show(error_message_modal(tb))
                

#%%Cell summary
      
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def preload_cell_summaries_table_analysis():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
    
        task = "Get cell metadata table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()

            if cell_dict is None:
                return  # Safety check
            Metadata_table = cell_dict['Metadata_table']
            for col in Metadata_table:
                if 'Unnamed' in col:
                    Metadata_table = Metadata_table.drop(columns=[col])
            Metadata_table = Metadata_table.T
            Metadata_table = Metadata_table.reset_index(drop=False)
            Metadata_table.columns = [" Metadata", " "]
            
            metadata_table_react_var.set(Metadata_table)
            
            is_metadata_table_error_react_var.set(None)
            metadata_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            metadata_table_react_var.set(pd.DataFrame(columns = [" Metadata", " "]))
            
            
            is_metadata_table_error_react_var.set(True)
            
            metadata_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
        
        task = "Get cell linear properties table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()

            if cell_dict is None:
                return  # Safety check
            #Same code as in ordifunc.create_summary_tables()
            Cell_IR, _ = ordifunc.compute_cell_input_resistance(cell_dict)
            
            if Cell_IR <=0:
                Cell_IR = np.nan

            
            Time_cst_mean, _ = ordifunc.compute_cell_time_constant(cell_dict)
            
            Resting_potential_mean, _ = ordifunc.compute_cell_resting_potential(cell_dict)
            
            
            
            Capacitance = Time_cst_mean/Cell_IR
            
            
            Linear_table = pd.DataFrame({'Input Resistance (GΩ)':[Cell_IR],
                                         'Time constant (ms)' : [Time_cst_mean],
                                         'Resting membrane potential (mV)' : [Resting_potential_mean],
                                         'Capacitance (pF)' : [Capacitance]})
            
            
            Linear_table = Linear_table.T
            Linear_table = Linear_table.reset_index(drop=False)
            Linear_table.columns = [" Linear properties", "Value"]
            Linear_table.loc[:,"Value"] = np.round(Linear_table.loc[:,"Value"], 2)
            Linear_table["Value"] = Linear_table["Value"].replace(np.nan, "--")  # or "—" if you prefer
            
            Linear_properties_table_react_var.set(Linear_table)
            
            is_linear_prop_table_error_react_var.set(None)
            linear_prop_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            linear_prop_table_error_react_var.set(pd.DataFrame(columns = [" Linear properties", "Value"]))
            
            
            is_linear_prop_table_error_react_var.set(True)
            
            linear_prop_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
        task = "Get cell firing properties table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()

            if cell_dict is None:
                return  # Safety check
            
            cell_feature_table = cell_dict['Cell_feature_table']
            cell_fit_table = cell_dict['Cell_fit_table']

            cell_feature_table = pd.merge(cell_feature_table, cell_fit_table.loc[:, ['Response_type', 'Output_Duration', 'I_O_obs', 'I_O_NRMSE']])
            cell_feature_table = cell_feature_table.loc[:,['Response_type',
                                                           'Output_Duration',
                                                           'I_O_obs',
                                                           'I_O_NRMSE',
                                                           "Gain", 
                                                           'Threshold',
                                                           'Saturation_Frequency', 
                                                           'Saturation_Stimulus', 
                                                           'Response_Fail_Frequency',
                                                           "Response_Fail_Stimulus"]]
            unit_dict = {"Gain" : "Gain (Hz/pA)",
                         "Threshold" : "Threshold (pA)",
                         "Saturation_Frequency" :  "Saturation Frequency (Hz)",
                         "Saturation_Stimulus" : "Saturation Stimulus (pA)",
                         "Response_Fail_Frequency" : "Response Fail Frequency (Hz)",
                         "Response_Fail_Stimulus" : "Response Fail Stimulus (pA)"}
            for col in ['I_O_NRMSE', "Gain", 'Threshold','Saturation_Frequency', 'Saturation_Stimulus', 'Response_Fail_Frequency',"Response_Fail_Stimulus"]:
                cell_feature_table.loc[:,col] = np.round(cell_feature_table.loc[:,col], 2)
            
                cell_feature_table.loc[:,col] = cell_feature_table.loc[:,col].replace(np.nan, "--")  # or "—" if you prefer
                                                                                                        
            cell_feature_table = cell_feature_table.rename(columns=unit_dict)
            
            firing_properties_table_react_var.set(cell_feature_table)
            
            is_firing_prop_table_error_react_var.set(None)
            firing_prop_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            firing_properties_table_react_var.set(pd.DataFrame(columns = [" Firing properties", "Value"]))
            
            
            is_firing_prop_table_error_react_var.set(True)
            
            firing_prop_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
            
        task = "Get cell Adaptation properties table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()
            if cell_dict is None:
                return  # Safety check
            cell_Adaptation = cell_dict['Cell_Adaptation']

            cell_Adaptation = cell_Adaptation.loc[:, ['Feature', 'Measure','Obs', 'Adaptation_Index', "C", 'M','RMSE']]
            for col in ['Adaptation_Index', "C", 'M','RMSE']:
                cell_Adaptation.loc[:,col] = np.round(cell_Adaptation.loc[:,col], 2)
            
                cell_Adaptation.loc[:,col] = cell_Adaptation.loc[:,col].replace(np.nan, "--")  # or "—" if you prefer
            
            Adaptation_properties_table_react_var.set(cell_Adaptation)
            
            is_adaptation_prop_table_error_react_var.set(None)
            adaptation_prop_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            Adaptation_properties_table_react_var.set(pd.DataFrame(columns = [" Adaptation properties", "Value"]))
            
            
            is_adaptation_prop_table_error_react_var.set(True)
            
            adaptation_prop_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
        
    




    @output
    @render.data_frame
    def Cell_Metadata_table():
        
        Metadata_table = metadata_table_react_var.get()
        
        return Metadata_table
    
    @render.ui
    def Cell_Metadata_table_button_ui():
        if is_metadata_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_metadata_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_metadata_table_saving", "💾 Save Metadata table")
    
    @reactive.Effect
    @reactive.event(input.show_metadata_table_error)
    def Metadata_info_table_error_triggered():
        d = error_dict.get().copy()
        d["Metadata_info_table_error"] = {
                            "trigger": 1,
                            "error_message": metadata_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_metadata_table_saving)
    def metadata_table_save_triggered():
        d = saving_dict.get().copy()
        d["Metadata_info_table"] = {
                            "trigger": 1,
                            "object": metadata_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def Cell_linear_properties_table():
        
        Linear_table = Linear_properties_table_react_var.get()
        return Linear_table
    
    @render.ui
    def Cell_linear_prop_table_button_ui():
        if is_linear_prop_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_linear_prop_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_linear_prop_table_saving", "💾 Save Linear properties table")
    
    @reactive.Effect
    @reactive.event(input.show_linear_prop_table_error)
    def Linear_prop_table_error_triggered():
        d = error_dict.get().copy()
        d["Linear_prop_table_error"] = {
                            "trigger": 1,
                            "error_message": linear_prop_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_linear_prop_table_saving)
    def Linear_prop_table_save_triggered():
        d = saving_dict.get().copy()
        d["Linear_prop_info_table"] = {
                            "trigger": 1,
                            "object": Linear_properties_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def Cell_Firing_properties_table():
         cell_feature_table= firing_properties_table_react_var.get()
         
         return cell_feature_table
     
        
    @render.ui
    def Cell_firing_prop_table_button_ui():
        if is_firing_prop_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_firing_prop_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_firing_prop_table_saving", "💾 Save firing properties table")
    
    @reactive.Effect
    @reactive.event(input.show_firing_prop_table_error)
    def Firing_prop_table_error_triggered():
        d = error_dict.get().copy()
        d["Firing_prop_table_error"] = {
                            "trigger": 1,
                            "error_message": firing_prop_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_firing_prop_table_saving)
    def firing_prop_table_save_triggered():
        d = saving_dict.get().copy()
        d["Firing_prop_info_table"] = {
                            "trigger": 1,
                            "object": firing_properties_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def Cell_Adaptation_table():
        cell_Adaptation = Adaptation_properties_table_react_var.get()
        return cell_Adaptation
    
    
    @render.ui
    def Cell_Adaptation_prop_table_button_ui():
        if is_adaptation_prop_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_adaptation_prop_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_adaptation_prop_table_saving", "💾 Save firing properties table")
    
    @reactive.Effect
    @reactive.event(input.show_adaptation_prop_table_error)
    def Adaptation_prop_table_error_triggered():
        d = error_dict.get().copy()
        d["Adaptation_prop_table_error"] = {
                            "trigger": 1,
                            "error_message": adaptation_prop_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_adaptation_prop_table_saving)
    def Adaptation_prop_table_save_triggered():
        d = saving_dict.get().copy()
        d["Adaptation_prop_info_table"] = {
                            "trigger": 1,
                            "object": Adaptation_properties_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
        
        
#%% Processing Report

    @output
    @render.data_frame
    @reactive.event(input.Update_cell_btn)
    def cell_processing_time_table():
        if input.JSON_config_file() and input.Cell_file_select() : # and cell_file_correctly_opened():
    
                
            cell_dict = get_cell_tables()
            processing_time_table = cell_dict["Processing_table"] 
            return processing_time_table

#%% Sweep information

    @output
    @render.text
    def display_cell_id_sweep_analysis():

        cell_id = cell_id_reactive_var.get()
        return cell_id

    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def preload_sweep_info_table_analysis():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
    
        task = "Get sweep information table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()
            Sweep_info_table = cell_dict["Sweep_info_table"]
            current_sweep_info_table_react_var.set(Sweep_info_table)
            for col in Sweep_info_table.columns:
                try:
                    if np.issubdtype(Sweep_info_table[col].dtype, np.floating):
                        Sweep_info_table[col] = np.round(Sweep_info_table[col], 2)
                except TypeError:
                    pass  # skip columns that can't be checked

            Sweep_info_table['Bridge_Error_extrapolated'] =  Sweep_info_table['Bridge_Error_extrapolated'].astype(str)
            Sweep_info_table = Sweep_info_table.replace(np.nan, 'NaN')
            Sweep_info_table = Sweep_info_table.loc[:,['Sweep',
                                                       'Protocol_id',
                                                       'Trace_id',
                                                       'Sampling_Rate_Hz',
                                                       'Stim_start_s', 
                                                       'Stim_end_s', 
                                                       'Bridge_Error_GOhms', 
                                                       'Bridge_Error_extrapolated', 
                                                       'Input_Resistance_GOhms',
                                                       'Time_constant_ms',
                                                       'Stim_amp_pA', 
                                                       'Stim_SS_pA', 
                                                       'Holding_current_pA',
                                                       'Resting_potential_mV',
                                                       'Holding_potential_mV',  
                                                        'SS_potential_mV']]
            
            is_sweep_table_error_react_var.set(None)
            Sweep_info_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            current_sweep_info_table_react_var.set(pd.DataFrame(columns = ['Sweep']))
            
            
            is_sweep_table_error_react_var.set(True)
            
            Sweep_info_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
        task = "Get sweep QC table"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()
            Sweep_QC_table = cell_dict['Sweep_QC_table']
            current_sweep_QC_table_react_var.set(Sweep_QC_table)
            
            
            is_sweep_QC_table_error_react_var.set(None)
            Sweep_QC_table_error_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            current_sweep_QC_table_react_var.set(pd.DataFrame(columns = ['Sweep']))
            
            
            is_sweep_QC_table_error_react_var.set(True)
            
            Sweep_QC_table_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task}failed ❌  {short_msg}")
            
    
    @render.ui
    def sweep_info_table_button_ui():
        if is_sweep_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_sweep_info_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_sweep_info_table_saving", "💾 Save Sweep properties table")
    
    @reactive.Effect
    @reactive.event(input.show_sweep_info_table_error)
    def Sweep_info_table_error_triggered():
        d = error_dict.get().copy()
        d["sweep_info_table_error"] = {
                            "trigger": 1,
                            "error_message": Sweep_info_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_sweep_info_table_saving)
    def sweep_info_table_save_triggered():
        d = saving_dict.get().copy()
        d["Sweep_info_table"] = {
                            "trigger": 1,
                            "object": current_sweep_info_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)

    @output
    @render.data_frame
    def Sweep_info_table():

        Sweep_info_table = current_sweep_info_table_react_var.get()
        return Sweep_info_table
        
    
    
    @output
    @render.data_frame
    def Sweep_QC_table():
        Sweep_QC_table = current_sweep_QC_table_react_var.get()

            
        Sweep_QC_table_emoji = replace_bools_with_emojis(Sweep_QC_table)
        

        return Sweep_QC_table_emoji
        
    
    @render.ui
    def sweep_QC_table_button_ui():
        if is_sweep_QC_table_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_sweep_QC_table_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_sweep_QC_table_saving", "💾 Save Sweep QC table")
    
    @reactive.Effect
    @reactive.event(input.show_sweep_info_table_error)
    def Sweep_QC_table_error_triggered():
        d = error_dict.get().copy()
        d["sweep_QC_table_error"] = {
                            "trigger": 1,
                            "error_message": Sweep_QC_table_error_react_var.get()
                        }
        error_dict.set(d)
    
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_sweep_QC_table_saving)
    def sweep_QC_table_save_triggered():
        d = saving_dict.get().copy()
        d["Sweep_QC_table"] = {
                            "trigger": 1,
                            "object": current_sweep_QC_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
#%% Spike analysis

    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Sweep_selected, input.BE_correction, input.Superimpose_BE_Correction)
    def preload_current_sweep_spike_plot_table():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select() or not input.Sweep_selected():
            return
        
        task = "Sweep Spike analysis"
        set_status(task, "⏳ Computing...")
        
        
        try:
            cell_dict = get_cell_tables()
            
            if cell_dict is None:
                return  # Safety check
            
            current_sweep = input.Sweep_selected()
            
            
            current_sweep_spike_plot, current_sweep_spike_table, current_TVC_table, TVC_plot_dict = get_sweep_spike_plot_table(cell_dict, 
                                                                                             current_sweep, 
                                                                                             input.BE_correction(),
                                                                                             input.Superimpose_BE_Correction(),
                                                                                             color_dict)
            
            
            
            sweep_spike_feature_plot_react_var.set(current_sweep_spike_plot)
            sweep_spike_feature_plot_dict_react_var.set(TVC_plot_dict)
            sweep_spike_feature_table_react_var.set(current_sweep_spike_table)
            current_sweep_TVC_table_react_var.set(current_TVC_table)
            is_TVC_spikes_feature_error_react_var.set(None)
            TVC_spikes_feature_error_react_var.set(None)
            
            set_status(task, f"{task} precomputed successfully ✅")
            
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            error_table = pd.DataFrame(columns = ['Feature', 'Spike_index'])
            sweep_spike_feature_plot_react_var.set(error_fig)
            sweep_spike_feature_plot_dict_react_var.set(dict())
            sweep_spike_feature_table_react_var.set(error_table)
            current_sweep_TVC_table_react_var.set(pd.DataFrame(columns = ['Time_s', 'Input_current_pA', "Membrane_potential_mV"]))
            is_TVC_spikes_feature_error_react_var.set(True)
            TVC_spikes_feature_error_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
            
    
    @output
    @render.data_frame
    def Sweep_spike_features_table():
        current_sweep_spike_feature_table = sweep_spike_feature_table_react_var.get()
        return current_sweep_spike_feature_table
    
    @output
    @render_widget
    def Sweep_spike_plotly():
        current_sweep_spike_feature_plot = sweep_spike_feature_plot_react_var.get()
        return current_sweep_spike_feature_plot
    
    # Update dict entries when buttons are pressed
    @reactive.Effect
    @reactive.event(input.trigger_tvc_table_saving)
    def TVC_table_save_triggered():
        d = saving_dict.get().copy()
        d["TVC_table"] = {
                            "trigger": 1,
                            "object": current_sweep_TVC_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)

    @reactive.Effect
    @reactive.event(input.trigger_spike_table_saving)
    def Spike_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Spike_feature_table"] = {
                            "trigger": 1,
                            "object": sweep_spike_feature_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @render.ui
    def TVC_plot_button_ui():
        if is_TVC_spikes_feature_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_TVC_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.layout_columns(
                                ui.input_action_button("trigger_tvc_table_saving", "💾 Save TVC table"),
                                ui.input_action_button("trigger_tvc_plot_saving", "📊 Save TVC plot data")
                            )
    
    @reactive.Effect
    @reactive.event(input.show_TVC_error)
    def TVC_plot_error_triggered():
        d = error_dict.get().copy()
        d["TVC_plot_error"] = {
                            "trigger": 1,
                            "error_message": TVC_spikes_feature_error_react_var.get()
                        }
        error_dict.set(d)
    @reactive.Effect
    @reactive.event(input.trigger_tvc_plot_saving)
    def TVC_plot_saving_triggered():
        d = saving_dict.get().copy()
        d["Spike_feature_table"] = {
                            "trigger": 1,
                            "object": sweep_spike_feature_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)

#%% Spike phase 

    @reactive.Calc
    @reactive.event(input.Update_cell_btn)
    def get_phase_spike_traces():
        if input.Cell_file_select() : # and cell_file_correctly_opened() :
            
            cell_dict = get_cell_tables()
            if cell_dict is None:
                return  # Safety check
            Full_TVC_table = cell_dict['Full_TVC_table']
            Full_SF_table = cell_dict['Full_SF_table']
            spike_trace_table = sp_an.get_cell_spikes_traces(Full_TVC_table, Full_SF_table)

            return spike_trace_table

    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def Spike_phase_plane_plot():
        spike_trace_table = get_phase_spike_traces()
        
        # Filter table based on the selected Sweep
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep'] == input.Sweep_selected(), :]
        discrete_colors = []

        # Determine the list of current spike indices
        if sub_spike_super_position_table.shape[0] == 0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index': 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                                              sub_spike_super_position_table['Spike_index'].astype(int).max() + 1)]
            # LG Fix known Plotly bug in sample_colorscale.
            n = len(current_spike_index_list)
            scaled = np.linspace(0, 0.999999, n)
            discrete_colors = sample_colorscale('plasma_r', scaled)

        # Handle colors for spikes not in the filtered table
        for x in range(spike_trace_table['Spike_index'].astype(int).min(), spike_trace_table['Spike_index'].astype(int).max() + 1):
            if x not in sub_spike_super_position_table['Spike_index'].astype(int).tolist():
                discrete_colors.append('rgb(173, 173, 173)')  # Default gray color for missing spikes

        # Prepare color mapping
        new_discrete_colors = []
        colored_spike_index_list = []
        uncolored_spike_index_list = []
        i = 0
        spike_trace_table['Spike_index'] = spike_trace_table['Spike_index'].astype(str)  # Ensure 'Spike_index' is a string
        color_dict = {}

        # Create full spike index list as strings
        # LG Shouldn't these be ints?
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                                       spike_trace_table['Spike_index'].astype(int).max() + 1)]

        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x] = 'rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)

        # Update color dictionary with discrete colors
        for x in colored_spike_index_list:
            color_dict[x] = discrete_colors[i]
            i += 1

        # Combine colored and uncolored lists
        full_spike_index_list_ordered = uncolored_spike_index_list + colored_spike_index_list

        # Sort data for plotting
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index", 'Time_s'])

        # Create Plotly figure using graph_objects
        fig = go.Figure()

        # Add traces for each Spike_index
        for spike_index in full_spike_index_list_ordered:
            spike_data = spike_trace_table[spike_trace_table['Spike_index'] == spike_index]
            fig.add_trace(
                go.Scatter(
                y=spike_data['Potential_first_time_derivative_mV/s'],
                x=spike_data['Membrane_potential_mV'],
                mode='lines',
                name=f'Spike {spike_index}',
                line=dict(color=color_dict[spike_index]),
                hovertemplate=(
                    'Spike Index: %{text}<br>'
                    'Sweep: %{customdata[0]}<br>'
                    'Membrane deriv: %{y:.2f} s<br>'
                    'Membrane Potential: %{x:.2f} mV'
                ),
                text=[spike_index] * len(spike_data),
                customdata=spike_data[['Sweep']],
                showlegend=True
            )

            )

        # Update figure layout
        fig.update_layout(
            yaxis_title='Membrane potential time derivative (mV/ms)',
            xaxis_title='Membrane Potential (mV)',
            legend_title='Spike Index'
        )
        return fig

    
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.Sweep_selected)
    def Spike_superposition_plot():
        spike_trace_table = get_phase_spike_traces()
        # Filter table based on the selected Sweep
        sub_spike_super_position_table = spike_trace_table.loc[spike_trace_table['Sweep'] == input.Sweep_selected(), :]
        discrete_colors = []

        # Determine the list of current spike indices
        if sub_spike_super_position_table.shape[0] == 0:
            current_spike_index_list = []
        else:
            sub_spike_super_position_table = sub_spike_super_position_table.astype({'Spike_index': 'category'})
            current_spike_index_list = [str(x) for x in range(sub_spike_super_position_table['Spike_index'].astype(int).min(),
                                                              sub_spike_super_position_table['Spike_index'].astype(int).max() + 1)]
            # LG Fix known Plotly bug in sample_colorscale.
            n = len(current_spike_index_list)
            scaled = np.linspace(0, 0.999999, n)
            discrete_colors = sample_colorscale('plasma_r', scaled)

        # Handle colors for spikes not in the filtered table
        for x in range(spike_trace_table['Spike_index'].astype(int).min(), spike_trace_table['Spike_index'].astype(int).max() + 1):
            if x not in sub_spike_super_position_table['Spike_index'].astype(int).tolist():
                discrete_colors.append('rgb(173, 173, 173)')  # Default gray color for missing spikes

        # Prepare color mapping
        new_discrete_colors = []
        colored_spike_index_list = []
        uncolored_spike_index_list = []
        i = 0
        spike_trace_table['Spike_index'] = spike_trace_table['Spike_index'].astype(str)  # Ensure 'Spike_index' is a string
        color_dict = {}

        # Create full spike index list as strings
        full_spike_index_list = [str(x) for x in range(spike_trace_table['Spike_index'].astype(int).min(),
                                                       spike_trace_table['Spike_index'].astype(int).max() + 1)]

        for x in full_spike_index_list:
            if x not in current_spike_index_list:
                new_discrete_colors.append('rgb(173, 173, 173)')
                color_dict[x] = 'rgb(173, 173, 173)'
                uncolored_spike_index_list.append(x)
            else:
                colored_spike_index_list.append(x)

        # Update color dictionary with discrete colors
        for x in colored_spike_index_list:
            color_dict[x] = discrete_colors[i]
            i += 1

        # Combine colored and uncolored lists
        full_spike_index_list_ordered = uncolored_spike_index_list + colored_spike_index_list

        # Sort data for plotting
        spike_trace_table = spike_trace_table.sort_values(by=["Spike_index", 'Time_s'])

        # Create Plotly figure using graph_objects
        fig = go.Figure()

        # Add traces for each Spike_index
        for spike_index in full_spike_index_list_ordered:
            spike_data = spike_trace_table[spike_trace_table['Spike_index'] == spike_index]
            fig.add_trace(
                go.Scatter(
                x=spike_data['Time_s'],
                y=spike_data['Membrane_potential_mV'],
                mode='lines',
                name=f'Spike {spike_index}',
                line=dict(color=color_dict[spike_index]),
                hovertemplate=(
                    'Spike Index: %{text}<br>'
                    'Sweep: %{customdata[0]}<br>'
                    'Time: %{x:.4f} s<br>'
                    'Membrane Potential: %{y:.2f} mV'
                ),
                text=[spike_index] * len(spike_data),
                customdata=spike_data[['Sweep']],
                showlegend=True
            )
            )

        # Update figure layout
        fig.update_layout(
            xaxis_title='Time (s)',
            yaxis_title='Membrane Potential (mV)',
            legend_title='Spike Index'
        )
        return fig

#%% Bridge Error Analysis
    
    
    
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Sweep_selected)
    def preload_BE_analysis():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select() or not input.Sweep_selected():
            return
    
        task = "Bridge Error Analysis"
        set_status(task, f"⏳ Computing {task}...")

        try:
            cell_dict = get_cell_tables()
            current_sweep = input.Sweep_selected()
            current_sweep_BE_plotly, current_sweep_BE_val, current_sweep_condition_table, BE_fit_values_dict, BE_plot_dict = get_current_sweep_BE_analysis(cell_dict, current_sweep)
            
            BE_plot_react_var.set(current_sweep_BE_plotly)
            BE_plot_dict_react_var.set(BE_plot_dict)
            BE_table_react_var.set(current_sweep_condition_table)
            BE_value_react_var.set(current_sweep_BE_val)
            BE_fit_values_dict_react_var.set(BE_fit_values_dict)
            is_BE_error_react_var.set(None)
            BE_error_message.set(None)
            
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            error_table = pd.DataFrame(columns = ['Conditions', 'Met', 'Details'])
            error_value = "Nan"
            BE_plot_react_var.set(error_fig)
            BE_table_react_var.set(error_table)
            BE_value_react_var.set(error_value)
            BE_fit_values_dict_react_var.set(error_table)
            is_BE_error_react_var.set(True)
            BE_plot_dict_react_var.set(dict())
            BE_error_message.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    @render.ui
    def BE_button_ui():
        if is_BE_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_BE_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_BE_plot_saving", "📊 Save BE plot data")
             
    @reactive.Effect
    @reactive.event(input.show_BE_error)
    def BE_plot_error_triggered():
        d = error_dict.get().copy()
        d["BE_error"] = {
                            "trigger": 1,
                            "error_message": BE_error_message.get()
                        }
        error_dict.set(d)
    @output
    @render_widget
    def BE_plotly():
        fig = BE_plot_react_var.get()
        
        return fig
    
    @reactive.Effect
    @reactive.event(input.trigger_BE_plot_saving)
    def BE_plot_save_triggered():
        d = saving_dict.get().copy()
        d["BE_plot"] = {
                            "trigger": 1,
                            "object": BE_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
            
    @output
    @render.data_frame
    def BE_fit_values_table():
        current_BE_fit_values_table = BE_fit_values_dict_react_var.get()
        
        
        return current_BE_fit_values_table
    
    @reactive.Effect
    @reactive.event(input.trigger_BE_fit_table_saving)
    def BE_fit_table_save_triggered():
        d = saving_dict.get().copy()
        d["BE_fit_table"] = {
                            "trigger": 1,
                            "object": BE_fit_values_dict_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def BE_conditions_table():
        current_BE_condition_table = BE_table_react_var.get()
        current_BE_condition_table = current_BE_condition_table.rename(columns={'Met':"Condition Respected"})
        current_BE_condition_table_emoji = replace_bools_with_emojis(current_BE_condition_table)
        
        return current_BE_condition_table_emoji
    
    @reactive.Effect
    @reactive.event(input.trigger_BE_conditions_table_saving)
    def BE_fit_conditions_save_triggered():
        d = saving_dict.get().copy()
        d["BE_conditions_table"] = {
                            "trigger": 1,
                            "object": BE_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.text
    def BE_value():
        current_BE_value = BE_value_react_var.get()
        return f"Estimated Bridge error = {round(current_BE_value, 5)} GOhms"
            

#%% Linear properties analysis

    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Sweep_selected)
    def preload_linear_analysis():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select() or not input.Sweep_selected():
            return
        task = "Linear properties analysis"
        set_status(task, f"⏳ Computing {task}...")
        waiting_plot = go.Figure()
        waiting_plot.add_annotation(
            text="Waiting for plot to render",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color="black", size=16)
        )
        current_sweep_time_cst_plot_react_var.set(waiting_plot)
        current_sweep_IR_plot_react_var.set(waiting_plot)
        
        
        try:
            cell_dict = get_cell_tables()
            current_sweep = input.Sweep_selected()
            Do_linear_analysis_table, TC_plotly,TC_plot_dict, R_in_plotly,R_in_plot_dict, properties_table, TC_table = get_sweep_linear_properties_analysis(cell_dict, current_sweep)
            
            current_sweep_linear_analysis_conditions_table_react_var.set(Do_linear_analysis_table)
            current_sweep_time_cst_plot_react_var.set(TC_plotly)
            current_sweep_TC_plot_dict_react_var.set(TC_plot_dict)
            current_sweep_IR_plot_react_var.set(R_in_plotly)
            current_sweep_IR_plot_dict_react_var.set(R_in_plot_dict)
            current_sweep_properties_table_react_vat.set(properties_table)
            current_sweep_time_cst_table_react_var.set(TC_table)
            is_current_sweep_linear_analysis_error_react_var.set(None)
            current_sweep_linear_analysis_error_message_react_var.set(None)

            set_status(task, f"{task} precomputed successfully ✅")
            
        except Exception as e:
            
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            error_table = pd.DataFrame(columns = ['', ''])
            
            current_sweep_linear_analysis_conditions_table_react_var.set(error_table)
            current_sweep_time_cst_plot_react_var.set(error_fig)
            current_sweep_TC_plot_dict_react_var.set(dict())
            current_sweep_IR_plot_react_var.set(error_fig)
            current_sweep_IR_plot_dict_react_var.set(dict())
            current_sweep_properties_table_react_vat.set(error_table)
            current_sweep_time_cst_table_react_var.set(error_table)
            is_current_sweep_linear_analysis_error_react_var.set(True)
            current_sweep_linear_analysis_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            
            set_status(task, f"{task} failed ❌  {short_msg}")
            
            
    @render.ui
    def Linear_analysis_button_ui():
        if is_current_sweep_linear_analysis_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_current_sweep_Linear_analysis_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
             
    @reactive.Effect
    @reactive.event(input.show_current_sweep_Linear_analysis_error)
    def current_sweep_linear_analysis_error_triggered():
        d = error_dict.get().copy()
        d["current_sweep_linear_analysis_error"] = {
                            "trigger": 1,
                            "error_message": current_sweep_linear_analysis_error_message_react_var.get()
                        }
        error_dict.set(d)
    

    @output
    @render_widget
    def cell_Input_Resitance_analysis():
        
        R_in_plot = current_sweep_IR_plot_react_var.get()
        if R_in_plot is not None:
            return R_in_plot
        else:
            return
    
    @reactive.Effect
    @reactive.event(input.trigger_IR_plot_saving)
    def IR_plot_saving_triggered():
        d = saving_dict.get().copy()
        d["IR_analysis_plot"] = {
                            "trigger": 1,
                            "object": current_sweep_IR_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def IR_validation_condition_table():
        
        Do_linear_analysis_table = current_sweep_linear_analysis_conditions_table_react_var.get()
        if Do_linear_analysis_table is not None:
        
            Do_linear_analysis_table_T = Do_linear_analysis_table.T
            
            
            if "Condition Respected" in Do_linear_analysis_table_T.columns:
                Do_linear_analysis_table_T["Condition Respected"] = Do_linear_analysis_table_T["Condition Respected"].replace({
                    True: "✅",
                    False: "❌"
                })
                
            Do_linear_analysis_table = Do_linear_analysis_table_T.T
            Do_linear_analysis_table.insert(0, "Condition", ["Measure", "Respected"])
            return Do_linear_analysis_table
        else:
            return
        
    
    @reactive.Effect
    @reactive.event(input.trigger_IR_table_saving)
    def IR_table_saving_triggered():
        d = saving_dict.get().copy()
        IR_analysis_table = current_sweep_linear_analysis_conditions_table_react_var.get().T

        d["IR_analysis_table"] = {
                            "trigger": 1,
                            "object": IR_analysis_table.T,
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    @output
    @render_widget
    def cell_Time_cst_analysis():
        
        TC_plotly = current_sweep_time_cst_plot_react_var.get()
        if TC_plotly is not None:
            return TC_plotly
        return
    
    @reactive.Effect
    @reactive.event(input.trigger_TC_analysis_plot_saving)
    def TC_plot_saving_triggered():
        d = saving_dict.get().copy()
        d["TC_analysis_plot"] = {
                            "trigger": 1,
                            "object": current_sweep_TC_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
        


    @output
    @render.data_frame
    def sweep_linear_properties_table():
        properties_table = current_sweep_properties_table_react_vat.get()
        if properties_table is not None:
            properties_table_T = properties_table.T
            for col in properties_table_T.columns:
    
                if np.issubdtype(properties_table_T[col].dtype, np.floating):
                    properties_table_T[col] = np.round(properties_table_T[col], 2)
            return properties_table_T
        else:
            return 
        
    @reactive.Effect
    @reactive.event(input.trigger_linear_properties_table_saving)
    def Linear_properties_saving_triggered():
        d = saving_dict.get().copy()
        properties_table = current_sweep_properties_table_react_vat.get()
        if properties_table is not None:
            properties_table_T = properties_table.T
            for col in properties_table_T.columns:
    
                if np.issubdtype(properties_table_T[col].dtype, np.floating):
                    properties_table_T[col] = np.round(properties_table_T[col], 2)
        d["Linear_properties_table"] = {
                            "trigger": 1,
                            "object": properties_table_T,
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def TC_fit_table():
        TC_table = current_sweep_time_cst_table_react_var.get()
        if TC_table is not None:
            TC_table = TC_table.T
            for col in TC_table.columns:
    
                if np.issubdtype(TC_table[col].dtype, np.floating):
                    TC_table[col] = np.round(TC_table[col], 2)
            #TC_table = TC_table.style.set_table_attributes('class="dataframe shiny-table table w-auto"').apply(highlight_rmse_a)
            return TC_table
        
        else:
            return 
    
    @reactive.Effect
    @reactive.event(input.trigger_TC_analysis_table_saving)
    def TC_saving_triggered():
        d = saving_dict.get().copy()



        d["TC_analysis_table"] = {
                            "trigger": 1,
                            "object": current_sweep_time_cst_table_react_var.get().T,
                            "format": "csv"
                        }
        saving_dict.set(d)
        
#%% I/O

    @output
    @render.text
    def display_cell_id_firing_analysis():

        cell_id = cell_id_reactive_var.get()
        return cell_id
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def update_IO_output_duration_list():
        

        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        
        cell_dict = get_cell_tables()

        Cell_fit_table = cell_dict['Cell_fit_table']
        Firing_type_cell_fit_table = Cell_fit_table.loc[(Cell_fit_table['Response_type']==input.Firing_response_type()) , : ]
        response_duration_list = Firing_type_cell_fit_table.loc[:,"Output_Duration"].unique()
        
        ui.update_select(
            "Firing_output_duration_new",
            label="Select output duration",
            choices=list(response_duration_list),
        )
        
    @render.ui
    def I_O_button_ui():
        if is_IO_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_IO_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return ui.input_action_button("trigger_IO_plot_saving", "📊 Save IO plot data")
    
    @reactive.Effect
    @reactive.event(input.show_IO_error)
    def IO_plot_error_triggered():
        d = error_dict.get().copy()
        d["IO_error"] = {
                            "trigger": 1,
                            "error_message": IO_error_message.get()
                        }
        error_dict.set(d)
    
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def preload_I_O_analysis():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
    
        task = "IO analysis"
        set_status(task, f"⏳ Computing {task}...")
    
        try:
            cell_dict = get_cell_tables()
            IO_figure, IO_dict = get_firing_analysis(cell_dict, input.Firing_response_type())
            Full_IO_figure_react_var.set(IO_figure)
            Full_IO_dict_figure_react_var.set(IO_dict)
            is_IO_error_react_var.set(None)
            IO_error_message.set(None)

            set_status(task, "IO analysis precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            
            
            Full_IO_figure_react_var.set(error_fig)
            Full_IO_dict_figure_react_var.set(dict())
            is_IO_error_react_var.set(True)
            
            IO_error_message.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"IO analysis failed ❌  {short_msg}")
        
    @output
    @render_widget
    def IO_plotly():
        """Display the precomputed IO plot."""
        fig = Full_IO_figure_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_IO_plot_saving)
    def IO_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["IO_full_plot_dict"] = {
                            "trigger": 1,
                            "object": Full_IO_dict_figure_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
        
    # -----------------------------
    # Gain
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_gain():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        task = "Gain regression IO plot"
        
        try:
            set_status(task, f"⏳ Computing {task}...")
            cell_dict = get_cell_tables()

            gain_table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table, plot_dict = get_regression_plot(gain_table, 'Gain', input.Firing_response_type(), unit_dict)
            Gain_regression_plotly_fig_react_var.set(plot)
            Gain_regression_table_react_var.set(table)
            Gain_regression_plot_dict_fig_react_var.set(plot_dict)
            is_Gain_regression_error_react_var.set(None)
            Gain_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Gain_regression_plotly_fig_react_var.set(error_fig)
            Gain_regression_table_react_var.set(pd.DataFrame(columns=['Gain', '']))
            Gain_regression_plot_dict_fig_react_var.set(dict())
            is_Gain_regression_error_react_var.set(True)
            Gain_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    # -----------------------------
    # Threshold
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_threshold():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        task = "Threshold regression"
        
        try:
            set_status(task, f"⏳ Computing {task}...")
            cell_dict = get_cell_tables()
            threshold_table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table, plot_dict = get_regression_plot(threshold_table, 'Threshold', input.Firing_response_type(), unit_dict)
            Threshold_regression_plotly_fig_react_var.set(plot)
            Threshold_regression_table_react_var.set(table)
            Threshold_regression_plot_dict_fig_react_var.set(plot_dict)
            is_Threshold_regression_error_react_var.set(None)
            Threshold_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Threshold_regression_plotly_fig_react_var.set(error_fig)
            Threshold_regression_table_react_var.set(pd.DataFrame(columns=['Threshold', '']))
            Threshold_regression_plot_dict_fig_react_var.set(dict())
            is_Threshold_regression_error_react_var.set(True)
            Threshold_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    # -----------------------------
    # Saturation_Frequency
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_saturation_frequency():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        
        try:
            task = "Saturation_Frequency regression"
            set_status(task, f"⏳ Computing {task}...")
            cell_dict = get_cell_tables()
            table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table_out, plot_dict = get_regression_plot(table, 'Saturation_Frequency', input.Firing_response_type(), unit_dict)
            Saturation_Frequency_regression_plotly_fig_react_var.set(plot)
            Saturation_Frequency_regression_table_react_var.set(table_out)
            Saturation_Frequency_regression_plot_dict_fig_react_var.set(plot_dict)
            is_Saturation_Frequency_regression_error_react_var.set(None)
            Saturation_Frequency_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Saturation_Frequency_regression_plotly_fig_react_var.set(error_fig)
            Saturation_Frequency_regression_table_react_var.set(pd.DataFrame(columns=['Saturation_Frequency', '']))
            Saturation_Frequency_regression_plot_dict_fig_react_var.set(dict())
            is_Saturation_Frequency_regression_error_react_var.set(True)
            Saturation_Frequency_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    # -----------------------------
    # Saturation_Stimulus
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_saturation_stimulus():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        task = "Saturation_Stimulus regression"
        try:
            
            set_status(task, f"⏳ Computing {task}...")

            cell_dict = get_cell_tables()
            table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table_out, plot_dict = get_regression_plot(table, 'Saturation_Stimulus', input.Firing_response_type(), unit_dict)
            Saturation_Stimulus_regression_plotly_fig_react_var.set(plot)
            Saturation_Stimulus_regression_table_react_var.set(table_out)
            Saturation_Stimulus_regression_plot_dict_fig_react_var.set(plot_dict)
            is_Saturation_Stimulus_regression_error_react_var.set(None)
            Saturation_Stimulus_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Saturation_Stimulus_regression_plotly_fig_react_var.set(error_fig)
            Saturation_Stimulus_regression_table_react_var.set(pd.DataFrame(columns=['Saturation_Stimulus', '']))
            Saturation_Stimulus_regression_plot_dict_fig_react_var.set(dict())
            is_Saturation_Stimulus_regression_error_react_var.set(True)
            Saturation_Stimulus_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    # -----------------------------
    # Response_Fail_Frequency
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_response_fail_frequency():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
        task = "Response_Fail_Frequency regression"
        try:
            
            set_status(task, f"⏳ Computing {task}...")

            cell_dict = get_cell_tables()
            table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table_out, plot_dict = get_regression_plot(table, 'Response_Fail_Frequency', input.Firing_response_type(), unit_dict)
            Response_Failure_Frequency_regression_plotly_fig_react_var.set(plot)
            Response_Failure_Frequency_regression_table_react_var.set(table_out)
            Response_Failure_Frequency_regression_plot_dict_react_var.set(plot_dict)
            is_Response_Failure_Frequency_regression_error_react_var.set(None)
            Response_Failure_Frequency_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Response_Failure_Frequency_regression_plotly_fig_react_var.set(error_fig)
            Response_Failure_Frequency_regression_table_react_var.set(pd.DataFrame(columns=['Response_Fail_Frequency', '']))
            Response_Failure_Frequency_regression_plot_dict_react_var.set(dict())
            is_Response_Failure_Frequency_regression_error_react_var.set(True)
            Response_Failure_Frequency_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
    
    
    # -----------------------------
    # Response_Fail_Stimulus
    # -----------------------------
    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_response_type)
    def compute_response_fail_stimulus():
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
            
        task = "Response_Fail_Stimulus regression"
        try:
            
            set_status(task, f"⏳ Computing {task}...")

            cell_dict = get_cell_tables()
            table = cell_dict['Cell_feature_table'].loc[
                cell_dict['Cell_feature_table']['Response_type'] == input.Firing_response_type(), :]
            plot, table_out, plot_dict = get_regression_plot(table, 'Response_Fail_Stimulus', input.Firing_response_type(), unit_dict)
            Response_Failure_Stimulus_regression_plotly_fig_react_var.set(plot)
            Response_Failure_Stimulus_regression_table_react_var.set(table_out)
            Response_Failure_Stimulus_regression_plot_dict_react_var.set(plot_dict)
            is_Response_Failure_Stimulus_regression_error_react_var.set(None)
            Response_Failure_Stimulus_regression_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            short_msg = get_error_message(traceback.extract_tb(e.__traceback__), e)
            error_fig = go.Figure()
            error_fig.add_annotation(text=f"⚠️ {short_msg}", x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color="red", size=16))
            Response_Failure_Stimulus_regression_plotly_fig_react_var.set(error_fig)
            Response_Failure_Stimulus_regression_table_react_var.set(pd.DataFrame(columns=['Response_Fail_Stimulus', '']))
            Response_Failure_Stimulus_regression_plot_dict_react_var.set(dict())
            is_Response_Failure_Stimulus_regression_error_react_var.set(True)
            Response_Failure_Stimulus_regression_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
     
    
    @render.ui
    def Gain_error_button_ui():
        if is_Gain_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Gain_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_Gain_error)
    def Gain_reg_error_triggered():
        d = error_dict.get().copy()
        d["Gain_reg_error"] = {
                            "trigger": 1,
                            "error_message": Gain_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    @output
    @render_widget
    def Gain_regression_plot():
        """Display the precomputed Gain regression  plot."""
        fig = Gain_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_gain_regression_plot_dict_saving)
    def gain_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Gain_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Gain_regression_plot_dict_fig_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Gain_regression_table():
        """Display the precomputed Gain regression  table."""
        table = Gain_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Gain', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_gain_regression_table_saving)
    def gain_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Gain_regression_table"] = {
                            "trigger": 1,
                            "object": Gain_regression_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
        
    
    @render.ui
    def Threshold_error_button_ui():
        if is_Threshold_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Threshold_error", "⚠️ Show Error", class_="btn-danger")
        # else:
        #     # Normal save button
        #     return 
    
    @reactive.Effect
    @reactive.event(input.show_Threshold_error)
    def Threshold_reg_error_triggered():
        d = error_dict.get().copy()
        d["Threshold_reg_error"] = {
                            "trigger": 1,
                            "error_message": Threshold_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    @output
    @render_widget
    def Threshold_regression_plot():
        """Display the precomputed Threshold regression  plot."""
        fig = Threshold_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_threshold_regression_plot_dict_saving)
    def threshold_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Threshold_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Threshold_regression_plot_dict_fig_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Threshold_regression_table():
        """Display the precomputed Threshold regression  table."""
        table = Threshold_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Threshold', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_threshold_regression_table_saving)
    def threshold_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Threshold_regression_table"] = {
                            "trigger": 1,
                            "object": Threshold_regression_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
        
    @render.ui
    def Sat_Freq_error_button_ui():
        if is_Saturation_Frequency_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Sat_Freq_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_Sat_Freq_error)
    def Sat_Freq_reg_error_triggered():
        d = error_dict.get().copy()
        d["Sat_Freq_reg_error"] = {
                            "trigger": 1,
                            "error_message": Saturation_Frequency_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    @output
    @render_widget
    def Saturation_Frequency_regression_plot():
        """Display the precomputed Saturation_Frequency regression  plot."""
        fig = Saturation_Frequency_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_sat_freq_regression_plot_dict_saving)
    def Saturation_frequency_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Sat_freq_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Saturation_Frequency_regression_plot_dict_fig_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Saturation_Frequency_regression_table():
        """Display the precomputed Saturation_Frequency regression  table."""
        table = Saturation_Frequency_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Saturation_Frequency', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_sat_freq_regression_table_saving)
    def Saturation_frequency_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Sat_freq_regression_table"] = {
                            "trigger": 1,
                            "object": Saturation_Frequency_regression_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
        
    @render.ui
    def Sat_Stim_error_button_ui():
        if is_Saturation_Stimulus_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Sat_stim_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_Sat_stim_error)
    def Sat_Stim_reg_error_triggered():
        d = error_dict.get().copy()
        d["Sat_Stim_reg_error"] = {
                            "trigger": 1,
                            "error_message": Saturation_Stimulus_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    @output
    @render_widget
    def Saturation_Stimulus_regression_plot():
        """Display the precomputed Saturation_Stimulus regression  plot."""
        fig = Saturation_Stimulus_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_sat_stim_regression_plot_dict_saving)
    def Saturation_stimulus_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Sat_stim_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Saturation_Stimulus_regression_plot_dict_fig_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Saturation_Stimulus_regression_table():
        """Display the precomputed Saturation_Stimulus regression  table."""
        table = Saturation_Stimulus_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Saturation_Stimulus', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_sat_stim_regression_table_saving)
    def Saturation_stimulus_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Sat_stim_regression_table"] = {
                            "trigger": 1,
                            "object": Saturation_Stimulus_regression_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    @render.ui
    def Response_Fail_Freq_error_button_ui():
        if is_Response_Failure_Frequency_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Resp_Fail_Freq_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_Resp_Fail_Freq_error)
    def Resp_Fail_Freq_reg_error_triggered():
        d = error_dict.get().copy()
        d["Resp_Fail_Freq_reg_error"] = {
                            "trigger": 1,
                            "error_message": Response_Failure_Frequency_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    @output
    @render_widget
    def Response_Failure_Frequency_regression_plot():
        """Display the precomputed Response_Failure_Frequency regression  plot."""
        fig = Response_Failure_Frequency_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_Response_fail_freq_regression_plot_dict_saving)
    def Response_fail_freq_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Resp_fail_freq_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Response_Failure_Frequency_regression_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Response_Failure_Frequency_regression_table():
        """Display the precomputed Response_Failure_Frequency regression  table."""
        table = Response_Failure_Frequency_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Response_Failure_Frequency', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_Response_fail_freq_regression_table_saving)
    def Response_fail_freq_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Resp_fail_freq_regression_table"] = {
                            "trigger": 1,
                            "object": Response_Failure_Frequency_regression_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    
    @render.ui
    def Response_Fail_Stim_error_button_ui():
        if is_Response_Failure_Stimulus_regression_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_Resp_Fail_Stim_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_Resp_Fail_Stim_error)
    def Resp_Fail_Stim_reg_error_triggered():
        d = error_dict.get().copy()
        d["Resp_Fail_Stim_reg_error"] = {
                            "trigger": 1,
                            "error_message": Response_Failure_Stimulus_regression_error_message_react_var.get()
                        }
        error_dict.set(d)
    
    
    @output
    @render_widget
    def Response_Failure_Stimulus_regression_plot():
        """Display the precomputed Response_Failure_Stimulus regression  plot."""
        fig = Response_Failure_Stimulus_regression_plotly_fig_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_Response_fail_stim_regression_plot_dict_saving)
    def Response_fail_stim_regression_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        d["Resp_fail_stim_regression_plot_dict"] = {
                            "trigger": 1,
                            "object": Response_Failure_Stimulus_regression_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Response_Failure_Stimulus_regression_table():
        """Display the precomputed Response_Failure_Stimulus regression  table."""
        table = Response_Failure_Stimulus_regression_table_react_var.get()
        return table if table is not None else pd.DataFrame(['Response_Failure_Stimulus', ''])  # or show an empty figure
    
    @reactive.Effect
    @reactive.event(input.trigger_Response_fail_stim_regression_table_saving)
    def Response_fail_stim_regression_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Resp_fail_stim_regression_table"] = {
                            "trigger": 1,
                            "object": Response_Failure_Stimulus_regression_plotly_fig_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
  
#%% IO detailed fit

    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Firing_output_duration_new, input.Firing_response_type)
    def preload_IO_fit_analysis():
        
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
    
        task = "IO fit analysis"
        set_status(task, f"⏳ Computing {task}...")
        
        try:
            cell_dict = get_cell_tables()

            IO_detailed_plotly, IO_plot_dict, stim_freq_table, pruning_obs, condition_table = get_detailed_IO_fit(cell_dict, input.Firing_response_type(), input.Firing_output_duration_new())
            IO_fit_plotly_react_var.set(IO_detailed_plotly)
            IO_fit_plot_dict_react_var.set(IO_plot_dict)
            
            stim_freq_table_react_var.set(stim_freq_table)
            pruning_obs_react_var.set(pruning_obs)
            IO_condition_table_react_var.set(condition_table)
            
            is_IO_fit_detailed_error_react_var.set(None)
            IO_fit_detailed_error_message_react_var.set(None)
            
            
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            error_table = pd.DataFrame(columns = ['Sweep', 'Stim_amp_pA', 'Passed_QC' "Frequency_Hz"])
            
            IO_fit_plotly_react_var.set(error_fig)
            IO_fit_plot_dict_react_var.set(dict())
            stim_freq_table_react_var.set(error_table)
            error_table = pd.DataFrame(columns = ['Condition', "Obs"])
            pruning_obs_react_var.set(error_table)
            IO_condition_table_react_var.set(error_table)
            is_IO_fit_detailed_error_react_var.set(True)
            IO_fit_detailed_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            
            set_status(task, f"{task} failed ❌  {short_msg}")

    @render.ui
    def I_O_detailed_button_ui():
        if is_IO_fit_detailed_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_IO_detailed_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_IO_detailed_error)
    def IO_plot_detailed_error_triggered():
        d = error_dict.get().copy()
        d["IO_detailed_error"] = {
                            "trigger": 1,
                            "error_message": IO_fit_detailed_error_message_react_var.get()
                        }
        error_dict.set(d)

    
    @output
    @render_widget
    def IO_analysis_plotly():
        current_IO_fit_plotly =  IO_fit_plotly_react_var.get()
        return current_IO_fit_plotly if current_IO_fit_plotly is not None else go.Figure() 
    
    @reactive.Effect
    @reactive.event(input.trigger_IO_detailed_plot_dict_saving)
    def IO_fit_plot_dict_saving_triggered():
        d = saving_dict.get().copy()
        
        d["IO_fit_plot_dict"] = {
                            "trigger": 1,
                            "object": IO_fit_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def IO_fit_stim_freq_table():
        current_IO_fit_stim_freq_table =  stim_freq_table_react_var.get()
        for col in current_IO_fit_stim_freq_table.columns:
            if col not in ['Sweep', "Stim_amp_pA", "Frequency_Hz"]:
                current_IO_fit_stim_freq_table[col] = current_IO_fit_stim_freq_table[col].astype(bool)
            elif col in ["Stim_amp_pA", "Frequency_Hz"]:
                current_IO_fit_stim_freq_table.loc[:, col] = np.round(current_IO_fit_stim_freq_table.loc[:,col], 2)

        # Desired first columns
        first_cols = ["Sweep", "Stim_amp_pA", "Frequency_Hz", "Passed_QC"]
        
        # All other columns in current order
        other_cols = [c for c in current_IO_fit_stim_freq_table.columns if c not in first_cols]
        
        # Reorder

        current_IO_fit_stim_freq_table = current_IO_fit_stim_freq_table[first_cols + other_cols]
        current_IO_fit_stim_freq_table_emoji = replace_bools_with_emojis(current_IO_fit_stim_freq_table)
        return current_IO_fit_stim_freq_table_emoji if current_IO_fit_stim_freq_table_emoji is not None else pd.DataFrame()
    
    @reactive.Effect
    @reactive.event(input.trigger_stim_freq_table_saving)
    def stim_freq_table_saving_triggered():
        d = saving_dict.get().copy()
        d["Stim_freq_table"] = {
                            "trigger": 1,
                            "object": stim_freq_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
        
    @output
    @render.data_frame
    def IO_fit_pruning_obs_table():
        current_IO_fit_pruning_obs_table =  IO_condition_table_react_var.get()
        current_IO_fit_pruning_obs_table_emoji = replace_bools_with_emojis(current_IO_fit_pruning_obs_table)
        
        return current_IO_fit_pruning_obs_table_emoji if current_IO_fit_pruning_obs_table_emoji is not None else pd.DataFrame()
    
    
    @reactive.Effect
    @reactive.event(input.trigger_IO_fit_conditions_table_saving)
    def IO_fit_pruning_obs_table_saving_triggered():
        d = saving_dict.get().copy()
        d["IO_conditions_table"] = {
                            "trigger": 1,
                            "object": IO_condition_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    
        
#%% Adaptation

    @reactive.Effect
    @reactive.event(cell_file_correctly_opened, input.Adaptation_feature_to_display)
    def preload_Adaptation_analysis():
        
        if not cell_file_correctly_opened.get() or not input.Cell_file_select():
            return
    
        task = "Adaptation analysis"
        set_status(task, f"⏳ Computing {task}...")
        
        try:
            cell_dict = get_cell_tables()
            current_adaptation_plotly, current_adaptation_plot_dict, current_adaptation_table = get_adaptation_analysis(cell_dict, input.Adaptation_feature_to_display())
            Adaptation_figure_plotly_react_var.set(current_adaptation_plotly)
            Adaptation_figure_plot_dict_react_var.set(current_adaptation_plot_dict)
            Adaptation_table_react_var.set(current_adaptation_table)
            is_Adaptation_error_react_var.set(None)
            Adaptation_error_message_react_var.set(None)
            set_status(task, f"{task} precomputed successfully ✅")
        except Exception as e:
            # Extract last relevant frame from traceback
            tb = traceback.extract_tb(e.__traceback__)
            
            short_msg = get_error_message(tb, e)
            
            # Build a simple error figure for the UI
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"⚠️ {short_msg}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(color="red", size=16)
            )
            error_table = pd.DataFrame(columns = ['Feature', "M", "C", "Adpataion index"])
            Adaptation_figure_plotly_react_var.set(error_fig)
            Adaptation_figure_plot_dict_react_var.set(dict())
            Adaptation_table_react_var.set(error_table)
            is_Adaptation_error_react_var.set(True)
            Adaptation_error_message_react_var.set("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            set_status(task, f"{task} failed ❌  {short_msg}")
            
    @render.ui
    def Adaptation_detailed_button_ui():
        if is_Adaptation_error_react_var.get() == True:
            # Button to show the modal
            return ui.input_action_button("show_adaptation_detailed_error", "⚠️ Show Error", class_="btn-danger")
        else:
            # Normal save button
            return 
    
    @reactive.Effect
    @reactive.event(input.show_adaptation_detailed_error)
    def adaptation_detailed_error_triggered():
        d = error_dict.get().copy()
        d["Adaptation_error"] = {
                            "trigger": 1,
                            "error_message": Adaptation_error_message_react_var.get()
                        }
        error_dict.set(d)

            
    @output
    @render_widget
    def Adaptation_plotly():
        fig = Adaptation_figure_plotly_react_var.get()
        return fig if fig is not None else go.Figure()  # or show an empty figure
    
    
    @reactive.Effect
    @reactive.event(input.trigger_adaptation_plot_dict_saving)
    def Adaptation_plot_dict_save_triggered():
        d = saving_dict.get().copy()
        d["Adaptation_plot_dict"] = {
                            "trigger": 1,
                            "object": Adaptation_figure_plot_dict_react_var.get(),
                            "format": "pkl"
                        }
        saving_dict.set(d)
    
    @output
    @render.data_frame
    def Adaptation_table():
        Adaptation_table = Adaptation_table_react_var.get()
        return Adaptation_table
    
    @reactive.Effect
    @reactive.event(input.trigger_adaptation_table_saving)
    def Adaptation_table_save_triggered():
        d = saving_dict.get().copy()
        d["Adaptation_table"] = {
                            "trigger": 1,
                            "object": Adaptation_table_react_var.get(),
                            "format": "csv"
                        }
        saving_dict.set(d)
    
    
#%% In progress
        
    @reactive.Calc
    @reactive.event(input.Update_cell_btn, input.x_Spike_feature, input.y_Spike_feature, input.y_Spike_feature_index)
    def get_spike_feature_index():
        if input.Cell_file_select() :#and cell_file_correctly_opened() :
            
            cell_dict=get_cell_tables()
            Full_SF_table = cell_dict['Full_SF_table']
            Sweep_info_table = cell_dict['Sweep_info_table']
            Full_TVC_table = cell_dict['Full_TVC_table']
            
            
            Spike_feature_x = input.x_Spike_feature()
            
            Spike_feature_y = input.y_Spike_feature()
            Spike_index_y = input.y_Spike_feature_index()

            First_Spike_feature_col = f'{str(Spike_feature_x)}_mV_Spike_N'
            Second_Spike_feature_col = f'{str(Spike_feature_y)}_mV_Spike_{input.y_Spike_feature_index()}'
            
            spike_feature_table=pd.DataFrame(columns=[First_Spike_feature_col, Second_Spike_feature_col, "Spike_index", "Sweep", "Stim_amp_pA"])

            Sweep_info_table = Sweep_info_table.sort_values(by=['Stim_amp_pA'])
            sweep_list = Sweep_info_table.loc[:,"Sweep"].unique()
            stim_amp_list = Sweep_info_table.loc[:,"Stim_amp_pA"].unique()

            for sweep, stim_amp in zip(sweep_list,stim_amp_list):
                sub_SF_table = Full_SF_table.loc[sweep,'SF']

                if sub_SF_table.shape[0]==0:
                    continue
                
                current_first_feature_table = sub_SF_table.loc[sub_SF_table['Feature']==Spike_feature_x,:]
                
                
                current_second_feature_table = sub_SF_table.loc[sub_SF_table['Feature']==Spike_feature_y,:]
                
                if Spike_feature_x == 'Trough' and Spike_index_y=="N+1":
                    # LG get_filtered_TVC_table -> get_sweep_TVC_table
                    sub_TVC = ordifunc.get_sweep_TVC_table(Full_TVC_table, sweep)
                    stim_start_time = Sweep_info_table.loc[sweep, "Stim_start_s"]
                    First_spike_time = list(sub_SF_table.loc[sub_SF_table['Feature']=="Threshold",'Time_s'])[0]
                    
                    minimum_voltage_before_first_spike = np.nanmin(np.array(sub_TVC.loc[(sub_TVC['Time_s']>=stim_start_time)&(sub_TVC['Time_s']<First_spike_time),"Membrane_potential_mV"]))
                    new_spike_feature_line = pd.DataFrame([np.nan,minimum_voltage_before_first_spike, np.nan, np.nan, np.nan, 'Trough', -1 ]).T
                    new_spike_feature_line.columns = current_first_feature_table.columns
                    
                    current_first_feature_table = pd.concat([new_spike_feature_line,current_first_feature_table], ignore_index=True)
                
                current_first_feature_table.index = current_first_feature_table.loc[:,"Spike_index"]
                current_first_feature_table.index = current_first_feature_table.index.astype(int)
                
                current_second_feature_table.index = current_second_feature_table.loc[:,"Spike_index"]
                current_second_feature_table.index = current_second_feature_table.index.astype(int)
                
                
                Spike_index_list = current_first_feature_table.loc[:,'Spike_index'].unique()
                for spike_index in Spike_index_list:
                    if input.y_Spike_feature_index() == 'N-1':
                        spike_index_second = spike_index-1
                    elif input.y_Spike_feature_index() == 'N+1':
                        spike_index_second = spike_index+1
                    else:
                        spike_index_second = spike_index
                    second_spike_features_index_list = current_second_feature_table.loc[:,"Spike_index"].unique()
                    
                    if spike_index_second not in second_spike_features_index_list:
                        continue
                    
                    
                    first_spike_feature = current_first_feature_table.loc[spike_index, 'Membrane_potential_mV']
                    Second_spike_feature = current_second_feature_table.loc[spike_index_second, 'Membrane_potential_mV']
                    
                    new_line = pd.DataFrame([first_spike_feature, Second_spike_feature, spike_index, sweep, stim_amp]).T
                    new_line.columns = spike_feature_table.columns
                    spike_feature_table = pd.concat([spike_feature_table, new_line], ignore_index=True)
                    
            spike_feature_table = spike_feature_table.astype({First_Spike_feature_col:'float',Second_Spike_feature_col:'float','Spike_index':'int', 'Stim_amp_pA':'float'})
                
            
            
            return spike_feature_table
        
    @output
    @render_widget
    @reactive.event(input.Update_cell_btn, input.x_Spike_feature, input.y_Spike_feature, input.y_Spike_feature_index)
    def Spike_features_index_correlation():
        
        if input.Cell_file_select() :#and cell_file_correctly_opened():
            
            Spike_feature_x = input.x_Spike_feature()
            
            Spike_feature_y = input.y_Spike_feature()

            
           
            spike_feature_table = get_spike_feature_index()
            
            First_Spike_feature_col = f'{str(Spike_feature_x)}_mV_Spike_N'
            Second_Spike_feature_col = f'{str(Spike_feature_y)}_mV_Spike_{input.y_Spike_feature_index()}'
            
            
            
            fig = px.scatter(spike_feature_table, 
                              x=First_Spike_feature_col, 
                              y=Second_Spike_feature_col, 
                              color='Stim_amp_pA',
                              hover_data=['Sweep',"Spike_index"])
            fig.update_layout(
                width=800, height=500)
    
            return fig
    
    
                
                
            
app = App(app_ui, server)