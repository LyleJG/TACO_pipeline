#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 19:05:50 2025

@author: julienballbe
"""
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.patches as patches

import Analysis_pipeline as analysis_pipeline
import Sweep_analysis as sw_an
import Spike_analysis as sp_an
import Firing_analysis as fir_an
import Ordinary_functions as ordifunc

#%% Overall_IO_plot

def plot_full_IO_fit(Full_IO_dict):
    
    Full_stim_freq_table = Full_IO_dict["Full_stim_freq_table"]
    Full_model_table = Full_IO_dict['Full_model_table']
    Full_IO_table = Full_IO_dict['Full_IO_table']
    Full_Saturation_table = Full_IO_dict['Full_Saturation_table']
    
    if Full_stim_freq_table.shape[0] > 0:
        # Define a color palette - different shades of blue
        blue_palette = px.colors.sequential.Blues[2:]  # Skip very light colors
        
        # Sort responses by time value (ascending)
        def extract_time_value(response):
            if 'ms' in response:
                return float(response.replace('ms', ''))
            else:
                time_str = response.split('_')[0]
                return float(time_str.replace('ms', ''))
        
        sorted_responses = sorted(Full_stim_freq_table['Response'].unique(), key=extract_time_value)
        response_colors = {
            response: blue_palette[i % len(blue_palette)]
            for i, response in enumerate(sorted_responses)
        }
        
        IO_figure = go.Figure()
        
        # Get all unique responses (already sorted by time)
        all_responses = sorted_responses
        
        # For each Response, add all trace types together
        for response in all_responses:
            color = response_colors[response]
            
            # # Add scatter plot (experimental data)
            # scatter_data = Full_stim_freq_table[Full_stim_freq_table['Response'] == response]
            # if scatter_data.shape[0] > 0:
            #     IO_figure.add_trace(go.Scatter(
            #         x=scatter_data['Stim_amp_pA'],
            #         y=scatter_data['Frequency_Hz'],
            #         mode='markers',
            #         name=response,
            #         marker=dict(size=8, color=color),
            #         showlegend=True
            #     ))
            
            # Add scatter plot (experimental data)
            scatter_data = Full_stim_freq_table[Full_stim_freq_table['Response'] == response]
            if scatter_data.shape[0] > 0:
                IO_figure.add_trace(go.Scatter(
                    x=scatter_data['Stim_amp_pA'],
                    y=scatter_data['Frequency_Hz'],
                    mode='markers',
                    name="Original_data",
                    marker=dict(size=8, color=color),
                    showlegend=True,
                    customdata=np.stack([
                        scatter_data['Stim_amp_pA'],
                        scatter_data['Frequency_Hz'],
                        scatter_data['Sweep'],
                        scatter_data['Response']
                    ], axis=-1),
                    hovertemplate="%{customdata[3]}<br>"
                    "Sweep %{customdata[2]}<br>"
                    "%{customdata[0]:.1f} pA<br>"
                                  "%{customdata[1]:.1f} Hz<br>",
                                 
                ))
            
            # Add model fit line
            model_data = Full_model_table[Full_model_table['Response'] == response]
            if model_data.shape[0] > 0:
                IO_figure.add_trace(go.Scatter(
                    x=model_data['Stim_amp_pA'],
                    y=model_data['Frequency_Hz'],
                    mode='lines',
                    name="IO_fit",
                    line=dict(color=color, width=2),
                    showlegend=True,
                    customdata=np.stack([
                        model_data['Stim_amp_pA'],
                        model_data['Frequency_Hz'],

                        model_data['Response']
                    ], axis=-1),
                    hovertemplate="%{customdata[2]}<br>"
                                  "%{customdata[0]:.1f} pA<br>"
                                  "%{customdata[1]:.1f} Hz<br>"
                ))
            
            # Add IO curve (dashed line)
            io_data = Full_IO_table[Full_IO_table['Response'] == response]
            if io_data.shape[0] > 0:
                IO_figure.add_trace(go.Scatter(
                    x=io_data['Stim_amp_pA'],
                    y=io_data['Frequency_Hz'],
                    mode='lines',
                    name="Linear fit",
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True,
                    customdata=np.stack([
                        io_data['Stim_amp_pA'],
                        io_data['Frequency_Hz'],
                        io_data['Response']
                    ], axis=-1),
                    hovertemplate="%{customdata[2]}<br>"
                                  "%{customdata[0]:.1f} pA<br>"
                                  "%{customdata[1]:.1f} Hz<br>"
                ))
            
            # Add saturation curve as horizontal lines if it exists
            if Full_Saturation_table.shape[0] > 0:
                sat_data = Full_Saturation_table[Full_Saturation_table['Response'] == response]
                if sat_data.shape[0] > 0:
                    for idx, row in sat_data.iterrows():
                        IO_figure.add_trace(go.Scatter(
                                x=[0, row['Stim_amp_pA']],
                                y=[row['Frequency_Hz'], row['Frequency_Hz']],
                                mode='lines',
                                name=response + " (sat)",
                                line=dict(color=color, width=2, dash='dot'),
                                showlegend=True,
                                hoverinfo='skip'
                            ))
        
        IO_figure.update_layout(
            autosize=True,
            xaxis_title="Input Current (pA)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            hovermode='closest'
        )
    else:
        IO_figure = go.Figure()
    
    return IO_figure

#%%Regression plots

def plot_regression_plots(regression_plot_dict, feature_name, x_legend, unit):
    
    Regression_feature_table = regression_plot_dict['Regression_feature_table']
    Linear_fit_table = regression_plot_dict['Linear_fit_table']
    
    fig = go.Figure()
    
    # --- Original data points ---
    fig.add_trace(go.Scatter(
        x=Regression_feature_table['Output_Duration'],
        y=Regression_feature_table[feature_name],
        mode='markers',
        marker=dict(
            symbol='circle',
            color='black',
            size=7,
            line=dict(width=1, color='white')
        ),
        name='Original Data'
    ))
    
    # --- Linear fit line ---
    fig.add_trace(go.Scatter(
        x=Linear_fit_table['Output_Duration'],
        y=Linear_fit_table[feature_name],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Linear Fit'
    ))
    
    # --- Layout & Style ---
    fig.update_layout(
        title=f'{feature_name} Linear Fit',
        xaxis_title=x_legend,
        yaxis_title=f'{feature_name} {unit}',
        width=450,
        height=420,
        template="plotly_white",
        font=dict(size=14),
        margin=dict(l=60, r=20, t=40, b=50),
        showlegend=False
    )
    
    # --- Axis styling ---
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
        linewidth=1,
        linecolor="black"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
        linewidth=1,
        linecolor="black"
    )
    
    return fig


#%% IO detailed plot
def plot_IO_detailed_fit_plot_choice(IO_plot_dict,stim_freq_table = None, plot_type = "matplotlib", do_fit = True):
    
    if do_fit == True:
        if plot_type == "plotly":
            fig = plot_IO_detailed_fit_plotly(IO_plot_dict, do_fit, stim_freq_table)
            
        elif plot_type == "matplotlib":
            fig = plot_IO_detailed_fit_matplotlib(IO_plot_dict, return_plot = True)[0]
            
        
    else:
        
        if plot_type == "plotly":
            fig = plot_stim_freq_table_plotly(stim_freq_table)
            
        elif plot_type == "matplotlib":
            fig = plot_stim_freq_table_matplotlib(stim_freq_table, return_plot = True)[0]
        
    return fig
    
    
def plot_IO_detailed_fit_matplotlib(IO_plot_dict, return_plot=False, saving_path=""):
    """
    Create a single figure with all fits plotted on one axis (PLOS ONE–compliant design):
    - Original data (QC passed/failed)
    - Polynomial fit
    - Final Hill Sigmoid fit
    - IO fit
    """
    
    # --- Default style adapted to PLOS ONE figure design ---
    default_style = {
        "figsize": (6, 6),          # ~15 cm wide — fits single-column width at 300 dpi
        "fontsize": 12,             # axis titles and legend
        "tick_font_size": 11,       # ticks slightly smaller
        "line_width": 2,            # standard scientific thickness
        "marker_size": 60,          # ensures readability in print
        "font_family": "Arial",     # PLOS recommends sans-serif, readable font
        "grid": True,
        "legend_frame": False,
    
        # Color-blind–friendly Okabe–Ito palette
        "passed_color": "#0072B2",      # Blue
        "failed_color": "#EE6677",     # Red–orange
        "trimmed_color": "#00694d",     # Green
        "poly_colors": None,
        "initial_color": "#56B4E9",     # Sky blue
        "final_color": "#D55E00",       # Vermilion
        "model_color": "#E69F00",       # Blue
        "gain_color": "#d52b00",        # Green
        "linear_color": "#000000",      # Black
        "threshold_color": "#000000",   # Black
        "saturation_color": "#D55E00",  # Gray
        "failure_color": "#D55E00",     # Red–orange
    }
    
    
    s = default_style
    
    plt.rcParams.update({
        "font.family": s["font_family"],
        "mathtext.fontset": "dejavusans",   # Utilise DejaVu Sans pour les maths
        "mathtext.default": "regular",       # Évite les polices mathématiques spéciales
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    plt.rcParams['text.usetex'] = False
    
    
        
    # --- Create single axis ---
    fig, ax = plt.subplots(figsize=s["figsize"])
    
    # ---------------- PANEL (Single): IO Fit ----------------
    original_data_table = IO_plot_dict["1-Polynomial_fit"]["original_data_table"]
    passed = original_data_table[original_data_table['Passed_QC']]
    failed = original_data_table[~original_data_table['Passed_QC']]
    
    poly_dict = IO_plot_dict["1-Polynomial_fit"]
    trimmed_stimulus_frequency_table = poly_dict['trimmed_stimulus_frequency_table']
    trimmed_3rd_poly_table = poly_dict['trimmed_3rd_poly_table']
    color_shape_dict_poly = poly_dict['color_shape_dict']
    
    hill_dict = IO_plot_dict['2-Final_Hill_Sigmoid_Fit']
    Initial_fit_color = hill_dict['Initial_Fit_table']
    color_shape_dict_hill = hill_dict['color_shape_dict']
    
    io_dict = IO_plot_dict['3-IO_fit']
    model_table = io_dict['model_table']
    gain_table = io_dict['gain_table']
    Intercept = io_dict['intercept']
    Gain = io_dict['Gain']
    Threshold_table = io_dict['Threshold']
    stimulus_for_max_freq = io_dict["stimulus_for_maximum_frequency"]
    maximum_frequency = np.nanmax(original_data_table.loc[:,'Frequency_Hz'])
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        maximum_linear_fit = np.nanmax([maximum_frequency, sat["Frequency_Hz"].iloc[0]])
    else:
        maximum_linear_fit = maximum_frequency
    
    # --- Plot all elements ---
    ax.scatter(
        passed['Stim_amp_pA'], passed['Frequency_Hz'],
        c=s["passed_color"], s=s["marker_size"], label="QC passed", alpha=0.8
    )
    ax.scatter(
        failed['Stim_amp_pA'], failed['Frequency_Hz'],
        c=s["failed_color"], s=s["marker_size"], label="QC failed", alpha=0.8
    )
    ax.scatter(
        trimmed_stimulus_frequency_table['Stim_amp_pA'],
        trimmed_stimulus_frequency_table['Frequency_Hz'],
        c=s["trimmed_color"], s=s["marker_size"] , label="Trimmed data", alpha=0.8
    )
    
    for legend in trimmed_3rd_poly_table['Legend'].unique():
        label_legend = "Polynomial fit" if legend == "3rd_order_poly" else legend
        legend_data = trimmed_3rd_poly_table[trimmed_3rd_poly_table['Legend'] == legend]
        ax.plot(
            legend_data['Stim_amp_pA'], legend_data['Frequency_Hz'],
            color=color_shape_dict_poly.get(legend, s["model_color"]),
            linewidth=s["line_width"], label=label_legend
        )
    
    ax.plot(
        Initial_fit_color['Stim_amp_pA'], Initial_fit_color['Frequency_Hz'],
        color=s["initial_color"], linewidth=s["line_width"], linestyle="--",
        label="Hill–Sigmoid (initial)"
    )
    ax.plot(
        model_table['Stim_amp_pA'], model_table['Frequency_Hz'],
        color=s["model_color"], linewidth=s["line_width"], label="IO fit"
    )
    ax.plot(
        gain_table['Stim_amp_pA'], gain_table['Frequency_Hz'],
        color=s["gain_color"], linewidth=s["line_width"] + 1.5, label="Linear IO portion"
    )
    
    xmin, xmax = ax.get_xlim()
    if "Saturation" in io_dict:
        sat = io_dict['Saturation']
        # ax.vlines(
        #     x=sat["Stim_amp_pA"].iloc[0],
        #     ymin=-5, ymax=sat["Frequency_Hz"].iloc[0],
        #     color=s["saturation_color"], linestyle="--", linewidth=s["line_width"], label="Saturation"
        # )
        ax.hlines(
            y=sat["Frequency_Hz"].iloc[0],
            xmin=xmin, xmax=sat["Stim_amp_pA"].iloc[0],
            color=s["saturation_color"], linestyle="--", linewidth=s["line_width"]
        )
    
    x_range = np.arange(original_data_table['Stim_amp_pA'].min(), stimulus_for_max_freq, 1)
    y_values = Intercept + Gain * x_range
    
    
    
    # Remove values above maximum_linear_fit AND below 0
    mask = (y_values <= maximum_linear_fit) & (y_values >= 0)
    x_range_masked = x_range[mask]
    y_values_masked = y_values[mask]
    
    ax.plot(
        x_range_masked, y_values_masked,
        color=s["linear_color"], linestyle=":", linewidth=s["line_width"], label="Linear fit"
    )
    
    ymin, ymax = ax.get_ylim()

    handles, labels = ax.get_legend_handles_labels()
    
    # --- Formatting ---
    ax.set_title("Input–Output Fit", fontsize=s["fontsize"], pad=10)
    ax.set_xlabel("Input current (pA)", fontsize=s["fontsize"])
    ax.set_ylabel("Firing frequency (Hz)", fontsize=s["fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=s["tick_font_size"])
    ax.set_ylim(bottom=-5)
    ax.set_xlim([xmin, xmax])
    if s["grid"]:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)


    # --- Threshold arrow below x-axis ---
    if Threshold_table is not None and not Threshold_table.empty:
        threshold_x = Threshold_table["Stim_amp_pA"].iloc[0]
        # Position sous l'axe des x
        threshold_y = -(ymax * 0.125)
        threshold_y = -5

        # Add arrow just below x-axis
        ax.annotate(
            '', 
            xy=(threshold_x, -5),  # arrow tip (x on axis, y=0)
            xytext=(threshold_x, -(ymax*0.125)),  # start of the arrow (below x-axis)
            arrowprops=dict(
                arrowstyle='-|>', 
                color=s["threshold_color"], 
                lw=1.8
            )
        )
        
        # Create proxy handle

        threshold_handle = Line2D(
            [0], [0], color=s["threshold_color"], 
            marker = "^",
            markersize=10, 
            linestyle='None'
        )

        
        
        
        
        handles.append(threshold_handle)
        labels.append("Threshold")
        
        # ax.plot(
        #     threshold_x, 
        #     threshold_y,
        #     marker='^',
        #     markersize=10,
        #     color=s["threshold_color"],
        #     markeredgewidth=1.5,
        #     markeredgecolor=s["threshold_color"],
        #     markerfacecolor=s["threshold_color"],
        #     clip_on=False,  # Important: permet d'afficher en dehors de la zone de plot
        #     zorder=10  # Assure que le marker est au-dessus
        # )
        
        # # Create proxy handle (identique au marker sur le plot)
        # threshold_handle = Line2D(
        #     [0], [0], 
        #     color=s["threshold_color"], 
        #     marker="^",
        #     markersize=10, 
        #     linestyle='None',
        #     markerfacecolor=s["threshold_color"],
        #     markeredgecolor=s["threshold_color"]
        # )
        
        # handles.append(threshold_handle)
        # labels.append("Threshold")
    
    
    if "Response_Failure" in io_dict:
        fail = io_dict['Response_Failure']
        
        failure_x = fail["Stim_amp_pA"].iloc[0]
    
    
        # Add arrow just below x-axis
        ax.annotate(
            '', 
            xy=(failure_x, -5),  # arrow tip (x on axis, y=0)
            xytext=(failure_x, -(ymax*0.125)),  # start of the arrow (below x-axis)
            arrowprops=dict(
                arrowstyle='-|>', 
                color=s["failure_color"], 
                lw=1.8
            )
        )
        failure_handle = Line2D(
            [0], [0], color=s["failure_color"], marker='^', markersize=14, linestyle='None'
        )
        handles.append(failure_handle)
        labels.append("Response failure")
        
    
    
    
    # --- Legend ---
    
    # Create a proxy arrow for the legend
    
    
    
    
    
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               fontsize=s["fontsize"]-1, ncol=4, frameon=s["legend_frame"])
    
    

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    

    
        
        
    # --- Saving ---
    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi=300)
        # optional: also save as SVG for vector graphics
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    
    if return_plot:
        return fig, ax
    else:
        plt.show()
        
def plot_stim_freq_table_matplotlib(stim_freq_table, return_plot = False, saving_path = None):
    
    # --- Default style adapted to PLOS ONE figure design ---
    default_style = {
        "figsize": (6, 6),          # ~15 cm wide — fits single-column width at 300 dpi
        "fontsize": 12,             # axis titles and legend
        "tick_font_size": 11,       # ticks slightly smaller
        "line_width": 2,            # standard scientific thickness
        "marker_size": 60,          # ensures readability in print
        "font_family": "Arial",     # PLOS recommends sans-serif, readable font
        "grid": True,
        "legend_frame": False,
    
        # Color-blind–friendly Okabe–Ito palette
        "passed_color": "#0072B2",      # Blue
        "failed_color": "#EE6677",     # Red–orange
        "trimmed_color": "#00694d",     # Green
        "poly_colors": None,
        "initial_color": "#56B4E9",     # Sky blue
        "final_color": "#D55E00",       # Vermilion
        "model_color": "#E69F00",       # Blue
        "gain_color": "#d52b00",        # Green
        "linear_color": "#000000",      # Black
        "threshold_color": "#000000",   # Black
        "saturation_color": "#D55E00",  # Gray
        "failure_color": "#D55E00",     # Red–orange
    }
    
    
    s = default_style
    
    # --- Create figure ---
           
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # === QC passed ===
    scatter = ax.scatter(
        stim_freq_table["Stim_amp_pA"],
        stim_freq_table["Frequency_Hz"],
        s=s["marker_size"],
        alpha=0.8,
        picker=True  # Pour permettre l'interactivité si nécessaire
    )
    
    # --- Layout ---
    ax.set_title("Input–Output Fit", fontsize=s.get("fontsize", 12), pad=10)
    ax.set_xlabel("Input current (pA)", fontsize=s.get("fontsize", 12))
    ax.set_ylabel("Firing frequency (Hz)", fontsize=s.get("fontsize", 12))
    
    # Tick font size
    ax.tick_params(axis='both', which='major', labelsize=s["tick_font_size"])
    
    # Grid
    ax.grid(True, color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)  # Grid behind data
    
    # Spines (axes lines)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Ticks outside
    ax.tick_params(
        axis='both',
        which='both',
        direction='out',
        top=True,      # Ticks on top
        right=True,    # Ticks on right
        bottom=True,
        left=True
    )
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Legend (horizontal, below plot)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,  # Ajuster selon le nombre d'éléments
        frameon=False
    )
    
    # Margins
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.15)
    
    # --- Saving ---
    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi=300)
        # optional: also save as SVG for vector graphics
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    
    if return_plot:
        return fig, ax
    else:
        plt.show()
        
    
def plot_IO_detailed_fit_plotly(IO_plot_dict, do_fit, stim_freq_table):

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

    # --- Extract dictionaries ---
    original_data_table = IO_plot_dict["1-Polynomial_fit"]["original_data_table"]
    passed = original_data_table[original_data_table["Passed_QC"]]
    failed = original_data_table[~original_data_table["Passed_QC"]]

    poly_dict = IO_plot_dict["1-Polynomial_fit"]
    trimmed_data = poly_dict["trimmed_stimulus_frequency_table"]
    poly_table = poly_dict["trimmed_3rd_poly_table"]
    poly_color_dict = poly_dict["color_shape_dict"]

    hill_dict = IO_plot_dict["2-Final_Hill_Sigmoid_Fit"]
    initial_table = hill_dict["Initial_Fit_table"]

    io_dict = IO_plot_dict["3-IO_fit"]
    model_table = io_dict["model_table"]
    gain_table = io_dict["gain_table"]
    intercept = io_dict["intercept"]
    gain = io_dict["Gain"]
    threshold_table = io_dict["Threshold"]
    stim_max = io_dict["stimulus_for_maximum_frequency"]
    
    maximum_frequency = np.nanmax(original_data_table.loc[:,'Frequency_Hz'])
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        maximum_linear_fit = np.nanmax([maximum_frequency, sat["Frequency_Hz"].iloc[0]])
    else:
        maximum_linear_fit = maximum_frequency
    
# --- Create figure ---
    fig = go.Figure()

    # === Stim Freq table
    for df, color, name in [
        (passed, s["passed_color"], "QC passed"),
        (failed, s["failed_color"], "QC failed"),
        (trimmed_data, s["trimmed_color"], "Trimmed data")]:
    
        fig.add_trace(go.Scatter(
            x=df["Stim_amp_pA"],
            y=df["Frequency_Hz"],
            mode="markers",
            marker=dict(size=s["marker_size"], color=color, opacity=0.8),
            name=name,
            customdata=df[["Sweep"]],
            hovertemplate=(
                    "Sweep: %{customdata[0]}<br>"
                    "Stim: %{x:.2f} pA<br>"
                    "Freq: %{y:.2f} Hz<extra></extra>"
                )

        ))

    # LG Remove poly fits
    # === Polynomial fits ===
    # for legend in poly_table["Legend"].unique():
    #     df_leg = poly_table[poly_table["Legend"] == legend]

    #     label = "Polynomial fit" if legend == "3rd_order_poly" else legend
    #     fig.add_trace(go.Scatter(
    #         x=df_leg["Stim_amp_pA"],
    #         y=df_leg["Frequency_Hz"],
    #         mode="lines",
    #         line=dict(color=poly_color_dict.get(legend, s["model_color"]),
    #                   width=s["line_width"]),
    #         name=label
    #     ))

    # === Hill sigmoid initial ===
    fig.add_trace(go.Scatter(
        x=initial_table["Stim_amp_pA"],
        y=initial_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["initial_color"], width=s["line_width"], dash="dash"),
        name="Hill–Sigmoid (initial)"
    ))

    # === IO model ===
    fig.add_trace(go.Scatter(
        x=model_table["Stim_amp_pA"],
        y=model_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["model_color"],
                  # width=s["line_width"]
                  width=4
                  ),
        name="IO fit"
    ))

    # === Linear IO portion ===
    fig.add_trace(go.Scatter(
        x=gain_table["Stim_amp_pA"],
        y=gain_table["Frequency_Hz"],
        mode="lines",
        line=dict(color=s["gain_color"], width=s["line_width"] + 1.5),
        name="Linear IO portion"
    ))

    # === Saturation line ===
    if "Saturation" in io_dict:
        sat = io_dict["Saturation"]
        fig.add_trace(go.Scatter(
            x=[-5, sat["Stim_amp_pA"].iloc[0]],
            y=[sat["Frequency_Hz"].iloc[0]] * 2,
            mode="lines",
            line=dict(color=s["saturation_color"], width=s["line_width"], dash="dash"),
            name="Saturation"
        ))

    # === Linear fit ===
    x_range = np.arange(original_data_table["Stim_amp_pA"].min(), stim_max, 1)
    y_vals = intercept + gain * x_range
    # Remove values above maximum_linear_fit AND below 0
    mask = (y_vals <= maximum_linear_fit) & (y_vals >= 0)
    fig.add_trace(go.Scatter(
        x=x_range[mask],
        y=y_vals[mask],
        mode="lines",
        line=dict(color=s["linear_color"], width=s["line_width"], dash="dot"),
        name="Linear fit"
    ))

    # ------------------------------------------------------------------
    # === Threshold arrow (Option A: inside plot at y = 0) ===
    # ------------------------------------------------------------------
    if threshold_table is not None and not threshold_table.empty:
        thr_x = threshold_table["Stim_amp_pA"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[thr_x],
            y=[-2],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=s["threshold_color"]
            ),
            name="Threshold",
            showlegend=True
        ))


    # === Failure arrow ===
    if "Response_Failure" in io_dict:
        fail = io_dict["Response_Failure"]
        fx = fail["Stim_amp_pA"].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[fx],
            y=[-2],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=s["failure_color"]
            ),
            name="Response Failure",
            showlegend=True
        ))

        

    # --- Layout ---
    fig.update_layout(
        # width=600,   # in pixels
        # height=600,  # in pixels
        plot_bgcolor="white",
        template = "none",
        title="Input–Output Fit",
        xaxis_title="Input current (pA)",
        yaxis_title="Firing frequency (Hz)",
        xaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        yaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=10, t=60, b=80)
    )
    
    
    #     # X axis
    # fig.update_layout(
    # plot_bgcolor="white",   # plotting area background
    
    # )
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    
    
        
    
    return fig

def plot_stim_freq_table_plotly(stim_freq_table):
    
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
    
    # --- Create figure ---
    fig = go.Figure()

    # === QC passed ===
    fig.add_trace(go.Scatter(
        x=stim_freq_table["Stim_amp_pA"],
        y=stim_freq_table["Frequency_Hz"],
        mode="markers",
        marker=dict(size=s["marker_size"], opacity=0.8),
        customdata=stim_freq_table[["Sweep"]],
        hovertemplate=(
                        "Sweep: %{customdata[0]}<br>"
                        "Stim: %{x:.2f} pA<br>"
                        "Freq: %{y:.2f} Hz<extra></extra>"
                    )

    ))
    # --- Layout ---
    fig.update_layout(
        # width=600,   # in pixels
        # height=600,  # in pixels
        plot_bgcolor="white",
        template = "none",
        title="Input–Output Fit",
        xaxis_title="Input current (pA)",
        yaxis_title="Firing frequency (Hz)",
        xaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        yaxis=dict(tickfont=dict(size=s["tick_font_size"])),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=10, t=60, b=80)
    )
    
    
    
    
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    

    return fig

#%% Adaptation plot


def plot_Adaptation_fit_plot_choice(adaptation_plot_dict,stim_freq_table = None, plot_type = "matplotlib", do_fit = True):
    
    if do_fit == True:
        if plot_type == "matplotlib":
            adaptation_fig = plot_adaptation_TACO_paper_one_panel(adaptation_plot_dict, return_plot=True)[0]
        elif plot_type == 'plotly':
            adaptation_fig = plotly_adaptation_TACO_one_panel(adaptation_plot_dict)
            
    return adaptation_fig

def plot_adaptation_TACO_paper_one_panel(plot_dict, return_plot=False, style_dict = None, saving_path=""):
    # --- Extract data ---
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict["C_ref"]
    interval_frequency_table = plot_dict["interval_frequency_table"]
    median_table = plot_dict["median_table"]
    M = plot_dict["M"]
    C = plot_dict["C"]

    # --- Default style (PLOS ONE & color-blind friendly) ---
    default_style = {
        "figsize": (6, 5),
        "xlabel": "Spike interval",
        "ylabel": "Normalized feature",
        "fontsize": 10,
        "tick_font_size": 9,
        "line_color_original": "#56B4E9",  # Sky Blue
        "line_color_sim": "#009E73",       # Bluish Green
        "area_color": "#F0E442",           # Yellow
        "rect_color": "#CC79A7",           # Reddish Purple
        "scatter_cmap": "viridis",
        "median_marker": "s",
        "median_color": "#D55E00",         # Vermilion
        "max_median_size": 18,
        "dpi": 300,
    }
    if style_dict:
        default_style.update(style_dict)
    s = default_style

    # --- Create single figure ---
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=s["dpi"])

    # --- Scatter: Original data ---
    sc = ax.scatter(
        interval_frequency_table["Spike_Interval"],
        interval_frequency_table["Normalized_feature"],
        c=interval_frequency_table["Stimulus_amp_pA"],
        cmap=s["scatter_cmap"],
        alpha=0.6,
        edgecolor="none",
        label="Original data",
    )

    # --- Median points ---
    sizes = (
        median_table["Count_weigths"] / median_table["Count_weigths"].max()
    ) * s["max_median_size"]
    ax.scatter(
        median_table["Spike_Interval"],
        median_table["Normalized_feature"],
        s=sizes,
        c=s["median_color"],
        marker=s["median_marker"],
        alpha=0.7,
        label="Median interval value",
    )

    # --- Simulation (exponential fit) ---
    ax.plot(
        sim_table["Spike_Interval"],
        sim_table["Normalized_feature"],
        color=s["line_color_sim"],
        linewidth=2,
        label="Exponential fit",
    )

    # --- Filled area (M) ---
    ax.fill_between(
        original_sim_table["Spike_Interval"],
        original_sim_table["Normalized_feature"],
        color=s["area_color"],
        alpha=0.6,
        label=f"M = {np.round(M, 2)}",
    )

    
    ax.fill_betweenx(
            y=[0, C_ref],
            x1=np.nanmin(sim_table["Spike_Interval"]),
            x2=(Na - 1) - 1,
            color=s["rect_color"],
            alpha=1,
            label=f"C = {np.round(C, 2)}",
        )
 
    # --- Formatting ---
    ax.set_xlabel(s["xlabel"], fontsize=s["fontsize"])
    ax.set_ylabel(s["ylabel"], fontsize=s["fontsize"])
    ax.tick_params(axis="both", which="major", labelsize=s["tick_font_size"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend (below plot) ---
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=s["fontsize"],
        ncol=2,
        frameon=False,
    )

    # --- Colorbar ---
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Input current (pA)", fontsize=s["fontsize"])
    cbar.ax.tick_params(labelsize=s["tick_font_size"])

    plt.tight_layout()
    
    if return_plot == True:
        return fig, ax

    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi = 300)
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    plt.show()
    

def plotly_adaptation_TACO_one_panel(plot_dict, style_dict=None):
    # --- Extract data ---
    original_sim_table = plot_dict["original_sim_table"]
    sim_table = plot_dict["sim_table"]
    Na = plot_dict["Na"]
    C_ref = plot_dict["C_ref"]
    interval_frequency_table = plot_dict["interval_frequency_table"]
    median_table = plot_dict["median_table"]
    M = plot_dict["M"]
    C = plot_dict["C"]

    # --- Default style ---
    default_style = {
        "xlabel": "Spike interval",
        "ylabel": "Normalized feature",
        "line_color_sim": "#009E73",
        "area_color": "#F0E442",
        "rect_color": "#CC79A7",
        "median_color": "#D55E00",
        "max_median_size": 18,
        "scatter_cmap": "Viridis",  # Plotly name
    }
    if style_dict:
        default_style.update(style_dict)
    s = default_style

    # --- Create figure ---
    fig = go.Figure()

    # === Scatter: Original data ===
    fig.add_trace(go.Scatter(
        x=interval_frequency_table["Spike_Interval"],
        y=interval_frequency_table["Normalized_feature"],
        mode="markers",
        marker=dict(
            color=interval_frequency_table["Stimulus_amp_pA"],
            colorscale=s["scatter_cmap"],
            size=6,
            opacity=0.6,
            colorbar=dict(title="Input current (pA)")
        ),
        name="Original data"
    ))

    # === Median points ===
    sizes = (
        median_table["Count_weigths"] /
        median_table["Count_weigths"].max()
    ) * s["max_median_size"]

    fig.add_trace(go.Scatter(
        x=median_table["Spike_Interval"],
        y=median_table["Normalized_feature"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=s["median_color"],
            symbol="square",
            opacity=0.7,
            line=dict(width=0)
        ),
        name="Median interval value"
    ))

    # === Simulation curve ===
    fig.add_trace(go.Scatter(
        x=sim_table["Spike_Interval"],
        y=sim_table["Normalized_feature"],
        mode="lines",
        line=dict(color=s["line_color_sim"], width=2),
        name="Exponential fit"
    ))

    # === Filled area (M) ===
    fig.add_trace(go.Scatter(
        x=sim_table["Spike_Interval"],
        y=sim_table["Normalized_feature"],
        fill="tozeroy",
        fillcolor="rgba(240, 228, 66, 0.6)",  # hex #F0E442 with alpha 0.1
        line=dict(color="rgba(0,0,0,0)"),
        name=f"M = {np.round(M, 2)}"
    ))

    # === Rectangle for C ===
    rect_x0 = np.nanmin(sim_table["Spike_Interval"])
    rect_x1 = (Na - 1) - 1

    fig.add_shape(
        type="rect",
        x0=rect_x0,
        x1=rect_x1,
        y0=0,
        y1=C_ref,
        fillcolor=s["rect_color"],
        opacity=1,
        line=dict(width=0),
    )

    # Dummy trace to show C in legend as a thick line
    fig.add_trace(go.Scatter(
        x=[0, 1],               # invisible range
        y=[None, None],         # will not plot any actual points
        mode="lines",
        line=dict(
            color=s["rect_color"],
            width=8              # thickness of the legend line
        ),
        name=f"C = {np.round(C, 2)}",
        showlegend=True
    ))

    # === Formatting ===
    fig.update_layout(
        xaxis_title=s["xlabel"],
        yaxis_title=s["ylabel"],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=20, t=20, b=80),
    )

    return fig

#%% Bridge Error


def plot_BE_TACO_choice(dict_plot, plot_type = 'matplotlib', do_fit = True):
    
    if do_fit == True:
        
        if plot_type == "plotly":
            
            fig = plot_BE_TACO_paper_plotly(dict_plot, style_dict=None, saving_path="", return_plot = True)
        elif plot_type == 'matplotlib':
            
            fig = plot_BE_TACO_paper_matplotlib(dict_plot, style_dict=None, saving_path="", return_plot = True)[0]

    return fig
    
def plot_BE_TACO_paper_matplotlib(dict_plot, style_dict=None, saving_path="", return_plot = False):
    """
    PLOS ONE–style Bridge Error analysis figure.
    5 rows × 2 columns (left = full view, right = zoomed view).
    All panels use color-blind–safe Okabe–Ito palette.
    """

    if style_dict is None:
        style_dict = {}
    actual_transition_time = dict_plot["Transition_time"]
    # --- Default PLOS ONE–ready style ---
    default_style = {
        "figsize": (14, 12),
        "font_size": 9,
        "title_fontsize": 10,
        "legend_fontsize": 8,
        "col_width_ratios": [2, 2],
        "x_lim_zoom": (actual_transition_time-0.010, actual_transition_time+0.010),
        "xlim_right": (actual_transition_time-0.010, actual_transition_time+0.010),
        "dpi": 300,
        # Okabe–Ito palette (color-blind safe)
        "colors": {
            "black": "#000000",
            "orange": "#E69F00",
            "skyblue": "#56B4E9",
            "bluish_green": "#009E73",
            "yellow": "#F0E442",
            "blue": "#0072B2",
            "vermillion": "#D55E00",
            "reddish_purple": "#CC79A7",
        },
    }

    style = {**default_style, **style_dict}
    c = style["colors"]

    plt.rcParams.update({
        "font.size": style["font_size"],
        "axes.labelsize": style["font_size"],
        "axes.titlesize": style["title_fontsize"],
        "legend.fontsize": style["legend_fontsize"],
        "xtick.labelsize": style["font_size"],
        "ytick.labelsize": style["font_size"],
    })

    # --- Extract data ---
    TVC_table = dict_plot["TVC_table"]
    min_time_current = dict_plot["min_time_current"]
    max_time_current = dict_plot["max_time_current"]
    
    alpha_FT = dict_plot["alpha_FT"]
    T_FT = dict_plot["T_FT"]
    delta_t_ref_first_positive = dict_plot["delta_t_ref_first_positive"]
    T_ref_cell = dict_plot["T_ref_cell"]

    V_fit_table = dict_plot["Membrane_potential_mV"]["V_fit_table"]
    V_table_to_fit = dict_plot["Membrane_potential_mV"]["V_table_to_fit"]
    V_pre_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Pre_transition"]
    V_post_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Post_transition"]

    pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
    post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
    current_trace_table = dict_plot["Input_current_pA"]["current_trace_table"]

    row_titles = [
        "Membrane potential",
        "Input current",
        "1st derivative of V",
        "2nd derivative of V",
        "Derivative of input current",
    ]

    y_labels = ["mV", "pA", "mV/ms", "mV/ms²", "pA/ms"]

    # --- Layout ---
    fig, axs = plt.subplots(
        5, 2, figsize=style["figsize"], dpi=style["dpi"],
        sharex="col", gridspec_kw={"width_ratios": style["col_width_ratios"]}
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    # --- Helper: zoom rectangle ---
    def add_zoom_rectangle(ax):
        ymin, ymax = ax.get_ylim()
        xmin, xmax = style["x_lim_zoom"]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color=c["blue"], ls="--", lw=1.2
        )
        ax.add_patch(rect)

    # --- Row plotting helpers ---
    def plot_membrane(ax, mode="right"):
        ax.plot(TVC_table["Time_s"], TVC_table["Membrane_potential_mV"],
                color=c["black"], lw=1, label="Membrane potential")
        if mode == "right":
            ax.plot(TVC_table["Time_s"], TVC_table["Membrane_potential_0_5_LPF"],
                    color=c["bluish_green"], lw=2.5, label="0.5 Hz LPF")
            ax.scatter([actual_transition_time], [V_pre_transition_time],
                       color=c["orange"], s=30, label="Pre-transition V")
            ax.scatter([actual_transition_time], [V_post_transition_time],
                       color=c["vermillion"], s=30, label="Post-transition V")
            # ax.plot(V_table_to_fit["Time_s"], V_table_to_fit["Membrane_potential_mV"],
            #         color=c["yellow"], lw=1.2, label="Fit window")
            ax.plot(V_fit_table["Time_s"], V_fit_table["Membrane_potential_mV"],
                    color=c["blue"], lw=2.5, label="Post-fit")
            return ax.get_legend_handles_labels()
        return [], []

    def plot_current(ax, mode="right"):
        ax.plot(TVC_table["Time_s"], TVC_table["Input_current_pA"],
                color=c["black"], lw=1, label="Input current")
        if mode == "right":
            mask_pre = (current_trace_table["Time_s"] <= (min_time_current + 0.005)) & \
                       (current_trace_table["Time_s"] >= min_time_current)
            mask_post = (current_trace_table["Time_s"] <= max_time_current) & \
                        (current_trace_table["Time_s"] >= (max_time_current - 0.005))
            ax.plot(current_trace_table.loc[mask_pre, "Time_s"],
                    [pre_T_current] * mask_pre.sum(),lw=2,
                    color=c["vermillion"], label="Pre-transition median")
            ax.plot(current_trace_table.loc[mask_post, "Time_s"],
                    [post_T_current] * mask_post.sum(),lw=2,
                    color=c["blue"], label="Post-transition median")
            return ax.get_legend_handles_labels()
        return [], []

    def plot_vdot(ax, mode="right"):
        ax.plot(TVC_table["Time_s"], TVC_table["V_dot_one_kHz"],
                color=c["black"], lw=1, label="dV/dt")
        if mode == "right":
            first_sign_change = delta_t_ref_first_positive + T_ref_cell
            T_start_fit = np.nanmin(V_table_to_fit["Time_s"])
            #ax.axvline(first_sign_change, color=c["bluish_green"], ls="--", label="First sign change")
            ax.axvline(T_ref_cell, color=c["vermillion"], ls="--", label="T_ref")
            ax.axvline(T_start_fit, color=c["yellow"], ls="--", label="Fit start")
            return ax.get_legend_handles_labels()
        return [], []

    def plot_vddot(ax, mode="right"):
        ax.plot(TVC_table["Time_s"], TVC_table["V_double_dot_five_kHz"],
                color=c["black"], lw=1, label="d2V/dt2")
        if mode == "right":
            Fast_ring_time = TVC_table.loc[
                (TVC_table["Time_s"] <= (actual_transition_time + 0.005)) &
                (TVC_table["Time_s"] >= actual_transition_time), "Time_s"
            ]
            ax.hlines(alpha_FT, Fast_ring_time.min(), Fast_ring_time.max(),
                      color=c["reddish_purple"], ls="--", label="± α FT")
            ax.hlines(-alpha_FT, Fast_ring_time.min(), Fast_ring_time.max(),
                      color=c["reddish_purple"], ls="--")
            ax.axvline(T_FT, color=c["vermillion"], ls="--", label="T_FT")
            return ax.get_legend_handles_labels()
        return [], []

    def plot_idot(ax, mode="right"):
        ax.plot(TVC_table["Time_s"], TVC_table["I_dot_five_kHz"],
                color=c["black"], lw=1, label="dI/dt")
        if mode == "right":
            ax.axvline(actual_transition_time, color=c["blue"], ls="--", label="T*")
            return ax.get_legend_handles_labels()
        return [], []

    funcs = [plot_membrane, plot_current, plot_vdot, plot_vddot, plot_idot]

    for i, func in enumerate(funcs):
        # Left (full view)
        func(axs[i, 0], mode="left")
        axs[i, 0].axvspan(style["x_lim_zoom"][0], style["x_lim_zoom"][1],
                          color=c["blue"], alpha=0.1, lw=0)
        add_zoom_rectangle(axs[i, 0])

        # Right (zoomed)
        handles, labels = func(axs[i, 1], mode="right")
        axs[i, 1].set_xlim(style["xlim_right"])

        axs[i, 0].set_ylabel(y_labels[i])
        axs[i, 0].set_title(row_titles[i], pad=5, loc="left", fontsize=style["title_fontsize"])
        axs[i, 1].set_yticklabels([])

        for j in [0, 1]:
            axs[i, j].grid(True, color="grey", alpha=0.4, ls="--", lw=0.6)
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)

        # Legend (right column)
        unique = dict(zip(labels, handles))
        axs[i, 1].legend(unique.values(), unique.keys(),
                         loc="center left", bbox_to_anchor=(1.02, 0.5),
                         frameon=False, handlelength=1.5)

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if saving_path:
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi= 300)
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    
    if return_plot == True:
        return fig, axs
    else:
        plt.show()
    

def plot_BE_TACO_paper_plotly(dict_plot, style_dict=None, saving_path="", return_plot = False):
    """
    PLOS ONE–style Bridge Error analysis figure (Plotly version).
    5 rows × 2 columns (left = full view, right = zoomed view).
    """

    actual_transition_time = dict_plot["Transition_time"]


    style_dict = {}

    # --- Default style ---
    default_style = {
        "figsize": (1400, 1200),
        "font_size": 12,
        "title_fontsize": 14,
        "legend_fontsize": 12,
        "col_width_ratios": [0.5, 0.5],
        "x_lim_zoom": (actual_transition_time - 0.010, actual_transition_time + 0.010),
        "xlim_right": (actual_transition_time - 0.010, actual_transition_time + 0.010),
        # Okabe–Ito palette (color-blind safe)
        "colors": {
            "black": "#000000",
            "orange": "#E69F00",
            "skyblue": "#56B4E9",
            "bluish_green": "#009E73",
            "yellow": "#F0E442",
            "blue": "#0072B2",
            "vermillion": "#D55E00",
            "reddish_purple": "#CC79A7",
        },
    }

    style = {**default_style, **style_dict}
    c = style["colors"]

    # --- Extract data ---
    TVC_table = dict_plot["TVC_table"]
    min_time_current = dict_plot["min_time_current"]
    max_time_current = dict_plot["max_time_current"]

    alpha_FT = dict_plot["alpha_FT"]
    T_FT = dict_plot["T_FT"]
    
    delta_t_ref_first_positive = dict_plot["delta_t_ref_first_positive"]
    T_ref_cell = dict_plot["T_ref_cell"]

    V_fit_table = dict_plot["Membrane_potential_mV"]["V_fit_table"]
    V_table_to_fit = dict_plot["Membrane_potential_mV"]["V_table_to_fit"]
    V_pre_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Pre_transition"]
    V_post_transition_time = dict_plot["Membrane_potential_mV"]["Voltage_Post_transition"]

    pre_T_current = dict_plot["Input_current_pA"]["pre_T_current"]
    post_T_current = dict_plot["Input_current_pA"]["post_T_current"]
    current_trace_table = dict_plot["Input_current_pA"]["current_trace_table"]

    row_titles = [
        "Membrane potential",
        "Input current",
        "1st derivative of V",
        "2nd derivative of V",
        "Derivative of input current",
    ]
    y_labels = ["mV", "pA", "mV/ms", "mV/ms²", "pA/ms"]

    # --- Layout ---
    fig = make_subplots(
        rows=5, cols=2,
        shared_xaxes=True,
        column_widths=style["col_width_ratios"],
        subplot_titles=[t + (" (full)" if i % 2 == 0 else " (zoom)") for t in row_titles for i in range(2)],
        vertical_spacing=0.06,
        horizontal_spacing=0.08
    )

    # Helper to add shaded zoom region on left panels
    def add_zoom_rectangle(fig, row, col):
        xmin, xmax = style["x_lim_zoom"]
        fig.add_vrect(
            x0=xmin, x1=xmax,
            fillcolor=c["blue"], opacity=0.1, line_width=0,
            row=row, col=col
        )

    # --- Plot functions ---
    def plot_membrane(row, col, zoom=False):
        fig.add_trace(go.Scatter(
            x=TVC_table["Time_s"], y=TVC_table["Membrane_potential_mV"],
            mode="lines", line=dict(color=c["black"], width=1),
            name="Membrane potential", legendgroup="Membrane"
        ), row=row, col=col)
        if zoom:
            fig.add_trace(go.Scatter(
                x=TVC_table["Time_s"], y=TVC_table["Membrane_potential_0_5_LPF"],
                mode="lines", line=dict(color=c["bluish_green"], width=1),
                name="Full trace 0.5 Hz LPF", legendgroup="Membrane"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=TVC_table["Time_s"], y=TVC_table["V_pre_T_0_5_LPF"],
                mode="lines", line=dict(color=c["reddish_purple"], width=2),
                name="Pre_T 0.5 Hz LPF", legendgroup="Membrane"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[actual_transition_time], y=[V_pre_transition_time],
                mode="markers", marker=dict(color=c["orange"], size=8),
                name="Pre-transition V", legendgroup="Membrane"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[actual_transition_time], y=[V_post_transition_time],
                mode="markers", marker=dict(color=c["vermillion"], size=8),
                name="Post-transition V", legendgroup="Membrane"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=V_fit_table["Time_s"], y=V_fit_table["Membrane_potential_mV"],
                mode="lines", line=dict(color=c["blue"], width=2.5),
                name="Post-fit", legendgroup="Membrane"
            ), row=row, col=col)

    def plot_current(row, col, zoom=False):
        fig.add_trace(go.Scatter(
            x=TVC_table["Time_s"], y=TVC_table["Input_current_pA"],
            mode="lines", line=dict(color=c["black"], width=1),
            name="Input current", legendgroup="Current"
        ), row=row, col=col)
        if zoom:
            mask_pre = (current_trace_table["Time_s"] <= (min_time_current + 0.005)) & \
                       (current_trace_table["Time_s"] >= min_time_current)
            mask_post = (current_trace_table["Time_s"] <= max_time_current) & \
                        (current_trace_table["Time_s"] >= (max_time_current - 0.005))
            fig.add_trace(go.Scatter(
                x=current_trace_table.loc[mask_pre, "Time_s"],
                y=[pre_T_current] * mask_pre.sum(),
                mode="lines", line=dict(color=c["vermillion"], width=2),
                name="Pre-transition median", legendgroup="Current"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=current_trace_table.loc[mask_post, "Time_s"],
                y=[post_T_current] * mask_post.sum(),
                mode="lines", line=dict(color=c["blue"], width=2),
                name="Post-transition median", legendgroup="Current"
            ), row=row, col=col)

    def plot_vdot(row, col, zoom=False):
        fig.add_trace(go.Scatter(
            x=TVC_table["Time_s"], y=TVC_table["V_dot_one_kHz"],
            mode="lines", line=dict(color=c["black"], width=1),
            name="dV/dt", legendgroup="Vdot"
        ), row=row, col=col)
        if zoom:
            T_start_fit = np.nanmin(V_table_to_fit["Time_s"])
            for x, color, name in [(T_ref_cell, c["vermillion"], "T_ref"),
                                   (T_start_fit, c["yellow"], "Fit start")]:
                fig.add_vline(
                    x=x, line=dict(color=color, dash="dash"),
                    row=row, col=col
                )

    def plot_vddot(row, col, zoom=False):
        fig.add_trace(go.Scatter(
            x=TVC_table["Time_s"], y=TVC_table["V_double_dot_five_kHz"],
            mode="lines", line=dict(color=c["black"], width=1),
            name="d2V/dt2", legendgroup="Vddot"
        ), row=row, col=col)
        if zoom:
            Fast_ring_time = TVC_table.loc[
                (TVC_table["Time_s"] <= (actual_transition_time + 0.005)) &
                (TVC_table["Time_s"] >= actual_transition_time), "Time_s"
            ]
            fig.add_trace(go.Scatter(
                x=[Fast_ring_time.min(), Fast_ring_time.max()],
                y=[alpha_FT, alpha_FT],
                mode="lines", line=dict(color=c["reddish_purple"], dash="dash"),
                name="± α FT", legendgroup="Vddot"
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[Fast_ring_time.min(), Fast_ring_time.max()],
                y=[-alpha_FT, -alpha_FT],
                mode="lines", line=dict(color=c["reddish_purple"], dash="dash"),
                showlegend=False, legendgroup="Vddot"
            ), row=row, col=col)
            
            if not np.isnan(T_FT):
                fig.add_trace(
                            go.Scatter(
                                x=[T_FT, T_FT],
                                y=[-alpha_FT, alpha_FT],  # set according to the subplot y-range
                                mode="lines",
                                line=dict(color=c["vermillion"]),
                                name="T_FT",
                                legendgroup="Vddot"
                            ),
                            row=row,
                            col=col
                        )


    def plot_idot(row, col, zoom=False):
        fig.add_trace(go.Scatter(
            x=TVC_table["Time_s"], y=TVC_table["I_dot_five_kHz"],
            mode="lines", line=dict(color=c["black"], width=1),
            name="dI/dt", legendgroup="Idot"
        ), row=row, col=col)
        if zoom:
            fig.add_vline(x=actual_transition_time, line=dict(color=c["blue"], dash="dash"),
                          row=row, col=col)

    # --- Combine all 5 rows × 2 columns ---
    funcs = [plot_membrane, plot_current, plot_vdot, plot_vddot, plot_idot]

    for i, func in enumerate(funcs, start=1):
        func(i, 1, zoom=False)
        add_zoom_rectangle(fig, i, 1)
        func(i, 2, zoom=True)
        fig.update_xaxes(range=style["xlim_right"], row=i, col=2)
        fig.update_yaxes(title_text=y_labels[i-1], row=i, col=1)

    # --- Global layout ---
    fig.update_layout(
        height=1200, width=1400,
        template="simple_white",
        font=dict(size=style["font_size"]),
        legend=dict(
            orientation="v",
            x=1.05, y=1,
            font=dict(size=style["legend_fontsize"])
        ),
        margin=dict(l=60, r=200, t=80, b=60),
    )
    
    

    #fig.show()
    # --- Export if requested ---
    if saving_path:
        fig.write_html(saving_path.replace(".pdf", ".html"))
        fig.write_image(saving_path.replace(".pdf", ".svg"))
        fig.write_image(saving_path)
    if return_plot == True:
        return fig
    else:
        fig.show()
    


#%% TVC_plots

def get_TVC_spike_features_data(
                                Full_TVC_table,
                                Full_SF_dict_table,
                                Sweep_info_table,
                                current_sweep,
                                BE,
                                BE_correction,
                                superimpose_BE_correction,
                                color_dict
                            ):

    # --- Filter TVC table ---
    # LG get_filtered_TVC_table -> get_sweep_TVC_table
    current_TVC_table = ordifunc.get_sweep_TVC_table(
        Full_TVC_table, current_sweep
    )

    # --- Create full SF table with or without BE correction ---
    Full_SF_table = sp_an.create_Full_SF_table(
        Full_TVC_table,
        Full_SF_dict_table,
        cell_sweep_info_table=Sweep_info_table,
        BE_correct=BE_correction,
    )

    # --- Apply BE correction to membrane potential ---
    if BE_correction and not np.isnan(BE):
        current_TVC_table = current_TVC_table.copy()
        current_TVC_table["Membrane_potential_mV"] = (
            current_TVC_table["Membrane_potential_mV"]
            - BE * current_TVC_table["Input_current_pA"]
        )

    # --- For optional superimposed original trace ---
    current_TVC_table_original = None
    if BE_correction and superimpose_BE_correction:
        # LG get_filtered_TVC_table -> get_sweep_TVC_table
        current_TVC_table_original = ordifunc.get_sweep_TVC_table(
            Full_TVC_table, current_sweep
        )

    # --- Spike Feature table for this sweep ---
    current_sweep_SF_table = Full_SF_table.loc[current_sweep, "SF"].copy()

    # --- Clean / round numeric columns ---
    if not current_sweep_SF_table.empty:
        for col in current_sweep_SF_table.columns:
            if col != "Time_s" and np.issubdtype(
                current_sweep_SF_table[col].dtype, np.floating
            ):
                current_sweep_SF_table[col] = current_sweep_SF_table[col].round(2)

        # Round Time_s except for spike width rows
        mask = current_sweep_SF_table["Feature"] != "Spike_width_at_half_heigth"
        current_sweep_SF_table.loc[mask, "Time_s"] = (
            current_sweep_SF_table.loc[mask, "Time_s"].round(2)
        )

        # Reorder
        current_sweep_SF_table = current_sweep_SF_table[
            [
                "Feature",
                "Spike_index",
                "Time_s",
                "Membrane_potential_mV",
                "Input_current_pA",
                "Potential_first_time_derivative_mV/s",
                "Potential_second_time_derivative_mV/s/s",
            ]
        ].sort_values("Spike_index")

    # --- Build and return dictionary ---
    return {
        "current_TVC_table": current_TVC_table,
        "current_TVC_table_original": current_TVC_table_original,
        "current_sweep_SF_table": current_sweep_SF_table,
        "BE_correction": BE_correction,
        "superimpose_BE_correction": superimpose_BE_correction,
        "color_dict": color_dict,
    }

def plot_TVC_spike_features_plotly(plot_dict, return_plot = False):

    current_TVC_table = plot_dict["current_TVC_table"]
    current_TVC_table_original = plot_dict["current_TVC_table_original"]
    current_sweep_SF_table = plot_dict["current_sweep_SF_table"]
    BE_correction = plot_dict["BE_correction"]
    superimpose_BE_correction = plot_dict["superimpose_BE_correction"]
    color_dict = plot_dict["color_dict"]

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Membrane Potential plot", "Input Current plot"),
        vertical_spacing=0.1,
    )

    # --- Main membrane potential trace ---
    name = (
        "Membrane potential BE Corrected"
        if BE_correction
        else "Membrane potential"
    )

    fig.add_trace(
        go.Scatter(
            x=current_TVC_table["Time_s"],
            y=current_TVC_table["Membrane_potential_mV"],
            mode="lines",
            name=name,
            line=dict(
                color=color_dict["Membrane potential"], width=1
            ),
        ),
        row=1,
        col=1,
    )

    # --- Optional original trace ---
    if BE_correction and superimpose_BE_correction and current_TVC_table_original is not None:
        fig.add_trace(
            go.Scatter(
                x=current_TVC_table_original["Time_s"],
                y=current_TVC_table_original["Membrane_potential_mV"],
                mode="lines",
                name="Membrane potential original",
                line=dict(
                    color=color_dict["Membrane potential original"]
                ),
            ),
            row=1,
            col=1,
        )

    # --- Spike Features ---
    if not current_sweep_SF_table.empty:
        for feature in current_sweep_SF_table["Feature"].unique():
            if feature in ["Spike_heigth", "Spike_width_at_half_heigth"]:
                continue

            subset = current_sweep_SF_table[
                current_sweep_SF_table["Feature"] == feature
            ]

            fig.add_trace(
                go.Scatter(
                    x=subset["Time_s"],
                    y=subset["Membrane_potential_mV"],
                    mode="markers",
                    name=feature,
                    marker=dict(color=color_dict[feature]),
                ),
                row=1,
                col=1,
            )

    # --- Input current ---
    fig.add_trace(
        go.Scatter(
            x=current_TVC_table["Time_s"],
            y=current_TVC_table["Input_current_pA"],
            mode="lines",
            name="Input current",
            line=dict(
                color=color_dict["Input_current_pA"], width=1
            ),
        ),
        row=2,
        col=1,
    )

    # --- Layout ---
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="TVC and SF Plots",
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Time_s", row=2, col=1)
    fig.update_yaxes(title_text="Membrane_potential_mV", row=1, col=1)
    fig.update_yaxes(title_text="Input_current_pA", row=2, col=1)

    if return_plot == True:
        return fig
    fig.show()


#%% Linear analysis


def plot_estimate_IR_RP_choice(plot_dict, sampling_step = 5, plot_type = "matplotlib"):
    
    if plot_type == "matplotlib":
        fig = plot_estimate_input_resistance_and_resting_potential_matplotlib(plot_dict, 
                                                                              sampling_step=sampling_step, 
                                                                              return_plot=True, 
                                                                              saving_path=None)
    elif plot_type == "plotly":
        fig = plot_estimate_input_resistance_and_resting_potential_plotly(plot_dict, 
                                                                              sampling_step=sampling_step, 
                                                                              return_plot=True)
        
    return fig
        
def plot_estimate_input_resistance_and_resting_potential_plotly(
        plot_dict, sampling_step=5, return_plot=False
    ):
    """
    Plot function of IR and Vrest analysis (clean Plotly version).
    """

    # Extract data
    Full_TVC_table = plot_dict["Full_TVC_table"]
    Pre = plot_dict["Pre_stim_start_table"]
    Dur = plot_dict["During_stim_table"]
    Post = plot_dict["Post_stim_end_table"]

    IR = plot_dict["IR"]
    V_rest = plot_dict["V_rest"]
    R2 = plot_dict["R2"]
    current_step = plot_dict["current_step"]

    # Downsample visible traces
    Pre_s = Pre.iloc[::sampling_step, :]
    Dur_s = Dur.iloc[::sampling_step, :]
    Post_s = Post.iloc[::sampling_step, :]

    # Combined fitted table
    Full_fitted = pd.concat([Pre, Dur, Post], ignore_index=True)

    # ----- Subplot layout -----
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.06,
        column_widths=[0.68, 0.32],
        specs=[
            [{}, {"rowspan": 2}],
            [{}, None],
        ],
    )

    # -------------------------------------------------------------------------
    # Panel 1 — Membrane potential vs time
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=Full_TVC_table["Time_s"],
            y=Full_TVC_table["Membrane_potential_mV"],
            mode="lines",
            line=dict(color="black", width=1),
            hovertemplate="Time: %{x:.4f}s<br>Vm: %{y:.2f} mV",
            name="Full trace Vm",
        ),
        row=1, col=1,
    )

    # Red overlays (Pre/Dur/Post)
    for tb in [Pre_s, Dur_s, Post_s]:
        fig.add_trace(
            go.Scatter(
                x=tb["Time_s"],
                y=tb["Membrane_potential_mV"],
                mode="lines",
                line=dict(color="red", width=2),
                hovertemplate="Time: %{x:.4f}s<br>Vm: %{y:.2f} mV",
                name="Fitted segment Vm",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # -------------------------------------------------------------------------
    # Panel 2 — Input current vs time
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=Full_TVC_table["Time_s"],
            y=Full_TVC_table["Input_current_pA"],
            mode="lines",
            line=dict(color="black", width=1),
            hovertemplate="Time: %{x:.4f}s<br>I: %{y:.2f} pA",
            name="Full trace I",
        ),
        row=2, col=1,
    )

    for tb in [Pre_s, Dur_s, Post_s]:
        fig.add_trace(
            go.Scatter(
                x=tb["Time_s"],
                y=tb["Input_current_pA"],
                mode="markers",
                marker=dict(color="red", size=5),
                hovertemplate="Time: %{x:.4f}s<br>I: %{y:.2f} pA",
                name="Fitted segment I",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # -------------------------------------------------------------------------
    # Panel 3 — IV scatter + fit
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=Full_fitted["Input_current_pA"],
            y=Full_fitted["Membrane_potential_mV"],
            mode="markers",
            marker=dict(color="red", size=5),
            hovertemplate="I: %{x:.2f} pA<br>Vm: %{y:.2f} mV",
            name="I–V data",
        ),
        row=1, col=2,
    )

    # Regression fit line
    line_dash = "dash" if (current_step < 40.0 or R2 < 0.8) else "solid"

    x_range = np.linspace(
        Full_fitted["Input_current_pA"].min(),
        Full_fitted["Input_current_pA"].max(),
        100
    )
    y_range = V_rest + IR * x_range

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            line=dict(color="blue", width=2, dash=line_dash),
            hovertemplate=(
                f"IR = {IR:.3f} GΩ<br>"
                f"Vrest = {V_rest:.2f} mV<br>"
                f"R² = {R2:.3f}<br>"
                f"Step = {current_step:.1f} pA"
            ),
            name="Linear fit",
        ),
        row=1, col=2,
    )

    # -------------------------------------------------------------------------
    # Layout styling
    # -------------------------------------------------------------------------
    fig.update_layout(
        height=780,
        width=900,
        title=dict(text="Input Resistance and Resting Potential", x=0.5),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        margin=dict(l=70, r=40, t=60, b=50),
        showlegend=False,    # You can switch this ON if you want
    )

    # Axes styling
    fig.update_xaxes(
        showgrid=True, gridcolor="#E5E5E5",
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#E5E5E5",
        zeroline=False
    )

    # Axis labels
    fig.update_yaxes(title_text="Membrane potential (mV)", row=1, col=1)
    fig.update_yaxes(title_text="Input current (pA)", row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig.update_xaxes(title_text="Input current (pA)", row=1, col=2)
    fig.update_yaxes(title_text="Membrane potential (mV)", row=1, col=2)

    # Output behavior (Shiny-compatible)
    if return_plot:
        return fig
    else:
        fig.show()

        
        
   

def plot_estimate_input_resistance_and_resting_potential_matplotlib(
        plot_dict, sampling_step=5, return_plot=False, saving_path=None):

    # Extract data
    Full_TVC_table = plot_dict["Full_TVC_table"]
    Pre_stim_start_table = plot_dict["Pre_stim_start_table"]
    During_stim_table = plot_dict["During_stim_table"]
    Post_stim_end_table = plot_dict['Post_stim_end_table']
    IR = plot_dict['IR']
    V_rest = plot_dict['V_rest']
    R2 = plot_dict['R2']
    current_step = plot_dict['current_step']

    # Color-blind friendly palette (Okabe–Ito)
    col_black = "black"
    col_highlight = "#D55E00"   # orange-red
    col_fit = "#0072B2"         # blue

    # Subsample highlight segments
    Pre = Pre_stim_start_table.iloc[::sampling_step, :]
    Dur = During_stim_table.iloc[::sampling_step, :]
    Post = Post_stim_end_table.iloc[::sampling_step, :]

    # Concatenate all fitted points
    Full_fitted_table = pd.concat([Pre_stim_start_table,
                                   During_stim_table,
                                   Post_stim_end_table],
                                   ignore_index=True)

    # ------------------------------
    # Publication style
    # ------------------------------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ------------------------------
    # Figure Layout
    # ------------------------------
    figsize = (9,6)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[0.7, 0.3],
        height_ratios=[1, 1],
        wspace=0.25,
        hspace=0.25
    )

    ax1 = fig.add_subplot(gs[0, 0])     # Vm vs time
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Current vs time
    ax3 = fig.add_subplot(gs[:, 1])     # I–V curve + fit

    # ----------------------------------------------------------
    # Plot 1: Membrane potential vs Time
    # ----------------------------------------------------------
    ax1.plot(Full_TVC_table["Time_s"],
             Full_TVC_table["Membrane_potential_mV"],
             color=col_black, lw=1)

    for table in [Pre, Dur, Post]:
        ax1.plot(table["Time_s"],
                 table["Membrane_potential_mV"],
                 color=col_highlight, lw=1)

    ax1.set_ylabel("Membrane Potential (mV)")

    # ----------------------------------------------------------
    # Plot 2: Input current vs Time
    # ----------------------------------------------------------
    ax2.plot(Full_TVC_table["Time_s"],
             Full_TVC_table["Input_current_pA"],
             color=col_black, lw=1)

    for table in [Pre, Dur, Post]:
        ax2.plot(table["Time_s"], table["Input_current_pA"],
                 linestyle="none", marker="o",
                 color=col_highlight, markersize=3)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Input Current (pA)")

    # ----------------------------------------------------------
    # Plot 3: I–V scatter + linear fit
    # ----------------------------------------------------------
    # Scatter
    ax3.scatter(Full_fitted_table["Input_current_pA"],
                Full_fitted_table["Membrane_potential_mV"],
                color=col_highlight, s=10, alpha=0.8,
                label="Linear fit data")

    # Linear fit
    x_range = np.linspace(Full_fitted_table["Input_current_pA"].min(),
                          Full_fitted_table["Input_current_pA"].max(), 200)
    y_range = V_rest + IR * x_range

    linestyle = "--" if (current_step < 40 or R2 < 0.8) else "-"

    ax3.plot(x_range, y_range,
             color=col_fit, lw=1.5, linestyle=linestyle,
             label="Linear fit")

    ax3.set_xlabel("Input Current (pA)")
    ax3.set_ylabel("Membrane Potential (mV)")

    # Summary box
    # summary_text = (
    #     f"IR = {IR:.3f} GΩ\n"
    #     f"Vrest = {V_rest:.2f} mV\n"
    #     f"R² = {R2:.3f}\n"
    #     f"Step = {current_step:.1f} pA"
    # )

    # ax3.text(0.05, 0.95, summary_text,
    #          transform=ax3.transAxes,
    #          va="top", ha="left",
    #          fontsize=9,
    #          bbox=dict(facecolor="white", edgecolor="black", alpha=0.8))

    # ----------------------------------------------------------
    # Aesthetics
    # ----------------------------------------------------------
    for ax in [ax1, ax2, ax3]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ----------------------------------------------------------
    # Figure-level legend (below the figure)
    # ----------------------------------------------------------
    handles, labels = ax3.get_legend_handles_labels()

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False
    )

    # Make space at the bottom for the legend
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if saving_path:
        
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi= 300)
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")
    if return_plot:
        return fig
    else:
        plt.show()
        
        


def plot_fit_membrane_time_cst_matplotlib(plot_dict, return_plot=False, saving_path = None):
    """
    Matplotlib version with two horizontal panels:
    - Left: full trace
    - Right: zoomed-in fitting interval
    """

    # ---------------------- Parameters dict ----------------------
    s = {
        "figsize": (12, 5),
        "exp_color": "black",
        "fit_color": "#D62728",
        "fit_linewidth": 2.5,
        "exp_linewidth": 1.5,
        "fit_alpha": 0.25,
        "fit_interval_color": "lightgray",
        "grid_color": "#E6E6E6",
        "font_family": "Arial",
        "font_size": 12,
        "legend_fontsize": 12,
        "legend_loc": "lower center",
        "legend_ncol": 3,
        "margin_bottom": 0.1
    }

    # ---------------------- Extract data ----------------------
    TVC = plot_dict["TVC_table"]
    Sim = plot_dict["Sim_table"]

    A = plot_dict["A"]
    tau = plot_dict["tau"]
    C = plot_dict["C"]
    RMSE = plot_dict["RMSE"]

    start_time = plot_dict["start_time"]
    end_time = plot_dict["end_time"]

    line_style = "-" if RMSE / A <= 0.1 else "--"

    # ---------------------- Figure and axes ----------------------
    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=s["figsize"], sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )

    # ---------------------- Left panel: full trace ----------------------
    ax = ax_full
    # Experimental trace
    ax.plot(
        TVC["Time_s"], TVC["Membrane_potential_mV"],
        color=s["exp_color"], lw=s["exp_linewidth"], label="Experimental data"
    )
    # Simulation fit
    ax.plot(
        Sim["Time_s"], Sim["Membrane_potential_mV"],
        color=s["fit_color"], lw=s["fit_linewidth"], linestyle=line_style,
        label="Fit"
    )
    # Fitting interval
    ax.axvspan(
        start_time, end_time,
        color=s["fit_interval_color"], alpha=s["fit_alpha"],
        label="Fitting interval"
    )

    ax.set_xlabel("Time (s)", fontsize=s["font_size"], family=s["font_family"])
    ax.set_ylabel("Membrane Potential (mV)", fontsize=s["font_size"], family=s["font_family"])
    ax.grid(True, color=s["grid_color"], linestyle=":", linewidth=0.8)
    ax.set_facecolor("white")

    # ---------------------- Right panel: zoom ----------------------
    ax = ax_zoom
    # Experimental trace
    ax.plot(
        TVC["Time_s"], TVC["Membrane_potential_mV"],
        color=s["exp_color"], lw=s["exp_linewidth"]
    )
    # Simulation fit
    ax.plot(
        Sim["Time_s"], Sim["Membrane_potential_mV"],
        color=s["fit_color"], lw=s["fit_linewidth"], linestyle=line_style
    )
    # Shaded interval
    ax.axvspan(start_time, end_time, color=s["fit_interval_color"], alpha=s["fit_alpha"])

    ax.set_xlim(start_time-0.01, end_time+0.01)
    ax.set_xlabel("Time (s)", fontsize=s["font_size"], family=s["font_family"])
    ax.grid(True, color=s["grid_color"], linestyle=":", linewidth=0.8)
    ax.set_facecolor("white")

    
    # ---------------------- Legend ----------------------
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc=s["legend_loc"],
        fontsize=s["legend_fontsize"],
        ncol=s["legend_ncol"],
        frameon=True,
        bbox_to_anchor=(0.5, -s["margin_bottom"]),
        fancybox=True
    )

    fig.tight_layout()
    
    if saving_path:
        
        plt.savefig(saving_path, format="pdf", bbox_inches="tight", dpi= 300)
        plt.savefig(saving_path.replace(".pdf", ".svg"), format="svg", bbox_inches="tight")

    if return_plot:
        return fig, (ax_full, ax_zoom)
    else:
        plt.show()
        
        
def plot_fit_membrane_time_cst_plotly(plot_dict, return_plot=False):
    """
    Nicer Plotly version of fit plot for membrane time constant estimation,
    with a legend for the fitting interval.
    """

    # Extract values
    TVC = plot_dict["TVC_table"]
    Sim = plot_dict["Sim_table"]

    A = plot_dict["A"]
    tau = plot_dict["tau"]
    C = plot_dict["C"]
    RMSE = plot_dict["RMSE"]

    start_time = plot_dict["start_time"]
    end_time = plot_dict["end_time"]

    # Determine dashed/solid fit
    line_style = "solid" if RMSE / A <= 0.1 else "dash"

    # Figure
    fig = go.Figure()

    # -------------------------------------------------------------
    # Experimental trace
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=TVC["Time_s"],
            y=TVC["Membrane_potential_mV"],
            mode="lines",
            line=dict(color="black", width=1.5),
            name="Experimental data",
            hovertemplate="Time: %{x:.4f}s<br>Vm: %{y:.2f} mV",
        )
    )

    # -------------------------------------------------------------
    # Simulation fit
    # -------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=Sim["Time_s"],
            y=Sim["Membrane_potential_mV"],
            mode="lines",
            line=dict(color="#D62728", width=2.5, dash=line_style),
            name="Simulation fit",
            hovertemplate=(
                "Time: %{x:.4f}s<br>Vm: %{y:.2f} mV"
                f"<br><br><b>A</b> = {A:.3f} mV"
                f"<br><b>τ</b> = {tau:.4f} s"
                f"<br><b>C</b> = {C:.3f} mV"
                f"<br><b>RMSE</b> = {RMSE:.3f}"
            ),
        )
    )

    # -------------------------------------------------------------
    # Fitting interval
    # -------------------------------------------------------------
    fig.add_vrect(
        x0=start_time,
        x1=end_time,
        fillcolor="lightgray",
        opacity=0.25,
        layer="below",
        line_width=0,
        # no name -> won't appear in legend
    )

    # Trick for legend: add invisible scatter trace with same color
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="lightgray"),
            showlegend=True,
            name="Fitting interval"
        )
    )

    
    # -------------------------------------------------------------
    # Axes
    # -------------------------------------------------------------
    fig.update_xaxes(
        title_text="Time (s)",
        range=[start_time - 0.01, end_time + 0.01],
        showgrid=True, gridcolor="#E6E6E6",
        zeroline=False,
    )

    fig.update_yaxes(
        title_text="Membrane potential (mV)",
        showgrid=True, gridcolor="#E6E6E6",
        zeroline=False,
    )

    # -------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------
    fig.update_layout(
        height=600,
        width=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=120, t=60, b=50),
        
            legend=dict(
            orientation="h",        # horizontal legend
            yanchor="top",           # anchor to the top of the legend box
            y=-0.2,                  # negative y moves it below the plot
            xanchor="center",        # anchor to center horizontally
            x=0.5,                   # centered
            bgcolor="rgba(255,255,255,0.7)",  # semi-transparent background
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12)
        ),

        font=dict(size=14),
    )

    if return_plot:
        return fig
    else:
        fig.show()

