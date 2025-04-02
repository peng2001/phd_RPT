import sys
sys.path.append('lib')  # Adjust the path accordingly
# main.ipynb
# Define your directory and calibration file paths
directory = 'data/Sabines Rig data/2C 25 Deg C'
calibration_data_file_path = 'Heat_flux_sensors_calibration_sabine.csv'

import os
import pandas as pd
import numpy as np

# Import functions from modules
from setup_sabine import (
    calculate_sensitivity,
    calculate_heatflux_vectorized,
    read_peltier_control_data,
    read_cycler_data,
    read_and_calibrate_heatflux
)
from calc_functions_sabine import (
    average_heatflux_during_cycles,
    get_temperature_during_cycles,
    calculate_total_heat_flux
)
from plot_functions_sabine import (
    plot_heatflux_and_current,
    plot_average_heatflux,
    plot_temperature_vs_heatflux,
    plot_heatflux_and_current_offset
)

# Other imports
from scipy.ndimage import gaussian_filter1d
import altair as alt
alt.data_transformers.disable_max_rows()

from colours_sabine import (
    get_grouped_colors,
    get_grouped_colors_averaged,
    get_grouped_colors_meanheatflux
)

# Get the color mappings
grouped_colors = get_grouped_colors()
grouped_colors_averaged = get_grouped_colors_averaged()
grouped_colors_meanheatflux = get_grouped_colors_meanheatflux()

# Read data
peltier_control_data = read_peltier_control_data(directory)
cycler_data = None
heatflux_data = None

# Process files in directory
for file in os.listdir(directory):
    if file.endswith('.txt'):
        txt_file_path = os.path.join(directory, file)
        cycler_data, metadata = read_cycler_data(txt_file_path)
    elif 'heatflux' in file:
        sensor_data_file_path = os.path.join(directory, file)
        heatflux_data = read_and_calibrate_heatflux(
            sensor_data_file_path, calibration_data_file_path, peltier_control_data, cycler_data
        )
    

# Average heat flux during cycles
cycle_values = list(range(25, 41))
if heatflux_data is not None and cycler_data is not None:
    averaged_heatflux = average_heatflux_during_cycles(heatflux_data, cycler_data, cycle_values)
    temperature_cycle = get_temperature_during_cycles(heatflux_data, cycler_data, cycle_values, peltier_control_data)
    total_heat_flux = calculate_total_heat_flux(averaged_heatflux, cycle_values)
    print("Total Heat flux: "+str(total_heat_flux))

    # plot_heatflux_and_current_offset(heatflux_data, cycler_data, grouped_colors_averaged)
    # Prepare data for plotting temperature vs heat flux
    temperatures = []
    mean_heatfluxes = []
    sensor_names = []

    for i, column in enumerate(averaged_heatflux.columns):
        if '_offset_smooth' in column:
            temperatures.append(temperature_cycle[i])
            mean_heatfluxes.append(averaged_heatflux[column].mean())
            sensor_names.append(column[8:14])

    melted_temperature_heatflux = pd.DataFrame({
        'Temperature': temperatures,
        'Mean_Heatflux': mean_heatfluxes,
        'Sensor_Name': sensor_names
    })

    # Plot averaged heat flux
    #plot_average_heatflux(averaged_heatflux, grouped_colors_averaged)

    # Plot temperature vs mean heat flux
    #plot_temperature_vs_heatflux(melted_temperature_heatflux, grouped_colors_meanheatflux)

    # Calculate total heat flux
    #top_side, bottom_side = calculate_total_heat_flux(averaged_heatflux, cycle_values)

averaged_heatflux = average_heatflux_during_cycles(heatflux_data, cycler_data, cycle_values)
temperature_cycle = get_temperature_during_cycles(heatflux_data, cycler_data, cycle_values, peltier_control_data)
top_side, bottom_side = calculate_total_heat_flux(averaged_heatflux, cycle_values)
print(np.sum(top_side))
print(np.sum(bottom_side))
print("Total Heat flux: "+str(abs(np.sum(top_side)+np.sum(bottom_side))))