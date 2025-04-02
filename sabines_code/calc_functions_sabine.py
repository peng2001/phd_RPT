import numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd

def average_heatflux_during_cycles(heatflux_data, cycler_data, cycle_values):
    """Averages heat flux during specified cycles."""
    index = (heatflux_data['time'].dt.tz_localize('UTC+01:00') - cycler_data['DPT Time'].iloc[0]).abs().idxmin()

    # Apply offset and smoothing
    for column in heatflux_data.columns:
        if column != 'time' and '_offset_smooth' not in column:
            offset = heatflux_data[column][index:index+100].mean()
            heatflux_data[column + '_offset_smooth'] = gaussian_filter1d(heatflux_data[column] - offset, sigma=2)

    # Initialize an empty list to store the results
    averaged_data = []

    # Loop over each specified cycle value
    for cycle_value in cycle_values:
        selected_cycle = cycler_data[cycler_data['Cycle C'] == cycle_value]
        if selected_cycle.empty:
            print(f'No cycle {cycle_value}')
            continue

        start_time = selected_cycle.iloc[0]['DPT Time']
        end_time = selected_cycle.iloc[-1]['DPT Time']

        period_data = heatflux_data[
            (heatflux_data['time'].dt.tz_localize('UTC+01:00') >= start_time) &
            (heatflux_data['time'].dt.tz_localize('UTC+01:00') < end_time)
        ]
        averages = period_data.filter(regex='_offset_smooth$').mean().to_dict()
        averages['start time average'] = start_time
        averages['end time average'] = end_time
        averages['Cycle Value'] = cycle_value
        averaged_data.append(averages)

    # Create a new DataFrame from the averaged data
    averaged_heatflux = pd.DataFrame(averaged_data)

    return averaged_heatflux

def get_temperature_during_cycles(heatflux_data, cycler_data, cycle_values, peltier_control_data):
    """Gets temperature during specified cycles."""
    selected_cycle = cycler_data[cycler_data['Cycle C'] == cycle_values[0]]
    start_time = selected_cycle.iloc[0]['DPT Time']
    sensor_index = (heatflux_data['time'].dt.tz_localize('UTC+01:00') - start_time).abs().idxmin()
    temperature_cycle = []

    for column in heatflux_data.columns:
        if column != 'time':
            prefix = column[8]
            peltier_df = peltier_control_data.get(prefix)
            if peltier_df is not None:
                cycle_temp = peltier_df['Current_temp_' + column[8:10]].iloc[sensor_index]
            else:
                cycle_temp = np.nan
                print(f"Column prefix {prefix} not recognized for column {column}")

            temperature_cycle.append(cycle_temp)

    return temperature_cycle

def calculate_total_heat_flux(averaged_heatflux, cycle_values):
    """Calculates total heat flux on top and bottom sides."""
    area = 0.04571 * 0.0505  # Area per Peltier element
    top_side = []
    bottom_side = []

    for cycle in cycle_values:
        idx = cycle - cycle_values[0]
        top_flux = (
            averaged_heatflux['HeatFluxD0_D07_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxC0_D01_offset_smooth'].iloc[idx] +
            2 * averaged_heatflux['HeatFluxD1_D08_offset_smooth'].iloc[idx] +
            2 * averaged_heatflux['HeatFluxD2_D11_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxD3_D12_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxC3_D03_offset_smooth'].iloc[idx] +
            2 * averaged_heatflux['HeatFluxD4_D13_offset_smooth'].iloc[idx] +
            2 * averaged_heatflux['HeatFluxD5_D14_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxD6_D16_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxC6_D04_offset_smooth'].iloc[idx]
        ) * area
        top_side.append(top_flux)

        bottom_flux = (
            averaged_heatflux['HeatFluxB0_C13_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxB6_C14_offset_smooth'].iloc[idx] +
            5 * averaged_heatflux['HeatFluxA2_C07_offset_smooth'].iloc[idx] +
            5 * averaged_heatflux['HeatFluxA4_C08_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxA0_C05_offset_smooth'].iloc[idx] +
            averaged_heatflux['HeatFluxA6_C11_offset_smooth'].iloc[idx]
        ) * area
        bottom_side.append(bottom_flux)

    top_side = np.array(top_side)
    bottom_side = np.array(bottom_side)
    print("Top Side Total Heat Flux:", top_side)
    print("Bottom Side Total Heat Flux:", bottom_side)

    return top_side, bottom_side