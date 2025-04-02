import os
import pandas as pd
import numpy as np
from io import StringIO

# Constants
DATA_COLUMNS_PELTIER = [
    'Current_temp_A0', 'Current_temp_A1', 'Current_temp_A2', 'Current_temp_A3',
    'Current_temp_A4', 'Current_temp_A5', 'Current_temp_A6', 'Current_temp_A7',
    'Current_temp_B0', 'Current_temp_B1', 'Current_temp_B2', 'Current_temp_B3',
    'Current_temp_B4', 'Current_temp_B5', 'Current_temp_B6', 'Current_temp_Falz-',
    'Current_temp_C0', 'Current_temp_C1', 'Current_temp_C2', 'Current_temp_C3',
    'Current_temp_C4', 'Current_temp_C5', 'Current_temp_C6', 'Current_temp_C7_Ableiter+',
    'Current_temp_D0', 'Current_temp_D1', 'Current_temp_D2', 'Current_temp_D3',
    'Current_temp_D4', 'Current_temp_D5', 'Current_temp_D6', 'Current_temp_D7_Ableiter-',
    'time'
]

def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    """Calculates the sensitivity of the heat flux sensor."""
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    """Calculates heat flux using vectorized operations."""
    return U / S

def read_peltier_control_data(directory):
    """Reads and processes Peltier control data files."""
    files = os.listdir(directory)
    files = sorted(files, key=lambda x: (not 'peltier_control' in x, x))
    peltier_control_data = {}

    for file in files:
        if 'peltier_control' in file:
            prefix = file.split('_')[2]
            filepath = os.path.join(directory, file)

            df = pd.read_csv(filepath, sep=',')
            df['time'] = pd.to_datetime(df[' unix_time_stamp'], unit='s')
            columns_to_keep = [col for col in df.columns if col in DATA_COLUMNS_PELTIER]
            df = df[columns_to_keep]
            peltier_control_data[prefix] = df

    return peltier_control_data

def read_cycler_data(txt_file_path):
    """Reads and processes Cycler data from a .txt file."""
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    metadata_lines = []
    data_lines = []
    data_section = False

    for line in lines:
        if not data_section:
            metadata_lines.append(line.strip())
            if line.strip().startswith('Rec'):
                data_section = True
                data_lines.append(line.strip())
        else:
            data_lines.append(line.strip())

    # Process metadata
    metadata = {}
    for line in metadata_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()

    # Create DataFrame for data section
    data = StringIO('\n'.join(data_lines))
    cycler_data = pd.read_csv(data, sep='\t')

    cycler_data['MD'] = cycler_data['MD'].astype('category')
    cycler_data.loc[cycler_data['MD'] == 'D', 'Current'] *= -1
    cycler_data['DPT Time'] = pd.to_datetime(cycler_data['DPT Time'], format='%d-%b-%y %I:%M:%S %p') 
    cycler_data['DPT Time'] = cycler_data['DPT Time'].dt.tz_localize('Europe/London')

    return cycler_data, metadata

def read_and_calibrate_heatflux(sensor_data_file_path, calibration_data_file_path, peltier_control_data, cycler_data=None):
    """Reads and calibrates heat flux data."""
    sensor_data = pd.read_csv(sensor_data_file_path, decimal=",")
    sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')
    sensor_data.columns = sensor_data.columns.str.replace(' Ave. \(µV\)', '', regex=True)

    # Read the calibration data CSV file
    calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
    calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
    calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

    heatflux_data = pd.DataFrame()
    heatflux_data['time'] = pd.to_datetime(sensor_data['Unnamed: 0'], unit='s')

    # Determine SensorIndex
    if cycler_data is not None:
        cycler_data['DPT Time Adjust'] = cycler_data['DPT Time']  - pd.Timedelta(hours=1)
        sensor_index = (heatflux_data['time'].dt.tz_localize('Europe/London') - cycler_data['DPT Time Adjust'].iloc[0]).abs().idxmin()
    else:
        sensor_index = 1011  # Default index if CyclerData is not provided

    for column in sensor_data.columns[1:]:
        sensor_id = '003066-' + column[-3:]

        # Determine T_S based on the column prefix
        prefix = column[0]
        peltier_df = peltier_control_data.get(prefix)
        if peltier_df is not None:
            T_S = peltier_df['Current_temp_' + column[0:2]].iloc[sensor_index]
        else:
            T_S = np.nan
            print(f"Column prefix {prefix} not recognized for column {column}")

        if np.isnan(T_S):
            print(f'Temperature is not a number for {column}')

        # Get calibration data
        calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
        S_0 = calibration_row['Sensitivity S0']
        S_C = calibration_row['Correction factor Sc']
        S = calculate_sensitivity(S_0, S_C, T_S)

        heatflux_data['HeatFlux' + column] = calculate_heatflux_vectorized(sensor_data[column], S)

    return heatflux_data
