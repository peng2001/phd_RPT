heatflux_file = 'data/test_J1_cell/...'

import os
import pandas as pd
import numpy as np
from io import StringIO
import toml

T_S = 20 # deg C, final temperature
L = 0.0057 # meters, 1/2 cell thickness (L)
deltaT = 5  # degrees C, magnitude of step change
fitting_time_skip = 30 # seconds, integer, ignore first points because of overshoot
cell_mass_density = 2465 # kg/m^3


def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    return U.astype(float)*(1000*1000) / S


sensor_data = pd.read_csv(heatflux_file, decimal=",")
sensor_data = sensor_data[['Unnamed: 0', 'heatflux']]
# sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')
sensor_data.columns = sensor_data.columns.str.replace(r'\sflux\sLast\s\(V\)', '', regex=True)

# Read the calibration data CSV file
calibration_data_file_path = 'Heat_flux_sensors_calibration.csv'

calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

#T_S=25
HeatfluxData=pd.DataFrame()
HeatfluxData['time']=pd.to_datetime(sensor_data['Unnamed: 0'], format='%Y-%m-%dT%H:%M:%S%z')

for column in sensor_data.columns[1:]:
    if column.startswith("heatflux_"):
        sensor_id = column.split("heatflux_")[1]

        calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
        S_0 = calibration_row['Sensitivity S0']
        S_C = calibration_row['Correction factor Sc']
        S = calculate_sensitivity(S_0, S_C, T_S)

        HeatfluxData['HeatFlux' + column] = calculate_heatflux_vectorized(sensor_data[column], S)



HeatfluxData.time_elapsed = (HeatfluxData.time - HeatfluxData.time.iloc[0]).dt.total_seconds()
HeatfluxData.average_heatflux = HeatfluxData.iloc[:, 1:].mean(axis=1) # * tab area divided by cell cross section

dq_dt = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed) # Find time values where dq/dt > 100 time_values = t[dq_dt > 100]
jump_times = HeatfluxData.time_elapsed[dq_dt < -100]
filtered_times = []
previous_time = None
for time in jump_times:
    if previous_time is None or (time - previous_time > 100):
        filtered_times.append(time)
        previous_time = time
print("Times where the step change starts")
print(str(filtered_times))

# try:
#     del metadata_lines, data_lines, lines, line, data, data_section, key, txt_file_path, value
# except:
#     print('')
# del directory, file, filepath, files, prefix, variable_name, calibration_data, calibration_data_file_path, calibration_row, column, T_S, sensor_id, sensor_data_file_path, sensor_data, columns_to_keep, data_columns_Peltier
