import os
import pandas as pd
import numpy as np
from io import StringIO
heatflux_file = 'data/J1PA aged cell 33/50soc/J1PAaged33_heatgen_50soc_1.5C_25deg.csv'
T_S = 25 # deg C, final temperature

heat_flux_sign = -1 # 1 or -1; 1 means heat flux entering the cell is positive; -1 means heat flux entering cell is negative
L = 0.0057 # meters, 1/2 cell thickness (L)
deltaT = 5  # degrees C, magnitude of step change
fitting_time_skip = 30 # seconds, integer, ignore first points because of overshoot
cell_mass_density = 2465 # kg/m^3
S_area = 0.032 # meters squared, cell surface area per face, used for heat gen calculation; 0.1 * 0.32

def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    return U.astype(float) / S


sensor_data = pd.read_csv(heatflux_file, decimal=",")
sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')
sensor_data.columns = sensor_data.columns.str.replace(' Ave. \(µV\)', '', regex=True)

# Read the calibration data CSV file
calibration_data_file_path = 'Heat_flux_sensors_calibration_Sabine.csv'

calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

#T_S=25
HeatfluxData=pd.DataFrame()
HeatfluxData['time'] = pd.to_datetime(sensor_data['Unnamed: 0'], unit='s')

for column in sensor_data.columns[1:]:
    sensor_id = '003066-' +column[-3:]
    # print(T_S)
    if np.isnan(T_S):
        print('Temperature is not a number for ' + column)
    calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
    S_0 = calibration_row['Sensitivity S0']
    S_C = calibration_row['Correction factor Sc']
    S = calculate_sensitivity(S_0, S_C, T_S)

    HeatfluxData['HeatFlux' + column] = calculate_heatflux_vectorized(sensor_data[column], S)

# HeatfluxData.drop(columns=['HeatFluxA0_C05', 'HeatFluxC0_D01', 'HeatFluxD0_D07', 'HeatFluxB0_C13'], inplace=True)
HeatfluxData["average_heatflux"] = HeatfluxData.iloc[:, 1:].mean(axis=1)
HeatfluxData["time_elapsed"] = HeatfluxData['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
HeatfluxData = HeatfluxData.dropna()

# graphing data
import matplotlib.pyplot as plt
# Filter columns that start with 'HeatFlux'
heatflux_columns = [col for col in HeatfluxData.columns if col.startswith('HeatFlux')]
# Plot each HeatFlux column
plt.figure(figsize=(12, 6))
for col in heatflux_columns:
    plt.plot(HeatfluxData['time_elapsed'], HeatfluxData[col], label=col)

plt.xlabel("Time Elapsed (s)")
plt.ylabel("Heat Flux (W/m²)")
plt.title("Heat Flux vs Time Elapsed")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()