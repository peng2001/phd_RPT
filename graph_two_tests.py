import os
import pandas as pd
import numpy as np
from io import StringIO
import toml

heatflux_file_1 = 'data/test_J1/J1_throughplane_15-20.csv'
heatflux_file_2 = 'data/test_J1PA/J1PA_throughplane_15-20.csv'
xlimit = [6000,8000]
align_data = True

T_S = 20 # deg C, final temperature

heat_flux_sign = -1 # 1 or -1; 1 means heat flux entering the cell is positive; -1 means heat flux entering cell is negative
L = 0.0057 # meters, 1/2 cell thickness (L)
deltaT = 5  # degrees C, magnitude of step change
fitting_time_skip = 50 # seconds, integer, ignore first points because of overshoot
cell_mass_density = 2465 # kg/m^3
S_area = 0.032 # meters squared, cell surface area per face, used for heat gen calculation; 0.1 * 0.32


def calculate_sensitivity(S_0, S_C, T_S, T_0=22.5):
    return S_0 + (T_S - T_0) * S_C

def calculate_heatflux_vectorized(U, S):
    return U.astype(float) / S

def setup(filename):
    sensor_data = pd.read_csv(filename, decimal=",")
    # sensor_data = sensor_data.apply(pd.to_numeric, errors='coerce')
    sensor_data.columns = sensor_data.columns.str.replace(r'\sLast\s\(µV\)', '', regex=True)

    # Read the calibration data CSV file
    calibration_data_file_path = 'Heat_flux_sensors_calibration.csv'

    calibration_data = pd.read_csv(calibration_data_file_path, sep=';')
    calibration_data['Sensitivity S0'] = pd.to_numeric(calibration_data['Sensitivity S0'], errors='coerce')
    calibration_data['Correction factor Sc'] = pd.to_numeric(calibration_data['Correction factor Sc'], errors='coerce')

    #T_S=25
    HeatfluxData=pd.DataFrame()
    HeatfluxData['time'] = pd.to_datetime(sensor_data['Unnamed: 0'], format='%H:%M:%S').dt.time

    for column in sensor_data.columns[1:]:
        if column.startswith("heatflux_"):
            sensor_id = column.split("heatflux_")[1]

            calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
            S_0 = calibration_row['Sensitivity S0']
            S_C = calibration_row['Correction factor Sc']
            S = calculate_sensitivity(S_0, S_C, T_S)

            HeatfluxData[column] = calculate_heatflux_vectorized(sensor_data[column], S)

    HeatfluxData["average_heatflux"] = HeatfluxData.iloc[:, 1:].mean(axis=1)
    HeatfluxData["time_elapsed"] = HeatfluxData['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    return HeatfluxData

def align_max_heatflux(df1, df2):
    # Find the time when maximum average_heatflux occurs for each dataset
    max_time_1 = df1.loc[df1["average_heatflux"].idxmax(), "time_elapsed"]
    max_time_2 = df2.loc[df2["average_heatflux"].idxmax(), "time_elapsed"]
    # Calculate the time shift needed to align the max heat flux values
    time_shift = max_time_1 - max_time_2

    # Shift the time data of the second dataset to align with the first dataset
    df2['time_elapsed'] = df2['time_elapsed'] + time_shift
    return df2

HeatfluxData_1 = setup(heatflux_file_1)
HeatfluxData_2 = setup(heatflux_file_2)
HeatfluxData_2["average_heatflux"] = np.multiply(HeatfluxData_2["average_heatflux"],-1)
if align_data:
    HeatfluxData_2 = align_max_heatflux(HeatfluxData_1, HeatfluxData_2)

# graphing data
import matplotlib.pyplot as plt
# Filter columns that start with 'HeatFlux'
plt.plot(HeatfluxData_1['time_elapsed'], HeatfluxData_1["average_heatflux"], label="J1")
plt.plot(HeatfluxData_2['time_elapsed'], HeatfluxData_2["average_heatflux"], label="J1PA")
    

plt.xlabel("Time Elapsed (s)")
plt.ylabel("Heat Flux (W/m²)")
plt.xlim(xlimit)
plt.title("Heat Flux vs Time Elapsed")
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

