import os
import pandas as pd
import numpy as np
from io import StringIO
import copy

heatflux_file = 'data/Sabines Rig J1PA/50soc/J1PA_throughplane_20-25_sabine.csv'
T_S = 25 # deg C, final temperature
monte_carlo_n = 500
heat_flux_sign = -1 # 1 or -1; 1 means heat flux entering the cell is positive; -1 means heat flux entering cell is negative
fitting_time_skip = 30 # seconds, integer, ignore first points because of overshoot
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
HeatfluxData["time_elapsed"] -= HeatfluxData["time_elapsed"][0]
HeatfluxData = HeatfluxData.dropna()


def monte_carlo_heatflux_iteration(sensor_data, calibration_data, T_S, uncertainty):
    global HeatfluxData, L, deltaT, cell_mass_density, S_area, S_initial, S_final

    # Sample global parameters
    L = np.random.normal(loc=0.0057, scale=uncertainty["L"])
    deltaT = np.random.normal(loc=5, scale=uncertainty["deltaT"])
    cell_mass_density = np.random.normal(loc=2465, scale=uncertainty["cell_mass_density"])
    cell_length = np.random.normal(loc=0.320, scale=uncertainty["cell_length"])
    cell_width = np.random.normal(loc=0.100, scale=uncertainty["cell_width"])
    S_area = cell_length * cell_width

    # Store for reproducibility/debug
    sampled_constants = {
        "L": L,
        "deltaT": deltaT,
        "cell_mass_density": cell_mass_density,
        'cell_length': cell_length,
        'cell_width': cell_width,
        "S_area": S_area
    }

    # HeatfluxData initialization
    HeatfluxData = pd.DataFrame()
    HeatfluxData['time'] = pd.to_datetime(sensor_data['Unnamed: 0'], unit='s')

    for column in sensor_data.columns[1:]:
        sensor_id = '003066-' + column[-3:]
        calibration_row = calibration_data[calibration_data['serial number'] == sensor_id].iloc[0]
        S_0 = calibration_row['Sensitivity S0']
        S_C = calibration_row['Correction factor Sc']
        S = calculate_sensitivity(S_0, S_C, T_S)

        # Calculate and add Gaussian noise to the heat flux
        heatflux = calculate_heatflux_vectorized(sensor_data[column], S)
        heat_flux_std_dev = uncertainty["heat_flux_pct"] * np.abs(heatflux)
        noise = np.random.normal(0, heat_flux_std_dev)
        HeatfluxData['HeatFlux' + column] = heat_flux_sign * (heatflux + noise)

    HeatfluxData["average_heatflux"] = HeatfluxData.iloc[:, 1:].mean(axis=1)
    HeatfluxData["time_elapsed"] = HeatfluxData['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    HeatfluxData["time_elapsed"] -= HeatfluxData["time_elapsed"].iloc[0]
    HeatfluxData = HeatfluxData.dropna()

    # Needed for loss correction in `run_fitting`
    S_initial = calculate_sensitivity(S_0, S_C, 22.5)
    S_final = calculate_sensitivity(S_0, S_C, T_S)

    return HeatfluxData.copy(), sampled_constants



import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
import math

##########################################################
# finding start time of the model
heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed)
if heat_flux_sign == 1:
    indices_step = np.where(heatflux_slopes > 5)[0] # indices where the slope is greater than X
    target_temperature_reached_time = HeatfluxData.time_elapsed[HeatfluxData.average_heatflux.idxmax()]
elif heat_flux_sign == -1:
    indices_step = np.where(heatflux_slopes < -5)[0] # indices where the slope is less than X (greater than -X)
    target_temperature_reached_time = HeatfluxData.time_elapsed[HeatfluxData.average_heatflux.idxmin()]
else:
    raise ValueError("initial_value_guess in setup.py should be 1 or -1")
start_change_tempreature_time = HeatfluxData.time_elapsed[indices_step[0]]-2 # time where target temperature for peltiers is changed 
# target_temperature_reached_time = HeatfluxData.time_elapsed[indices_step[-1]] # time where the target temperature is reached


slopes_after_step = heatflux_slopes[indices_step[-1] + 20:] # only the data points after the step change
seconds_after_step = HeatfluxData.time_elapsed[indices_step[-1] + 20:]
moving_averages = np.convolve(slopes_after_step, np.ones(100)/100, mode='valid') # window size of 50, to get moving average of last 50 slopes
indices_steady = slopes_after_step[indices_step[-1] + 20:]
if heat_flux_sign == 1:
    indices_end = np.where(moving_averages > -0.001)[0] # indices where the slope is less than than Y (greater than negative y)
elif heat_flux_sign == -1:
    indices_end = np.where(moving_averages < 0.001)[0] # indices where the slope is greater than than Y
else:
    raise ValueError("initial_value_guess in setup.py should be 1 or -1")
steady_state_time = seconds_after_step.iloc[indices_end[0]]

start_time = target_temperature_reached_time # start time elapsed to fit equation, seconds, where the target temperature for step change is reached
# start_time = (start_change_tempreature_time + target_temperature_reached_time) / 2 # start time elapsed to fit equation, seconds, halfway between start of step change and end of step change
end_time = steady_state_time # start time elapsed to fit equation, seconds, where steady state is reached

def step_change_heat_flux(t, conductivity,diffusivityEminus5,heat_flux_offset):
    # t is time since step change in seconds, conductivity and diffusivity are fitting parameters
    # Equation used: qdot = k*infinite series for odd indices((-2*deltaT/L)*exp(-t/tau))
    #   tau = 1/(diffusivity*s^2)
    #   s = n*pi/2L
    summation = 0 # start at zero, add each term in series
    for n in range(1, 6, 2): # loop through odd numbers to 100 to approximate infinite series
        s = n*3.14159265/(2*L)
        tau = 1/((diffusivityEminus5*10**(-5))*(s**2))
        summation += (-2*deltaT/L)*np.exp(-t/tau)
    return conductivity*summation + heat_flux_offset

def exponential(t, initial, asymptote, tau):
    return asymptote + (initial - asymptote) * np.exp(-t / tau)

def fit_exponential(time_list, heat_flux_list):
    if heat_flux_sign == 1:
        initial_value_guess = 500
    elif heat_flux_sign == -1:
        initial_value_guess = -500
    else:
        raise ValueError("initial_value_guess in setup.py should be 1 or -1")
    asymptote_guess = HeatfluxData["average_heatflux"].iloc[-1]
    tau_guess = 50
    popt, pcov = curve_fit(exponential, time_list, heat_flux_list, p0=[initial_value_guess, asymptote_guess, tau_guess], bounds=([-np.inf, -np.inf, 30], [np.inf, np.inf, 100]))
    fitted_initial, fitted_asymptote, fitted_tau = popt
    # print(fitted_tau)
    return fitted_initial, fitted_asymptote, fitted_tau

def fit_heat_flux_equation(time_list, heat_flux_list):
    model = Model(step_change_heat_flux)
    if heat_flux_sign == 1:
        k_guess = -0.7
        alpha_guess = 0.00001
        offset_guess = 0
    elif heat_flux_sign == -1:
        k_guess = -0.7
        alpha_guess = 0.00001
        offset_guess = 0
    params = model.make_params(conductivity=k_guess,diffusivityEminus5=alpha_guess,heat_flux_offset=offset_guess)
    params['heat_flux_offset'].set(value=offset_guess, vary=False) # FIX IT SO THAT IT WONT BE FITTED
    result = model.fit(heat_flux_list, params, t=time_list)
    return result

def round_sig(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def graph_heat_vs_time_and_fitted_eqn(exp_time, exp_heatflux, adjusted_heat_flux, conductivity, diffusivity, heat_flux_offset):
    linspace_time = np.arange(exp_time[0]+fitting_time_skip, exp_time[-1], 1)
    fitted_heat_flux = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time]
    # plt.plot(exp_time, exp_heatflux, label="Experimental", color="blue")
    plt.plot(exp_time, adjusted_heat_flux, label="Experimental with Losses Removed", color="purple")
    plt.plot(linspace_time, fitted_heat_flux, label="Final Fitted Equation", color="red")
    linspace_time_overshoot = np.arange(2, fitting_time_skip+1, 1)
    fitted_heat_flux_overshoot = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time_overshoot]
    plt.plot(linspace_time_overshoot, fitted_heat_flux_overshoot, color="red", linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heat Flux (W/m^2)')
    plt.legend()
    plt.title('Heat Flux over Time')
    plt.show()

def graph_heat_vs_time(exp_time, exp_heatflux):
    plt.plot(exp_time, exp_heatflux)
    plt.plot(exp_time, np.gradient(exp_heatflux, exp_time), label = 'slope')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heat Flux (W/m^2)')
    plt.title('Heat Flux over Time')
    plt.axvline(x=start_time, color='r', linestyle='--')
    plt.axvline(x=end_time, color='r', linestyle='--')
    plt.legend()

    plt.show()

def calculate_fit_error(exp_time, exp_heatflux, conductivity,diffusivity,heat_flux_offset):
    linspace_time = np.arange(exp_time[0]+fitting_time_skip, exp_time[-1], 0.1)
    fitted_heat_flux = [step_change_heat_flux(t, conductivity, diffusivity, heat_flux_offset) for t in linspace_time]
    # filter values in experimental value to only include points that are used to fit equation
    filtered_exp_time = [time for time in exp_time if linspace_time[0] <= time <= linspace_time[-1]]
    filtered_exp_heat = [exp_heatflux[i] for i in range(len(exp_time)) if linspace_time[0] <= exp_time[i] <= linspace_time[-1]]
    interpolated_fitted_heat_flux = np.interp(filtered_exp_time, linspace_time, fitted_heat_flux)
    avg_abs_relative_err = np.sum(np.abs(np.subtract(filtered_exp_heat,interpolated_fitted_heat_flux)))/np.sum(np.abs(filtered_exp_heat))
    print("Average absolute relative error of heat equation flux fit: "+str(100*avg_abs_relative_err)+" %")

def run_fitting():
    # graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)

    heat_flux_column = HeatfluxData.average_heatflux
    time_window = np.subtract([time for time in HeatfluxData.time_elapsed if start_time <= time <= end_time], start_time)
    heat_fluxes = [heat_flux_column[i] for i in range(len(HeatfluxData.time_elapsed)) if start_time <= HeatfluxData.time_elapsed[i] <= end_time]
    time_window_for_fitting = [time for time in time_window if time >= fitting_time_skip] # skips first few seconds to ignore overshoots, as defined on top
    heat_fluxes_for_fitting = [heat_fluxes[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]
    #fitting the analytical solution
    # result_direct = fit_heat_flux_equation(time_window_for_fitting, heat_fluxes_for_fitting)
    # heat_flux_offset = result_direct.params['heat_flux_offset'].value
    # conductivity = result_direct.params['conductivity'].value
    # conductivity_error = result_direct.params['conductivity'].stderr
    # diffusivityEminus5 = result_direct.params['diffusivityEminus5'].value
    # diffusivityEminus5_error = result_direct.params['diffusivityEminus5'].stderr
    # diffusivity = (diffusivityEminus5)*10**(-5)
    # diffusivity_error = (diffusivityEminus5_error)*10**(-5)
    # print("**Results**")
    # print("Conductivity: "+str(round_4_sig(conductivity))+" W/(m*K)")
    # print("Diffusivity: "+str(round_4_sig(diffusivity))+" m^2/s")
    # print("Conductivity stderr: "+str(conductivity_error)+" W/(m*K)")
    # print("Diffusivity stderr: "+str(diffusivity_error)+" m^2/s")
    # print("Heat flux offset: "+str(round_4_sig(heat_flux_offset))+" W/m^2")
    initial_loss_index = next(i for i, t in enumerate(HeatfluxData.time_elapsed) if t >= start_change_tempreature_time) - 5 # 5 points before the start time to get initial loss
    prev_100_points = HeatfluxData.average_heatflux[(initial_loss_index-100):initial_loss_index] # get prev 100 points to get the average initial loss
    initial_loss_estimate = (sum(prev_100_points) / len(prev_100_points)) * S_final / S_initial # multiply by S_final / S_initial because different calibration temperature before step change

    fitted_initial, fitted_asymptote, fitted_tau = fit_exponential(time_window_for_fitting, heat_fluxes_for_fitting)
    # print("Final loss: "+str(fitted_asymptote))
    # print("HEat flux fitted to exponential:")
    # print(fitted_initial)
    # print(fitted_asymptote)
    # print(fitted_tau)
    losses = [exponential(t, initial=initial_loss_estimate, asymptote=fitted_asymptote, tau=fitted_tau) for t in time_window]
    # losses = np.zeros(len(time_window))+fitted_asymptote
    # print("Initial loss: "+str(initial_loss_estimate))
    # print("Final loss: "+str(fitted_asymptote))
    linspace_time = np.arange(time_window[0]+fitting_time_skip, time_window[-1], 1)
    adjusted_heat_flux = np.subtract(heat_fluxes, losses)
    adjusted_heat_fluxes_for_fitting = [adjusted_heat_flux[i] for i in range(len(time_window)) if time_window[i] >= fitting_time_skip]
    fitted_exponential = [exponential(t, fitted_initial, fitted_asymptote, fitted_tau) for t in linspace_time]
    # plt.plot(time_window, heat_fluxes, label="Experimental", color="blue")
    # plt.plot(time_window, losses, label="Losses", color="green")
    # plt.plot(time_window, adjusted_heat_flux, label="Heat flux with losses adjusted", color="purple")
    # plt.axhline(y=initial_loss_estimate, color='green', linestyle='--')
    # plt.axhline(y=fitted_asymptote, color='green', linestyle='--')
    # linspace_time_overshoot = np.arange(2, fitting_time_skip+1, 1)
    # plt.plot(linspace_time, fitted_exponential, color="black", label="Fitted Exponential to Data")
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Heat Flux (W/m^2)')
    # plt.legend()
    # plt.title('Heat Flux over Time')
    # plt.show()

    result = fit_heat_flux_equation(time_window_for_fitting, adjusted_heat_fluxes_for_fitting)
    # heat_flux_offset = result.params['heat_flux_offset'].value
    conductivity = result.params['conductivity'].value
    conductivity_error = result.params['conductivity'].stderr
    diffusivityEminus5 = result.params['diffusivityEminus5'].value
    diffusivityEminus5_error = result.params['diffusivityEminus5'].stderr
    diffusivity = (diffusivityEminus5)*10**(-5)
    diffusivity_error = (diffusivityEminus5_error)*10**(-5)
    # print("Heat flux offset: "+str(round_4_sig(heat_flux_offset))+" W/m^2")
    # graph_heat_vs_time(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux)
    # graph_heat_vs_time_and_fitted_eqn(time_window, heat_fluxes, adjusted_heat_flux, conductivity,diffusivityEminus5,heat_flux_offset=0)

    return conductivity, diffusivity, conductivity_error, diffusivity_error

uncertainty = {
    'L': 0.000005,  # meters used insulated caliper, 0.005mm uncertainty
    'deltaT': 0.05,  # deg C, 1% of 5 degrees
    'cell_mass_density': 50,  # kg/m^3
    "cell_length": 0.0005, # used ruler, 0.5mm uncertainty
    "cell_width": 0.0005,
    "heat_flux_pct": 0.03  # 3% uncertainty
}


if __name__ == "__main__":
    print("Running Monte Carlo Fitting")

    results = []

    for i in range(monte_carlo_n):  # Run n Monte Carlo simulations
        HeatfluxData, sampled_constants = monte_carlo_heatflux_iteration(sensor_data, calibration_data, T_S, uncertainty)
        try:
            conductivity, diffusivity, conductivity_error, diffusivity_error = run_fitting()
            specific_heat = conductivity / (diffusivity * sampled_constants["cell_mass_density"])
            results.append({
                "conductivity": conductivity,
                "diffusivity": diffusivity,
                "specific_heat": specific_heat,
                **sampled_constants
            })
            # print(f"Iteration {i+1}: k={conductivity:.4f}, α={diffusivity:.4e}, Cp={specific_heat:.2f}")
        except Exception as e:
            print(f"Iteration {i+1} failed: {e}")

    # Optionally convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    summary = results_df.describe()
    cv_percent = abs((summary.loc["std"] / summary.loc["mean"]) * 100)
    cv_percent.name = "std % of mean"
    summary = pd.concat([summary, pd.DataFrame([cv_percent])])
    print("\nSummary Statistics:")
    print(summary)