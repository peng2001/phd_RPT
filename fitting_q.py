import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
import matplotlib.pyplot as plt
from setup import *
from scipy.signal import savgol_filter
import math

def round_sig(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

##########################################################
# finding start time of the model
# fig, ax1 = plt.subplots()
# ax1.plot(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux, label='Heat Flux')
# ax1.set_xlabel('Time (seconds)')
# ax1.set_ylabel('Heat Flux (W/m^2)')
# plt.title('Heat Flux over Time')
# heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed)
# ax2 = ax1.twinx()
# ax2.plot(HeatfluxData.time_elapsed, heatflux_slopes, label='Slope', color="purple")
# ax1.set_ylabel('Heat Flux (W/m^2)')
# ax2.set_ylabel('Heat Flux Slope (W/(m^2 * s))')
# ax1.set_ylim(-40, 40)
# ax2.set_ylim(-8,8)
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# plt.show()

HeatfluxData["average_heatflux"] = savgol_filter(HeatfluxData["average_heatflux"], 300, 3) # smoothing data because it's too noisy to make out any peaks in the slope
heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed)

# fig, ax1 = plt.subplots()
# ax1.plot(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux, label='Heat Flux')
# ax1.set_xlabel('Time (seconds)')
# ax1.set_ylabel('Heat Flux (W/m^2)')
# plt.title('Heat Flux over Time')
# heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed)
# ax2 = ax1.twinx()
# ax2.plot(HeatfluxData.time_elapsed, heatflux_slopes, label='Slope', color="purple")
# ax1.set_ylabel('Heat Flux (W/m^2)')
# ax2.set_ylabel('Heat Flux Slope (W/(m^2 * s))')
# ax1.set_ylim(-40, 40)
# ax2.set_ylim(-8, 8)
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# plt.show()

min_slope_index = np.argmax(heatflux_slopes)  # Index of min slope, find start and ends of the heat generation
max_slope_index = np.argmin(heatflux_slopes)  # Index of max slope
time_at_min_slope = HeatfluxData.time_elapsed[min_slope_index]
time_at_max_slope = HeatfluxData.time_elapsed[max_slope_index]

start_time = min(time_at_max_slope, time_at_min_slope) + 300
end_time = max(time_at_max_slope, time_at_min_slope) - 300
HeatfluxData = HeatfluxData[HeatfluxData.time_elapsed <= end_time + 3000] # cut out all extra data points at end
heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed) # recalculate slopes

heat_fluxes_gen = [heatflux for t, heatflux in zip(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux) if start_time <= t <= end_time] # the heat fluxes within start time and end time (during the pulsing)
heat_fluxes_gen_average = np.mean(heat_fluxes_gen)
loss_start_index = len(HeatfluxData.average_heatflux) - 500 # 500 seconds before final measurement
loss_calc_times = HeatfluxData.time_elapsed[loss_start_index:]  # Corresponding time values
loss_calc_heatflux = HeatfluxData.average_heatflux[loss_start_index:]  # Last 500 heat flux values
losses_average = np.mean(loss_calc_heatflux)

plt.plot(HeatfluxData.time_elapsed, HeatfluxData.average_heatflux, label="Heat flux")
plt.plot(HeatfluxData.time_elapsed, heatflux_slopes, label="Heat flux slope")
plt.plot()
plt.xlabel('Time (seconds)')
plt.ylabel('Heat Flux (W/m^2)')
plt.title('Heat Flux over Time')
plt.axvline(x=start_time, color='green', linestyle='--')
plt.axvline(x=end_time, color='green', linestyle='--')
plt.hlines(y=heat_fluxes_gen_average, xmin=start_time, xmax=end_time, colors='red', linewidth=2, label="Average heat flux during heat gen")
plt.hlines(y=losses_average, xmin=loss_calc_times.iloc[0], xmax=loss_calc_times.iloc[-1], colors='purple', linewidth=2, label="Average heat flux loss")
plt.legend()
plt.show()
print()

##########
total_loss = losses_average*S_area*2
total_heatgen = heat_fluxes_gen_average*S_area*2
total_heatgen_subtract_losses = total_heatgen-total_loss
print("**Results**")
print("Total heat generation: "+str(round_sig(abs(total_heatgen_subtract_losses),4))+" W")