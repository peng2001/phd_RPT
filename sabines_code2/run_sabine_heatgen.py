from setup_sabine_heatgen import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

def round_sig(x, sig):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def calculate_heatflux_subtract_losses(heatflux_data):
    loss_start_index = len(HeatfluxData.average_heatflux) - 500 # 500 seconds before final measurement; to determine the average loss using final 1000 points
    loss_calc_times = HeatfluxData["time_elapsed"][loss_start_index:]  # Corresponding time values
    heatflux_subtract_losses = pd.DataFrame()
    for col in HeatfluxData.columns:
        if col.startswith("HeatFlux"):  
            avg_value = HeatfluxData[col].iloc[loss_start_index:].mean()  # Compute average loss
            heatflux_subtract_losses[col] = HeatfluxData[col] - avg_value  # Subtract from each value in the column
    heatflux_subtract_losses["average_heatflux"] = heatflux_subtract_losses.filter(like="HeatFlux").mean(axis=1)
    heatflux_subtract_losses["average_heatflux"] = savgol_filter(heatflux_subtract_losses["average_heatflux"], 300, 3)
    heatflux_subtract_losses["time_elapsed"] = HeatfluxData["time_elapsed"]
    return heatflux_subtract_losses


def calculate_total_heatflux(heatflux_subtract_losses): # input output from heatflux_subtract_losses
    """Calculates total heat flux on top and bottom sides."""
    area = 0.04571 * 0.0505  # Area per Peltier element
    top_side = []
    bottom_side = []

    heatflux_subtract_losses["top_flux"] = (
        heatflux_subtract_losses['HeatFluxD0_D07'] +
        heatflux_subtract_losses['HeatFluxC0_D01'] +
        2 * heatflux_subtract_losses['HeatFluxD1_D08'] +
        2 * heatflux_subtract_losses['HeatFluxD2_D11'] +
        heatflux_subtract_losses['HeatFluxD3_D12'] +
        heatflux_subtract_losses['HeatFluxC3_D03'] +
        2 * heatflux_subtract_losses['HeatFluxD4_D13'] +
        2 * heatflux_subtract_losses['HeatFluxD5_D14'] +
        heatflux_subtract_losses['HeatFluxD6_D16'] +
        heatflux_subtract_losses['HeatFluxC6_D04']
    ) * area

    heatflux_subtract_losses["bottom_flux"] = (
        heatflux_subtract_losses['HeatFluxB0_C13'] +
        heatflux_subtract_losses['HeatFluxB6_C14'] +
        5 * heatflux_subtract_losses['HeatFluxA2_C07'] +
        5 * heatflux_subtract_losses['HeatFluxA4_C08'] +
        heatflux_subtract_losses['HeatFluxA0_C05'] +
        heatflux_subtract_losses['HeatFluxA6_C11']
    ) * area

    filtered_data = heatflux_subtract_losses[(heatflux_subtract_losses["time_elapsed"] >= start_time) & (heatflux_subtract_losses["time_elapsed"] <= end_time)]
    top_side = filtered_data["top_flux"]
    bottom_side = filtered_data["bottom_flux"]

    top_side = np.array(top_side)
    print(np.average(top_side)/(14*area))
    bottom_side = np.array(bottom_side)
    # print("Top Side Total Heat Flux:", top_side)
    # print("Bottom Side Total Heat Flux:", bottom_side)

    return top_side, bottom_side


#########################################################################
HeatfluxData["average_heatflux"] = savgol_filter(HeatfluxData["average_heatflux"], 300, 3) # smoothing data because it's too noisy to make out any peaks in the slope
heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed)
min_slope_index = np.argmax(heatflux_slopes)  # Index of min slope, find start and ends of the heat generation
max_slope_index = np.argmin(heatflux_slopes)  # Index of max slope
time_at_min_slope = HeatfluxData.time_elapsed[min_slope_index]
time_at_max_slope = HeatfluxData.time_elapsed[max_slope_index]

start_time = min(time_at_max_slope, time_at_min_slope) + 300
end_time = max(time_at_max_slope, time_at_min_slope) - 300

HeatfluxData = HeatfluxData[HeatfluxData.time_elapsed <= end_time + 3000] # cut out all extra data points at end
heatflux_slopes = np.gradient(HeatfluxData.average_heatflux, HeatfluxData.time_elapsed) # recalculate slopes
heatflux_subtract_losses = calculate_heatflux_subtract_losses(HeatfluxData)

# for col in heatflux_subtract_losses.columns:
#     if col.startswith("HeatFlux"):  # Filter columns
#         plt.plot(HeatfluxData["time_elapsed"], heatflux_subtract_losses[col], label=col)
# plt.plot()
# plt.xlabel('Time (seconds)')
# plt.ylabel('Heat Flux (W/m^2)')
# plt.title('Heat Flux over Time')
# plt.axvline(x=start_time, color='green', linestyle='--')
# plt.axvline(x=end_time, color='green', linestyle='--')
# plt.legend()
# plt.show()
# print()

total_heat_flux_top, total_heat_flux_bottom = calculate_total_heatflux(heatflux_subtract_losses)
print("Total Heat Flux: "+str(round_sig(abs(np.average(total_heat_flux_top) + np.average(total_heat_flux_bottom)),4)) + " W")
