# plotting.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

def plot_heatflux_and_current(heatflux_data, cycler_data, grouped_colors):
    """Plots heat flux data along with current from CyclerData."""
    data = heatflux_data.copy()
    data_melted = data.melt(id_vars='time', var_name='Sensor', value_name='Value')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sensor in data_melted['Sensor'].unique():
        sensor_data = data_melted[data_melted['Sensor'] == sensor]
        fig.add_trace(go.Scatter(
            x=(sensor_data['time'].dt.tz_localize('UTC+01:00') - cycler_data['DPT Time'].iloc[0]).dt.total_seconds() / 3600,
            y=sensor_data['Value'],
            line=dict(color=grouped_colors.get(sensor, '#000000')),
            mode='lines',
            name=sensor),
            secondary_y=False,
        )

    # Plot CyclerData Current
    fig.add_trace(go.Scatter(
        x=(cycler_data['DPT Time'] - cycler_data['DPT Time'].iloc[0]).dt.total_seconds() / 3600,
        y=cycler_data['Current'],
        line_color="#101010",
        mode='lines',
        name='Current'),
        secondary_y=True,
    )

    # Set axis titles and ranges
    fig.update_xaxes(title_text="Time in h", title_font={"size": 20}, tickfont={"size": 20})
    fig.update_yaxes(title_text="Heat Flux in W/m²", title_font={"size": 20}, tickfont={"size": 20}, range=[0, 400], secondary_y=False)
    fig.update_yaxes(title_text="Current in A", title_font={"size": 20}, tickfont={"size": 20}, range=[-100, 100], secondary_y=True)

    fig.show()

def plot_heatflux_and_current_offset(heatflux_data, cycler_data, grouped_colors):
    """Plots heat flux data along with current from CyclerData."""
    index = (heatflux_data['time'].dt.tz_localize('UTC+01:00') - cycler_data['DPT Time'].iloc[0]).abs().idxmin()

    # Apply offset and smoothing
    for column in heatflux_data.columns:
        if column != 'time' and '_offset_smooth' not in column:
            offset = heatflux_data[column][index:index+100].mean()
            heatflux_data[column + '_offset_smooth'] = gaussian_filter1d(heatflux_data[column] - offset, sigma=2)

    data = heatflux_data.copy()
    data_melted = data.melt(id_vars='time', var_name='Sensor', value_name='Value')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sensor in data_melted['Sensor'].unique():
        if '_offset_smooth' not in sensor:
            continue
        sensor_data = data_melted[data_melted['Sensor'] == sensor]
        fig.add_trace(go.Scatter(
            x=(sensor_data['time'].dt.tz_localize('UTC+01:00') - cycler_data['DPT Time'].iloc[0]).dt.total_seconds() / 3600,
            y=sensor_data['Value'],
            line=dict(color=grouped_colors.get(sensor[:-14], '#000000')),
            mode='lines',
            name=sensor),
            secondary_y=False,
        )

    # Plot CyclerData Current
    fig.add_trace(go.Scatter(
        x=(cycler_data['DPT Time'] - cycler_data['DPT Time'].iloc[0]).dt.total_seconds() / 3600,
        y=cycler_data['Current'],
        line_color="#101010",
        mode='lines',
        name='Current'),
        secondary_y=True,
    )

    # Set axis titles and ranges
    fig.update_xaxes(title_text="Time in h", title_font={"size": 20}, tickfont={"size": 20})
    fig.update_yaxes(title_text="Heat Flux in W/m²", title_font={"size": 20}, tickfont={"size": 20}, range=[0, 400], secondary_y=False)
    fig.update_yaxes(title_text="Current in A", title_font={"size": 20}, tickfont={"size": 20}, range=[-100, 100], secondary_y=True)

    fig.show()





def plot_average_heatflux(averaged_heatflux, grouped_colors):
    """Plots the averaged heat flux over cycles."""
    average_heatflux_melted = averaged_heatflux.melt(id_vars='Cycle Value', var_name='Sensor', value_name='Value')
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sensor in average_heatflux_melted['Sensor'].unique():
        if 'offset_smooth' in sensor:
            sensor_data = average_heatflux_melted[average_heatflux_melted['Sensor'] == sensor]
            fig.add_trace(go.Scatter(
                x=sensor_data['Cycle Value'],
                y=sensor_data['Value'],
                line=dict(color=grouped_colors.get(sensor, '#000000')),
                mode='markers',
                name=sensor),
                secondary_y=False,
            )

    fig.update_xaxes(title_text="Cycle Number")
    fig.update_yaxes(title_text="Average Heat Flux in W/m²")
    fig.show()

def plot_temperature_vs_heatflux(melted_temperature_heatflux, grouped_colors):
    """Plots temperature vs. mean heat flux."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for sensor in melted_temperature_heatflux['Sensor_Name'].unique():
        sensor_data = melted_temperature_heatflux[melted_temperature_heatflux['Sensor_Name'] == sensor]
        fig.add_trace(go.Scatter(
            x=sensor_data['Temperature'],
            y=sensor_data['Mean_Heatflux'],
            line=dict(color=grouped_colors.get(sensor, '#000000')),
            mode='markers',
            name=sensor),
            secondary_y=False,
        )

    fig.update_xaxes(title_text="Temperature in °C", title_font={"size": 20}, tickfont={"size": 20})
    fig.update_yaxes(title_text="Average Heat Flux in W/m²", title_font={"size": 20}, tickfont={"size": 20})
    fig.update_layout(
        xaxis=dict(range=[5, 45]),
        yaxis=dict(range=[0, 300]),
        width=800, height=600
    )
    fig.update_traces(marker=dict(size=15))
    fig.show()
