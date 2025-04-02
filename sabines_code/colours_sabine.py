# colors.py

def get_grouped_colors():
    """Returns the color mappings for grouped colors."""
    return {
        # Group 1: Blues - plus terminal
        "HeatFluxC0_D01": "#2171b5",  # Deep Blue
        "HeatFluxA0_C05": "#2171b5",  # Deep Blue
        "HeatFluxB0_C13": "#2171b5",  # Deep Blue
        "HeatFluxD0_D07": "#2171b5",  # Deep Blue

        "HeatFluxD1_D08": "#63b7d5",  # Light Cyan Blue

        "HeatFluxD3_D12": "#74c476",  # Light Green
        "HeatFluxC3_D03": "#74c476",  # Light Green

        "HeatFluxD2_D11": "#2ca25f",  # Bright Green
        "HeatFluxA2_C07": "#2ca25f",  # Bright Green

        "HeatFluxD4_D13": "#FFC300",  # Golden Yellow
        "HeatFluxA4_C08": "#FFC300",  # Golden Yellow

        "HeatFluxC6_D04": "#C70039",  # Crimson
        "HeatFluxA6_C11": "#C70039",  # Crimson
        "HeatFluxD6_D16": "#C70039",  # Crimson
        "HeatFluxB6_C14": "#C70039",  # Crimson

        "HeatFluxD5_D14": "#fdae6b",  # Soft Orange
    }

def get_grouped_colors_averaged():
    """Returns the color mappings for averaged grouped colors."""
    return {
        # Group 1: Blues - plus terminal
        "HeatFluxA0_C05_offset_smooth": "#4292c6",  # Base Blue
        "HeatFluxC6_D04_offset_smooth": "#2171b5",  # Deep Blue
        "HeatFluxA6_C11_offset_smooth": "#6baed6",  # Sky Blue
        "HeatFluxC0_D01_offset_smooth": "#08519c",  # Dark Navy
        "Sensor#05": "#1d91c0",  # Teal Blue
        "Sensor#06": "#3182bd",  # Medium Blue
        "Sensor#07": "#63b7d5",  # Light Cyan Blue
        "Sensor#08": "#0b559f",  # Ocean Blue

        # Group 2: Greens - minus terminal
        "HeatFluxB0_C13_offset_smooth": "#41ab5d",  # Base Green
        "HeatFluxD6_D16_offset_smooth": "#238b45",  # Forest Green
        "HeatFluxB6_C14_offset_smooth": "#74c476",  # Light Green
        "HeatFluxD0_D07_offset_smooth": "#006d2c",  # Dark Green
        "Sensor#13": "#66c2a4",  # Mint Green
        "Sensor#14": "#2ca25f",  # Bright Green
        "Sensor#15": "#4daf4a",  # Vivid Green
        "Sensor#16": "#009f4d",  # Emerald Green

        # Group 3: Oranges
        "HeatFluxD1_D08_offset_smooth": "#f16913",  # Base Orange
        "HeatFluxD2_D11_offset_smooth": "#d94801",  # Deep Orange
        "HeatFluxD3_D12_offset_smooth": "#fd8d3c",  # Bright Orange
        "HeatFluxD4_D13_offset_smooth": "#e6550d",  # Burnt Orange
        "HeatFluxD5_D14_offset_smooth": "#fdae6b",  # Soft Orange
        "HeatFluxC3_D03_offset_smooth": "#f27023",  # Vivid Orange
        "HeatFluxA2_C07_offset_smooth": "#e34a33",  # Crimson Orange
        "HeatFluxA4_C08_offset_smooth": "#ff7f0e",  # Medium Orange
    }

def get_grouped_colors_meanheatflux():
    """Returns the color mappings for mean heat flux."""
    return {
        # Group 1: Blues - plus terminal
        "C0_D01": "#2171b5",  # Deep Blue
        "A0_C05": "#2171b5",  # Deep Blue
        "B0_C13": "#2171b5",  # Deep Blue
        "D0_D07": "#2171b5",  # Deep Blue

        "D1_D08": "#63b7d5",  # Light Cyan Blue

        "D3_D12": "#74c476",  # Light Green
        "C3_D03": "#74c476",  # Light Green

        "D2_D11": "#2ca25f",  # Bright Green
        "A2_C07": "#2ca25f",  # Bright Green

        "D4_D13": "#FFC300",  # Golden Yellow
        "A4_C08": "#FFC300",  # Golden Yellow

        "C6_D04": "#C70039",  # Crimson
        "A6_C11": "#C70039",  # Crimson
        "D6_D16": "#C70039",  # Crimson
        "B6_C14": "#C70039",  # Crimson

        "D5_D14": "#fdae6b",  # Soft Orange
    }
