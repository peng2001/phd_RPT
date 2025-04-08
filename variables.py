heatflux_file = 'data/test_J1_cell/J1_throughplane_15-20.csv'

heat_flux_sign = 1 # 1 or -1; 1 means heat flux entering the cell is positive; -1 means heat flux entering cell is negative
T_S = 20 # deg C, final temperature
L = 0.0057 # meters, 1/2 cell thickness (L)
deltaT = 5  # degrees C, magnitude of step change
fitting_time_skip = 30 # seconds, integer, ignore first points because of overshoot
cell_mass_density = 2465 # kg/m^3
S_area = 0.032 # meters squared, cell surface area per face, used for heat gen calculation; 0.1 * 0.32