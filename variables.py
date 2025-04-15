heatflux_file = 'data/test_J1PA/J1PA_throughplane_25-30_again_4.csv'
T_S = 30 # deg C, final temperature

heat_flux_sign = -1 # 1 or -1; 1 means heat flux entering the cell is positive; -1 means heat flux entering cell is negative
L = 0.0057 # meters, 1/2 cell thickness (L)
deltaT = 5  # degrees C, magnitude of step change
fitting_time_skip = 50 # seconds, integer, ignore first points because of overshoot
cell_mass_density = 2465 # kg/m^3
S_area = 0.032 # meters squared, cell surface area per face, used for heat gen calculation; 0.1 * 0.32