import numpy as np
from scipy.optimize import fsolve
import streamlit as st

from src.funcs import equilibrium_equations_at_pH, constant_data
from src.plots import plot_equilibrium_results_plotly

# initial fitting of constants
constant_data['a'].fit_power()
constant_data['b'].fit_linear()
constant_data['c'].fit_power()
constant_data['d'].fit_linear()

with st.sidebar:
    temperature = st.slider('Temperature, °C', min_value=25.0, max_value=150.0, value=125.0, step=5.0)

    # Set the total concentrations of species in the system
    concentration_format = "%.2f"
    N_total_val = st.number_input('Total NH3, mol/l', min_value=0.0, max_value=10.0, value=2.0, format=concentration_format)
    S_total_val = st.number_input('Total SO2, mol/l', min_value=0.0, max_value=10.0, value=1.0, format=concentration_format)
    Ac_total_val = st.number_input('Total Ac, mol/l', min_value=0.0, max_value=10.0, value=0.4, format=concentration_format)

    # Interpolate constants based on temperature
    constants = {
        'a': constant_data['a'].constants_at_temp(temperature),
        'b': constant_data['b'].constants_at_temp(temperature),
        'c': constant_data['c'].constants_at_temp(temperature),
        'd': constant_data['d'].constants_at_temp(temperature),
    }

    with st.expander("Equilibrium Constants", expanded=True):
        # Set equilibrium constants (at a specific temperature)
        constant_format = "%.2e"
        a = st.number_input('SO2_H2O = HSO3', min_value=0.0, max_value=1.0e14, value=constants['a'], step=1e3, format=constant_format)
        b = st.number_input('HSO3 = SO3', min_value=0.0, max_value=1.0e14, value=constants['b'], format=constant_format)
        c = st.number_input('NH4 = NH3_H2O', min_value=0.0, max_value=1.0e14, value=constants['c'], format=constant_format)
        d = st.number_input('HAc = Ac', min_value=0.0, max_value=1.0e14, value=constants['d'], format=constant_format)

tab_results, tab_constants = st.tabs(["Equilibrium Results", "Constant fitting"])

with tab_constants:
    st.title("Equilibrium Constants")

    st.write(
        """
        The equilibrium constants are fitted to the following equations:
        - a: SO2_H2O = HSO3
        - b: HSO3 = SO3
        - c: NH4 = NH3_H2O
        - d: HAc = Ac
        """
    )

    # Plot fitted curves for each constant as a function of temperature
    import plotly.graph_objs as go
    temp_range = np.linspace(25, 150, 200)
    a_vals = [constant_data['a'].constants_at_temp(t) for t in temp_range]
    b_vals = [constant_data['b'].constants_at_temp(t) for t in temp_range]
    c_vals = [constant_data['c'].constants_at_temp(t) for t in temp_range]
    d_vals = [constant_data['d'].constants_at_temp(t) for t in temp_range]

    # Measured data points from constants_at_temp
    a_x = list(constant_data['a'].x_data)
    a_y = list(constant_data['a'].y_data)
    b_x = list(constant_data['b'].x_data)
    b_y = list(constant_data['b'].y_data)
    c_x = list(constant_data['c'].x_data)
    c_y = list(constant_data['c'].y_data)
    d_x = list(constant_data['d'].x_data)
    d_y = list(constant_data['d'].y_data)

    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=temp_range, y=a_vals, mode='lines', name='a (fit)'))
    fig_a.add_trace(go.Scatter(x=a_x, y=a_y, mode='markers', name='a (measured)', marker=dict(size=10, color='red')))
    fig_a.update_layout(title='Constant a vs Temperature', xaxis_title='Temperature (°C)', yaxis_title='a', yaxis_type='linear')
    st.plotly_chart(fig_a, use_container_width=True)

    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=temp_range, y=b_vals, mode='lines', name='b (fit)'))
    fig_b.add_trace(go.Scatter(x=b_x, y=b_y, mode='markers', name='b (measured)', marker=dict(size=10, color='red')))
    fig_b.update_layout(title='Constant b vs Temperature', xaxis_title='Temperature (°C)', yaxis_title='b', yaxis_type='log')
    st.plotly_chart(fig_b, use_container_width=True)

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=temp_range, y=c_vals, mode='lines', name='c (fit)'))
    fig_c.add_trace(go.Scatter(x=c_x, y=c_y, mode='markers', name='c (measured)', marker=dict(size=10, color='red')))
    fig_c.update_layout(title='Constant c vs Temperature', xaxis_title='Temperature (°C)', yaxis_title='c', yaxis_type='log')
    st.plotly_chart(fig_c, use_container_width=True)

    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=temp_range, y=d_vals, mode='lines', name='d (fit)'))
    fig_d.add_trace(go.Scatter(x=d_x, y=d_y, mode='markers', name='d (measured)', marker=dict(size=10, color='red')))
    fig_d.update_layout(title='Constant d vs Temperature', xaxis_title='Temperature (°C)', yaxis_title='d', yaxis_type='log')
    st.plotly_chart(fig_d, use_container_width=True)

# st.title(f'NH3-SO2-Ac-H2O system at {temperature:.0f} °C')
pH_values: np.ndarray = np.linspace(14, 0, 500) # Generate pH values

results: list[list[float]] = []
initial_guess_1: list[float] = [
    1e-6, # SO2_H2O
    1e-6, # HSO3
    S_total_val, # SO3
    1e-6, # NH4
    N_total_val, # NH3_H2O
    1e-6, # HAc
    Ac_total_val, # Ac
]

for pH in pH_values:
    solution, infodict, ier, mesg = fsolve(equilibrium_equations_at_pH, initial_guess_1, args=(pH, a, b, c, d, S_total_val, N_total_val, Ac_total_val), xtol=1e-10, full_output=True)
    if ier == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution
        h_plus_eq = 10**-pH
        results.append([pH, h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq, h_plus_eq])
        initial_guess_1 = solution
    else:
        print(f"Convergence failed for pH = {pH}: {mesg}")
        results.append([pH] + [np.nan] * 9) # Append NaNs if no convergence

# Convert results to a NumPy array for easier handling
results_array = np.array(results)


with tab_results:
    # Plotting the results
    fig = plot_equilibrium_results_plotly(results_array, temperature)
    st.plotly_chart(fig)

    st.write("## Component Concentrations at certain pH")
    ph_col_1, ph_col_2 = st.columns(2)
    pH_input_1 = ph_col_1.number_input('pH', min_value=0.0, max_value=14.0, value=5.0, step=0.1, format="%.1f", key="pH_input_1")
    pH_input_2 = ph_col_2.number_input('pH', min_value=0.0, max_value=14.0, value=8.0, step=0.1, format="%.1f", key="pH_input_2")
    #initial guess from results_array
    initial_guess_1 = results_array[np.abs(results_array[:, 0] - pH_input_1).argmin(), 1:8].tolist()  # Get the closest initial guess from results
    solution_1, infodict_1, ier_1, mesg_1 = fsolve(equilibrium_equations_at_pH, initial_guess_1, args=(pH_input_1, a, b, c, d, S_total_val, N_total_val, Ac_total_val), xtol=1e-10, full_output=True)
    initial_guess_2 = results_array[np.abs(results_array[:, 0] - pH_input_2).argmin(), 1:8].tolist()  # Get the closest initial guess from results
    solution_2, infodict_2, ier_2, mesg_2 = fsolve(equilibrium_equations_at_pH, initial_guess_2, args=(pH_input_2, a, b, c, d, S_total_val, N_total_val, Ac_total_val), xtol=1e-10, full_output=True)

    solution_dict = {}
    if ier_1 == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution_1
        solution_dict[pH_input_1] = {
            "SO\u2082==H\u2082O": h2so3_eq,
            "HSO\u2083\u207B": hso3_eq,
            "SO\u2083\u00B2\u207B": so3_eq,
            "NH\u2084\u207A": nh4_eq,
            "NH\u2083==H\u2082O": nh3_h2o_eq,
            "HAc": hac_eq,
            "Ac\u207B": ac_eq,
        }
    else:
        st.write(f"Convergence failed for pH = {pH}: {mesg}")

    if ier_2 == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution_2
        solution_dict[pH_input_2] = {
            "SO\u2082==H\u2082O": h2so3_eq,
            "HSO\u2083\u207B": hso3_eq,
            "SO\u2083\u00B2\u207B": so3_eq,
            "NH\u2084\u207A": nh4_eq,
            "NH\u2083==H\u2082O": nh3_h2o_eq,
            "HAc": hac_eq,
            "Ac\u207B": ac_eq,
        }
    else:
        st.write(f"Convergence failed for pH = {pH}: {mesg}")


    st.dataframe(solution_dict)