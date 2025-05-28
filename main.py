import numpy as np
from scipy.optimize import fsolve
import streamlit as st

from src.constant_data import constant_data
from src.funcs import equilibrium_equations_at_pH
from src.plots import plot_constant, plot_equilibrium_results_plotly

temp_min = 25.0
temp_max = 175.0

with st.sidebar:
    temperature = st.slider(
        "Temperature, °C", min_value=temp_min, max_value=temp_max, value=125.0, step=5.0
    )

    # Set the total concentrations of species in the system
    concentration_format = "%.2f"
    N_total_val = st.number_input(
        "Total NH3, mol/l",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        format=concentration_format,
    )
    S_total_val = st.number_input(
        "Total SO2, mol/l",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        format=concentration_format,
    )
    Ac_total_val = st.number_input(
        "Total Ac, mol/l",
        min_value=0.0,
        max_value=10.0,
        value=0.4,
        format=concentration_format,
    )

    # Interpolate constants based on temperature
    constants = {
        "a": constant_data["a"].constants_at_temp(temperature),
        "b": constant_data["b"].constants_at_temp(temperature),
        "c": constant_data["c"].constants_at_temp(temperature),
        "d": constant_data["d"].constants_at_temp(temperature),
    }

    with st.expander("Equilibrium Constants", expanded=True):
        # Set equilibrium constants (at a specific temperature)
        constant_format = "%.2e"
        a = st.number_input(
            "SO2_H2O = HSO3",
            min_value=0.0,
            max_value=1.0e14,
            value=constants["a"],
            step=1e3,
            format=constant_format,
        )
        b = st.number_input(
            "HSO3 = SO3",
            min_value=0.0,
            max_value=1.0e14,
            value=constants["b"],
            format=constant_format,
        )
        c = st.number_input(
            "NH4 = NH3_H2O",
            min_value=0.0,
            max_value=1.0e14,
            value=constants["c"],
            format=constant_format,
        )
        d = st.number_input(
            "HAc = Ac",
            min_value=0.0,
            max_value=1.0e14,
            value=constants["d"],
            format=constant_format,
        )

tab_results, tab_constants = st.tabs(["Equilibrium Results", "Constant fitting"])

with tab_constants:
    st.title("Equilibrium Constants")

    # Plot fitted curves for each constant as a function of temperature
    temp_range = np.linspace(temp_min, temp_max, int((temp_max - temp_min) * 2 + 1))

    fig_a = plot_constant(
        temp_range,
        constant_data["a"],
    )
    st.plotly_chart(fig_a, use_container_width=True)

    fig_b = plot_constant(
        temp_range,
        constant_data["b"],
    )
    st.plotly_chart(fig_b, use_container_width=True)

    fig_c = plot_constant(
        temp_range,
        constant_data["c"],
        yaxis_type="log",
    )
    st.plotly_chart(fig_c, use_container_width=True)

    fig_d = plot_constant(
        temp_range,
        constant_data["d"],
    )
    st.plotly_chart(fig_d, use_container_width=True)

# st.title(f'NH3-SO2-Ac-H2O system at {temperature:.0f} °C')
pH_values: np.ndarray = np.linspace(14, 0, 500)  # Generate pH values

results: list[list[float]] = []
initial_guess_1: list[float] = [
    1e-6,  # SO2_H2O
    1e-6,  # HSO3
    S_total_val,  # SO3
    1e-6,  # NH4
    N_total_val,  # NH3_H2O
    1e-6,  # HAc
    Ac_total_val,  # Ac
]

for pH in pH_values:
    solution, infodict, ier, mesg = fsolve(
        equilibrium_equations_at_pH,
        initial_guess_1,
        args=(pH, a, b, c, d, S_total_val, N_total_val, Ac_total_val),
        xtol=1e-10,
        full_output=True,
    )
    if ier == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution
        h_plus_eq = 10**-pH
        results.append(
            [
                pH,
                h2so3_eq,
                hso3_eq,
                so3_eq,
                nh3_h2o_eq,
                nh4_eq,
                hac_eq,
                ac_eq,
                h_plus_eq,
            ]
        )
        initial_guess_1 = solution
    else:
        print(f"Convergence failed for pH = {pH}: {mesg}")
        results.append([pH] + [np.nan] * 9)  # Append NaNs if no convergence

# Convert results to a NumPy array for easier handling
results_array = np.array(results)


with tab_results:
    # Plotting the results
    fig = plot_equilibrium_results_plotly(results_array, temperature)
    st.plotly_chart(fig, use_container_width=True)

    st.write("## Component Concentrations at certain pH")
    ph_col_1, ph_col_2 = st.columns(2)
    pH_input_1 = ph_col_1.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=5.0,
        step=0.1,
        format="%.1f",
        key="pH_input_1",
    )
    pH_input_2 = ph_col_2.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=8.0,
        step=0.1,
        format="%.1f",
        key="pH_input_2",
    )
    # initial guess from results_array
    initial_guess_1 = results_array[
        np.abs(results_array[:, 0] - pH_input_1).argmin(), 1:8
    ].tolist()  # Get the closest initial guess from results
    solution_1, infodict_1, ier_1, mesg_1 = fsolve(
        equilibrium_equations_at_pH,
        initial_guess_1,
        args=(pH_input_1, a, b, c, d, S_total_val, N_total_val, Ac_total_val),
        xtol=1e-10,
        full_output=True,
    )
    initial_guess_2 = results_array[
        np.abs(results_array[:, 0] - pH_input_2).argmin(), 1:8
    ].tolist()  # Get the closest initial guess from results
    solution_2, infodict_2, ier_2, mesg_2 = fsolve(
        equilibrium_equations_at_pH,
        initial_guess_2,
        args=(pH_input_2, a, b, c, d, S_total_val, N_total_val, Ac_total_val),
        xtol=1e-10,
        full_output=True,
    )

    solution_dict = {}
    if ier_1 == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution_1
        solution_dict[pH_input_1] = {
            "SO\u2082==H\u2082O": h2so3_eq,
            "HSO\u2083\u207b": hso3_eq,
            "SO\u2083\u00b2\u207b": so3_eq,
            "NH\u2084\u207a": nh4_eq,
            "NH\u2083==H\u2082O": nh3_h2o_eq,
            "HAc": hac_eq,
            "Ac\u207b": ac_eq,
        }
    else:
        st.write(f"Convergence failed for pH = {pH}: {mesg}")

    if ier_2 == 1:
        h2so3_eq, hso3_eq, so3_eq, nh3_h2o_eq, nh4_eq, hac_eq, ac_eq = solution_2
        solution_dict[pH_input_2] = {
            "SO\u2082==H\u2082O": h2so3_eq,
            "HSO\u2083\u207b": hso3_eq,
            "SO\u2083\u00b2\u207b": so3_eq,
            "NH\u2084\u207a": nh4_eq,
            "NH\u2083==H\u2082O": nh3_h2o_eq,
            "HAc": hac_eq,
            "Ac\u207b": ac_eq,
        }
    else:
        st.write(f"Convergence failed for pH = {pH}: {mesg}")

    st.dataframe(solution_dict)
