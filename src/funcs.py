import numpy as np
from scipy.optimize import curve_fit

def equilibrium_equations_at_pH(
    unknowns: list[float],
    pH: float, 
    a: float, 
    b: float, 
    c: float, 
    d: float, 
    S_total: float, 
    N_total: float, 
    Ac_total: float
) -> list[float]:
    SO2_H2O, HSO3, SO3, NH4, NH3_H2O, HAc, Ac = unknowns
    h_plus = 10 ** -pH

    eq1 = a * h_plus - SO2_H2O / HSO3
    eq2 = b * h_plus - HSO3 / SO3
    eq3 = c * h_plus - NH4 / NH3_H2O
    eq4 = d * h_plus - HAc / Ac
    eq5 = S_total - SO2_H2O - HSO3 - SO3
    eq6 = N_total - NH4 - NH3_H2O
    eq7 = Ac_total - Ac - HAc

    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

constants_at_temp: dict[str, dict[int, float]] = {
    'a': {
        25: 90.0,
        100: 4.5e2,
        150: 3.5e3, 
    },
    'b': {
        25: 1.6e7,
        100: 1.6e7,
        150: 1.6e7,
    },
    'c': {
        25: 2.0e9,
        100: 2.0e7,
        150: 2.0e6,
    },
    'd': {
        25: 6.0e4,
        100: 8.0e4,
        150: 10.0e4,
    },
}

# interpolation function for constants
def interpolate_constants_temp(temp: float) -> dict[str, float]:
    constants = {}
    for key, temp_values in constants_at_temp.items():
        temps = np.array(list(temp_values.keys()))
        values = np.array(list(temp_values.values()))
        constants[key] = np.interp(temp, temps, values, left=0, right=200)
    return constants

def power_law(x, a, b):
    return a * x ** b

def fit_power_law(x_data, y_data):
    popt, _ = curve_fit(power_law, x_data, y_data)
    return popt
