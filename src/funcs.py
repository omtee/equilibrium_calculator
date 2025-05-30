import numpy as np


def equilibrium_equations_at_pH(
    unknowns: list[float],
    pH: float,
    a: float,
    b: float,
    c: float,
    d: float,
    S_total: float,
    N_total: float,
    Ac_total: float,
) -> list[float]:
    SO2_H2O, HSO3, SO3, NH4, NH3_H2O, HAc, Ac = unknowns
    h_plus = 10**-pH

    eq1 = a * h_plus - SO2_H2O / HSO3
    eq2 = b * h_plus - HSO3 / SO3
    eq3 = c * h_plus - NH4 / NH3_H2O
    eq4 = d * h_plus - HAc / Ac
    eq5 = S_total - SO2_H2O - HSO3 - SO3
    eq6 = N_total - NH4 - NH3_H2O
    eq7 = Ac_total - Ac - HAc

    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]


def linear(x: float, a: float, b: float) -> float:
    return a * x + b


def quadratic(x: float, a: float, b: float, c: float) -> float:
    return a * x**2 + b * x + c


def power_law(x: float, a: float, b: float) -> float:
    return a * x**b


def exponential(x: float, a: float, b: float) -> float:
    return a * np.exp(b * x)


def logarithmic(x: float, a: float, b: float) -> float:
    return a * np.log(b * x)


def reciprocal(x: float, a: float, b: float) -> float:
    return a / (b * x)


def constant(x: float, a: float) -> np.ndarray:
    return a * np.ones_like(x)


def van_t_hoff(T_celsius: float, delta_H: float, delta_S: float) -> float:
    """
    Calculates the natural logarithm of the 'c' constant using a form
    derived from the van 't Hoff equation for the temperature dependence of Ka.

    Parameters:
    T_celsius (float): Temperature in Celsius.
    delta_H (float): Standard enthalpy change of the reaction (J/mol).
                     This corresponds to the reaction NH4+ + H2O <=> NH3 + H3O+.
    delta_S (float): Standard entropy change of the reaction (J/(mol*K)).
                     This corresponds to the reaction NH4+ + H2O <=> NH3 + H3O+.

    Returns:
    float: The natural logarithm of the 'c' constant at temperature T_celsius.
    """
    # Universal gas constant
    R = 8.314  # J/(mol*K)

    # Convert temperature to Kelvin and calculate inverse
    T_kelvin = T_celsius + 273.15
    inv_T = 1 / T_kelvin

    # Van't Hoff equation for ln(Ka): ln(Ka) = - (delta_H / R) * (1/T) + (delta_S / R)
    # Since c = 1/Ka, then ln(c) = -ln(Ka)
    # So, ln(c) = (delta_H / R) * (1/T) - (delta_S / R)

    # Calculate ln(c) based on the derived Van't Hoff form
    ln_c_val = (delta_H / R) * inv_T - (delta_S / R)

    return ln_c_val
