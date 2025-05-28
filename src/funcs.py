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
