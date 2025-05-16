from typing import Callable, Any, Dict, Iterable, List
import numpy as np
from scipy.optimize import curve_fit

def equilibrium_equations_at_pH(
    unknowns: List[float],
    pH: float, 
    a: float, 
    b: float, 
    c: float, 
    d: float, 
    S_total: float, 
    N_total: float, 
    Ac_total: float
) -> List[float]:
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

def linear(x: float, a: float, b: float) -> float:
    return a * x + b

def quadratic(x: float, a: float, b: float, c: float) -> float:
    return a * x ** 2 + b * x + c

def power_law(x: float, a: float, b: float) -> float:
    return a * x ** b

def exponential(x: float, a: float, b: float) -> float:
    return a * np.exp(b * x)

def logarithmic(x: float, a: float, b: float) -> float:
    return a * np.log(b * x)

def reciprocal(x: float, a: float, b: float) -> float:
    return a / (b * x)

def constant(x: float, a: float) -> np.ndarray:
    return a * np.ones_like(x)

def fit(func: Callable, x_data: Iterable[float], y_data: Iterable[float]) -> Any:
    popt, _ = curve_fit(func, x_data, y_data)
    return popt

class EquilibriumConstant:
    def __init__(self, name: str, equation: str, data: Dict[float, float]):
        self.name = name
        self.equation = equation
        self.data = data
        self._fit_params = None
        self._fit_func = None

    @property
    def x_data(self) -> np.ndarray:
        return np.array(list(self.data.keys()))
    
    @property
    def y_data(self) -> np.ndarray:
        return np.array(list(self.data.values()))

    def _fit_func_template(self, func: Callable) -> None:
        popt, _ = curve_fit(func, self.x_data, self.y_data)
        self._fit_params = popt
        self._fit_func = func

    def fit_power(self) -> None:
        self._fit_func_template(power_law)
    
    def fit_linear(self) -> None:
        self._fit_func_template(linear)

    def fit_quadratic(self) -> None:
        self._fit_func_template(quadratic)

    def fit_exponential(self) -> None:
        self._fit_func_template(exponential)

    def fit_logarithmic(self) -> None:
        self._fit_func_template(logarithmic)

    def fit_reciprocal(self) -> None:
        self._fit_func_template(reciprocal)

    def fit_constant(self) -> None:
        self._fit_func_template(constant)
    
    def constants_at_temp(self, temp: float) -> float:
        if self._fit_func is None:
            raise ValueError("Fit the data first using fit_power or fit_linear.")
        return self._fit_func(temp, *self._fit_params)

constant_data: Dict[str, EquilibriumConstant] = {
    'a': EquilibriumConstant(
        name='a',
        equation='SO2_H2O = HSO3',
        data={
            25: 90.0,
            100: 4.5e2,
            150: 3.5e3, 
        },
    ),
    'b': EquilibriumConstant(
        name='b',
        equation='HSO3 = SO3',
        data={
            25: 1.6e7,
            100: 1.6e7,
            150: 1.6e7,
        },
    ),
    'c': EquilibriumConstant(
        name='c',
        equation='NH4 = NH3_H2O',
        data={
            25: 2.0e9,
            100: 2.0e7,
            150: 2.0e6,
        },
    ),
    'd': EquilibriumConstant(
        name='d',
        equation='HAc = Ac',
        data={
            25: 6.0e4,
            100: 8.0e4,
            150: 10.0e4,
        },
    ),
}
