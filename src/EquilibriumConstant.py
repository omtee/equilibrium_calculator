from typing import Callable, Dict

import numpy as np
from scipy.optimize import curve_fit

from src.funcs import (
    constant,
    exponential,
    linear,
    logarithmic,
    power_law,
    quadratic,
    reciprocal,
)


class EquilibriumConstant:
    def __init__(
        self,
        name: str,
        equation: str,
        data: Dict[float, float],
        fit_func: Callable = None,
    ):
        self.name = name
        self.equation = equation
        self.data = data
        self._fit_func = fit_func
        self._fit_params = None

        if fit_func is not None:
            self._fit_func_template(fit_func)

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
            raise ValueError(
                "Data is not fitted. Use one of the fit methods e.g, fit_power() or fit_linear()."
            )
        return self._fit_func(temp, *self._fit_params)
