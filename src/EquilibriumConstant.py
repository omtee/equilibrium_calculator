from typing import Callable, Dict

import numpy as np
from scipy.optimize import curve_fit

from src.funcs import van_t_hoff


class EquilibriumConstant:
    def __init__(
        self,
        name: str,
        equation: str,
        data: Dict[float, float],
        fit_func: Callable,
    ):
        if fit_func is None:
            raise ValueError("fit_func must be a callable function.")

        self.name = name
        self.equation = equation
        self.data = data
        self._fit_func = fit_func
        self._fit_params = None

        self._curve_fit()

    @property
    def x_data(self) -> np.ndarray:
        return np.array(list(self.data.keys()))

    @property
    def y_data(self) -> np.ndarray:
        return np.array(list(self.data.values()))

    def _curve_fit(self, **kwargs) -> None:
        y_data = self.y_data
        # For van 't Hoff, we use the natural log of the y_data (temperature in Celsius)
        if self._fit_func == van_t_hoff:
            y_data = np.log(y_data)

        popt, _ = curve_fit(self._fit_func, self.x_data, y_data, **kwargs)
        self._fit_params = popt

    def constants_at_temp(self, temp: float) -> float:
        if self._fit_func is None:
            raise ValueError(
                "Data is not fitted. Use one of the fit methods e.g, fit_power() or fit_linear()."
            )
        # For van 't Hoff, we take exponent of the fitted value
        if self._fit_func == van_t_hoff:
            return np.exp(self._fit_func(temp, *self._fit_params))
        return self._fit_func(temp, *self._fit_params)
