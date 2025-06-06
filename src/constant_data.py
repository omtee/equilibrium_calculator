from typing import Dict

from EquilibriumConstant import EquilibriumConstant
from funcs import power_law, constant, quadratic, van_t_hoff


constant_data: Dict[str, EquilibriumConstant] = {
    "a": EquilibriumConstant(
        name="a",
        equation="SO2_H2O = HSO3",
        data={
            25: 90.0,
            100: 4.5e2,
            150: 3.5e3,
        },
        fit_func=power_law,
    ),
    "b": EquilibriumConstant(
        name="b",
        equation="HSO3 = SO3",
        data={
            25: 1.6e7,
            100: 1.6e7,
            150: 1.6e7,
        },
        fit_func=constant,
    ),
    "c": EquilibriumConstant(
        name="c",
        equation="NH4 = NH3_H2O (power)",
        data={
            25: 2.0e9,
            100: 2.0e7,
            150: 2.0e6,
        },
        fit_func=power_law,
    ),
    "c2": EquilibriumConstant(
        name="c2",
        equation="NH4 = NH3_H2O (van 't Hoff)",
        data={
            25: 2.0e9,
            100: 2.0e7,
            150: 2.0e6,
        },
        fit_func=van_t_hoff,
    ),
    "d": EquilibriumConstant(
        name="d",
        equation="HAc = Ac",
        data={
            25: 6.0e4,
            100: 8.0e4,
            150: 10.0e4,
        },
        fit_func=quadratic,
    ),
}
