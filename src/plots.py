from typing import Literal
import numpy as np
import plotly.graph_objects as go

from EquilibriumConstant import EquilibriumConstant


def plot_equilibrium_results_plotly(results_array: np.ndarray, temp: float) -> None:
    rounded_results = np.round(results_array, 4)  # Round results to 4 decimal places
    pH_values = rounded_results[:, 0]  # First column is pH values
    results = rounded_results[
        :, 1:
    ]  # Exclude the pH column for filtering, round to 4 decimal places

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 0],
            mode="lines",
            name="SO\u2082=H\u2082O",
            line=dict(color="red", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 1],
            mode="lines",
            name="HSO\u2083\u207b",
            line=dict(color="red", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 2],
            mode="lines",
            name="SO\u2083\u207b\u00b2",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 3],
            mode="lines",
            name="NH\u2084\u207a",
            line=dict(color="blue", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 4],
            mode="lines",
            name="NH\u2083=H\u2082O",
            line=dict(color="blue", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 5],
            mode="lines",
            name="HAc",
            line=dict(color="green", dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pH_values,
            y=results[:, 6],
            mode="lines",
            name="Ac\u207b",
            line=dict(color="green", dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"NH3-SO2-Ac-H2O system at {temp:.0f} °C",
            yanchor="top",
            y=1,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        xaxis_title="pH",
        yaxis_title="mol/l",
        yaxis_type="log",
        xaxis_tickvals=np.arange(min(pH_values), max(pH_values) + 1, 1),
        # yaxis_tickvals=[0.01, 0.1, 1, 10],
        yaxis_range=[-2, 1],
        xaxis_tickmode="linear",
        xaxis_side="top",
        xaxis_showgrid=True,
        xaxis_showspikes=True,
        xaxis_spikemode="across",
        yaxis_showspikes=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
        hovermode=None,
        height=600,
        width=800,
    )

    return fig


def plot_constant(
    temp_range: np.ndarray,
    const_data: EquilibriumConstant,
    yaxis_type: Literal["linear", "log"] = "linear",
):
    fit_vals = const_data.constants_at_temp(temp_range)
    x = list(const_data.x_data)
    y = list(const_data.y_data)
    eqn = const_data.equation

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=temp_range,
            y=fit_vals,
            mode="lines",
            name=f"fit ({const_data._fit_func.__name__})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="measured",
            marker=dict(size=10, color="red"),
        )
    )
    fig.update_layout(
        title=f"Constant for {eqn}",
        xaxis_title="Temperature (°C)",
        xaxis_dtick=25,
        yaxis_type=yaxis_type,
    )
    return fig
