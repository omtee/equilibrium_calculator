import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import plotly.graph_objects as go

def plot_equilibrium_results_plt(results_array: np.ndarray) -> plt.Figure:
    pH_values = results_array[:, 0]
    filtered_results = results_array[:, 1:]  # Exclude the pH column for filtering

    plt.figure(figsize=(10, 6))
    plt.plot(pH_values, filtered_results[:, 0], label='$SO_2$==$H_2O$', color='red', linestyle='-')
    plt.plot(pH_values, filtered_results[:, 1], label='$HSO_3^-$', color='red', linestyle='-.')
    plt.plot(pH_values, filtered_results[:, 2], label='$SO_3^{-2}$', color='red', linestyle='--')
    plt.plot(pH_values, filtered_results[:, 3], label='$NH_4^+$', color='blue', linestyle='-')
    plt.plot(pH_values, filtered_results[:, 4], label='$NH_3$==$H_2O$', color='blue', linestyle='--')
    plt.plot(pH_values, filtered_results[:, 5], label='$HAc$', color='green', linestyle='-')
    plt.plot(pH_values, filtered_results[:, 6], label='$Ac^-$', color='green', linestyle='--')

    plt.xlabel('pH')
    plt.ylabel('mol/l')
    plt.yscale('log')
    plt.ylim(0.01, 10)
    plt.yticks([0.01, 0.1, 1, 10])
    plt.xlim(min(pH_values), max(pH_values))
    plt.xticks(np.arange(min(pH_values), max(pH_values) + 1, 1))  # Set x-axis tick interval to 1 pH
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format y-axis with decimals
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=7)  # Legend at the bottom

    # Add x-axis at the top
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    return plt

def plot_equilibrium_results_plotly(results_array: np.ndarray, temp: float) -> None:
    rounded_results = np.round(results_array, 4)  # Round results to 4 decimal places
    pH_values = rounded_results[:, 0]  # First column is pH values
    results = rounded_results[:, 1:]  # Exclude the pH column for filtering, round to 4 decimal places

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 0], mode='lines', name='SO\u2082=H\u2082O', line=dict(color='red', dash='solid')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 1], mode='lines', name='HSO\u2083\u207B', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 2], mode='lines', name='SO\u2083\u207B\u00B2', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 3], mode='lines', name='NH\u2084\u207A', line=dict(color='blue', dash='solid')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 4], mode='lines', name='NH\u2083=H\u2082O', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 5], mode='lines', name='HAc', line=dict(color='green', dash='solid')))
    fig.add_trace(go.Scatter(x=pH_values, y=results[:, 6], mode='lines', name='Ac\u207B', line=dict(color='green', dash='dash')))

    fig.update_layout(
        title=dict(
            text=f'NH3-SO2-Ac-H2O system at {temp:.0f} °C',
            yanchor='top',
            y=1,
            xanchor='center',
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
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
        hovermode="x unified",
        height=600,
        width=800,
    )

    return fig