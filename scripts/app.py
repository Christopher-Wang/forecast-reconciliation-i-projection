import numpy as np
import pandas as pd
import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True, cwd=True)

from src.utils.general_utils import ListableEnum
from src.datasets_manager import HierarchicalDatasetNames, HierarchicalDataset
from src.experiments.results_collector import ResultsCollector, MetricName, ResultKey
import streamlit as st
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import io
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def add_create_pdf_button(fig: go.Figure | plt.Figure, file_name: str, width: float | None, height: float | None) -> None:
    # Create PDF in memory
    pdf_bytes = io.BytesIO()

    if isinstance(fig, go.Figure):
        # fig.update_layout(autosize=False)
        pio.write_image(fig, pdf_bytes, width=1000, format="pdf")

    elif isinstance(fig, plt.Figure):
        fig.savefig(pdf_bytes, format="pdf", bbox_inches='tight')

    else:
        st.error("Unsupported figure type. Please pass a Plotly or Matplotlib figure.")
        return

    pdf_bytes.seek(0)

    # Show download button
    st.download_button(
        label="📄 Download PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
        key=file_name
    )


def hex_to_rgba(hex_color: str, alpha=1.0) -> str:
    """ Convert hex color to RGBA format

    Args:
        hex_color: color in hex format
        alpha: target transparency

    Returns:
          rgba: color in RGBA format
    """
    hex_color = hex_color.lstrip("#")
    rgba = f"rgba({int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}, {alpha})"
    return rgba


def plot_perf_violin(
        list_exp_to_perfs: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        list_exp_to_distrib_color: list[dict[str, str]] | dict[str, str],
        annotations_list: list[str | None] | str | None,
        sort_exps_per_perfs: bool,
        yaxis_title: str,
        width_per_res: float = 300,
        height: float = 600
) -> go.Figure:
    """
    Plot quantile violin plots per group (groups involve several methods being compared -- if only one group it's fine)

    Args:
        list_exp_to_perfs: list of dictionary mapping method to quantiles
        list_exp_to_distrib_color: list of dictionary mapping method to color (for violin and box)
        annotations_list: list of group annotations
        sort_exps_per_perfs: whether to show violin plots ordered by performance
        width_per_res: width for each result plot
        short_medal_form: whether to use short medal name (e.g. "B" instead of "Bronze") in tick labels
        height: figure height

    Returns:

    """
    # Prepare the violin plot traces for each category
    fig = go.Figure()

    if isinstance(list_exp_to_perfs, dict):
        list_exp_to_perfs = [list_exp_to_perfs]
    if isinstance(list_exp_to_distrib_color, dict):
        list_exp_to_distrib_color = [list_exp_to_distrib_color]
    if not isinstance(annotations_list, list):
        annotations_list = [annotations_list]

    assert len(list_exp_to_perfs) == len(list_exp_to_distrib_color)

    n_groups = len(list_exp_to_perfs)

    x_tick_labels = []
    x_tick_vals = []
    i = 0

    for group_ind in range(n_groups):
        results_dict = list_exp_to_perfs[group_ind]
        distrib_colors_dict = list_exp_to_distrib_color[group_ind]
        annotation = annotations_list[group_ind]

        if i > 0:  # vertical line to separate the different groups
            # fig.add_trace(
            #     go.Scatter(
            #         x=[i - .2, i - .2],
            #         y=[0, 115],
            #         mode='lines',
            #         line=dict(color='grey', width=5)
            #     )
            # )
            i += .5

        group_inds = []

        methods = list(results_dict.keys())
        if sort_exps_per_perfs:
            inds = np.argsort([np.mean(results_dict[method]) for method in methods])
            methods = [methods[i] for i in inds]

        for method in methods:
            x_tick_labels.append(method)
            x_tick_vals.append(i)
            group_inds.append(i)
            perfs = results_dict[method]
            method_color = distrib_colors_dict[method]
            fig.add_trace(
                go.Scatter(
                    x=i - .2 - np.random.rand(len(perfs)) / 8,
                    y=perfs,
                    mode='markers',
                    name=method,
                    marker=dict(size=10, opacity=0.75, color=method_color, line=dict(color=None, width=0)),
                )
            )

            fig.add_trace(
                go.Violin(
                    x=[i - .05 for _ in range(len(perfs))],
                    y=perfs,
                    box_visible=False,
                    meanline_visible=True,
                    line_color=method_color,
                    fillcolor=method_color,
                    name=method,
                    points=False,
                    side="positive",
                    # span=[0, 100],
                    # spanmode='manual',
                    # bandwidth=15
                )
            )

            fig.add_trace(
                go.Box(
                    x=[i + .35 for _ in range(len(perfs))],
                    y=perfs,
                    name=method,
                    marker_color=method_color,
                    width=0.2,
                    boxpoints='all',
                    boxmean=True,
                )
            )
            i += 1

        if annotation is not None:
            # fig.add_annotation(
            #     x=np.mean(group_inds),
            #     y=110,
            #     text=annotation,
            #     showarrow=False,
            #     font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            #     align="center",
            #     xanchor="center",
            #     yanchor="middle"
            # )
            pass

    fig.update_layout(
        # xaxis_title="Method",
        yaxis_title=yaxis_title,
        showlegend=False,
        plot_bgcolor="white",
        height=height,
        width=width_per_res * len(x_tick_vals),
        xaxis=dict(
            color=hex_to_rgba(hex_color="#000000", alpha=0),
            tickvals=x_tick_vals,
            ticktext=x_tick_labels,
            tickangle=0,
            title_font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            tickfont=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000)
        ),
        yaxis=dict(
            # tickvals=np.arange(20, 120, 20),
            # ticktext=np.arange(20, 120, 20),
            title_font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            tickfont=dict(size=18, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            showgrid=False,
        ),
    )

    return fig


def plot_perf_boxplot(
        list_exp_to_perfs: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
        list_exp_to_distrib_color: list[dict[str, str]] | dict[str, str],
        annotations_list: list[str | None] | str | None,
        sort_exps_per_perfs: bool,
        yaxis_title: str,
        width_per_res: float = 300,
        height: float = 600
) -> go.Figure:
    """
    Plot perf box-plots per group (groups involve several methods being compared -- if only one group it's fine)

    Args:
        list_exp_to_perfs: list of dictionary mapping method to perfs
        list_exp_to_distrib_color: list of dictionary mapping method to color
        annotations_list: list of group annotations
        sort_exps_per_perfs: whether to show plots ordered by performance
        yaxis_title:  name for the y-axis
        width_per_res: width for each result plot
        height: figure height

    Returns:
        The figure
    """
    # Prepare the violin plot traces for each category
    fig = go.Figure()

    if isinstance(list_exp_to_perfs, dict):
        list_exp_to_perfs = [list_exp_to_perfs]
    if isinstance(list_exp_to_distrib_color, dict):
        list_exp_to_distrib_color = [list_exp_to_distrib_color]
    if not isinstance(annotations_list, list):
        annotations_list = [annotations_list]

    assert len(list_exp_to_perfs) == len(list_exp_to_distrib_color)

    n_groups = len(list_exp_to_perfs)

    x_tick_labels = []
    x_tick_vals = []
    i = 0

    for group_ind in range(n_groups):
        results_dict = list_exp_to_perfs[group_ind]
        distrib_colors_dict = list_exp_to_distrib_color[group_ind]
        annotation = annotations_list[group_ind]

        if i > 0:  # vertical line to separate the different groups
            # fig.add_trace(
            #     go.Scatter(
            #         x=[i - .2, i - .2],
            #         y=[0, 115],
            #         mode='lines',
            #         line=dict(color='grey', width=5)
            #     )
            # )
            i += .5

        group_inds = []

        methods = list(results_dict.keys())
        if sort_exps_per_perfs:
            inds = np.argsort([np.mean(results_dict[method]) for method in methods])
            methods = [methods[i] for i in inds]

        for method in methods:
            x_tick_labels.append(method)
            x_tick_vals.append(i)
            group_inds.append(i)
            perfs = results_dict[method]
            method_color = distrib_colors_dict[method]
            fig.add_trace(
                go.Scatter(
                    x=i + .15 - .3 * np.random.rand(len(perfs)),
                    y=perfs,
                    mode='markers',
                    name=method,
                    marker=dict(size=10, opacity=0.75, color=method_color, line=dict(color='black', width=1)),
                )
            )
            lowerfence, upperfence = None, None
            if len(perfs) < 9:
                lowerfence, upperfence = (min(perfs),), (max(perfs),)
            fig.add_trace(
                go.Box(
                    x=[i for _ in range(len(perfs))],
                    y=perfs,
                    name=method,
                    marker_color=method_color,
                    width=0.4,
                    boxpoints=None,
                    upperfence=upperfence,
                    lowerfence=lowerfence,
                    boxmean=True,
                )
            )
            i += 1

        if annotation is not None:
            # fig.add_annotation(
            #     x=np.mean(group_inds),
            #     y=110,
            #     text=annotation,
            #     showarrow=False,
            #     font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            #     align="center",
            #     xanchor="center",
            #     yanchor="middle"
            # )
            pass

    fig.update_layout(
        # xaxis_title="Method",
        yaxis_title=yaxis_title,
        showlegend=False,
        plot_bgcolor="white",
        height=height,
        width=width_per_res * len(x_tick_vals),
        xaxis=dict(
            color=hex_to_rgba(hex_color="#000000", alpha=0),
            tickvals=x_tick_vals,
            ticktext=x_tick_labels,
            tickangle=0,
            title_font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            tickfont=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000)
        ),
        yaxis=dict(
            # tickvals=np.arange(20, 120, 20),
            # ticktext=np.arange(20, 120, 20),
            title_font=dict(size=20, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            tickfont=dict(size=18, color='black', family="Computer Modern, Times New Roman, serif", weight=1000),
            showgrid=False,
        ),
    )

    return fig


st.set_page_config(layout="wide")  # wide layout

dataset_names = [
    HierarchicalDatasetNames.TOURISM_SMALL,
    HierarchicalDatasetNames.LABOUR,
    HierarchicalDatasetNames.WIKI2,
    HierarchicalDatasetNames.TRAFFIC,
]
n_cross_vals = 5
horizons = [HierarchicalDataset.get_hierarchical_dataset(dataset_name=ds).default_horizon for ds in dataset_names]


class StateKey(ListableEnum):

    @staticmethod
    def get_all_results_key(dataset_names_: list[HierarchicalDatasetNames]) -> str:
        return "-".join(sorted([d.value for d in dataset_names_]))


if StateKey.get_all_results_key(dataset_names_=dataset_names) not in st.session_state:
    try:
        all_results = pd.read_csv("./all_results_.csv", index_col=0)
    except FileNotFoundError:
        all_results = ResultsCollector.collect_results(
            dataset_names=dataset_names, horizons=horizons, n_cross_vals=n_cross_vals,
            metric_names=[MetricName.RMSE, MetricName.MAE]
        )
        all_results.to_csv("./all_results.csv")
    st.session_state[StateKey.get_all_results_key(dataset_names_=dataset_names)] = all_results

# st.title("Results Dashboard")

# Create one tab per dataset
tabs = st.tabs([d.value for d in dataset_names])

for i, dataset_name in enumerate(dataset_names):
    with (tabs[i]):
        st.subheader(f"Results for {dataset_name.value}")

        st.markdown("#### Raw results:")
        metrics_to_show = [MetricName.RMSE.value]
        data = st.session_state[StateKey.get_all_results_key(dataset_names_=dataset_names)]
        data = data[data[ResultKey.DATASET.value] == dataset_name.value]
        for metric_name, metric_data in data.groupby(ResultKey.METRIC.value):
            if metric_name not in metrics_to_show:
                continue

            st.markdown(f"##### Metric: {metric_name}")
            bottom_level = max(data[ResultKey.LEVEL.value].unique().tolist(), key=lambda lev: lev.count("/"))
            if "/" not in bottom_level:
                bottom_level = max([lev for lev in data[ResultKey.LEVEL.value].unique().tolist() if lev != "Overall"])
            overall_level = "Overall"

            level_descrs = [(bottom_level, "Lowest"), (overall_level, "Overall")]

            for level_descr in level_descrs:
                level, description = level_descr

                st.markdown(f"Level: {level}")
                level_data = metric_data[metric_data[ResultKey.LEVEL.value] == level]
                exp_to_perfs = {}
                exp_to_color = {}
                for method, method_data in level_data.groupby(ResultKey.METHOD.value):
                    assert len(method_data) == n_cross_vals, len(method_data)
                    method_label = method
                    if "MinT" in method_label:
                        mint_spec = "-".join(method_label.split("-")[1:]).replace("nonneg", "+")
                        method_label = f"MinT<br>{mint_spec}"
                    exp_to_perfs[method_label] = method_data[ResultKey.VALUE.value].values.flatten()
                    exp_to_color[method_label] = method_data[ResultKey.METHOD_COLOR.value].values.flatten()[0]

                height = 450
                width_per_res = 150
                width = width_per_res * len(exp_to_perfs)
                fig = plot_perf_boxplot(
                    list_exp_to_perfs=exp_to_perfs,
                    list_exp_to_distrib_color=exp_to_color,
                    annotations_list=None,
                    sort_exps_per_perfs=True,
                    yaxis_title=f'{metric_name} ({dataset_name.value} {description} level)',
                    height=height,
                    width_per_res=width_per_res
                )

                fig_name = f"boxplot-{dataset_name.value}-{description}_level-{metric_name}"
                add_create_pdf_button(fig=fig, file_name=f"{fig_name}.pdf", height=height, width=width)
                st.plotly_chart(fig, use_container_width=False, key=fig_name)

            for fold, fold_data in metric_data.groupby(ResultKey.FOLD.value):
                st.markdown(f"###### Fold {fold + 1}")
                fold_data = fold_data[[ResultKey.LEVEL.value, ResultKey.METHOD.value, ResultKey.VALUE.value]]
                fold_data = fold_data.pivot(
                    index=ResultKey.LEVEL.value, columns=ResultKey.METHOD.value, values=ResultKey.VALUE.value
                ).reset_index().set_index(ResultKey.LEVEL.value)

                ranks = fold_data.rank(axis=1, method="min")


                # formatter for significant digits
                def format_sig(x, sig=4):
                    return np.format_float_positional(x, precision=sig, unique=False, fractional=False, trim='k')


                for col in fold_data.columns:
                    fold_data[col] = [
                        f"{np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')} ({int(rank)})"
                        for val, rank in zip(fold_data[col], ranks[col])
                    ]

                st.dataframe(fold_data)

            # levels = data[ResultKey.LEVEL.value].unique()
            # bottom_level = max(levels, key=lambda x: x.count("/"))

            # Filter for dataset
            # df_ds = data[data["dataset"] == ds]
            #
            # # Create a plotly bar chart (you can change to line/scatter/etc.)
            # fig = px.bar(
            #     df_ds,
            #     x="method",
            #     y="score",
            #     title=f"Comparison of methods on {ds}",
            #     color="method"
            # )
            #
            # st.plotly_chart(fig, use_container_width=True)
