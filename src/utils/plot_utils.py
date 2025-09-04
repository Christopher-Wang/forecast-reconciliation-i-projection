import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.core.display import SVG
from IPython.core.display_functions import display

from src.utils.general_utils import get_project_root


def is_running_in_pycharm() -> bool:
    """ Return whether a notebook is running in pycharm """
    return "LC_ALL" in os.environ


def plot_show(fig: go.Figure | plt.Figure) -> None:
    if is_running_in_pycharm():
        svg_path = f"{get_project_root()}/tmp/tmp.svg"
        os.makedirs(os.path.dirname(svg_path), exist_ok=True)
        if isinstance(fig, go.Figure):
            fig.write_image(svg_path)
        elif isinstance(fig, plt.Figure):
            plt.savefig(svg_path)
        else:
            raise NotImplementedError
        display(SVG(svg_path))
    else:
        fig.show()
