from __future__ import annotations

import plotly.graph_objects as go


def empty_figure(title: str, height: int = 450) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=height, autosize=False)
    return fig
