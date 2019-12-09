import plotly
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import ipdb
from igraph import Graph
import networkx as nx
import colorlover as cl
from itertools import chain, combinations
from collections import Counter, OrderedDict
from functools import reduce
import json
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import app_layout


graph = Graph.Read_Ncol("data/gov_to_gov/edges.txt")
start_vertex = "cdc.gov"
start_sensitivity = 0.75


def labeled_pagerank(graph):
    result = zip(graph.vs["name"], graph.pagerank())
    return Counter(dict(result))


original_pagerank = labeled_pagerank(graph)
rankings = {
    page: rank + 1 for rank, (page, score) in enumerate(original_pagerank.most_common())
}


def base_normalize(sub, orig, sensitivity=0.75):
    return sub / (orig ** sensitivity)


def relative_pagerank(subgraph, normalize=base_normalize, sensitivity=0.75):
    subgraph_pagerank = labeled_pagerank(subgraph)
    # for each vertex v, normalize it's subgraph pagerank by its original pagerank
    # according to the normalization function
    return Counter(
        {
            v: normalize(subgraph_pagerank[v], original_pagerank[v], sensitivity)
            for v in subgraph_pagerank.keys()
        }
    )


def get_adjacent_subgraph(vertex, mode="ALL", include_self=False):
    vertex_id = graph.vs.find(name=vertex).index
    adjacent_vertices = graph.neighbors(vertex, mode=mode)
    if not include_self:
        proper_adjacents = [v for v in adjacent_vertices if v != vertex_id]
        return graph.subgraph(proper_adjacents)
    else:
        adjacent_vertices.append(vertex_id)
        return graph.subgraph(adjacent_vertices)


def adjacent_pagerank(vertex, mode="ALL", normalize=base_normalize, sensitivity=0.75):
    subgraph = get_adjacent_subgraph(vertex, mode=mode)
    return relative_pagerank(subgraph, normalize=normalize, sensitivity=sensitivity)


def processed_pagerank(
    vertex, mode="ALL", n=10, normalize=base_normalize, sensitivity=0.75
):
    vertex_ranks = adjacent_pagerank(
        vertex, mode=mode, normalize=normalize, sensitivity=sensitivity
    ).most_common(n)
    vertices, scores = zip(*vertex_ranks)
    scores = divide_by_max(scores)
    return list(vertices), list(scores)


def cocitation(g, vertices):
    A = np.array(g.get_adjacency().data)
    v_ids = [g.vs.find(name=v).index for v in vertices]
    return {
        (g.vs[i]["name"], g.vs[j]["name"]): A[i] @ A[j]
        for i, j in combinations(v_ids, 2)
    }


def biblio(g, vertices):
    A = np.array(g.get_adjacency().data)
    v_ids = [g.vs.find(name=v).index for v in vertices]
    return {
        (g.vs[i]["name"], g.vs[j]["name"]): A[:, i] @ A[:, j]
        for i, j in combinations(v_ids, 2)
    }


def get_hyperlink(website):
    return f"<a href='https://{website}'> {website}</a>"


def divide_by_max(X):
    A = np.array(list(X))
    m = np.max(A)
    A = 1 / m * A
    return A


def list_concat(lists):
    return reduce(lambda a, b: a + b, lists, [])


def get_starting_positions(vertices):
    if not vertices:
        raise ValueError("There are no vertices")
    y_coords = np.linspace(0.5, -0.5, len(vertices))
    return {
        vertex: (np.random.uniform(-0.01, 0.01), y)
        for vertex, y in zip(vertices, y_coords)
    }


def top_edges_for_vertex(v, edge_weights):
    vertex_edge_weights = Counter(
        {edge: weight for edge, weight in edge_weights.items() if v in edge}
    )
    top_edges_for_vertex = [edge for edge, weight in vertex_edge_weights.most_common(2)]
    return top_edges_for_vertex


def get_positions(edge_weights, subgraph_vertices, mode):
    starting_positions = get_starting_positions(subgraph_vertices)
    if len(subgraph_vertices) > 1:
        weighted_edgelist = [(u, v, w) for (u, v), w in edge_weights.items()]
        g = nx.Graph()
        g.add_weighted_edges_from(weighted_edgelist)
        positions = nx.spring_layout(
            g, weight="weight", scale=1 / 2, pos=starting_positions
        ).values()
    else:
        positions = starting_positions.values()
    if mode == "OUT":
        positions = [p + np.array([2.5, 0]) for p in positions]
    return positions


def get_subgraph_edge_weights(vertex, adjacent_subgraph, subgraph_vertices, mode):
    all_edge_weights = biblio(adjacent_subgraph, subgraph_vertices + [vertex])
    adjacent_edge_weights = {
        edge: max(weight, 0.1)
        for edge, weight in all_edge_weights.items()
        if vertex in edge
    }
    local_edge_weights = {
        edge: weight for edge, weight in all_edge_weights.items() if vertex not in edge
    }
    top_local_edges = list_concat(
        top_edges_for_vertex(v, local_edge_weights) for v in subgraph_vertices
    )
    top_local_edge_weights = {
        edge: max(weight, 0.1)
        for edge, weight in local_edge_weights.items()
        if edge in top_local_edges
    }
    return top_local_edge_weights, adjacent_edge_weights


def get_node_data(vertex, subgraph_vertices, positions, sizes, mode):
    subgraph_node_data = {
        v: {"x": x, "y": y, "size": size, "type": mode.lower()}
        for v, (x, y), size in zip(subgraph_vertices, positions, sizes)
    }
    center_node_data = {vertex: {"x": 1.25, "y": 0, "size": 1.1, "type": "center"}}
    node_data = {**subgraph_node_data, **center_node_data}
    return node_data


def get_normalized_edge_weights(local_edge_weights, adjacent_edge_weights):
    edge_weights = {**local_edge_weights, **adjacent_edge_weights}
    m = max(edge_weights.values())
    normalized_edge_weights = {
        edge: weight / m for edge, weight in edge_weights.items()
    }
    return normalized_edge_weights


def get_edge_data(normalized_edge_weights, node_data):
    edge_data = [
        {"source": source, "target": target, "weight": weight}
        for (source, target), weight in normalized_edge_weights.items()
    ]
    for edge in edge_data:
        edge["u_x"] = node_data[edge["source"]]["x"]
        edge["u_y"] = node_data[edge["source"]]["y"]
        edge["v_x"] = node_data[edge["target"]]["x"]
        edge["v_y"] = node_data[edge["target"]]["y"]
    return edge_data


def get_graph_info(subgraph, mode):
    mode_text = "incoming" if mode == "IN" else "outgoing"
    n_vertices = subgraph.vcount() - 1
    return f"Number of {mode_text} edges: {n_vertices}"


def get_subgraph_data(vertex, mode="IN", sensitivity=0.75, n=6):
    # ipdb.set_trace(context=10)
    subgraph_vertices, sizes = processed_pagerank(
        vertex, mode=mode, sensitivity=sensitivity, n=n
    )
    adjacent_subgraph = get_adjacent_subgraph(vertex, mode=mode, include_self=True)
    local_edge_weights, adjacent_edge_weights = get_subgraph_edge_weights(
        vertex, adjacent_subgraph, subgraph_vertices, mode
    )
    positions = get_positions(local_edge_weights, subgraph_vertices, mode)
    node_data = get_node_data(vertex, subgraph_vertices, positions, sizes, mode)
    normalized_edge_weights = get_normalized_edge_weights(
        local_edge_weights, adjacent_edge_weights
    )
    edge_data = get_edge_data(normalized_edge_weights, node_data)
    info = get_graph_info(adjacent_subgraph, mode)
    return node_data, edge_data, info


def make_edge_trace(edge):
    return go.Scatter(
        x=(edge["u_x"], edge["v_x"], None),
        y=(edge["u_y"], edge["v_y"], None),
        line=dict(width=5 * edge["weight"], color="#999"),
        showlegend=False,
        text=f"{edge['source']} -> {edge['target']}",
        hoverinfo="none",
        mode="lines",
    )


def get_node_df(node_data):
    df = pd.DataFrame(node_data).T.reset_index()
    n_colors = len(node_data.keys())
    n_colors = str(max(3, n_colors))
    colors = cl.scales[n_colors]["qual"]["Paired"]
    df["color"] = colors[: len(df)]
    df["label_y"] = [0.5 - 0.2 * i for i in range(len(df))]
    np.linspace(-0.5, 0.5, len(df))
    df = df.rename(columns={"index": "name"})
    df["link"] = df.name.apply(get_hyperlink)
    return df


def get_half_of_graph(node_df, edge_data):
    node_traces = [
        go.Scatter(
            x=node_df.x,
            y=node_df.y,
            mode="markers",
            hoverinfo="text",
            customdata=node_df.name,
            text=node_df.name,
            legendgroup="nodes",
            marker=dict(size=0.25 * node_df.size, color=node_df.color),
            line_width=2,
            showlegend=False,
        )
    ]
    edge_traces = [make_edge_trace(edge) for edge in edge_data]
    return node_traces, edge_traces


def get_legend(df, info):
    legend_trace = go.Scatter(
        x=[0] * len(df[:-1]),
        y=df.label_y,
        text=df[:-1].link,
        marker=(dict(size=20, color=df[:-1].color)),
        showlegend=False,
        mode="markers+text",
        textposition="middle right",
        hoverinfo="none",
    )
    invis_trace = go.Scatter(
        x=[8] * len(df[:-1]),
        y=df.label_y,
        name="incoming",
        text=df[:-1].link,
        marker=(dict(size=0, color=df[:-1].color)),
        showlegend=False,
        mode="none",
        hoverinfo="none",
    )
    return {
        "data": [legend_trace, invis_trace],
        "layout": go.Layout(
            title=info,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        ),
    }


def get_state(vertex, sensitivity, cache):
    if (f"{vertex}, {sensitivity}") in cache:
        (
            in_node_data,
            in_edge_data,
            left_info,
            out_node_data,
            out_edge_data,
            right_info,
        ) = cache[f"{vertex}, {sensitivity}"]
    else:
        in_node_data, in_edge_data, left_info = get_subgraph_data(
            vertex, mode="IN", sensitivity=sensitivity
        )
        out_node_data, out_edge_data, right_info = get_subgraph_data(
            vertex, mode="OUT", sensitivity=sensitivity
        )

        cache[f"{vertex}, {sensitivity}"] = (
            in_node_data,
            in_edge_data,
            left_info,
            out_node_data,
            out_edge_data,
            right_info,
        )
    in_node_df = get_node_df(in_node_data)
    left_node_traces, left_edge_traces = get_half_of_graph(in_node_df, in_edge_data)
    out_node_df = get_node_df(out_node_data)
    right_node_traces, right_edge_traces = get_half_of_graph(out_node_df, out_edge_data)
    left_legend = get_legend(in_node_df, left_info)
    right_legend = get_legend(out_node_df, right_info)
    graph = {
        "data": left_edge_traces
        + right_edge_traces
        + left_node_traces
        + right_node_traces,
        "layout": go.Layout(
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        ),
    }
    title = f"#### [{vertex}](http://vertex): pagerank {rankings[vertex]} of {len(rankings)}"
    return (title, left_info, left_legend, graph, right_info, right_legend, cache)


external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = app_layout.layout
app.title = 'Web Flyover'


@app.callback(
    [
        Output("title-text", "children"),
        Output("left-info-panel", "children"),
        Output("left-legend-panel", "figure"),
        Output("graph", "figure"),
        Output("right-info-panel", "children"),
        Output("right-legend-panel", "figure"),
        Output("cache", "children"),
    ],
    [
        Input("graph", "clickData"),
        Input("back-button", "n_clicks"),
        Input("sensitivity-slider", "value"),
        Input("node-search", "value"),
    ],
    [
        State("left-legend-panel", "figure"),
        State("graph", "figure"),
        State("right-legend-panel", "figure"),
        State("cache", "children"),
        State("history", "children"),
    ],
)
def update_view(
    clickData,
    n_clicks,
    sensitivity,
    searched_node,
    left_legend,
    graph,
    right_legend,
    cache,
    history,
):
    history = json.loads(history) if history else []
    cache = json.loads(cache) if cache else {}
    inputs = dash.callback_context.inputs
    if app_start(inputs):
        state = get_state(start_vertex, start_sensitivity, cache)
    else:
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "graph" and clickData["points"][0].get("customdata"):
            new_vertex = clickData["points"][0]["customdata"].split()[0]
            state = get_state(new_vertex, sensitivity, cache)
        elif trigger == "back-button" and len(history) > 1:
            previous_vertex = history[-2]
            state = get_state(previous_vertex, sensitivity, cache)
        elif trigger == "sensitivity-slider":
            current_vertex = history[-1]
            state = get_state(current_vertex, sensitivity, cache)
        elif trigger == "node-search" and searched_node:
            new_vertex = searched_node
            state = get_state(new_vertex, sensitivity, cache)
        else:
            raise PreventUpdate
    (title, left_info, left_legend, graph, right_info, right_legend, cache) = state
    return (
        title,
        "",  # left_info
        left_legend,
        graph,
        "",  # right_info
        right_legend,
        json.dumps(cache),
    )


@app.callback(
    Output("sensitivity-display", "children"), [Input("sensitivity-slider", "value")]
)
def update_sensitivity_display(value):
    return f"Sensitivity: {value:.2f}"


@app.callback(
    Output("history", "children"),
    [
        Input("graph", "clickData"),
        Input("back-button", "n_clicks"),
        Input("node-search", "value"),
    ],
    [State("history", "children")],
)
def update_history(clickData, n_clicks, searched_node, history):
    if history:
        history = json.loads(history)
    else:
        history = []
    inputs = dash.callback_context.inputs
    if app_start(inputs):
        history.append(start_vertex)
    else:
        trigger = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "graph" and clickData["points"][0].get("customdata"):
            new_point = clickData["points"][0]["customdata"].split()[0]
            history.append(new_point)
        elif trigger == "back-button":
            if len(history) > 1:
                history.pop()
        elif trigger == "node-search" and searched_node:
            history.append(searched_node)
    return json.dumps(history)


@app.callback(
    Output("input-data", "children"),
    [
        Input("graph", "clickData"),
        Input("back-button", "n_clicks"),
        Input("sensitivity-slider", "value"),
        Input("node-search", "value"),
    ],
    [State("input-data", "children")],
)
def dump_input_data(clickData, n_clicks, sensitivity, searched_node, data):
    data = json.loads(data) if data else []
    inputs = dash.callback_context.inputs
    inputs["app_start"] = app_start(inputs)
    data.append(inputs)
    return json.dumps(data, indent=2)


@app.callback(
    Output("trigger-data", "children"),
    [
        Input("graph", "clickData"),
        Input("back-button", "n_clicks"),
        Input("sensitivity-slider", "value"),
        Input("node-search", "value"),
    ],
    [State("trigger-data", "children")],
)
def dump_trigger_data(clickData, n_clicks, sensitivity, searched_node, data):
    data = json.loads(data) if data else []
    trigger = dash.callback_context.triggered
    data.append(trigger)
    return json.dumps(data, indent=2)
    # dash.callback_context.triggered[0]["prop_id"].split(".")[0]


@app.callback(Output("click-data", "children"), [Input("graph", "clickData")])
def dump_click_data(clickData):
    return json.dumps(clickData, indent=2)


def app_start(inputs):
    graph_clicked = bool(inputs.get("graph.clickData"))
    sensitivity_adj = bool(
        inputs.get("sensitivity-slider.value")
        and inputs.get("sensitivity-slider.value") != 0.75
    )
    back_pressed = bool(inputs.get("back-button.n_clicks"))
    node_been_searched = inputs.get("node-search.value") != start_vertex
    if not any([graph_clicked, sensitivity_adj, back_pressed, node_been_searched]):
        return True
    return False


if __name__ == "__main__":
    app.run_server(debug=False)
