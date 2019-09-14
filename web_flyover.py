import colorlover as cl
import plotly
import plotly.graph_objects as go
from itertools import chain
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from igraph import Graph
from collections import Counter, OrderedDict
colors = cl.scales['8']['qual']['Paired']
from itertools import combinations
import pandas as pd
from functools import reduce
from igraph import plot
import plotly.io as pio
import networkx as nx
import json

graph = Graph.Read_Ncol('edgelists/gov_to_gov.txt')    

def labeled_pagerank(graph):
    result = zip(graph.vs['name'], graph.pagerank())
    return Counter(dict(result))

original_pagerank = labeled_pagerank(graph)

def base_normalize(sub, orig, sensitivity=0.75):
    return sub / (orig ** sensitivity)

def relative_pagerank(subgraph, normalize=base_normalize, sensitivity=0.75):
    subgraph_pagerank = labeled_pagerank(subgraph)
    return Counter({key : normalize(subgraph_pagerank[key], original_pagerank[key], sensitivity) 
                        for key in subgraph_pagerank.keys()})

def adjacent_subgraph(vertex, mode='ALL', include_self=False):
    vertex_id = graph.vs.find(name=vertex).index
    adjacent_vertices = graph.neighbors(vertex, mode=mode)
    if not include_self:
        proper_adjacents = [v for v in adjacent_vertices if v != vertex_id]
        return graph.subgraph(proper_adjacents)
    else:
        adjacent_vertices.append(vertex_id)
        return graph.subgraph(adjacent_vertices)

def adjacent_pagerank(vertex, mode='ALL', normalize=base_normalize, sensitivity=0.75):
    subgraph = adjacent_subgraph(vertex, mode=mode)
    return relative_pagerank(subgraph, normalize=normalize, sensitivity=sensitivity)

def processed_pagerank(vertex, mode='ALL', n=10, normalize=base_normalize, sensitivity=0.75):
    vertex_ranks = adjacent_pagerank(vertex, mode=mode, normalize=normalize, sensitivity=sensitivity).most_common(n)
    vertices, scores = zip(*vertex_ranks)
    scores = divide_by_max(scores)
    return vertices, scores

def get_default_layout():
    node_x = [.1]*5 + [.25] + [.4]*5
    node_y = [.1 * i  for i in range(6, 1, -1)] + [.4] + [.1 * i  for i in range(6, 1, -1)]
    edge_x = [.1, .25, None]*5 + [.25, .4, None]*5
    edge_y = [(.1 * i, .4, None) for i in range(6, 1, -1)] + [(.4, .1 * i, None) for i in range(6, 1, -1)]
    edge_y = list(chain(*edge_y))
    return node_x, node_y, edge_x, edge_y

def get_vs_and_sizes(vertex, sensitivity=0.75):
    in_v, in_s = processed_pagerank(vertex, mode='IN', n=5, sensitivity=sensitivity)
    out_v, out_s = processed_pagerank(vertex, mode='OUT', n=5, sensitivity=sensitivity)
    
    vertices = list(in_v) + [vertex] + list(out_v)
    sizes = list(in_s) + [1.15] + list(out_s)
    labels = ["<a href='https://{}'> {}</a>".format(i, i) for i in vertices]
    sizes = [30 * (size**3) for size in sizes]
    return tuple(vertices), tuple(sizes)

def cocitation(g, vertices):
    A = np.array(g.get_adjacency().data)
    v_ids = [g.vs.find(name=v).index for v in vertices]
    return {(g.vs[i]['name'], g.vs[j]['name']) : A[i] @ A[j] 
             for i, j in combinations(v_ids, 2)}

def biblio(g, vertices):
    A = np.array(g.get_adjacency().data)
    v_ids = [g.vs.find(name=v).index for v in vertices]
    return {(g.vs[i]['name'], g.vs[j]['name']) : A[:,i] @ A[:,j]
                for i, j in combinations(v_ids, 2)}

def divide_by_max(X):
    A = np.array(list(X))
    m = np.max(A)
    A = 1/m * A
    return list(A)

def list_concat(lists):
    return reduce(lambda a, b: a + b, lists, [])
    

center = {'IN' : (0, 0), 'OUT' : (2.5, 0)}

def get_subgraph_data(vertex, mode='IN', sensitivity=0.75, n=6, density=2):
    subgraph = adjacent_subgraph(vertex, mode=mode, include_self=True)
    vertices, sizes = processed_pagerank(vertex, mode=mode, sensitivity=sensitivity, n=n)
    vertices = list(vertices)
    vertices.append(vertex)
    edge_weights = biblio(subgraph, vertices)
    
    subgraph_edge_weights = {edge : weight for edge, weight in edge_weights.items() if vertex not in edge}
    incoming_edge_weights = {edge : weight for edge, weight in edge_weights.items() if vertex in edge}
    
    def top_edges_for_vertex(v, weights, n=2):
        return Counter({edge : weight for edge, weight in weights.items() if v in edge}).most_common(density)
    
    top_subgraph_edge_weights = list_concat(top_edges_for_vertex(v, subgraph_edge_weights) for v in vertices)
    local_weighted_edgelist = [(*x[0], x[1]) for x in top_subgraph_edge_weights]
    
    g = nx.Graph()
    g.add_weighted_edges_from((u, v, w**3) for u, v, w in local_weighted_edgelist)
    layout = nx.kamada_kawai_layout(g, weight='weight', scale=1/2, center=center[mode])
    positions = list(layout.values())
    
    edges = [edge for edge, weight in top_subgraph_edge_weights] + list(incoming_edge_weights.keys())
    weights = divide_by_max([weight for edge, weight in top_subgraph_edge_weights]  + list(incoming_edge_weights.values()))
    node_data = {vertex : {'x' : x, 'y': y, 'size' : size, 'type' : mode.lower()} 
                     for vertex, (x, y), size in zip(vertices, positions, sizes)}
    center_node_data = {vertex : {'x' : 1.25, 'y' : 0, 'size' : 1.1, 'type' : 'center'}}
    node_data = {**node_data, **center_node_data}
    edge_data = [{'source' : source, 'target' : target, 'weight': weight} 
                     for (source, target), weight in zip(edges, weights)]
    for edge in edge_data:
        edge['u_x'] = node_data[edge['source']]['x']
        edge['u_y'] = node_data[edge['source']]['y']
        edge['v_x'] = node_data[edge['target']]['x']
        edge['v_y'] = node_data[edge['target']]['y']
    return node_data, edge_data

def make_edge_trace(edge):
    return  go.Scatter(
                x=(edge['u_x'], edge['v_x'], None),
                y=(edge['u_y'], edge['v_y'], None),
                line=dict(width=5*edge['weight'],color='#888'),
                showlegend=False,
                hoverinfo='none',
                mode='lines')


def get_new_graph(vertex, sensitivity=0.75):   
    incoming_node_data, incoming_edge_data = get_subgraph_data(vertex, mode='IN', sensitivity=sensitivity)
    outgoing_node_data, outgoing_edge_data = get_subgraph_data(vertex, mode='OUT', sensitivity=sensitivity)

    n_colors = str(max(len(incoming_node_data.keys()), len(outgoing_node_data.keys())))
    colors = cl.scales[n_colors]['qual']['Paired']

    in_df = pd.DataFrame(incoming_node_data).T.reset_index()
    in_df['color'] = colors[:len(in_df)]

    out_df = pd.DataFrame(outgoing_node_data).T.reset_index()
    out_df = out_df[out_df.type == 'out']
    out_df['color'] = colors[:len(out_df)]

    df = pd.concat([in_df, out_df]).rename(columns={'index' : 'name'})

    node_trace = go.Scatter(
        x=df.x, y=df.y,
        mode='markers',
        hoverinfo='text',
        #ids=ids,
        text=df.name,
        legendgroup='nodes',
        marker=dict(size=.25*df.size, color=df.color),
        line_width=2,
        showlegend=False)
    
    edge_data = incoming_edge_data + outgoing_edge_data
    edge_traces = [make_edge_trace(edge) for edge in edge_data]
    
    legend_traces = []
    in_label = go.Scatter(x=[0], y=[0], name='incoming', marker=(dict(size=0, color='black')), visible='legendonly', legendgroup='Incoming')
    legend_traces.append(in_label)
    
    
    
    for _, (name, size, _, x, y, color) in df[df.type == 'in'].iterrows():
        trace = go.Scatter(x=[x], y=[y],
                           mode='markers',
                           hoverinfo='text', 
                           name=name,
                           text=name, 
                           marker=dict(size=50, color=color), 
                           showlegend=True,
                           visible='legendonly',
                           legendgroup='Incoming')
        legend_traces.append(trace)

    
    out_label = go.Scatter(x=[0], y=[0], name='outgoing', marker=(dict(size=0, color='black')), visible='legendonly', legendgroup='Outgoing')
    legend_traces.append(out_label)
    
    for _, (name, size, _, x, y, color) in df[df.type == 'out'].iterrows():
        trace = go.Scatter(x=[x], y=[y],
                           mode='markers',
                           hoverinfo='text', 
                           name=name,
                           text=name, 
                           marker=dict(size=50, color=color), 
                           showlegend=True,
                           visible='legendonly',
                           legendgroup='Outgoing')
        legend_traces.append(trace)

    return {
         'data' : edge_traces + [node_trace] + legend_traces,
         'layout' : go.Layout(
                    title='<br>%s' % vertex, titlefont_size=20,
                    width=1260, height=700,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=go.layout.Legend(traceorder="grouped", font=dict(size=16, color="black"))
                )
        }

app = dash.Dash(__name__, 
    requests_pathname_prefix='/web-flyover/')

def app_start(context):
    inputs = context.inputs
    graph_clicked = bool(inputs['graph.clickData'])
    sensitivity_adj = inputs['sensitivity-slider.value'] != 0.75
    back_pressed = bool(inputs['back-button.n_clicks'])
    
    if not any([graph_clicked, sensitivity_adj, back_pressed]):
        return True
    return False

app.layout = html.Div(children=[dcc.Graph(id='graph'),
                                html.Button(id='back-button', children='back'),
                                dcc.Slider(
                                    id='sensitivity-slider',
                                    min=0,
                                    max=1,
                                    step=0.025,
                                    value=0.75,
                                ),
                                html.Div(id='history'), ##, style={'display': 'none'}),
                                html.Pre(id='click-data', style={'display': 'none'}),
                                html.Pre(id='ctx-data', style={'display': 'none'}),
                                html.Pre(id='raw-data')]
                                )
    
@app.callback(
     Output('graph', 'figure'),
    [Input('graph', 'clickData'),
     Input('back-button', 'n_clicks'),
     Input('sensitivity-slider', 'value')],
    [State('graph', 'figure'),
     State('history', 'children')])
def update_graph(clickData, n_clicks, sensitivity, figure, history):
    if history:
        history = json.loads(history)
    else:
        history = []
    
    ctx = dash.callback_context
    if app_start(ctx):
        graph = get_new_graph('cdc.gov')
        
    else:
        trig = ctx.triggered[0]['prop_id'].split('.')[0]
        if trig == 'graph':
            new_point = clickData["points"][0]['text'].split()[0]
            graph = get_new_graph(new_point, sensitivity)
        if trig == 'back-button':
            if len(history) > 1:
                previous = history[-2]
                graph = get_new_graph(previous, sensitivity)
            else:
                graph = figure
        if trig == 'sensitivity-slider':
            current = history[-1]
            graph = get_new_graph(current, sensitivity)

    return graph
    
@app.callback(    
     Output('history', 'children'),
    [Input('graph', 'clickData'),
     Input('back-button', 'n_clicks'),
     Input('sensitivity-slider', 'value')],
    [State('history', 'children')])
def update_history(clickData, n_clicks, sensitivity, history):
    if history:
        history = json.loads(history)
    else:
        history = []
    
    ctx = dash.callback_context
    
    if app_start(ctx):
        history.append('cdc.gov')
    else:   
        trig = ctx.triggered[0]['prop_id'].split('.')[0]
        if trig == 'graph':
            new_point = clickData["points"][0]['text'].split()[0]
            history.append(new_point)
        if trig == 'back-button':
            if len(history) > 1:
                history.pop()

    return json.dumps(history)

@app.callback(    
    Output('ctx-data', 'children'),
    [Input('graph', 'clickData'),
    Input('back-button', 'n_clicks'),
    Input('sensitivity-slider', 'value')],
    [State('history', 'children')])
def dump_input_data(clickData, n_clicks, sensitivity, history):
    ctx = dash.callback_context
    return json.dumps(ctx.inputs, indent=2)

@app.callback(
    Output('click-data', 'children'),
    [Input('graph', 'clickData')])
def dump_click_data(clickData):
    return json.dumps(clickData, indent=2)