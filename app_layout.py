import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc

starting_vertex = "cdc.gov"

with open("data/gov_to_gov/nodes.txt", "r") as f:
    nodes = [line.strip() for line in f.readlines()]

graph = dcc.Graph(id="graph", config={"displayModeBar": False})
controls = dbc.Row(
    dbc.Col(
        dbc.Row(
            [
                dbc.Col(html.Button("Back", id="back-button"), width=2),
                dbc.Col(
                    dcc.Slider(
                        id="sensitivity-slider", min=0, max=1, step=0.025, value=0.75
                    ),
                ),
                dbc.Col(html.Div(id="sensitivity-display"), width=3,),
            ],
        ),
        width={"offset": 3, "size": 6},
    ),
    align="start",
)

center_panel = dbc.Col(graph, width=6)
left_panel = dbc.Col(
    [
        html.Div(dcc.Markdown(id="left-info-panel"), className="text-center",),
        dcc.Graph(id="left-legend-panel", config={"displayModeBar": False}),
    ],
    width=3,
)


right_panel = dbc.Col(
    [
        html.Div(dcc.Markdown(id="right-info-panel"), className="text-center",),
        dcc.Graph(id="right-legend-panel", config={"displayModeBar": False}),
    ],
    width=3,
)


search_nodes = dbc.Col(
    html.Div(
        dcc.Dropdown(
            id="node-search",
            options=[{"label": node, "value": node} for node in nodes],
            value=starting_vertex,
        ),
        style={"verticalAlign": "middle"},
    ),
    width={"offset": 1, "size": 2},
    align="end",
)
title = dbc.Col(
    html.Div(dcc.Markdown(id="title-text"), className="text-center"), width=6
)
title_bar = dbc.Row([search_nodes, title])
body = dbc.Row([left_panel, center_panel, right_panel], no_gutters=True)
debug = dbc.Row(
    [
        html.Div(id="history", style={"display": "none"}),
        html.Div(id="cache", style={"display": "none"}),
        html.Pre(id="trigger-data", style={"display": "none"}),
        html.Pre(id="click-data", style={"display": "none"}),
        html.Pre(id="input-data", style={"display": "none"}),
    ]
)
spacer = dbc.Row(dbc.Col(html.Div(style={"height": "75px"})))
documentation = dbc.Row(
    dbc.Col(
        dcc.Markdown(
            """
            This app provides an graph interface to the network of all the hyperlinks between websites ending in ".gov", as scraped by 
            [Common Crawl](https://commoncrawl.org/2019/11/host-and-domain-level-web-graphs-aug-sep-oct-2019/).  

            The center node is the current website, whose title is at the top. On the left hand side are
            the sites most representative of those that link to it, and on the right are those most representative
            of those it links to. The sensitivity, which you can adjust with the slider, is what calibrates this "representativeness:" the higher the sensitivity, the more 
            idiosyncratic representatives you will get. The lower the sensitivity, the more the sites will be representative 
            of the network as a whole.  

            When you click on a node for a site in the graph, it will take you to that site's graph. You can in this way explore a path of relatedness,
            as well as find your way back with the back button. If you are curious about what any of the sites are, the legends on the right and
            left are hyperlinked. If you have a specific site in mind you would like to start from, you can search for it in the dropdown.  

            The edge thicknesses indicate the strength of the relatedness between two nodes.  

            Enjoy. If you would like to share any ideas or experiences, please find me at trent wat son 1 at gmail.
            """
        ),
        width={"offset": 2, "size": 8},
    ),
    align="end",
)

layout = html.Div(
    children=[
        dbc.Container([title_bar, body, controls], fluid=True),
        dbc.Container([spacer, documentation]),
        debug,
    ]
)

