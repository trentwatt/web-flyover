import dash
import dash_html_components as html

app = dash.Dash(
    __name__,
    requests_pathname_prefix='/placeholder/'
)

app.layout = html.Div("Other thing here") 