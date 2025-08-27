from dash import Input, Output, State, html
from app import app, df
import dash
from app.layout import create_scatter_figure  # assuming you put the function in layout.py


@app.callback(
    Output("scatter-plot", "figure"),
    Input("my-slider", "value")
)
def update_scatter(num_points):
    if not num_points:
        num_points = 100
    return create_scatter_figure(df, num_points=num_points)

@app.callback(
    Output("sidebar", "style"),
    Output("main-content", "style"),
    Output("sidebar-content", "children"),
    Output("scatter-plot", "clickData"),
    Input("scatter-plot", "clickData"),
    Input("close-sidebar", "n_clicks"),
    State("sidebar", "style"),
    State("main-content", "style")
)
def toggle_sidebar(clickData, close_clicks, sidebar_style, main_style):
    ctx = dash.callback_context

    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "close-sidebar":
        sidebar_style["width"] = "0%"
        sidebar_style["padding"] = "0"
        main_style["margin-left"] = "0%"
        return sidebar_style, main_style, "", None

    if clickData:
        point = clickData["points"][0]
        x = point["x"]
        y = point["y"]
        info = point.get("text", "Nessuna info")
        sidebar_style["width"] = "25%"
        sidebar_style["padding"] = "2%"
        main_style["margin-left"] = "25%"
        content = html.Div([
            html.H4("Dettagli Punto"),
            html.P(f"Punto selezionato: ({x}, {y})"),
            html.P(f"Info: {info}", style={"color": "#269C8B"})
        ], style={"display": "flex", "flex-direction": "column", "gap": "1em", "margin-top": "1em"})
        return sidebar_style, main_style, content, clickData

    return sidebar_style, main_style, dash.no_update, dash.no_update
