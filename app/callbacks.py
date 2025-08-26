from dash import Input, Output, State, html
from app import app
import dash

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

    # Close sidebar
    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "close-sidebar":
        sidebar_style["width"] = "0%"
        sidebar_style["padding"] = "0"
        main_style["margin-left"] = "0%"
        return sidebar_style, main_style, "", None

    # Open sidebar
    if clickData:
        point = clickData["points"][0]
        x = point["x"]
        y = point["y"]
        info = point.get("text", "Nessuna info")
        sidebar_style["width"] = "25%"
        sidebar_style["padding"] = "2%"
        main_style["margin-left"] = "25%"
        content = html.Div([
            html.H4("Dettagli Punto", style={"color": "black"}),
            html.P(f"Punto selezionato: ({x}, {y})", style={"color": "black"}),
            html.P(f"Info: {info}", style={"color": "black"})
        ], style={"display": "flex", "flex-direction": "column", "gap": "1em", "margin-top": "1em"})
        return sidebar_style, main_style, content, clickData

    return sidebar_style, main_style, dash.no_update, dash.no_update
