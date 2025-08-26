from dash import html, dcc
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [10,9,8,7,6,5,4,3,2,1],
    "info": ["A","B","C","D","E","F","G","H","I","J"]
})

def create_layout():
    return html.Div([
        html.Div([
            html.H2("ViT Grad-Cam Dashboard", className="m-0")
        ], className="navbar-custom d-flex align-items-center justify-content-center"),

        html.Div(id="sidebar", children=[
            html.Div([
                html.Button("×", id="close-sidebar",
                            style={
                                "background": "transparent",
                                "border": "none",
                                "cursor": "pointer",
                            })
            ], style={"display": "flex", "justify-content": "flex-end"}),

            html.Div(id="sidebar-content", children=[
                html.H4("Dettagli Punto", style={"color": "#269C8B"}),
                html.P("Seleziona un punto sul grafico per vedere le informazioni qui.",
                       style={"color": "#269C8B"})
            ], style={"padding": "1em"})
        ], style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "height": "100%",
            "width": "0%",  # inizialmente chiusa
            "overflow": "hidden",
            "padding": "0",
            "transition": "width 0.3s ease",
            "z-index": 1000,
            "box-shadow": "2px 0 0.5em rgba(0,0,0,0.1)"
        }),

        html.Div(id="main-content", children=[
            html.Div([
                html.Div([
                    html.H5("Selezioni", className="card-title"),
                    html.Div([
                        dcc.Dropdown(
                            id="dropdown-1",
                            options=[{"label": f"Opzione {i}", "value": str(i)} for i in range(1,4)],
                            value="1",
                            clearable=False
                        ),
                        dcc.Dropdown(
                            id="dropdown-2",
                            options=[{"label": c, "value": c} for c in ["A","B","C"]],
                            value="A",
                            clearable=False
                        )
                    ], style={"display": "flex", "gap": "1em"})
                ], className="card p-3"),

                html.Div([
                    html.H5("Seleziona valore", className="card-title"),
                    dcc.Slider(
                        id='my-slider',
                        min=0,
                        max=10,
                        step=1,
                        value=5,
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="card p-3"),

                html.Div([
                    html.H5("Scatter Plot", className="card-title"),
                    dcc.Graph(
                        id='scatter-plot',
                        figure=px.scatter(df, x="x", y="y", text="info",
                                          template="plotly_white")
                    )
                ], className="card p-3")

            ], className="container mt-4")
        ], style={"margin-left": "0%", "transition": "margin-left 0.3s",
                  "min-height": "100vh", "padding": "1em"})
    ])
