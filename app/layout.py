from dash import html, dcc
import plotly.express as px
import pandas as pd

df = pd.read_csv("data/dataset.csv")

def create_layout():
    return html.Div([
        html.Div([
            html.H2("ViT Grad-Cam Dashboard", className="navbar-title"),

            html.Div([  # Buttons container
                html.A(
                    html.Button(
                        html.Img(src="/assets/images/github_logo.png", style={"height": "24px"}),
                        className="nav-button"
                    ),
                    href="https://github.com/fbizza/gradcam-ViT",
                    target="_blank"
                ),
                html.A(
                    html.Button(
                        html.Img(src="/assets/images/uni_logo.svg", style={"height": "24px"}),
                        className="nav-button"
                    ),
                    href="https://www.unifi.it",
                    target="_blank"
                )
            ], className="nav-buttons")
        ], className="navbar-custom"),

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
            "width": "0%",
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
                            options=[{"label": f"Opzione {i}", "value": str(i)} for i in range(1, 4)],
                            value="1",
                            clearable=False,
                            style={"flex": "1"}
                        ),
                        dcc.Dropdown(
                            id="dropdown-2",
                            options=[{"label": c, "value": c} for c in ["A", "B", "C"]],
                            value="A",
                            clearable=False,
                            style={"flex": "1"}
                        )
                    ], style={
                        "display": "flex",
                        "gap": "1em",
                        "justifyContent": "center",
                        "alignItems": "center"
                    })

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
                        figure=px.scatter(
                            df,
                            x="tsne_1",
                            y="tsne_2",
                            color="predicted_label",
                        ).update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),

                            legend=dict(
                                title=dict(
                                    text="Predicted Label",
                                    font=dict(
                                        size=14,  # title of the legend
                                        color="#269C8B",
                                        weight=1000,
                                        #family="Arial",
                                    )
                                ),
                                font=dict(
                                    size=12,  # font of the entries in the legend
                                    color="#269C8B"
                                ),
                                bgcolor="#282B33",
                                bordercolor="#269C8B",
                                borderwidth=2
                            )
                        ),
                        config={
                            "scrollZoom": True,
                            "displayModeBar": False
                        }
                    )
                ], className="card p-3")

            ], className="container mt-4")
        ], style={"margin-left": "0%", "transition": "margin-left 0.3s",
                  "min-height": "100vh", "padding": "1em"})
    ])
