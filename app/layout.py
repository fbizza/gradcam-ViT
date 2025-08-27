from dash import html, dcc
import plotly.express as px
import pandas as pd

df = pd.read_csv("data/dataset.csv")

def create_layout():
    return html.Div([
        html.Div([
            html.H2("ViT Grad-Cam Dashboard", className="navbar-title"),

            html.Div([
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
                    #html.H5("Selections", className="card-title"),

                    html.Div([
                        # Dropdown 1
                        html.Div([
                            html.Label("Dropdown 1", style={"textAlign": "center", "margin-bottom": "0.5em"}),
                            dcc.Dropdown(
                                id="dropdown-1",
                                options=[{"label": f"Opzione {i}", "value": str(i)} for i in range(1, 4)],
                                value="1",
                                clearable=False,
                                style={"width": "100%"}
                            )
                        ], style={"display": "flex", "flexDirection": "column", "flex": "1"}),

                        # Dropdown 2
                        html.Div([
                            html.Label("Dropdown 2", style={"textAlign": "center", "margin-bottom": "0.5em"}),
                            dcc.Dropdown(
                                id="dropdown-2",
                                options=[{"label": c, "value": c} for c in ["A", "B", "C"]],
                                value="A",
                                clearable=False,
                                style={"width": "100%"}
                            )
                        ], style={"display": "flex", "flexDirection": "column", "flex": "1"})

                    ], style={
                        "display": "flex",
                        "gap": "1em",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "margin-bottom": "1em"
                    }),

                    # Slider
                    html.Div([
                        html.Label("Number of points", style={"textAlign": "center", "margin-bottom": "0.5em"}),
                        dcc.Slider(
                            id='my-slider',
                            min=0,
                            max=5000,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="drag"
                        )
                    ], style={"display": "flex", "flexDirection": "column", "width": "100%", "margin-top": "1em"})
                ], className="card p-3"),



                html.Div([
                    html.H5(
                        "Scatter Plot",
                        className="card-title",
                        style={
                            "position": "absolute",
                            "inset": "0 auto auto 0",
                            "margin": "0",
                            "padding": "0.5em 1em",
                            "zIndex": "10",
                            "color": "#269C8B",
                            "background": "rgba(40,43,51,0.6)",
                            "borderBottomRightRadius": "0.75em"
                        }
                    ),
                    dcc.Graph(
                        id='scatter-plot',
                        figure=px.scatter(
                            df,
                            x="tsne_1",
                            y="tsne_2",
                            color="predicted_label",
                        ).update_layout(
                            dragmode='pan',
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(
                                visible=False,
                                range=[
                                    df["tsne_1"].min() - (df["tsne_1"].max() - df["tsne_1"].min()) * 0.1,
                                    df["tsne_1"].max() + (df["tsne_1"].max() - df["tsne_1"].min()) * 0.25
                                ]
                            ),
                            yaxis=dict(
                                visible=False,
                                range=[
                                    df["tsne_2"].min() - (df["tsne_2"].max() - df["tsne_2"].min()) * 0.1,
                                    df["tsne_2"].max() + (df["tsne_2"].max() - df["tsne_2"].min()) * 0.1
                                ]
                            ),
                            margin=dict(l=0, r=0, t=0, b=0),

                            legend=dict(
                                title=dict(
                                    text="Predicted Label",
                                    font=dict(size=14, color="#269C8B")
                                ),
                                font=dict(size=12, color="#269C8B"),
                                bgcolor="#2F323B",
                                borderwidth=0,
                                x=0.993,
                                y=1,
                                xanchor="right",
                                yanchor="top"
                            )

                        ),
                        config={
                            "scrollZoom": True,
                            "displayModeBar": False
                        },
                        style={"width": "100%", "height": "100%"}
                    )
                ], className="card p-0",
                    style={
                        "position": "relative",
                        "overflow": "hidden"
                    })

            ], className="container mt-4")
        ], style={"margin-left": "0%", "transition": "margin-left 0.3s",
                  "min-height": "100vh", "padding": "1em"})
    ])
