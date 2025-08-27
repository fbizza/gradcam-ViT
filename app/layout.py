from dash import html, dcc
import plotly.express as px
from app import df

def create_scatter_figure(df, num_points=None, dim_reduction="tsne"):

    if num_points:
        df_plot = df.sample(n=num_points, random_state=29)
    else:
        df_plot = df

    if dim_reduction == "umap":
        x_col, y_col = "umap_1", "umap_2"
    else:
        x_col, y_col = "tsne_1", "tsne_2"

    sorted_labels = sorted(df_plot['predicted_label'].unique())

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color="predicted_label",
        category_orders={"predicted_label": sorted_labels},
        render_mode="webgl"
    )

    fig.update_layout(
        dragmode='pan',
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            visible=False,
            range=[
                df[x_col].min() - (df[x_col].max() - df[x_col].min()) * 0.1,
                df[x_col].max() + (df[x_col].max() - df[x_col].min()) * 0.25
            ]
        ),
        yaxis=dict(
            visible=False,
            range=[
                df[y_col].min() - (df[y_col].max() - df[y_col].min()) * 0.1,
                df[y_col].max() + (df[y_col].max() - df[y_col].min()) * 0.1
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
    )

    return fig



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
                        # Dropdown 1 (predictions)
                        html.Div([
                            html.Label("Predictions", style={"textAlign": "center", "margin-bottom": "0.5em", "fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="dropdown-predictions",
                                options=[
                                    {"label": "All predictions", "value": "all"},
                                    {"label": "Correct predictions", "value": "correct"},
                                    {"label": "Wrong predictions", "value": "wrong"}
                                ],
                                value="all",  # default
                                clearable=False,
                                style={"width": "100%"}
                            )
                        ], style={"display": "flex", "flexDirection": "column", "flex": "1", "fontWeight": "bold"}),

                        # Dropdown 2 (dim reduction)
                        html.Div([
                            html.Label(
                                "Dimensionality Reduction",
                                style={"textAlign": "center", "margin-bottom": "0.5em", "fontWeight": "bold"}
                            ),
                            dcc.Dropdown(
                                id="dropdown-dim-reduction",
                                options=[
                                    {"label": "t-SNE", "value": "tsne"},
                                    {"label": "umap", "value": "umap"}
                                ],
                                value="tsne",
                                clearable=False,
                                style={"width": "100%"}
                            )
                        ], style={"display": "flex", "flexDirection": "column", "flex": "1", "fontWeight": "bold"})

                    ], style={
                        "display": "flex",
                        "gap": "1em",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "margin-bottom": "1em"
                    }),

                    # Number of points to show
                    html.Div([
                        html.Label(
                            "Number of points",
                            style={
                                "textAlign": "center",
                                "margin-bottom": "0.5em",
                                "width": "100%",
                                "fontWeight": "bold"
                            }
                        ),

                        html.Div([
                            html.Button("1000", id="btn-1000", n_clicks=0, className="point-button"),
                            html.Button("2000", id="btn-2000", n_clicks=0, className="point-button"),
                            html.Button("3000", id="btn-3000", n_clicks=0, className="point-button"),
                            html.Button("4000", id="btn-4000", n_clicks=0, className="point-button"),
                            html.Button("5000", id="btn-5000", n_clicks=0, className="point-button selected"),
                        ], style={
                            "display": "flex",
                            "gap": "1em",
                            "justify-content": "center"
                        })
                    ], style={"display": "flex", "flexDirection": "column", "alignItems": "center"})
                ], className="card p-3"),


                # Scatter plot
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
                        figure=create_scatter_figure(df),
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
