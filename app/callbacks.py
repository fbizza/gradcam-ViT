from dash import Input, Output, State, html
from app import app, df
import dash
import base64

from app.layout import create_scatter_figure
from models import generate_gradcam_images


@app.callback(
    [Output("scatter-plot", "figure")] +
    [Output(f"btn-{n}", "className") for n in [1000, 2000, 3000, 4000, 5000]],

    [Input(f"btn-{n}", "n_clicks") for n in [1000, 2000, 3000, 4000, 5000]] +
    [Input("dropdown-predictions", "value"),
     Input("dropdown-dim-reduction", "value")],
    prevent_initial_call=True
)
def update_scatter(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    pred_filter = args[-2]
    dim_reduction = args[-1] or "tsne"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id.startswith("btn-"):
        num_points = int(button_id.split("-")[1])
    else:
        num_points = 5000
        button_id = "btn-5000"

    filtered_df = df.copy()
    if pred_filter == "correct":
        filtered_df = filtered_df[filtered_df["correct_classification"] == 1]
    elif pred_filter == "wrong":
        filtered_df = filtered_df[filtered_df["correct_classification"] == 0]

    if num_points > len(filtered_df):
        num_points = len(filtered_df)

    fig = create_scatter_figure(filtered_df, num_points=num_points, dim_reduction=dim_reduction)

    classes = []
    for n in [1000, 2000, 3000, 4000, 5000]:
        if f"btn-{n}" == button_id:
            classes.append("point-button selected")
        else:
            classes.append("point-button")

    return [fig] + classes




from dash import Input, Output, State, html, dcc
import plotly.graph_objects as go
import base64
from app import app, df
from models import generate_gradcam_images

@app.callback(
    Output("sidebar", "style"),
    Output("main-content", "style"),
    Output("sidebar-content", "children"),
    Output("scatter-plot", "clickData"),
    Input("scatter-plot", "clickData"),
    Input("close-sidebar", "n_clicks"),
    Input("dropdown-dim-reduction", "value"),
    State("sidebar", "style"),
    State("main-content", "style")
)
def toggle_sidebar(clickData, close_clicks, dim_reduction, sidebar_style, main_style):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Close or reset sidebar
    if triggered_id in ["close-sidebar", "dropdown-dim-reduction"]:
        sidebar_style.update({
            "width": "0%",
            "padding": "0",
            "overflowY": "auto",
            "height": "100vh",
            "transition": "0.3s",
            "backgroundColor": "#353842"
        })
        main_style.update({
            "marginLeft": "0%",
            "transition": "0.3s"
        })
        return sidebar_style, main_style, "", None

    if clickData:
        point = clickData["points"][0]
        x = point["x"]
        y = point["y"]

        x_col, y_col = ("umap_1", "umap_2") if dim_reduction == "umap" else ("tsne_1", "tsne_2")
        matches = df[(df[x_col] == x) & (df[y_col] == y)]
        if matches.empty:
            return sidebar_style, main_style, dash.no_update, None

        row = matches.iloc[0]
        correctness = "Correct" if row["correct_classification"] else "Wrong"
        pred_color = "#269C8B" if row["correct_classification"] else "#dc3545"

        true_label = row["true_label"]
        predicted_label = row["predicted_label"]
        img_name_fixed = row['img_name'].replace("\\", "/")
        img_path = f"data/images/{img_name_fixed}"

        # Generate Grad-CAM images
        generate_gradcam_images(img_path, "app/tmp")
        gradcam_image = 'app/tmp/gradcam.jpg'
        original_image = 'app/tmp/original_image.jpg'

        encoded_gradcam_image = base64.b64encode(open(gradcam_image, 'rb').read())
        encoded_original_image = base64.b64encode(open(original_image, 'rb').read())

        # Create toy softmax bar chart (replace with real values if available)
        softmax_values = [0.7, 0.2, 0.1]  # toy values
        labels = ["Class A", "Class B", "Class C"]
        fig = go.Figure(go.Bar(
            x=labels,
            y=softmax_values,
            marker_color=["#269C8B", "#1f7f6e", "#dc3545"]
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(range=[0, 1])
        )

        # Sidebar content
        content = html.Div([
            html.H4("Details", style={"color": "#269C8B", "marginBottom": "0.5em"}),

            html.Table([
                html.Tr([html.Td("Classification:"), html.Td(correctness, style={"color": pred_color, "fontWeight": "bold"})]),
                html.Tr([html.Td("Predicted label:"), html.Td(predicted_label)]),
                html.Tr([html.Td("True label:"), html.Td(true_label)]),
            ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "1em", "color": "#269C8B"}),

            html.H5("Grad-CAM Explanation", style={"color": "#269C8B", "marginBottom": "0.5em"}),

            html.Div([
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_gradcam_image.decode()),
                    style={"width": "50%", "borderRadius": "0.5rem", "border": "2px solid #269C8B"}
                ),
                html.Img(
                    src='data:image/jpg;base64,{}'.format(encoded_original_image.decode()),
                    style={"width": "50%", "borderRadius": "0.5rem", "border": "2px solid #269C8B"}
                )
            ], style={"display": "flex", "justifyContent": "space-between", "gap": "2%", "marginBottom": "1em"}),

            html.H5("Top 3 Softmax", style={"color": "#269C8B", "marginBottom": "0.5em"}),
            dcc.Graph(figure=fig, style={"height": "200px"})
        ], style={"display": "flex", "flexDirection": "column", "gap": "0.8em", "marginTop": "0.5em"})

        # Open sidebar
        sidebar_style.update({
            "width": "25%",
            "padding": "2%",
            "overflowY": "auto",
            "height": "100vh",
            "transition": "0.3s",
            "backgroundColor": "#353842"
        })
        main_style.update({
            "marginLeft": "25%",
            "transition": "0.3s"
        })

        return sidebar_style, main_style, content, clickData

    return sidebar_style, main_style, dash.no_update, dash.no_update


