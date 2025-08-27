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
    [Input("dropdown-predictions", "value")],

    prevent_initial_call=True
)
def update_scatter(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    pred_filter = args[-1]

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id.startswith("btn-"):
        num_points = int(button_id.split("-")[1])
    else:
        num_points = 5000
        button_id = "btn-5000"

    filtered_df = df.copy()
    if pred_filter == "correct":
        filtered_df = filtered_df[filtered_df["correct_classification"] == True]
    elif pred_filter == "wrong":
        filtered_df = filtered_df[filtered_df["correct_classification"] == False]

    if num_points > len(filtered_df):
        num_points = len(filtered_df)

    fig = create_scatter_figure(filtered_df, num_points=num_points)

    classes = []
    for n in [1000, 2000, 3000, 4000, 5000]:
        if f"btn-{n}" == button_id:
            classes.append("point-button selected")
        else:
            classes.append("point-button")

    return [fig] + classes

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

        row = df[(df["tsne_1"] == x) & (df["tsne_2"] == y)].iloc[0]

        correctness = "Correct" if row["correct_classification"] else "Wrong"
        true_label = row["true_label"]
        predicted_label = row["predicted_label"]

        img_name_fixed = row['img_name'].replace("\\", "/")
        img_path = f"data/images/{img_name_fixed}"

        generate_gradcam_images(img_path, "app/tmp")
        gradcam_image = 'app/tmp/gradcam.jpg'
        original_image = 'app/tmp/original_image.jpg'
        encoded_gradcam_image = base64.b64encode(open(gradcam_image, 'rb').read())
        encoded_original_image = base64.b64encode(open(original_image, 'rb').read())

        content = html.Div([
            html.H4("Dettagli Punto"),
            html.P(f"Predizione: {correctness}", style={"fontWeight": "bold", "color": "#269C8B"}),
            html.P(f"True label: {true_label}", style={"color": "#269C8B"}),
            html.P(f"Predicted label: {predicted_label}", style={"color": "#269C8B"}),
            html.Img(src='data:image/png;base64,{}'.format(encoded_gradcam_image.decode()), style={"width": "100%"}),
            html.Img(src='data:image/jpg;base64,{}'.format(encoded_original_image.decode()), style={"width": "100%"})
        ], style={"display": "flex", "flexDirection": "column", "gap": "1em", "marginTop": "1em"})

        sidebar_style["width"] = "25%"
        sidebar_style["padding"] = "2%"
        main_style["margin-left"] = "25%"

        return sidebar_style, main_style, content, clickData

    return sidebar_style, main_style, dash.no_update, dash.no_update

