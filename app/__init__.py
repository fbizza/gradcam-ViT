from dash import Dash
import pandas as pd

external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"]
app = Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_csv("data/dataset.csv").drop(columns=["cls_vector"])
server = app.server

from app import layout, callbacks

app.layout = layout.create_layout()
