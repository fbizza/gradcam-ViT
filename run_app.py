from dash_app import app

if __name__ == "__main__":
    server = app.server
    server.run(debug=False, host='0.0.0.0', port=8050)
