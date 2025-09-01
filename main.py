from dash_app import app
server = app.server
server.run(debug=False, host='0.0.0.0', port=8050)

