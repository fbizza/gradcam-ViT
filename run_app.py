from app import app

if __name__ == "__main__":
    app.server.run(debug=False, host='0.0.0.0', port=8050)
