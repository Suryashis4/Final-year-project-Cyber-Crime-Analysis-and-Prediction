from backend.react_api import app


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
