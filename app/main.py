from flask import Flask
app= Flask(__name__)
@app.route('/')
def index():
    return "<h1>Welcome to our Data Science Final :))</h1>"


@app.route('/hi')
def hi():
    return "This is just a test"

