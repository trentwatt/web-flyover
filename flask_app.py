from flask import Flask

flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return '<h2>Logspace</h2><a href="/web-flyover">web flyover</a>'