from flask import Flask

from app import extensions

from app import routes


def create_app():
    app = Flask(__name__)
    extensions.init_app(app)
    routes.init_app(app)

    return app
