import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Api
from app.resources import auth

def init_app(app):

    api = Api(app,prefix="/api")
    api.add_resource(auth.Login,"/auth/login")