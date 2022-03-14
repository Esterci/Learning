from os import putenv
from flask_restful import Resource

class Login(Resource):

    def get(self):

        return "página de login"