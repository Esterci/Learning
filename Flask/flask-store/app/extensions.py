from dynaconf import FlaskDynaconf
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_app(app):
    FlaskDynaconf(app)
    JWTManager(app)
    CORS(app)

    db.init_app(app)
    Migrate(app, db)

    from app.models import User

    @app.shell_context_processor
    def context_processor():

        return dict(app=app, db=db, User=User)
