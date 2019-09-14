from werkzeug.wsgi import DispatcherMiddleware
from flask_app import flask_app
from web_flyover import app as web_flyover
from placeholder import app as placeholder

application = DispatcherMiddleware(flask_app, {
    '/web-flyover': web_flyover.server,
    '/other': placeholder.server,
})  