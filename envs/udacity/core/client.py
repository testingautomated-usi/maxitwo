import socketio
import eventlet.wsgi
from flask import Flask
from socketio import Server


def start_app(application: Flask, socket_io: Server, port: int):
    app = socketio.Middleware(socket_io, application)
    eventlet.wsgi.server(eventlet.listen(('', port)), app)
