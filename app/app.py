from flask import Flask
import os

UPLOAD_FOLDER = os.getcwd() + "/pictures"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "super secret key"
