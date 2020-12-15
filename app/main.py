from flask import render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import os
from app.app import app
import logging


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/hi")
def hi():
    return "This is just a test"


@app.route("/", methods=["POST"])
def submit_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected for uploading")
            return redirect(request.url)
        if file:
            # do sth
            filename = secure_filename(file.filename)
            logging.warning(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            logging.warning(app.config["UPLOAD_FOLDER"])
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            flash("Still in work please check later\n")
            flash("Still in work please check later")
            #     getPrediction(filename)
            #     label, acc = getPrediction(filename)
            #     flash(label)
            #     flash(acc)
            #     flash(filename)
            return redirect("/")
