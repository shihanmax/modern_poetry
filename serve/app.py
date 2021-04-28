import random

from flask import Flask, render_template, request

app = Flask(__name__)


def get_random():
    res = []

    for i in range(10):
        res.append(
            " ".join([str(i) for i in range(random.choice([1, 3, 4, 5, 6, 7, 9, 6]))])
        )

    return res


@app.route("/")
def index():
    return render_template('query.html')


@app.route("/gen/", methods=['POST'])
def forward():
    hint = request.form["hint"]
    res = get_random()
    res.insert(0, hint)
    return render_template('query.html', GenerateText=res)
