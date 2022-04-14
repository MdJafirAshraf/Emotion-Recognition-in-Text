import json
import numpy as np
import text2emotion as te
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['sentence_input']
        print(text)
        pred = te.get_emotion(text)

        class_name = list(pred.keys())
        values = list(pred.values())

        pred_value = class_name[np.argmax(values)]
        print(pred_value)

    return json.dumps(pred_value)


if __name__ == '__main__':
    app.run(port=5000)
