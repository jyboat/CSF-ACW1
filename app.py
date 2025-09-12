from flask import Flask, render_template, url_for, redirect, request, flash
from flask_toastr import Toastr


app = Flask(__name__)


toastr = Toastr(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pass

    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')


if __name__ == "__main__":
    app.debug = True
    app.run()
