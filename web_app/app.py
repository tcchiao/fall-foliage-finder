from flask import Flask, request, render_template
import cPickle as pickle

app = Flask(__name__)

# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def submit():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
