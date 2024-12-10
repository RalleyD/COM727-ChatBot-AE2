from flask import Flask, render_template, request
from chatbot import handle_query

'''
    setup a flask app - required for all flask applications
    passing in the name of the model to be the name of the
    app
'''
app = Flask(__name__)


@app.route('/')  # decorator lets Flask know this is our homepage
def home():
    # by default, looks for a 'templates' directory
    return render_template("index.html")


@app.post('/post')
def bot_reply():
    '''aligns with the index.html JS script POST URL'''
    user_query = request.get_json()['msg']
    return handle_query(user_query)


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # for development use:
    app.run(debug=True)
