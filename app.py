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


@app.route('/get')
def bot_reply():
    '''aligns with the JQuery get request in index.html'''
    # get the user's query text accessed from the JSON key 'msg'
    user_query = request.args.get('msg')
    return handle_query(user_query)


@app.route('/about')
def about():
    # TODO, add contributors names and project year
    return "Solent Five Study Budy Chatbot Project"


if __name__ == "__main__":
    # for development use:
    app.run(debug=True)
