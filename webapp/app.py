from flask import Flask, render_template, request

''' 
    setup a flask app - required for all flask applications
    passing in the name of the moduel to be the name of the
    app
'''
app = Flask(__name__)


# decorator lets Flask know this is our homepage
@app.route('/')
def home():
    return render_template("index.html")

# @app.route('/Your-Message', methods=['GET', 'POST'])
# def user_msg():
#     if request.method == 'POST':
        

@app.route('/about')
def about():
    return "Solent Five Study Budy Chatbot Project"


if __name__ == "__main__":
    app.run()