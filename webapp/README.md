# Solent Five ChatBot Web App

Flask-based application wtih a simple UI to send and receive messages from the chatbot.

## Requirements

- the user can send a message to the chatbox
- the web app can call upon the chatbot as required
- the webapp can display the responses on the page
- the webapp keeps a history of interactions


## Environment

Due to the flat file strucutre in this repo, either app.py or main.py can be run
to call up the flask back end by pressing the play (|>) button in vscode or pycharm
when the file is in the active editor.

If the name of the python module has changed from 'run.py', we need to tell flask the name of the new file e.g. 'frontpage'

To set a different name name of the chatbot webapp, run:

```
export FLASK_APP=app
```

## Development

By directly running app.py, the flask backend will run in debug mode and any
changes made to the webapp will automatically be refreshed without having to restart
the server.

By directly running main.py this runs the webapp in a proudction deployment mode.

Optionally...

To set a development environment for working on the webapp. Open a terminal window and run:

```
export FLASK_ENV=development
```

### debug

In order to have the flask app continuously running and picking up your changes as you save the file. Run flask like so:

```
flask run --debug
```

### templates

webpage templates live in the 'templates' subdirectory. This is a default name that Flask apps look for.

## Usage

Run the web application from either main.py or app.py

Click the IP address that appears in the terminal, e.g:

```
/semsester-1/COM727-ChatBot-AE2/main.py
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

Which should open a browser window, and start chatting.

When the user posts a message in the chatbox, the index.html frontend calls in some JavaScript to send an HTML POST request to the Flask backend

The app.py implements the Flask backend which calls on a function in chatbot.py to handle the user's query and return a reply.

### n.b
if the model has not been trained or served, training.py will need to be run first. Then it will all work from the webapp.

### Alternatively...
To run the webapp, using the terminal where the environment variables are set:

```
flask run
```
