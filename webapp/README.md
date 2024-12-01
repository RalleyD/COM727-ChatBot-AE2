# Solent Five ChatBot Web App

Flask-based application wtih a simple UI to send and receive messages from the chatbot.

## Requirements

- the user can send a message to the chatbox
- the web app can call upon the chatbot as required
- the webapp can display the responses on the page
- the webapp keeps a history of interactions
- the webapp provides a RESTful API to the chatbot


## Environment

If the name of the python module has changed from 'run.py', we need to tell flask the name of the new file e.g. 'frontpage'

To set the name of the chatbot app, run:

```
export FLASK_APP=webapp/frontpage
```

## Delopment

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

To run the webapp, using the terminal where the environment variables are set:

```
flask run
```

## TODOs

- figure out a nice way to run the script with the environment already setup e.g bash script