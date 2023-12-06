import json
from flask import Flask, request, jsonify

from chatbot import get_response, predict_class

app = Flask(__name__)

with open("intents.json") as file:
    intents = json.load(file)

@app.route('/chat', methods=["GET", "POST"])
def chatSpiceBot():
    chatInput = request.form['chatInput']
    ints = predict_class(chatInput)
    return jsonify(spiceBotReply=get_response(ints, intents))
    
if __name__ == '__main__':
    app.run(debug=True)