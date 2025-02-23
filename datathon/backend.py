from flask import Flask, request, jsonify, render_template
import random

app = Flask(__name__)

# Priority levels
PRIORITY_LEVELS = ["Low", "Medium Low", "Medium High", "High"]

@app.route('/')
def index():
    return render_template('datathon.html')  # Load datathon.html for the root page

@app.route('/commit')
def commit():
    return render_template('commit.html')  # Load commit.html when visiting /commit route

@app.route('/pull')
def pull():
    return render_template('pull.html')  # Load pull.html when visiting /pull route

@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.json
    message = data.get("message", "").strip()  # Get message from the request body
    
    if not message:
        return jsonify({"error": "No message provided"}), 400  # Return error if message is empty
    
    # Choose a random priority level
    priority = random.choice(PRIORITY_LEVELS)
    
    # Return the message and its associated priority
    return jsonify({"message": message, "priority": priority})

if __name__ == '__main__':
    app.run(debug=True)
