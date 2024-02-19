from flask import Flask, render_template, request, redirect, url_for,jsonify
from flask_cors import CORS
from twitterscraper import tweet_return
from predict_types import predict_type,recreate_model

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/tweet_pred', methods=['GET', 'POST'])
def tweet():
    # GET request
    if request.method == 'GET':
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)
    # POST request
    if request.method == 'POST':
        data = request.get_json()  # parse as JSON
        user_handle = data["handle"]
        tweets = tweet_return(user_handle)
        print("tweets :",tweets)
        user_type = predict_type(tweets)
        print("Personality is: ",user_type)
        return jsonify(user_type)



if __name__ == '__main__' :
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, port=5000)
