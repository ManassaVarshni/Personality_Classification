import pickle
import pandas as pd
from flask import Flask, request, render_template, url_for
from model_files.ml_model import predict_personality, calculateScores, normalized_all_types_scores
from forms import SimpleForm

app = Flask("Personality_classification")
app.config['SECRET_KEY'] = '0x00000246994F8700'


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', title = 'Home')

@app.route('/test', methods = ['POST', 'GET'])
def project():
    form = SimpleForm()
    return render_template('test.html',title = 'Test', form = form)

@app.route('/predict', methods = ['POST'])
def predict():
    payload = [x for x in request.form.values()]

    # Removing first and last element from payload because one is CSRF token and the other is Submit.
    payload = payload[1:len(payload)-1]

    # We do transpose because we need a single row and 50 columns and not the vice versa
    final_answers = pd.DataFrame(payload).T
    
    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    
    predictions = predict_personality(final_answers, model)
    scores = calculateScores(predictions, normalized_all_types_scores)

    response = {
        'personality_type': predictions,
        'scores': scores
    }

    return render_template('prediction_text.html', prediction_text='You are of personality type {}'.format(response['personality_type'])
    , score='And your score for each 5 different categories are {}:'.format(response['scores']))



if __name__ =='__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)


