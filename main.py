import pickle
import pandas as pd
from flask import Flask, request, jsonify
from model_files.ml_model import predict_personality

app = Flask("Personality_classification")

@app.route('/', methods = ['POST'])
def predict():
    payload = request.stream
    final_answers = pd.read_json(payload, typ='frame', orient='records')
    #answers = [request.form.values()]
    #final_answers = pd.DataFrame(answers)

    with open('./model_files/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_personality(final_answers, model)

    response = {
        'personality type': predictions
    }

    return response


if __name__ =='__main__':
    app.run(debug = True, host = '0.0.0.0', port = 9696)