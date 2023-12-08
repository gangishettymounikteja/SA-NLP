import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense  
from keras.models import Sequential, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from statistics import mode

app = Flask(__name__)
CORS(app)

# # Load data
# data = pd.read_csv('./IMDB.csv', nrows=5000)

# def map_sentiment(x):
#   if x == 'positive':
#       return 1
#   elif x == 'negative':
#       return 0
      
# data['sentiment'] = data['sentiment'].apply(map_sentiment)

# # Preprocess data
# tokenizer = Tokenizer(num_words=5000, split=' ')
# tokenizer.fit_on_texts(data['review'].values)
# X = tokenizer.texts_to_sequences(data['review'].values)
# X = pad_sequences(X)

# Load model
model = load_model('./sentiment_model_Hybrid.h5')


tokenizer = Tokenizer(num_words=5000)


model_names = [ 'RNN', 'Hybrid', 'CNN', 'BLSTM', 'BERT']
results = []
@app.route('/predict', methods=['POST'])
def predict():
  for model in model_names:
    model = load_model('./sentiment_model_' + model + '.h5')
    if(model == 'CNN'):
      seq = tokenizer.texts_to_sequences([text])
      print('CNN')
    elif(model == 'RNN'):
      seq = tokenizer.texts_to_sequences([text])
      seq = pad_sequences(seq, maxlen=X.shape[1])
      pred = model.predict(seq)
      results.append(pred)
    elif(model == 'BLSTM'):
       seq = tokenizer.texts_to_sequences([text])
       padded = pad_sequences(seq, maxlen=64)
       pred = model.predict(padded)[0]
    elif(model == 'Hybrid'):
        seq = tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=100)
        pred = model.predict([padded, padded])
        print('Hybrid')
    else:
        print('Error')
    
    # pred = model.predict(seq)
    avg = np.mean(results)

    if avg > 0.5:
      sentiment = 'positive'
    else:
      sentiment = 'negative'

    result = {'prediction': sentiment}
  
  return jsonify(result)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*') 
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response

if __name__ == '__main__':
  app.run(debug=True)