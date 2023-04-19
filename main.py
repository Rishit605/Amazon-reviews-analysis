from flask import Flask, request, jsonify
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

app = Flask(__name__)

# load the pre-trained Keras model
model = load_model('path/to/model.h5')

# load the tokenizer used for preprocessing the text
with open('path/to/tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(json.load(f))


# define the endpoint for the API
@app.route('/classify', methods=['POST'])
def classify_review():
    # get the text data from the request
    text = request.json['text']

    token = Tokenizer(num_words=10000, oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    token.fit_on_texts(text)

    word_idx = token.word_index

    # Sequencing the encoded text

    train_seq = token.texts_to_sequences(text)
    train_padded = pad_sequences(train_seq, maxlen=120, truncating='pre')

    # predict the class label for the preprocessed text
    predictions = model.predict(train_padded)
    class_label = 'Positive' if predictions[0][0] > 0.5 else 'Negative'

    # return the predicted class label as a response
    response = {'class_label': class_label}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)