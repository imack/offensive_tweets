import os
import pickle
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

model_path = os.path.join('../', 'model')
MAX_SEQUENCE_LENGTH=100

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None
    tokenizer = None

    @classmethod
    def get_model(cls):
        if cls.model == None:
            json_file = open(os.path.join(model_path, 'model.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            cls.model = model_from_json(loaded_model_json)
            # load weights into new model
            cls.model.load_weights(os.path.join(model_path, 'model.h5'), 'r')
            print("Loaded model from disk")
        return cls.model

    @classmethod
    def get_tokenizer(cls):
        if cls.tokenizer == None:
            with open(os.path.join(model_path, 'tokenizer.pickle'), 'rb') as handle:
                cls.tokenizer = pickle.load(handle)
            print("Loaded tokenizer from disk")
        return cls.tokenizer

    @classmethod
    def predict(cls, twitter_texts):
        """For the input, do the predictions and return them.

        Args:
            input (an array of strings): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        tokenizer = cls.get_tokenizer()

        sequences = tokenizer.texts_to_sequences(twitter_texts)
        padded_input = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        return clf.predict(padded_input)