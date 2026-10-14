import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


#Load the LSTM model

model = load_model('next_word_lstm.h5')


# load the tokenizer

with open ('tokenizer.pkl', 'rb') as handle:
    token = pickle.load(handle)


# Function to predict the next words

def predict_next_words(model, token, text, max_sequence_length):
    token_list = token.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]            #Ensure the Sequence Length matches the max_sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in token.word_index.items():
        if index == predicted_word_index:
            return word
    return None



# Streamlit app

st.title("Next Word Prediction With LSTM")
input_text = st.text_input("Enter the Sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] + 1
    next_word = predict_next_words(model, token, input_text, max_sequence_length)
    st.write(f"Next word: {next_word}")
    