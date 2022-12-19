import random
import json
import torch
from model import NeuralNet# we import our model
from nltk_utils import bag_of_words, tokenize
import streamlit as st
import keyboard
from streamlit_chat import message
from langdetect import detect
from googletrans import Translator

trans = Translator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

# ======================================================================


def translate_input_model(input_msg, input_lang):
    model_input_lag = trans.translate(input_msg, dest="en", src=input_lang).text
    return model_input_lag

def translate_output_model(out_msg,input_lang):
    model_out_lang = trans.translate(out_msg,dest=input_lang, src='en').text
    return model_out_lang
    
# def processed_response():
    
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence !="":
        if sentence == "quit":
            break
    input_lang = detect(sentence)
    input_sentence = translate_input_model(sentence,input_lang)
    model_out =get_response(input_sentence)
    out_sentence = translate_output_model(model_out,input_lang)
    print(out_sentence)
    # return out_sentence


# if __name__ == "__main__":
    # print("Let's chat! (type 'quit' to exit)")
    # while True:
    #     # sentence = "do you use credit cards?"
    #     sentence = input("You: ")
    #     if sentence == "quit":
    #         break

    #     resp = get_response(sentence)
    #     print(resp)
        
    # print(processed_response())

