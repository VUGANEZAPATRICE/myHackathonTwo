import random # Because later we need to do a random choice from the possible answers
import json # 
import torch
from model import NeuralNet# we import our model
from nltk_utils import bag_of_words, tokenize
import streamlit as st
import keyboard
from streamlit_chat import message
from langdetect import detect
from googletrans import Translator

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

def input_predict(sentence):

    if sentence == "quit":
        # break
        pass

    # sentence = tokenize(sentence)# we need to tokenize the input sentence
    sentence = tokenize(sentence)#####################
    X = bag_of_words(sentence, all_words) # check for bag of word=>remember it returns an numpy array
    X = X.reshape(1, X.shape[0])# we need to give it one row because we have one sample(1,X.shape[0]==>the number of columns)==>our model expects this shape
    X = torch.from_numpy(X).to(device) # turn it into pytorch tensor and then pass it to a device

    # Prediction
    output = model(X) #this will give us the prediction
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()] #to get the actual tag
    # we need to check if the probability of this tag is high enough
    # look at the capture.png image again
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return prob,tag

def get_text():
    # input_text = st.text_input("You: ","Hello, how are you?", key=f"input")
    input_text=input("sentence: ")
    return input_text 

def final_pred():
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    return random.choice(intent['responses'])

query = get_text()
prob,tag = input_predict(query)

if prob.item()>0.75:
    pred = final_pred()
    print(pred)
    
else:
    print("I do not understand...")






# def get_response(msg):
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
    
#     return "I do not understand..."


# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)

