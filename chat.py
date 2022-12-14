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


st.title("Awosome ChatBot")
menu= ["Home", "Back"]
choice = st.sidebar.selectbox("Menu",menu)

answers_list = []
sentence1_list = []
not_understood_list=[]

if choice=='Home':
        
    translater = Translator()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#we can check if gpu is available

    with open('intents.json', 'r') as json_data:# we use our file
        intents = json.load(json_data)# we load it

    FILE = "data.pth"
    data = torch.load(FILE)# using torch we load our saved file after training

    # we want to get the same information
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state) 
    model.eval()# we set it into evaluation model

    #st.title("Chat bot is here")################

##############
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
    
    col1,col2 = st.columns(2)

    with col1:
        st.title('HAYSTACK')

    with col2:
        bot_name = "Sam"
        #st.markdown(bot_name)####################################
        # print("Let's chat! (type 'quit' to exit)")
        # st.write(f"Hello? My name is {bot_name}. I can here Every language.Let us chat!!!!") 
        st.subheader(f"Hello? My name is {bot_name}. I can here Every language.Let us chat!!!!")        

            
        def get_text():
            input_text = st.text_input("You: ","Hello, how are you?", key=f"input")
            return input_text 

        # while True:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []
        # sentence = "do you use credit cards?"
        # sentence1 = input(f"You: {st.text_area(label=f'chat{ccc}', key=f'kkk{ccc}')}")
        sentence1 = get_text()
        sentence1_list.append(sentence1)
        #     # if sentence == "quit":
        input_lang = detect(sentence1)
        

        if input_lang !='en':
            sentence2 = translater.translate(sentence1,dest="en").text
            prob,tag = input_predict(sentence2)
        #============================funct================================================================================== 
        # def input_predict():
        #     if sentence2 == "quit":
        #         # break
        #         pass

        #     # sentence = tokenize(sentence)# we need to tokenize the input sentence
        #     sentence = tokenize(sentence2.text)#####################
        #     X = bag_of_words(sentence, all_words) # check for bag of word=>remember it returns an numpy array
        #     X = X.reshape(1, X.shape[0])# we need to give it one row because we have one sample(1,X.shape[0]==>the number of columns)==>our model expects this shape
        #     X = torch.from_numpy(X).to(device) # turn it into pytorch tensor and then pass it to a device

        #     # Prediction
        #     output = model(X) #this will give us the prediction
        #     _, predicted = torch.max(output, dim=1)

        #     tag = tags[predicted.item()] #to get the actual tag
        #     # we need to check if the probability of this tag is high enough
        #     # look at the capture.png image again
        #     probs = torch.softmax(output, dim=1)
        #     prob = probs[0][predicted.item()]

        #    return prob
        #============================funct==================================================================================

            if prob.item() > 0.75:
                
                # we need to find the corresponding intents: we loop over all the intents and check if the tag matches
                for intent in intents['intents']:
                    if tag == intent["tag"]:# if predicted tag is == to tag in our json file do the below things
                        # print(f"{bot_name}: {random.choice(intent['responses'])}")# make a random choice in corresponding responses
                        # st.write(f"{bot_name}: {random.choice(intent['responses'])}")##########################
                        
                        pred_sentence = random.choice(intent['responses'])
                        sentence_pred_original = translater.translate(pred_sentence,dest=input_lang)
                        
                        st.session_state.past.append(sentence1)
                        # st.session_state.generated.append(random.choice(intent['responses']))
                        st.session_state.generated.append(sentence_pred_original.text)  
                        if st.session_state['generated']:

                            for i in range(len(st.session_state['generated'])-1, -1, -1):
                                message(st.session_state["generated"][i], key=str(i))
                                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  
                                answers_list.append(st.session_state["generated"][i])            
            else:

                message(f"{bot_name}: I do not understand...")
                
                not_understood_list.append(sentence1)

        else:
            prob,tag= input_predict(sentence1)
            
            if prob.item() > 0.75:
                
                # we need to find the corresponding intents: we loop over all the intents and check if the tag matches
                for intent in intents['intents']:
                    if tag == intent["tag"]:# if predicted tag is == to tag in our json file do the below things
                        # print(f"{bot_name}: {random.choice(intent['responses'])}")# make a random choice in corresponding responses
                        # st.write(f"{bot_name}: {random.choice(intent['responses'])}")##########################
                        
                        pred_sentence = random.choice(intent['responses'])
                        #sentence_pred_original = translater.translate(pred_sentence,dest=input_lang)
                        
                        st.session_state.past.append(sentence1)
                        # st.session_state.generated.append(random.choice(intent['responses']))
                        st.session_state.generated.append(pred_sentence)  
                        if st.session_state['generated']:

                            for i in range(len(st.session_state['generated'])-1, -1, -1):
                                message(st.session_state["generated"][i], key=str(i))
                                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')              
            else:

                message(f"{bot_name}: I do not understand...")
                
else:
    
    options1=["Training","DataView"]
    choice2 = st.sidebar.selectbox("Options:",options1)
    
        
    if choice2=="Training":

        st.subheader("Collecting data")
        with st.form(key="form1"):
            Tag = st.text_input(label="Enter the tag name: ")
            pattern1 = st.text_input(label="1st Pattern: ")
            pattern2 = st.text_input(label="2nd Pattern: ")
            pattern3 = st.text_input(label="3rd Pattern: ")
            pattern4 = st.text_input(label="4th Pattern: ")
            pattern5 = st.text_input(label="5th Pattern: ")
            response1 = st.text_input(label="1st response: ")
            response2 = st.text_input(label="2nd response: ")
            response3 = st.text_input(label="3rd response: ")
            response4 = st.text_input(label="4th response: ")
            response5 = st.text_input(label="5th response: ")
            submit = st.form_submit_button(label='SaveJson')
        if submit:
            if pattern1 and pattern2 and pattern3 and pattern4 and pattern5 !='':
                if response1 and response2 and response3 and response4 and response5 !='':
                    with open('intents.json') as json_data:
                        intents = json.load(json_data)
                    temp = {"tag": Tag,
                            "patterns": [pattern1,pattern2,pattern3,pattern4,pattern5],
                            "responses": [response1,response2,response3,response4,response5]
                            }
                    intents['intents'].append(temp)
                    with open('intents.json', 'w') as json_file:
                        json.dump(intents, json_file)
                    print(st.success("1 tag is enough!!!."))
                    print(st.success("go to the next form!!!."))
            else:
                print(st.error("please fill all the patterns!!!."))        
        print(st.success("1 tag is enough!!!."))
        print(st.success("go to the next form!!!."))
        if st.button("Train"):
            st.write("Training is in progress...")
            st.warning("Please wait for a while!!!")
            exec(open('train.py').read())
            
            print(st.success("Training is done!!!."))
            

        
        
    if choice2=="DataView":
        if st.button("View Data"):
                with open('non_understood_View.json','w+') as json_dataView:
                    resp = json.load(json_dataView)
                for indexx,item in enumerate(not_understood_list):
                    temp = {f"tag{indexx}":item,
                            f"patterns{indexx}": item,
                            f"responses{indexx}": 'Sorry, I do not understand'
                    }
                temp1 = {"asked_questions":sentence1_list,
                    "responses": answers_list
                    }
                resp['QUESTIONS'].append(temp)
                intents['MISUNDERSTOOD'].append(temp1)
                with open('respMisunderstood.json', 'w+') as f:
                    json.dump(resp, f)

        else:
            print(st.error("please fill all the patterns!!!.")) 
        st.write()
        