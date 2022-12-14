import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem,bag_of_words
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
# print(intents)

# OUR PIPELINE

# sentence ==>Tokenize ['er','rt','t','tt','rr'] 
# ==> lower+stem, ==> Remove punctuations, ['any','fer','fgfhfh']
# ===> Bag of words, ==>[0,1,0,0,0,1]
all_words =[]
# ['Hi', 'Hey', 'How', 'are', 'you', 'Is', 'anyone', 'there', '?', 'Hello', 
# 'Good', 'day', 'Bye', 'See', 'you', 'later', 'Goodbye', 'Thanks', 'Thank',
#  'you', 'That', "'s", 'helpful', 'Thank', "'s", 'a', 'lot', '!', 'Which', 
# 'items', 'do', 'you', 'have', '?', 'What', 'kinds', 'of', 'items', 'are', 
# 'there', '?', 'What', 'do', 'you', 'sell', '?', 'Do', 'you', 'take', 'cred
# it', 'cards', '?', 'Do', 'you', 'accept', 'Mastercard', '?', 'Can', 'I', '
# pay', 'with', 'Paypal', '?', 'Are', 'you', 'cash', 'only', '?', 'How', 'lo
# ng', 'does', 'delivery', 'take', '?', 'How', 'long', 'does', 'shipping', '
# take', '?', 'When', 'do', 'I', 'get', 'my', 'delivery', '?', 'Tell', 'me',
#  'a', 'joke', '!', 'Tell', 'me', 'something', 'funny', '!', 'Do', 'you', '
# know', 'a', 'joke', '?']


# ['Hi', 'Hey', 'How', 'are', 'you', 'Is', 'anyone', 'there', '?', 'Hello', 
# 'Good', 'day', 'Bye', 'See', 'you', 'later', 'Goodbye', 'Thanks', 'Thank',
#  'you', 'That', "'s", 'helpful', 'Thank', "'s", 'a', 'lot', '!', 'Which', 
# 'items', 'do', 'you', 'have', '?', 'What', 'kinds', 'of', 'items', 'are', 
# 'there', '?', 'What', 'do', 'you', 'sell', '?', 'Do', 'you', 'take', 'cred
# it', 'cards', '?', 'Do', 'you', 'accept', 'Mastercard', '?', 'Can', 'I', '
# pay', 'with', 'Paypal', '?', 'Are', 'you', 'cash', 'only', '?', 'How', 'lo
# ng', 'does', 'delivery', 'take', '?', 'How', 'long', 'does', 'shipping', '
# take', '?', 'When', 'do', 'I', 'get', 'my', 'delivery', '?', 'Tell', 'me',
#  'a', 'joke', '!', 'Tell', 'me', 'something', 'funny', '!', 'Do', 'you', '
# know', 'a', 'joke', '?']

tags = []

# ['greeting', 'goodbye', 'thanks', 'items', 'payments', 'delivery', 'funny'
# ]

xy = [] # An empty list that will later holds patterns and tags
        #   '''
#           [(['Hi'], 'greeting'), (['Hey'], 'greeting'), (['How', 'are', 'you'], 'gre
# eting'), (['Is', 'anyone', 'there', '?'], 'greeting'), (['Hello'], 'greeti
# ng'), (['Good', 'day'], 'greeting'), (['Bye'], 'goodbye'), (['See', 'you',
#  'later'], 'goodbye'), (['Goodbye'], 'goodbye'), (['Thanks'], 'thanks'), (
# ['Thank', 'you'], 'thanks'), (['That', "'s", 'helpful'], 'thanks'), (['Tha
# nk', "'s", 'a', 'lot', '!'], 'thanks'), (['Which', 'items', 'do', 'you', '
# have', '?'], 'items'), (['What', 'kinds', 'of', 'items', 'are', 'there', '
# ?'], 'items'), (['What', 'do', 'you', 'sell', '?'], 'items'), (['Do', 'you
# ', 'take', 'credit', 'cards', '?'], 'payments'), (['Do', 'you', 'accept', 
# 'Mastercard', '?'], 'payments'), (['Can', 'I', 'pay', 'with', 'Paypal', '?
# '], 'payments'), (['Are', 'you', 'cash', 'only', '?'], 'payments'), (['How
# ', 'long', 'does', 'delivery', 'take', '?'], 'delivery'), (['How', 'long',
#  'does', 'shipping', 'take', '?'], 'delivery'), (['When', 'do', 'I', 'get'
# , 'my', 'delivery', '?'], 'delivery'), (['Tell', 'me', 'a', 'joke', '!'], 
# 'funny'), (['Tell', 'me', 'something', 'funny', '!'], 'funny'), (['Do', 'y
# ou', 'know', 'a', 'joke', '?'], 'funny')]
          
#           '''
   
   
   
   
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    
    # Looping over all different patterns
    for pattern in intent['patterns']:
        
        # tokenize each word in the sentence
        w = tokenize(pattern)
        
        # add to our words list
        all_words.extend(w)
        
        # add to xy pair
        xy.append((w, tag))

# print(all_words)
# print(xy)
# print(tags)

# stem and lower each word + removing any punctuation
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]


# remove duplicates and sort
all_words = sorted(set(all_words))#we need unique words
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag) # gives us the index of each lebel
    y_train.append(label) # You have to pay attention sometims we need one hot encoding, but here is not necessary
                          #crossEntropy
# We need to convert them to numpy array
                          
X_train = np.array(X_train)
y_train = np.array(y_train)

# Let us go and implement the bag of words


# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])#input size is equal to bag of words or lenth of X_train[0], because they are equals

hidden_size = 8
output_size = len(tags) # output size is equal to number of classes
print(input_size, output_size)

# =======================================================================================================
# Let us create a new dataset
class ChatDataset(Dataset):

    def __init__(self):
        
        # here we will store number of samples
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
        
    # Later we can access dataset by index: dataset[idx]
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# Here we create our dataset by calling this class
dataset = ChatDataset()
# we also want to create a dataLoader from the above
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
# dataLoader was created to be able to iterate automatically over the dataset
# ================================================================================================================
#This was to check the size of input or output
# print(input_size,len(all_words))
# print(output_size,tags)

# To check if gpu is available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# we push our model to the available device
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer(we create the loss and optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # we want to optimize model.parameters() with learning rate

# =======================================================================================================================================
# Train the model(Let us create our actual training loop)
for epoch in range(num_epochs):
    for (words, labels) in train_loader:#here we unpack our training loader
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass(to pass the above to the forword)
        outputs = model(words) # we call the model and pass input which are words
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)# we calculate the loss at the end : here we pass the predicted outputs(predicted) and actual labels
        
        # Backward and optimize
        optimizer.zero_grad()# to initialize the gradient to zero
        loss.backward()# to calculate the back propagation
        optimizer.step()# 
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')# the loss after training

# ===========================================================================================


# SAVING THE MODEL AND CHAT IMPLEMENTATION
# ==========================================

# we create a dictionary because we want to save different things
data = {
"model_state": model.state_dict(),#we want to save modeel state and we will get tis by calling 'model.state_dict()' 
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,#all the words we collected
"tags": tags #all the tags we collected
}

FILE = "data.pth" #pth means pytoch file
torch.save(data, FILE) #save the above file but we put the data we want to save

print(f'training complete. file saved to {FILE}')

# The rest is to implement the chat