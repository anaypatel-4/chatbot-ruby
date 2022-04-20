import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lematizer = WordNetLemmatizer()
intents = json.loads(open('chatbot\\ruby\intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.model')

def cleansent(sentance):
    sentwords = nltk.word_tokenize(sentance)
    sentwords = [lematizer.lemmatize(word) for word in sentwords]
    return sentwords

def bag_of_words(sentance):
    sentwords = cleansent(sentance)
    bag = [0] * len(words)
    for w in sentwords:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict(sentance):
    bow = bag_of_words(sentance)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('The Bot is Online...')

while True:
    message = input("user--> ")
    ints = predict(message)
    res = get_response(ints, intents)
    print("bot--> "+res)
    if ints[0]['intent'] == 'goodbye':
        exit(0)
