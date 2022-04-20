from asyncio.windows_events import NULL
from django.shortcuts import redirect, render
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lematizer = WordNetLemmatizer()
intents = json.loads(open('E:\Personal Files\My Projects\Websites\chatbot\\ruby\intents.json').read())

words = pickle.load(open('E:\Personal Files\My Projects\Websites\words.pkl','rb'))
classes = pickle.load(open('E:\Personal Files\My Projects\Websites\classes.pkl','rb'))
model = load_model('E:\Personal Files\My Projects\Websites\chatbot_model.model')

def home(request):
    ans = 'online'
    que = 'online'
    if request.method == 'POST':
        que = request.POST.get('chat')
    
        message = str(que)
        ints = predict(message)
        res = get_response(ints, intents)
        ans = res
        if ints[0]['intent'] == 'goodbye':
            ans = "Good Bye!"
    return render(request,'index.html',{"reply":ans, "ques":que})
    

def error(request):
    return render(request,'error.html')
    

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
    if True:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "Sorry I don't know that"
    return result
            

