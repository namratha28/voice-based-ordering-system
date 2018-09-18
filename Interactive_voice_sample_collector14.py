##Collect voice samples with outputs

import Record_audio
import os
import keyboard as kb
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import pickle
from Record_audio import get_features
import os
import warnings
warnings.filterwarnings('ignore')


def rec():
        
    Speech_samples_required = ['Zero', 'One', 'Two','Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'yes', 'no', ]
    Username = input("What is your ID? ")
    #frames = []
    dict = {'Eight': 8, 'Four': 4, 'Nine': 9, 'One': 1, 'Seven': 7,'Five': 5, 'Six': 6, 'Three': 3, 'Two': 2, 'Zero': 0, 'no': 11, 'yes': 10}
    
    
    confidence_array = np.zeros(12)
      
    for v, i in enumerate(Speech_samples_required *1):
        print("When ready, please press esc and say {}".format(i))
        kb.wait('esc')
        print(i)
        audio_file_path = Record_audio.record_voice(Username, dict[i], v , "team_data/")
        Record_audio.play_audio(audio_file_path)
    
        #get_features in Record_audio.py should be completed
        features = Record_audio.get_features(audio_file_path, sr=8000)
        label = int(audio_file_path.split("/")[-1].split("_")[0])
    
    
        #######use your classifier to return a confidence and predict for the input
        #######digit is the prediction and confidence should be confidence measure
        confidence, digit = 0, 0#use the model you defined here to get the confidence and digit
    
        if digit == label :
            confidence_array[dict[i]] = confidence
        else : 
            pass
            
        print("Confidence : {}  Prediction : {} label : {} ".format(confidence, digit, label))
    
    ##############################################################################################################################
    #Every sample recorded during this time will be classified on real time and the classfier should adjust to your voice samples#
    ##############################################################################################################################
    #your code to update the model here


def update_pickle():
   wavveclive=pickle.load(open("wavvec_team_d.pkl","rb"))
   print(len(wavveclive))
   file_name = "C:/Users/vivek/AI_ML/AIML_Lab_exercises/Hackathon I/team_data/1_userinput_1.wav"
   deep_features = get_features(file_name, sr=8000, n_mfcc=30, n_mels=128, frames = 15)
   #print(deep_features)
   label = int(file_name.split("/")[-1].split("_")[0])
   #print (label)
#   fn=os.path.basename(file_name)
#   label=fn.split(sep='_')[0]
   wavveclive.append((deep_features,label))
   print(len(wavveclive))
   return wavveclive
 
#rec()
wavveclive = update_pickle()


    ##############################################################################################################################
#if v%12 == 0 :
 #   Record_audio.plotchart(Speech_samples_required,confidence_array)