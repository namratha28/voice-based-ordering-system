#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:39:44 2018

@author: root
"""



import keyboard as kb
import Record_audio




Speech_samples_required = ['One', 'Two','Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Zero', 'yes', 'no', ]
Username = input("What is your ID? ")
#frames = []
dict = {'Eight': 8, 'Four': 4, 'Nine': 9, 'One': 1, 'Seven': 7,'Five': 5, 'Six': 6, 'Three': 3, 'Two': 2, 'Zero': 0, 'no': 11, 'yes': 10}

for v, i in enumerate(Speech_samples_required *10):
    print("When ready, please press esc and say {}".format(i))
    kb.wait('esc')
    print(i)
    audio_file_plath = Record_audio.record_voice(Username, dict[i], v , 'team_data/')
    