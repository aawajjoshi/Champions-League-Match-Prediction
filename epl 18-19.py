# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:12:34 2019

@author: S521315
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

import tweepy
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
#from twitter_tokens import *
import json
import csv
from googletrans import Translator


from twitter_keys import consumer_key, consumer_secret, access_token, access_secret



#FORZA JUVE
serie = pd.read_csv("serie - 1819.csv")
serie = serie[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
serie = serie.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
serie.head()

serie_home = serie[serie['HomeTeam'] == 'Juventus'][['HomeGoals']]
serie_away = serie[serie['AwayTeam'] == 'Juventus'][['AwayGoals']]

#home goal mean for Juventus
juve_home = serie_home.mean()

#away goal mean for Juventus
juve_away = serie_away.mean()

juve_win = 0
juve_loss = 0 
juve_draw = 0

df = pd.DataFrame(serie)

for index, row in df.iterrows():
    
    if(row['HomeTeam'] == 'Juventus'):# or row['AwayTeam'] == 'Juventus'):
        if(row['HomeGoals'] > row['AwayGoals']):
            juve_win += 1
        elif(row['HomeGoals'] < row['AwayGoals']):
            juve_loss += 1
        else:
            juve_draw += 1
            


for index, row in df.iterrows():
    
    if(row['AwayTeam'] == 'Juventus'):
        if(row['HomeGoals'] < row['AwayGoals']):
            juve_win += 1
        elif(row['HomeGoals'] > row['AwayGoals']):
            juve_loss += 1
        else:
            juve_draw += 1
            

#Aupa Atleti
laliga = pd.read_csv("laliga - 1819.csv")
laliga = laliga[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
laliga = laliga.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
laliga.head()

laliga_home = laliga[laliga['HomeTeam'] == 'Ath Madrid'][['HomeGoals']]
laliga_away = laliga[laliga['AwayTeam'] == 'Ath Madrid'][['AwayGoals']]

#home goal mean for Athletico 
athmadrid_home = laliga_home.mean()

#away goal mean for Athletico 
athmadrid_away = laliga_away.mean()

athmadrid_win = 0
athmadrid_loss = 0 
athmadrid_draw = 0

df = pd.DataFrame(laliga)

for index, row in df.iterrows():
    
    if(row['HomeTeam'] == 'Ath Madrid'):
        if(row['HomeGoals'] > row['AwayGoals']):
            athmadrid_win += 1
        elif(row['HomeGoals'] < row['AwayGoals']):
            athmadrid_loss += 1
        else:
            athmadrid_draw += 1
            


for index, row in df.iterrows():
    
    if(row['AwayTeam'] == 'Ath Madrid'):
        if(row['HomeGoals'] < row['AwayGoals']):
            athmadrid_win += 1
        elif(row['HomeGoals'] > row['AwayGoals']):
            athmadrid_loss += 1
        else:
            athmadrid_draw += 1
        






#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)
#api = tweepy.API(auth,wait_on_rate_limit=True)
#translator = Translator()


#class Listener(StreamListener):
#
#    def on_data(self, status):
#        print(status)
#        with open('Juve_vs_AthMadrid.json', 'a') as f:
#            f.write(status)
#        return True
#    def on_error(self, status):
#        print(status)
#        return True
#    
#twitter_stream = Stream(auth, Listener())
#twitter_stream.filter(track=['#Juve', '#juve', '#JuveAtleti', '#turin',
#                             '#AúpaAtleti', '#ForzaJuve', '#AtléticosAroundTheWorld!', '#VamosAtleti',
#                             '#AtléticosPorElMundo'])
#
#class Listener(StreamListener):
#
#    def on_data(self, status):
#        print(status)
#        with open('ManCity_vs_Sch04.json', 'a') as f:
#            f.write(status)
#        return True
#    def on_error(self, status):
#        print(status)
#        return True
#    
#twitter_stream = Stream(auth, Listener())
#twitter_stream.filter(track=['#mancity', '#ManCity', '#cityvs04', '#S04MCI',
#                             '#MCIS04', '#S04', '#s04', '#SCHALKE!'])


