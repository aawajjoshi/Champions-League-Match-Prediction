# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:12:34 2019

@author: S521315
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
#import seaborn
from scipy.stats import poisson,skellam
import math

import tweepy
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener
#from twitter_tokens import *
import json
import csv
from googletrans import Translator


#from twitter_keys import consumer_key, consumer_secret, access_token, access_secret


###----------------------------------------------------------------------------------------###


#FORZA JUVE
serie = pd.read_csv("serie - 1819.csv")
serie = serie[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
serie = serie.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
serie.head()

serie_home = serie[serie['HomeTeam'] == 'Juventus'][['HomeGoals']]
serie_away = serie[serie['AwayTeam'] == 'Juventus'][['AwayGoals']]

#home game count for Juventus
juve_home_game_count = int(serie_home.count())

#away game count for Juventus
juve_away_game_count = int(serie_away.count())

#home goal count for Juventus
juve_home = int(serie_home.sum())

#away goal count for Juventus
juve_away = int(serie_away.sum())

#home goal mean for Juventus
juve_home_mean = float(round(serie_home.mean(), 2))

#away goal mean for Juventus
juve_away_mean = float(round(serie_away.mean(), 2))

#acquiring win, loss, and draw numbers for Juventus
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


###----------------------------------------------------------------------------------------###


#Aupa Atleti
laliga = pd.read_csv("laliga - 1819.csv")
laliga = laliga[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
laliga = laliga.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
laliga.head()

laliga_home = laliga[laliga['HomeTeam'] == 'Ath Madrid'][['HomeGoals']]
laliga_away = laliga[laliga['AwayTeam'] == 'Ath Madrid'][['AwayGoals']]

#home game count for Athletico
athmadrid_home_game_count = int(laliga_home.count())

#away game count for Juventus
athmadrid_away_game_count = int(laliga_away.count())

#home goal count for Athletico
athmadrid_home = int(laliga_home.sum())

#away goal count for Athletico
athmadrid_away = int(laliga_away.sum())

#home goal mean for Athletico 
athmadrid_home_mean = float(round(laliga_home.mean()))

#away goal mean for Athletico 
athmadrid_away_mean = float(round(laliga_away.mean()))

athmadrid_win = 0
athmadrid_loss = 0 
athmadrid_draw = 0

#acquiring win, loss, and draw numbers for Athletico Madrid
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

            
###----------------------------------------------------------------------------------------###         


#creating a vertical barplot to show the season results of each team till date
n_groups = 3
ind = np.arange(n_groups)
width = 0.27

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [juve_win, juve_draw, juve_loss]
rects1 = ax.bar(ind, yvals, width, color = 'teal')
zvals = [athmadrid_win, athmadrid_draw, athmadrid_loss]
rects2 = ax.bar(ind+width, zvals, width, color = 'mediumseagreen')

ax.set_ylabel('Matches')
ax.set_xlabel('Results')
ax.set_xticks(ind+width)
ax.set_xticklabels(('Wins', 'Draws', 'Loss'))
ax.legend( (rects1[0], rects2[0]), ('Juventus', 'Athletico Madrid') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')


plt.title("Results of Juventus's and Athletico's matches so far in the 18/19 season")
plt.show()


###----------------------------------------------------------------------------------------###

            
#creating a horizontal barplot to show number of home and away goals scored by each team till date
df = pd.DataFrame(dict(graph=['Away\nGames', 'Home\nGames'],
                           n=[athmadrid_away, athmadrid_home], m=[juve_away, juve_home])) 

ind = np.arange(len(df))
width = 0.27

fig, ax = plt.subplots()

ax.barh(ind + width, df.m, width, color='teal', label='Juventus')
ax.barh(ind, df.n, width, color='mediumseagreen', label='Athletico Madrid')

ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
ax.legend()
ax.set_xlabel('Number of Goals Scored')

plt1.title("Goals scored by Juventus and Athletico so far in the 18/19 season")
plt1.show()


###----------------------------------------------------------------------------------------###


#Creating a Poisson distribution of goals scored by each team (the match will be played in Juventus's home, so only Juventus's home goals and Athletico's away goals are taken into account)
fig, ax = plt.subplots()
bars = ('1', '2','3', '4', '5', '6', '7', '8', '9')
y_pos = np.arange(len(bars))


#getting home goals for Juventus
juve_home = serie[serie['HomeTeam']=='Juventus'][['HomeGoals']].apply(pd.value_counts,normalize=True)
juve_home_pois = [poisson.pmf(i,np.sum(np.multiply(juve_home.values.T,juve_home.index.T),axis=1)[0]) for i in range(8)]

#getting away goals for Athletico
athmadrid_away = laliga[laliga['AwayTeam']=='Ath Madrid'][['AwayGoals']].apply(pd.value_counts,normalize=True)
athmadrid_away_pois = [poisson.pmf(i,np.sum(np.multiply(athmadrid_away.values.T,athmadrid_away.index.T),axis=1)[0]) for i in range(8)]

#making the plot       
pois1, = ax.plot([i for i in range(8)], juve_home_pois,
                  linestyle='-', marker='o',label="Juventus", color = 'teal')
pois1, = ax.plot([i for i in range(8)], athmadrid_away_pois,
                  linestyle='-', marker='o',label="Athletico Madrid", color = 'mediumseagreen')

ax.set_xlim([-0.5,7.5])
ax.set_ylim([-0.01,0.65])
ax.set_xticklabels(y_pos)

ax.set_ylabel('Proportion of Matches')
ax.set_xlabel('Goals')
ax.set_title("Number of Goals per Match (Poisson Distribution)\n",size=14,fontweight='bold')

plt.legend()
plt.tight_layout()
plt.show()




        
        
        
##setting tweepy up
#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_secret)
#api = tweepy.API(auth,wait_on_rate_limit=True)
#translator = Translator()
#
#
##gathering live tweets with probable hashtags for the fixture for an hour before the game starts
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








