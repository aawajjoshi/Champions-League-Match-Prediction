# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 01:12:34 2019

@author: S521315
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import numpy as np
import seaborn
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



#FORZA JUVE
serie = pd.read_csv("serie - 1819.csv")
serie = serie[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
serie = serie.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
serie.head()

serie_home = serie[serie['HomeTeam'] == 'Juventus'][['HomeGoals']]
serie_away = serie[serie['AwayTeam'] == 'Juventus'][['AwayGoals']]

#home goal count for Juventus
juve_home = int(serie_home.sum())

#away goal count for Juventus
juve_away = int(serie_away.sum())

#home goal mean for Juventus
juve_home_mean = serie_home.mean()

#away goal mean for Juventus
juve_away_mean = serie_away.mean()

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


         

#Aupa Atleti
laliga = pd.read_csv("laliga - 1819.csv")
laliga = laliga[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
laliga = laliga.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
laliga.head()

laliga_home = laliga[laliga['HomeTeam'] == 'Ath Madrid'][['HomeGoals']]
laliga_away = laliga[laliga['AwayTeam'] == 'Ath Madrid'][['AwayGoals']]

#home goal count for Athletico
athmadrid_home = int(laliga_home.sum())

#away goal count for Athletico
athmadrid_away = int(laliga_away.sum())

#home goal mean for Athletico 
athmadrid_home_mean = laliga_home.mean()

#away goal mean for Athletico 
athmadrid_away_mean = laliga_away.mean()

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






#poisson_pred = np.column_stack([[poisson.pmf(i, serie.mean()[j]) for i in range(8)] for j in range(2)])
#
## plot histogram of actual goals
#plt.hist(serie[['HomeGoals', 'AwayGoals']].values, range(9), 
#         alpha=0.7, label=['Home', 'Away'],normed=True, color=["#FFA07A", "#20B2AA"])
#
## add lines for the Poisson distributions
#pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0],
#                  linestyle='-', marker='o',label="Home", color = '#CD5C5C')
#pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1],
#                  linestyle='-', marker='o',label="Away", color = '#006400')
#
#leg=plt.legend(loc='upper right', fontsize=13, ncol=2)
#leg.set_title("Poisson           Actual        ", prop = {'size':'14', 'weight':'bold'})
#
#plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)])
#plt.xlabel("Goals per Match",size=13)
#plt.ylabel("Proportion of Matches",size=13)
#plt.title("Number of Goals per Match (EPL 2016/17 Season)",size=14,fontweight='bold')
#plt.ylim([-0.004, 0.4])
#plt.tight_layout()
#plt.show()
            


#fig,(ax1,ax2) = plt.subplots(2, 1)
#
#
#juve_home = serie[serie['HomeTeam']=='Juventus'][['HomeGoals']].apply(pd.value_counts,normalize=True)
#juve_home_pois = [poisson.pmf(i,np.sum(np.multiply(juve_home.values.T,juve_home.index.T),axis=1)[0]) for i in range(8)]
#athmadrid_home = laliga[laliga['HomeTeam']=='Athletico Madrid'][['HomeGoals']].apply(pd.value_counts,normalize=True)
#athmadrid_home_pois = [poisson.pmf(i,np.sum(np.multiply(athmadrid_home.values.T,athmadrid_home.index.T),axis=1)[0]) for i in range(8)]
#
#juve_away = serie[serie['AwayTeam']=='Chelsea'][['AwayGoals']].apply(pd.value_counts,normalize=True)
#juve_away_pois = [poisson.pmf(i,np.sum(np.multiply(juve_away.values.T,juve_away.index.T),axis=1)[0]) for i in range(8)]
#athmadrid_away = laliga[laliga['AwayTeam']=='Sunderland'][['AwayGoals']].apply(pd.value_counts,normalize=True)
#athmadrid_away_pois = [poisson.pmf(i,np.sum(np.multiply(athmadrid_away.values.T,athmadrid_away.index.T),axis=1)[0]) for i in range(8)]
#
#
#
#ax1.bar(juve_home.index-0.4,juve_home.values,width=0.4,color="#034694",label="Juventus")
#ax1.bar(athmadrid_home.index,athmadrid_home.values,width=0.4,color="#EB172B",label="Athletico Madrid")
#        
#pois1, = ax1.plot([i for i in range(8)], juve_home_pois,
#                  linestyle='-', marker='o',label="Juventus", color = "#0a7bff")
#pois1, = ax1.plot([i for i in range(8)], athmadrid_home_pois,
#                  linestyle='-', marker='o',label="Athletico Madrid", color = "#ff7c89")
#
#leg=ax1.legend(loc='upper right', fontsize=12, ncol=2)
#leg.set_title("Poisson                 Actual                ", prop = {'size':'14', 'weight':'bold'})
#ax1.set_xlim([-0.5,7.5])
#ax1.set_ylim([-0.01,0.65])
#ax1.set_xticklabels([])
#
#ax1.text(7.65, 0.585, '                Home                ', rotation=-90,
#        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
#ax2.text(7.65, 0.585, '                Away                ', rotation=-90,
#        bbox={'facecolor':'#ffbcf6', 'alpha':0.5, 'pad':5})
#
#ax2.bar(juve_away.index-0.4,juve_away.values,width=0.4,color="#034694",label="Juventus")
#ax2.bar(athmadrid_away.index,athmadrid_away.values,width=0.4,color="#EB172B",label="Athletico Madrid")
#
#pois1, = ax2.plot([i for i in range(8)], juve_away_pois,
#                  linestyle='-', marker='o',label="Juventus", color = "#0a7bff")
#pois1, = ax2.plot([i for i in range(8)], athmadrid_away_pois,
#                  linestyle='-', marker='o',label="Athletico Madrid", color = "#ff7c89")
#
#ax2.set_xlim([-0.5,7.5])
#ax2.set_ylim([-0.01,0.65])
#ax1.set_title("Number of Goals per Match (EPL 2016/17 Season)",size=14,fontweight='bold')
#ax2.set_xlabel("Goals per Match",size=13)
#ax2.text(-1.15, 0.9, 'Proportion of Matches', rotation=90, size=13)
#plt.tight_layout()
#plt.show()


        
        
        
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








