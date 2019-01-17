import nltk
from nltk import TweetTokenizer
from nltk.corpus import movie_reviews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sb

import random
p = [{"id":1,"category":"crime","headline":"There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV","short_description":"She left her husband. He killed their children. Just another day in America."},{"id":2,"category":"entertainment","headline":"Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup's Official Song","short_description":"Of course it has a song."},{"id":3,"category":"entertainment","headline":"Hugh Grant Marries For The First Time At Age 57","short_description":"The actor and his longtime girlfriend Anna Eberstein tied the knot in a civil ceremony."},{"id":4,"category":"entertainment","headline":"Jim Carrey Blasts 'Castrato' Adam Schiff And Democrats In New Artwork","short_description":"The actor gives Dems an ass-kicking for not fighting hard enough against Donald Trump."},{"id":5,"category":"entertainment","headline":"Julianna Margulies Uses Donald Trump Poop Bags To Pick Up After Her Dog","short_description":"The \"Dietland\" actress said using the bags is a \"really cathartic, therapeutic moment.\""},{"id":6,"category":"entertainment","headline":"Morgan Freeman 'Devastated' That Sexual Harassment Claims Could Undermine Legacy","short_description":"\"It is not right to equate horrific incidents of sexual assault with misplaced compliments or humor,\" he said in a statement."}]
q=[]
r=[]
for i in range(len(p)):
   # print(p[i]['headline'])
    q.append(p[i]['category'])
    r.append(p[i]['headline'])
    text = p[i]['headline']
    tkznr = TweetTokenizer()
    print(tkznr.tokenize(text))

for text in r:
    sentence = nltk.word_tokenize(text)
    for s in sentence:
        words = nltk.word_tokenize(s)
        tagword=nltk.pos_tag(words)
        newtagword=nltk.ne_chunk(tagword)
        print(newtagword)



print(type(q[1]))