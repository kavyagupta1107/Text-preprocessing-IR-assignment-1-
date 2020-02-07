#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from bs4 import BeautifulSoup
from nltk.collocations import *
from nltk.probability import FreqDist 
from matplotlib import pyplot
import numpy as np
import operator
import collections


# In[2]:


pathfile=(r"C:\Users\lkg15\Desktop\IR assignment\wiki_90")
textfileobj=open(pathfile,'r',encoding="utf8")
soupobject=BeautifulSoup(textfileobj,"html.parser")


# In[3]:


wotags=soupobject.text
#without tags-- removes html tags


# In[4]:


tokens=nltk.word_tokenize(wotags)
#tokenizing the file using word_tokenize
print(len(tokens))
print(tokens[:1000])


# In[5]:


# obtaining unigrams,bigrams and trigrams
unigrams=nltk.ngrams(tokens,1)
bigrams=nltk.ngrams(tokens,2)
trigrams=nltk.ngrams(tokens,3)
unilist=list(unigrams)
bilist=list(bigrams)
trilist=list(trigrams)
print(len(unilist))
print(len(bilist))
print(len(trilist))


# In[6]:


uniset=list(set(unilist)) #set ensures unique unigrams
biset=list(set(bilist))  #set ensures unique bigrams
triset=list(set(trilist))  #set ensures unique trigrams
print(len(uniset))  
print(len(biset))
print(len(triset))


# In[66]:


#plotting
graph1=nltk.probability.FreqDist(unilist)  #graph1 is the FreqDist object for unigrams
pyplot.yscale('log')
graph1.plot(70,cumulative=False)


# In[67]:


pyplot.yscale('log')
graph2=nltk.probability.FreqDist(bilist)
pyplot.yscale('log')
graph3=nltk.probability.FreqDist(trilist)
graph2.plot(70,cumulative=False)
graph3.plot(70,cumulative=False)


# In[9]:


unisfrequency=np.array(list(reversed(sorted([z for _,z in graph1.items()]))))
count_uni= np.argmin(unisfrequency.cumsum() < graph1.N()*0.9)
print(count_uni) #count_uni number of tokens comprise 90%ofthe corpus


# In[10]:


bisfrequency=np.array(list(reversed(sorted([z for _,z in graph2.items()]))))
count_bi=np.argmin(bisfrequency.cumsum()<graph2.N()*0.8)
print(count_bi)


# In[11]:


trisfrequency=np.array(list(reversed(sorted([z for _,z in graph3.items()]))))
count_tri=np.argmin(trisfrequency.cumsum()<graph3.N()*0.7)
print(count_tri)


# In[12]:


#code for stemming
from nltk.stem import PorterStemmer
st=PorterStemmer()
words_stem = [st.stem(word) for word in tokens]
print(words_stem[:100])


# In[13]:


#code for lemmatization
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
words_lemma = [lm.lemmatize(word) for word in tokens]
print(words_lemma[:100])


# In[14]:

#ANALYSING TOKENS AFTER STEMMING
stem_unigrams=nltk.ngrams(words_stem,1)
stem_bigrams=nltk.ngrams(words_stem,2)
stem_trigrams=nltk.ngrams(words_stem,3)
stem_unilist=list(stem_unigrams)
stem_bilist=list(stem_bigrams)
stem_trilist=list(stem_trigrams)


# In[15]:


stem_uniset=list(set(stem_unilist))
stem_biset=list(set(stem_bilist))
stem_triset=list(set(stem_trilist))
print(len(stem_uniset))
print(len(stem_biset))
print(len(stem_triset))


# In[16]:


stem_graph1=nltk.probability.FreqDist(stem_unilist)
pyplot.yscale('log')
stem_graph1.plot(70,cumulative=False)
pyplot.yscale('log')
stem_graph2=nltk.probability.FreqDist(stem_bilist)
stem_graph3=nltk.probability.FreqDist(stem_trilist)
stem_graph2.plot(70,cumulative=False)
pyplot.yscale('log')
stem_graph3.plot(70,cumulative=False)


# In[17]:


stem_unisfrequency=np.array(list(reversed(sorted([z for _,z in stem_graph1.items()]))))
stem_count_uni= np.argmin(stem_unisfrequency.cumsum() < stem_graph1.N()*0.9)
print(stem_count_uni)

stem_bisfrequency=np.array(list(reversed(sorted([z for _,z in stem_graph2.items()]))))
stem_count_bi=np.argmin(stem_bisfrequency.cumsum()<stem_graph2.N()*0.8)
print(stem_count_bi)

stem_trisfrequency=np.array(list(reversed(sorted([z for _,z in stem_graph3.items()]))))
stem_count_tri=np.argmin(stem_trisfrequency.cumsum()<stem_graph3.N()*0.7)
print(stem_count_tri)


# In[18]:

#ANALYISNG TOKENS AFTER LEMMATIZATION
lemma_unigrams=nltk.ngrams(words_lemma,1)
lemma_bigrams=nltk.ngrams(words_lemma,2)
lemma_trigrams=nltk.ngrams(words_lemma,3)
lemma_unilist=list(lemma_unigrams)
lemma_bilist=list(lemma_bigrams)
lemma_trilist=list(lemma_trigrams)
lemma_uniset=list(set(lemma_unilist))
lemma_biset=list(set(lemma_bilist))
lemma_triset=list(set(lemma_trilist))
print(len(lemma_uniset))
print(len(lemma_biset))
print(len(lemma_triset))


# In[19]:


lemma_graph1=nltk.probability.FreqDist(lemma_unilist)
pyplot.yscale('log')
lemma_graph1.plot(70,cumulative=False)
pyplot.yscale('log')
lemma_graph2=nltk.probability.FreqDist(lemma_bilist)
lemma_graph3=nltk.probability.FreqDist(lemma_trilist)
lemma_graph2.plot(70,cumulative=False)
pyplot.yscale('log')
lemma_graph3.plot(70,cumulative=False)


# In[20]:


lemma_unisfrequency=np.array(list(reversed(sorted([z for _,z in lemma_graph1.items()]))))
lemma_count_uni= np.argmin(lemma_unisfrequency.cumsum() < lemma_graph1.N()*0.9)
print(lemma_count_uni)

lemma_bisfrequency=np.array(list(reversed(sorted([z for _,z in lemma_graph2.items()]))))
lemma_count_bi=np.argmin(lemma_bisfrequency.cumsum()<lemma_graph2.N()*0.8)
print(lemma_count_bi) 

lemma_trisfrequency=np.array(list(reversed(sorted([z for _,z in lemma_graph3.items()]))))
lemma_count_tri=np.argmin(lemma_trisfrequency.cumsum()<lemma_graph3.N()*0.7)
print(lemma_count_tri)


# In[40]:



#print(graph2.freq(('provide', 'the'))*graph2.N()) #plus plus
#print(graph1.freq(('provide',))*graph1.N()) #this-plusplus gives plusminus
#print(graph1.freq(('the',))*graph1.N())#this-plusplus gives minus plus
#print(graph2.N()-graph2.freq(('provide', 'the'))*graph2.N())#this gives minusminus


# In[43]:


#chi-square test
chi_dic={}
for comb in biset:
    plusplus=graph2.freq(comb)*graph2.N()
    plusminus=graph1.freq((comb[0],))*graph1.N()-plusplus
    minusplus=graph1.freq((comb[1],))*graph1.N()-plusplus
    minusminus=graph2.N()-plusplus
    chi_n=((plusplus*minusminus)-(plusminus*minusplus))**2
    chi_d=(plusplus+plusminus)*(plusplus+minusplus)*(plusminus+minusminus)*(minusplus+minusminus)
    chi=graph2.N()*chi_n/chi_d
    chi_dic[comb]=chi
    
    


# In[44]:


print(chi_dic)


# In[51]:



sorted_chidic = sorted(chi_dic.items(), key=operator.itemgetter(1))


# In[63]:


print(sorted_chidic[-20:])


# In[ ]:




