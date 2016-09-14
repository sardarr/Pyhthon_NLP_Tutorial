#########################################################################
#Author: Sardar Hamidian
#Date:09/10/2016
#Python for NLP
#########################################################################
#Reading and writing to or from file
#########################################################################
M_list=[]
input_file=open('sample_one.txt','r')
my_list=input_file.readlines()
# for lines in input_file:
#     my_list.append(lines)



new_content=['My name is Alex this is the first content should be written in txt file']
newf =  open('new_file.txt', 'w')
for line in new_content:
    newf.write(line)


#########################################################################
#Word and Sentence Tokenization
#
#########################################################################
from nltk import word_tokenize,sent_tokenize

# word_tok=word_tokenize(my_list[0])
# print (word_tok)
# sent_token=sent_tokenize(my_list[0])
# print (sent_token)

#########################################################################
#Topwords
#
#########################################################################
from nltk.corpus import stopwords
from string import punctuation
punc_word=set(punctuation)
stop_word=set(stopwords.words("English"))
# for word in word_tok:
#     if word not in stop_word and word not in punc_word:
#         print (word)
# filtered_word=[w for w in word_tok if w not in stop_word ]

#########################################################################
#Stemming and lemmatizing
#
#########################################################################
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

wd_nt_lem=WordNetLemmatizer()


ps=PorterStemmer()
exp_for_python=["cats","are","driving","drived",'drivable','drivability','walking','walked','walkable']

# print(wd_nt_lem.lemmatize('cats'))
# for all_words in exp_for_python:
#     print (ps.stem(all_words),end=', ')

#########################################################################
#POS
# labling pos for every isngle word
#########################################################################
import nltk
from nltk.corpus import state_union #there are the corpus from previous presidnets
from nltk.tokenize import PunktSentenceTokenizer # it's unsuppervised ml tokenizer you can also train it your self
test_text=state_union.raw('2005-GWBush.txt')#loading corporas

pt_train=PunktSentenceTokenizer()
tokenized_text=pt_train.tokenize(test_text)
for i in tokenized_text:
    word=nltk.word_tokenize(i)
    tags=nltk.pos_tag(word)
    ner_of_content=nltk.ne_chunk(tags)
    ner_of_content=nltk.ne_chunk(tags,binary=True)#only to get Name Entilty
    # print(ner_of_content,end=',')
    # ner_of_content.draw()
    # ner_of_content.draw()

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
# print (nltk.pos_tag(text))

#########################################################################
#Wordnet
#
#########################################################################
from nltk.corpus import wordnet

syns_of_word=wordnet.synsets("program")
#sysnset
# print (syns_of_word)
#lemmas
# print(syns_of_word[0].lemmas())
#definition
# print(syns_of_word[0].definition())
#examples
print(syns_of_word[0].examples())
#synonyms and antonyms
synonyms=[]
antonyms=[]
for sy_i in wordnet.synsets("good"):
    for lm_sy in sy_i.lemmas():
            synonyms.append(lm_sy.name())
            if lm_sy.antonyms():
                antonyms.append(lm_sy.antonyms()[0].name())

# print(set(synonyms))
# print(antonyms)

########################################################
#Semantic similarity using word net
#
########################################################
#
# word_one=wordnet.synset("ship.n.01")
# word_two=wordnet.synset("boat.n.01")
# print(word_one.wup_similarity(word_two))
#
# word_one=wordnet.synset("ship.n.01")
# word_two=wordnet.synset("car.n.01")
# print(word_one.wup_similarity(word_two))
#
# word_one=wordnet.synset("ship.n.01")
# word_two=wordnet.synset("dog.n.01")
# print(word_one.wup_similarity(word_two))

########################################################
# N-gram Language Model
#
#########################################################
from nltk import bigrams
# from nltk import trigrams
# text=" this is just a sample for bigram and trigram, this is just a beginning and nothing else matter"
#
#
# # split the texts into tokens
# tokens = nltk.word_tokenize(text)
# tokens = [token.lower() for token in tokens if len(token) > 1] #same as unigrams
# bi_tokens = bigrams(tokens)
# tri_tokens = trigrams(tokens)
#
# fdist = nltk.FreqDist(bi_tokens) #counting word frequency
# for k,v in fdist.items():
#     print (k,v)
# # print(bi_tokens)
# # print ([(item, tri_tokens.count(item)) for item in sorted(set(tri_tokens))])