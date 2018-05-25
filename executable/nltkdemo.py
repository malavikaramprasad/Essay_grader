
# coding: utf-8

# In[21]:


#!/usr/bin/env python2

"""
Created on Fri Apr  6 09:59:45 2018

@author: kksaikrishna
"""
from __future__ import print_function
import nltk
import scipy.stats
from enchant.checker import SpellChecker
import spacy
import pandas as pd
import numpy  as np
import sys
import re


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('treebank')
nltk.download('stopwords')
nltk.download('names')


from nltk.corpus import stopwords
from nltk.corpus import wordnet
from neuralcoref import Coref

from collections import Counter

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

import gensim

from nltk.corpus import names
import random


# In[22]:


#For Pronouns to Nouns
coref = Coref()

#Resource Files Path
results_file_path = '../output/results.txt'
test_csv_file_path = '../input/testing/index.csv'
test_file_path = '../input/testing/essays/'
test_delimiter = ';'

train_file_path = '../input/training/essays/'
train_csv_file_path = '../input/training/index.csv'
train_delimiter = ';'

trained_values_path = 'resources/trained_values.txt'

stops = set(stopwords.words('english'))
en_nlp = spacy.load('en')

male_path = 'resources/male.txt'
female_path = 'resources/female.txt'


# In[23]:


#Reads index.CSV and returns filenames, prompt and grade as seperate lists
def retDataFromCsv(indexPath,delim):
    data = pd.read_csv(indexPath,delimiter = delim)
    fileNames = data.filename.tolist()
    topics = data.prompt.tolist()
    return fileNames,topics

#Util to Read a file in the given path, and returs it as string
def returnFileContentAsStr(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
        return data
    
#Util to read float values from file
def readFileAsFloat(path):
    with open(path, 'r') as f:
        floats = map(float, f)
    return floats

#Writes the result to the file
def writeResult(output,path):
    with open(path,'w+') as f:   #************** Change it to w+ ***********
        for content in output:
            f.write(content)   

#Used to format output and write it into results.txt
def getOutputString(fileName,scoreA,scoreB,scoreC1,scoreC2,scoreC3,scoreD1,scoreD2,finalScore,highOrLow):  
    ret = ''
    ret = fileName + ';' + str(np.round(scoreA,2)) + ';' + str(np.round(scoreB,2)) + ';' 
    ret += str(np.round(scoreC1,2)) + ';' + str(np.round(scoreC2,2)) + ';' + str(np.round(scoreC3,2)) + ';'
    ret += str(np.round(scoreD1,2)) + ';' + str(np.round(scoreD2,2))
    ret += ';' + str(np.round(finalScore,2)) + ';' + highOrLow + '\n'
    return ret

#Util to Calculate Score from normal distribution, and give the score in a range according to the resolution and offselt
def calculateScore(mean,sd,value,res = 4,offset = 1):
    prob = scipy.stats.norm.cdf(x=value,loc=mean,scale=sd)
    return (prob * res) + offset

#Finds if the file is high or low from the score
def isHighOrLow(val):
    if val >= 2.5:
        return 'high'
    else:
        return 'low'

#Util that changes the pronouns to nouns
def resolvePronounToNoun(text):
    uText = unicode(text, "utf-8")
    coref.one_shot_coref(utterances=uText)
    resolvedText = coref.get_resolved_utterances()
    return resolvedText[0]
    
#Util to stem the words in the input list of words
def remStopWords(words):
    ret = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words:
            ret.append(word)
    return ret

#Util that takes a word and returns the list of synonyms
def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

#Method that finds the synonyms for each word in the list and appends it to the list
def appendSynonyms(words):
    ret = []
    for word in words:
        ret.append(word)
        synonyms = getSynonyms(word)
        for syn in synonyms:            
            ret.append(syn)
    return ret    

#Util to stem words
def stemWords(words):
    ret = []
    for word in words:
        ret.append(stemmer.stem(word))
    return ret

#Method that does text preprocessing
def textPreProcess(text,isSplitSent = 1, isAppendSynonyms = 1):
    ret = []
    if(isSplitSent != 0):
        sents = nltk.sent_tokenize(text)
        words_by_sents = [[w.lower() for w in nltk.word_tokenize(sent)]
                           for sent in sents]
        
        for words in words_by_sents:
            temp = []
            temp = remStopWords(words)
            if(isAppendSynonyms != 0):
                temp = appendSynonyms(temp)
            temp = stemWords(temp)
            ret.append(list(set(temp)))
    else:
        words = [w.lower() for w in nltk.word_tokenize(text)]
        ret = remStopWords(words)
        if(isAppendSynonyms != 0):
            ret = appendSynonyms(ret)
        ret = stemWords(list(set(ret)))
    
    return ret

#util to find top n frequent words from text
def findTopNFreqWords(text,n):
    ret = []
    words = [word.lower() for word in nltk.word_tokenize(text)]
    word_count = Counter(words)
    most_common = word_count.most_common(n)
    for word in most_common:
        ret.append(word[0])
    return ret


#Method to return the last letter of the word, in a formatted form
def gender_features(word):
    return {'last_letter': word[-1]}

#Classifier for Gender Classification
labeled_names = ([(name, 'male') for name in names.words()] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[24]:


#Class to display progress bar
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


# In[25]:


#Class that does various counts using nltk
class NltkCounter:
    #Tokenizes the text as sentene, words, gets it's pos tokens, and gets the dependency using spacy
    def preProcessData(self,file_content):
        self.file_content = file_content
        self.sent_tokens = nltk.sent_tokenize(file_content)
        self.word_tokens = []
        self.pos_tokens = []
        self.spacy_tokens = []
        for sent in self.sent_tokens:
            words = nltk.word_tokenize(sent)
            self.word_tokens.append(words)
            
            posTags = nltk.pos_tag(words)
            self.pos_tokens.append(posTags)
            
            uSent = unicode(sent, "utf-8")
            spacyOut = en_nlp(uSent)
            self.spacy_tokens.append(spacyOut)
    
    #returns number of characters in the current file
    def retCharCount(self):
        return len(self.file_content)
    
    #returns number of words in the current file
    def retWordCount(self):
        wc = 0
        for words in self.word_tokens:
            wc += len(words)
        return wc

    #returns number of sentences in the current file
    def retSentCount(self):
        return len(self.sent_tokens)
    
    #returns number of spelling errors in the current file
    def retSpErrCount(self):
        #considers both UK and US dictionary
        chkr = SpellChecker('en_US','en_GB')
        chkr.set_text(self.file_content)
        count = 0
        for err in chkr:
            count += 1
        return count
    
    #returns number of sentences that has agreeing Subject-Verb in the current file
    def retSvAgrCount(self):
        count = 0
        setSNoun = ['NN','NNP']
        setPNoun = ['NNS','NNPS']
        setSPrp = ['he','she','it']
        setPPrp = ['i','you','we','they']
        for i in range(len(self.sent_tokens)):
            for j in range(len(self.spacy_tokens[i])):
                if(self.spacy_tokens[i][j].dep_ == u'nsubj'):
                    for k in range(j,len(self.spacy_tokens[i])):
                        if(self.spacy_tokens[i][k].pos_ == u'VERB'):
                            if(self.spacy_tokens[i][k].tag_ == 'VBZ'):
                                if(self.spacy_tokens[i][j].tag_ in setSNoun or self.spacy_tokens[i][j].text.lower() in setSPrp):
                                    count += 1
                            elif(self.spacy_tokens[i][k].tag_ == 'VBP'):
                                if(self.spacy_tokens[i][j].tag_ in setPNoun or self.spacy_tokens[i][j].text.lower() in setPPrp):
                                    count += 1
                            break
        return count          
    
    #returns number of sentences that has agreeing Verb-Tense in the current file
    def retVTAgrCount(self):
        count = 0
        present_tense_verbs={"VBP", "VBZ", "VB", "VBG"}
        past_tense_verbs={"VBD", "VBN"}
        for i in range(len(self.sent_tokens)):
            count_present=0
            count_past=0
            verb_tags = [tag[1] for tag in self.pos_tokens[i] if tag[1] in present_tense_verbs or tag[1] in past_tense_verbs]
            for v in verb_tags:
                if(v in present_tense_verbs):
                    count_present = count_present + 1;
                if(v in past_tense_verbs):
                    count_past = count_past + 1;
            if(count_present==0 and count_past>0):
                count += 1
            elif(count_present>0 and count_past==0):
                count += 1
        return count 
    
    #returns number of verbs in the current file
    def retVerbCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            verb_postags =["VBP", "VBZ", "VB", "VBG","VBD","VBN"]
            for tag in self.pos_tokens[i]:
                if(tag[1] in verb_postags):
                    count = count + 1     
        return count
    
    #Returns the number of Subject in the current file
    def retSubjCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            for j in range(len(self.spacy_tokens[i])):
                if(self.spacy_tokens[i][j].dep_ == u'nsubj'):
                    count += 1
                    break
        return count
    
    #Returns number of sentences that start with Capital letter and ends with '.'
    def retCapCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            if(self.sent_tokens[i][0].isupper() and self.sent_tokens[i][-1]=='.'):
                count += 1
        return count
    
    #Returns a tf-idf score for the topic coherence, sentence by sentence in a file
    def retTopicCoherence(self,topic):
        res_text = resolvePronounToNoun(self.file_content)
        words_by_sent = textPreProcess(res_text,1,0)        
        dictionary = gensim.corpora.Dictionary(words_by_sent)
        corpus = [dictionary.doc2bow(words) for words in words_by_sent]
        tf_idf = gensim.models.TfidfModel(corpus)
        sims = gensim.similarities.Similarity('',tf_idf[corpus],num_features=len(dictionary))
        
        filtered_topic = topic.split('\t')
        res_topic = resolvePronounToNoun(filtered_topic[2])
        words_topic = textPreProcess(res_topic,0,1)
        topic_bow = dictionary.doc2bow(words_topic)
        topic_doc_tf_idf = tf_idf[topic_bow]
        topicCoherence = sims[topic_doc_tf_idf]
        add = 0
        for coh in topicCoherence:
            add += coh
        return add
    
    #Returns number of sentences with correct gender/number matching for possessive pronouns
    def textcoherency(self):
        malePRP = ["he","his"]
        femalePRP = ["she","her"]
        pluralPRP = ["they","them","their"]
        totalSingPrp = ["he","his","she","her"]
        errorcount = 0
        if(len(self.sent_tokens) > 2):
            twosent = self.sent_tokens[0] + self.sent_tokens[1]
            gender = ''
            for i in range(2 , len(self.sent_tokens)):
                for pos in self.pos_tokens[i]:
                    prev_tokens = nltk.word_tokenize(twosent)
                    prev_pos_tokens = nltk.pos_tag(prev_tokens)
                    if(pos[1] == 'PRP' or pos[1] == 'PRP$' and pos[0].lower() in totalSingPrp):
                        for pos_prev in prev_pos_tokens:
                            if(pos_prev[1] == "NNP" or pos_prev[1] == "NNPS"):
                                gender = classifier.classify(gender_features(pos_prev[0]))
                                if(gender == "male" and pos[0].lower() in malePRP):
                                    errorcount = errorcount + 1
                                if(gender == "female" and pos[0].lower() in femalePRP):
                                    errorcount = errorcount + 1
                    if(pos[1] == 'PRP' or pos[1] == 'PRP$' and pos[0].lower() in pluralPRP):
                        for pos_prev in prev_pos_tokens:
                            if(pos_prev[1] == "NNPS"):
                                errorcount = errorcount + 1
                            
                #find all word Noun tags in prev sentence and look for singular/plural and gender
                twosent = self.sent_tokens[i-1] + self.sent_tokens[i]
        return errorcount
    
    #Returns a score, with each sentence scored with number of Parts of Speech elements used
    def distribution(self):
        score = 0
        i = 0
        noun_list = ["NN","NNS","NNP","NNPS"]
        verb_list = ["VBP", "VBZ", "VB", "VBG","VBD","VBN"]
        adv_list = ["RB","RBR","RBS"]
        adj_list = ["JJ","JJR","JJS"]   
        
        for sentence in self.sent_tokens:
            pos_tokens = self.pos_tokens[i]
            list_of_noun = [tag[1] for tag in pos_tokens if tag[1] in noun_list]
            list_of_verbs = [tag[1] for tag in pos_tokens if tag[1] in verb_list]
            list_of_adv = [tag[1] for tag in pos_tokens if tag[1] in adv_list]
            list_of_adj = [tag[1] for tag in pos_tokens if tag[1] in adj_list]
            if(list_of_noun and list_of_verbs and list_of_adj and list_of_adv):
                score = score + 4
            elif(list_of_noun and list_of_verbs and list_of_adv):
                score = score + 3
            elif(list_of_noun and list_of_verbs and list_of_adj):
                score = score + 3
            elif(list_of_noun and list_of_verbs):
                score = score + 2 
            i += 1
        return score
    
    #Returns number of sentences with subject associated with the main verb, or a verb that is dependent to the main verb
    def nsubjcheck(self):
        count = 0
        i = 0
        for sentence in self.sent_tokens:
            doc = self.spacy_tokens[i]
            root = ""
            listofverb = []
            for token in doc:
                if(token.dep_ == "nsubj"):
                    listofverb.append(token.head.text)
                if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                    root = token.text
            if(root in listofverb):
                count = count +1
            for verb in listofverb:
                if(verb != root):
                    for token in doc:
                        if(token.text == verb):
                            if(token.head.text == root):
                                count = count + 1
            i += 1
        return count
    
    #Returns number of sentences with object associated with the main verb, or a verb that is dependent to the main verb
    def dobjcheck(self):
        count = 0
        i = 0
        for sentence in self.sent_tokens:
            doc = self.spacy_tokens[i]
            root = ""
            listofverb = []
            for token in doc:
                if(token.dep_ == "dobj"):
                    listofverb.append(token.head.text)
                if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                    root = token.text
            if(root in listofverb):
                count = count +1
            for verb in listofverb:
                if(verb != root):
                    for token in doc:
                        if(token.text == verb):
                            if(token.head.text == root):
                                count = count + 1
            i += 1
        return count
    
    #Returns number of sentences with the root of the sentence having verb tag
    def checkforrootverb(self):
        count = 0
        i = 0
        for sentence in self.sent_tokens:
            doc = self.spacy_tokens[i]
            for token in doc:
                if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                    count = count + 1       
            i += 1
        return count 
    


# In[27]:


#Class that does training and test the tag from Train and Test Folder
class AutomaticEssayGrader:
    #Method that trains the mean and SD values
    def train(self):
        print('Training Started')
        listFileNames,listPrompt = retDataFromCsv(train_csv_file_path,train_delimiter)
        objCounter = NltkCounter()
        listCharCount = []
        listWordCount = []
        listSentCount = []
        listSpErrCount = []
        listVerbCount = []
        listSvAgrCount = []
        listVTAgrCount = []
        
        listSubjCount = []
        listSubjRoot = []
        listObjRoot = []
        listRoot = []
        listCapCount = []
        
        listDist = []
        listTC = []
        
        listTfIdf = []
        
        listFinalScore = []
        
        progressTrain = ProgressBar(len(listFileNames), fmt=ProgressBar.FULL)
        
        for files,topic in zip(listFileNames,listPrompt):
            progressTrain.current += 1
            progressTrain()
            path = train_file_path + files
            file_content = returnFileContentAsStr(path)
            objCounter.preProcessData(file_content)
            
            #Finds counts of Nlp elements of the current file
            listCharCount.append(objCounter.retCharCount())
            listWordCount.append(objCounter.retWordCount())
            listSentCount.append(objCounter.retSentCount())
            listSpErrCount.append(objCounter.retSpErrCount())
            listSvAgrCount.append(objCounter.retSvAgrCount())
            listVTAgrCount.append(objCounter.retVTAgrCount())
            listVerbCount.append(objCounter.retVerbCount())
            
            listSubjCount.append(objCounter.retSubjCount())
            listSubjRoot.append(objCounter.nsubjcheck())
            listObjRoot.append(objCounter.dobjcheck())
            listRoot.append(objCounter.checkforrootverb())
            listCapCount.append(objCounter.retCapCount())
            
            listDist.append(objCounter.distribution())
            listTC.append(objCounter.textcoherency())

            listTfIdf.append(objCounter.retTopicCoherence(topic))
            
        
        #Finds mean and sd
        m_char = np.mean(listCharCount)
        sd_char = np.std(listCharCount)
        
        m_word = np.mean(listWordCount)
        sd_word = np.std(listWordCount)
        
        m_sent = np.mean(listSentCount)
        sd_sent = np.std(listSentCount)
        
        m_sp = np.mean(listSpErrCount)
        sd_sp = np.std(listSpErrCount)
        
        m_sv = np.mean(listSvAgrCount)
        sd_sv = np.std(listSvAgrCount)
        
        m_vt = np.mean(listVTAgrCount)
        sd_vt = np.std(listVTAgrCount)
        
        m_verb = np.mean(listVerbCount)
        sd_verb = np.std(listVerbCount)
        
        m_sub = np.mean(listSubjCount)
        sd_sub = np.std(listSubjCount)
        
        m_cap = np.mean(listCapCount)
        sd_cap = np.std(listCapCount)
        
        m_tfidf = np.mean(listTfIdf)
        sd_tfidf = np.std(listTfIdf)
        
        m_sr = np.mean(listSubjRoot)
        sd_sr = np.std(listSubjRoot)
        
        m_or = np.mean(listObjRoot)
        sd_or = np.std(listObjRoot)
        
        m_rt = np.mean(listRoot)
        sd_rt = np.std(listRoot)
        
        m_d = np.mean(listDist)
        sd_d = np.std(listDist)
        
        m_tc = np.mean(listTC)
        sd_tc = np.std(listTC)
        
        #Calculates Scores, and finds the distribution for final score
        for i in range(len(listFileNames)):
            scoreA_1 = calculateScore(m_char,sd_char,listCharCount[i])
            scoreA_2 = calculateScore(m_word,sd_word,listWordCount[i])
            scoreA_3 = calculateScore(m_sent,sd_sent,listSentCount[i])
            scoreA = 0.1*scoreA_1 + 0.1*scoreA_2 + 0.8*scoreA_3
            
            scoreB = calculateScore(m_sp,sd_sp,listSpErrCount[i],4,0)
            
            scoreCi_1 = calculateScore(m_sv,sd_sv,listSvAgrCount[i])
            scoreCi_2 = calculateScore(m_vt,sd_vt,listVTAgrCount[i])
            scoreCi = 0.5*scoreCi_1 + 0.5*scoreCi_2
            
            scoreCii = calculateScore(m_verb,sd_verb,listVerbCount[i])
            
            
            scoreCiii_1 = calculateScore(m_sub,sd_sub,listSubjCount[i])
            scoreCiii_2 = calculateScore(m_cap,sd_cap,listCapCount[i])
            scoreCiii_3 = calculateScore(m_sr,sd_sr,listSubjRoot[i])
            scoreCiii_4 = calculateScore(m_or,sd_or,listObjRoot[i])
            scoreCiii_5 = calculateScore(m_rt,sd_rt,listRoot[i])
            scoreCiii = 0.15*scoreCiii_1 + 0.35*scoreCiii_2 + 0.1*scoreCiii_3 + 0.1*scoreCiii_4 + 0.3*scoreCiii_5
            
            scoreDi_1 = calculateScore(m_d,sd_d,listDist[i])
            scoreDi_2 = calculateScore(m_tc,sd_tc,listTC[i])
            scoreDi = 0.9*scoreDi_1 + 0.1*scoreDi_2
            
            scoreDii = calculateScore(m_tfidf,sd_tfidf,listTfIdf[i])
            
            finalScore = 4*(scoreA) - 1*(scoreB) + 1*(scoreCi) + 1*scoreCii + 1*(scoreCiii) + 1*(scoreDi) + 1*(scoreDii)
#             finalScore = scoreDii #######Need to Take it OUt########
            
            listFinalScore.append(finalScore)
            
        
        m_fs = np.mean(listFinalScore)
        sd_fs = np.std(listFinalScore)
        
        #Writes trained values to file
        trained_out = str(m_char) + '\n' + str(sd_char) + '\n'
        trained_out += str(m_word) + '\n' + str(sd_word) + '\n'
        trained_out += str(m_sent) + '\n' + str(sd_sent) + '\n'
        trained_out += str(m_sp) + '\n' + str(sd_sp) + '\n'
        trained_out += str(m_sv) + '\n' + str(sd_sv) + '\n'
        trained_out += str(m_vt) + '\n' + str(sd_vt) + '\n'
        trained_out += str(m_verb) + '\n' + str(sd_verb) + '\n'
        trained_out += str(m_sub) + '\n' + str(sd_sub) + '\n'
        trained_out += str(m_cap) + '\n' + str(sd_cap) + '\n'  
        trained_out += str(m_sr) + '\n' + str(sd_sr) + '\n'   
        trained_out += str(m_or) + '\n' + str(sd_or) + '\n'   
        trained_out += str(m_rt) + '\n' + str(sd_rt) + '\n'    
        trained_out += str(m_d) + '\n' + str(sd_d) + '\n'   
        trained_out += str(m_tc) + '\n' + str(sd_tc) + '\n'
        trained_out += str(m_tfidf) + '\n' + str(sd_tfidf) + '\n' 
        trained_out += str(m_fs) + '\n' + str(sd_fs) + '\n'   
            
        writeResult(list(trained_out),trained_values_path)
        print('\nTraining Complete')
        
    def test(self):
        print('\nTesting Started')
        tv = readFileAsFloat(trained_values_path)
        lowScore = 0
        highScore = 0
        output = []
        objCounter = NltkCounter()

        listFileNames,listPrompt = retDataFromCsv(test_csv_file_path,train_delimiter) #need to change

        progressTest = ProgressBar(len(listFileNames), fmt=ProgressBar.FULL)
        for files,topic in zip(listFileNames,listPrompt):
            progressTest.current += 1
            progressTest()
            path = test_file_path + files
            file_content = returnFileContentAsStr(path)
            
            objCounter.preProcessData(file_content)
            
            #calculate Score for Each File
            scoreA_1 = calculateScore(tv[0],tv[1],objCounter.retCharCount())
            scoreA_2 = calculateScore(tv[2],tv[3],objCounter.retWordCount())
            scoreA_3 = calculateScore(tv[4],tv[5],objCounter.retSentCount())
            scoreA = 0.1*scoreA_1 + 0.1*scoreA_2 + 0.8*scoreA_3
            
            scoreB = calculateScore(tv[6],tv[7],objCounter.retSpErrCount(),4,0)
            
            scoreCi_1 = calculateScore(tv[8],tv[9],objCounter.retSvAgrCount())
            scoreCi_2 = calculateScore(tv[10],tv[11],objCounter.retVTAgrCount())
            scoreCi = 0.5*scoreCi_1 + 0.5*scoreCi_2
            
            scoreCii = calculateScore(tv[12],tv[13],objCounter.retVerbCount())
            
            
            scoreCiii_1 = calculateScore(tv[14],tv[15],objCounter.retSubjCount())
            scoreCiii_2 = calculateScore(tv[16],tv[17],objCounter.retCapCount())
            scoreCiii_3 = calculateScore(tv[18],tv[19],objCounter.nsubjcheck())
            scoreCiii_4 = calculateScore(tv[20],tv[21],objCounter.dobjcheck())
            scoreCiii_5 = calculateScore(tv[22],tv[23],objCounter.checkforrootverb())
            scoreCiii = 0.15*scoreCiii_1 + 0.35*scoreCiii_2 + 0.1*scoreCiii_3 + 0.1*scoreCiii_4 + 0.3*scoreCiii_5
            
            
            scoreDi_1 = calculateScore(tv[24],tv[25],objCounter.distribution())
            scoreDi_2 = calculateScore(tv[26],tv[27],objCounter.textcoherency())
            scoreDi = 0.9*scoreDi_1 + 0.1*scoreDi_2
            
            
            scoreDii = calculateScore(tv[28],tv[29],objCounter.retTopicCoherence(topic))
                 
            
            #Calculate final score
            total = 4*(scoreA) - 1*(scoreB) + 1*(scoreCi) + 1*scoreCii + 1*(scoreCiii) + 1*(scoreDi) + 1*(scoreDii)
        
            finalScore = calculateScore(tv[30],tv[31],total)
            highOrLow = isHighOrLow(finalScore)
            output.append(getOutputString(files,scoreA,scoreB,scoreCi,scoreCii,scoreCiii,scoreDi,scoreDii,finalScore,highOrLow))
            if highOrLow == 'high':
                highScore += 1
            elif highOrLow == 'low':
                lowScore += 1
        writeResult(output,results_file_path)
        print('\nTesting Complete')
        print('Number of Essays graded High:',highScore)
        print('Number of Essays graded Low:', lowScore)
        print('\nResult is available in result.txt file under the output folder.')
        return [highScore,lowScore]


# In[29]:

objEssayGrader = AutomaticEssayGrader()
if len(sys.argv) != 2:
    print('Invalid Arguments. \nExecute\n      nltkdemo.py -h\nto know more.')
elif sys.argv[1] == '-train':
    objEssayGrader.train()
elif sys.argv[1] == '-test':
    objEssayGrader.test()
elif sys.argv[1] == '-both':
    objEssayGrader.train()
    objEssayGrader.test()
elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\n\n*********Automatic Essay Grader*********\n\n')
    print('Format:   nltkdemo.py <task>')
    print('\n<task>\n-train : To train with new set of data \n-test : To run the test with pre-computed values\n-both : To run the training and then test based on that\n-h or --help : For Help')
    print('\nExample: nltkdemo.py -test\nnltkdemo.py -both\n\n\n\n')
    print('\n\n****************************************\n\n')
else:
    print('Invalid Arguments. Execute\n nltkdemo.py -h\n to know more...')

