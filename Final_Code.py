# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:23:26 2019

@author: sbhardwa

https://github.com/binoydutt/Resume-Job-Description-Matching

"""
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import sys, getopt, re
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pattern import en
from os import listdir
from os.path import isfile, join
from gensim.summarization import keywords
from sklearn.manifold import MDS
import pandas as pd
from gensim.summarization import keywords
import matplotlib.pyplot as plt
from flashtext.keyword import KeywordProcessor
import io
import re
import nltk
import spacy
import docx2txt
import subprocess
from datetime import datetime
from dateutil import relativedelta
#from . import constants as cs
from pdfminer.pdfparser import PDFSyntaxError

dir_cvs= '//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/New folder/CV'   
sys.path.append('//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/')
import constants as cs
job_id='Java Specialist'

########converts pdf, returns its text content as a string######
def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text 

######extract raw CV text#######
def preprocess_training_data_raw(dir_cvs):    
    dircvs = [join(dir_cvs, f) for f in listdir(dir_cvs) if isfile(join(dir_cvs, f))]
    allresume_raw = []
    
    for cv in dircvs:
        alltext = convert(cv,pages=None)
        #alltext=alltext.lower()
        #alltext.translate(str.maketrans('','', string.punctuation))
        #stop_resume=(stopword_i(alltext))
        allresume_raw.append(alltext)
    return allresume_raw

#######extract CV sections from raw resume#######
def extract_entity_sections(text):
    text_split = [i.strip() for i in text.split('\n')]
    # sections_in_resume = [i for i in text_split if i.lower() in sections]
    entities = {}
    key = False
    for phrase in text_split:
        if len(phrase) == 1:
            p_key = phrase
        else:
            p_key = set(phrase.lower().split()) & set(cs.RESUME_SECTIONS_PROFESSIONAL)
        try:
            p_key = list(p_key)[0]
        except IndexError:
            pass
        if p_key in cs.RESUME_SECTIONS_PROFESSIONAL:
            entities[p_key] = []
            key = p_key
        elif key and phrase.strip():
            entities[key].append(phrase)
    
    return entities

########extract no of years of experience from raw resume######
def get_total_experience(experience_list):
    
    if  max(re.findall(r'\d{4}', str(experience_list)),default=0):
        experience = re.findall(r'\d{4}', str(experience_list))
        total_experience_in_years=int(max(experience))-int(min(experience))
    else:
        total_experience_in_years="Not Found"

    return total_experience_in_years


###########extract the education from raw resume############
def extract_education(nlp_text):
    edu = {}
    try:
        for index, text in enumerate(nlp_text):
            for tex in text.split():
                tex = re.sub(r'[?|$|.|!|,]', r'', tex)
                if tex.lower() in cs.EDUCATION and tex not in cs.STOPWORDS:
                    edu[tex] = text + nlp_text[index + 1]
    except IndexError:
        pass

    # Extract year
    education = []
    for key in edu.keys():
        year = re.search(re.compile(cs.YEAR), edu[key])
        if year:
            education.append((key, ''.join(year.group(0))))
        else:
            education.append(key)
    return education


#########extract skills from raw resume######
def ski_extract(keylist,resume):
    keylisthit=[]
    for i in range(len(resume)):
        keylist_hit=[]
        for j in range(len(keylist)):
            k2=re.findall(keylist[j],resume[i].lower(), re.MULTILINE)
            if len(k2)>0:
                keylist_hit1=(keylist[j]+"{"+str(len(k2))+"}")
                #skill_hit=[skill[i]+str(len(k2))] 
            else:
                keylist_hit1=(keylist[j]+"{"+str(0)+"},")
            keylist_hit.append(keylist_hit1)   
    
        keylisthit.append(keylist_hit)
    
    return (keylisthit)   

allresume_raw=preprocess_training_data_raw(dir_cvs) 
skills=ski_extract(cs.SKILLS[job_id],allresume_raw)

education=[]
experience=[]
experience_years=[]
for i in range(len(allresume_raw)):
    educa=extract_entity_sections(allresume_raw[i])['education']
    exper=extract_entity_sections(allresume_raw[i])['experience']
    exper_yr=get_total_experience(extract_entity_sections(allresume_raw[i])['experience'])
    education.append(educa)
    experience.append(exper)
    experience_years.append(exper_yr)

#############
#####perform NLP text extraction##### 
#############

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    punctuations="?:!.,;!@#$%^&*()_-+=[]{}|/?><"
    stem_sentence=[]
    ps=PorterStemmer()
    for word in token_words:
        if word in punctuations:
            token_words.remove(word)
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)    

def stopword_i(file):
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    stop_res= pattern.sub('', file)
    output=re.sub('[0-9]+','', stop_res)
    return output
    
def preprocess_training_data_nlp(dir_cvs):    
    dircvs = [join(dir_cvs, f) for f in listdir(dir_cvs) if isfile(join(dir_cvs, f))]
    allresume = []
    
    for cv in dircvs:
        alltext = convert(cv,pages=None)
        alltext=alltext.lower()
        alltext.translate(str.maketrans('','', string.punctuation))
        
        stem_alltext=stemSentence(alltext)
        lemmatizer= WordNetLemmatizer() 
        lem_resume=lemmatizer.lemmatize(stem_alltext,pos="v")
        stop_resume=(stopword_i(lem_resume))
        allresume.append(stop_resume)
    return allresume

###Create tags of each resume####
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
        #start=allFiles.find('\\')+2
        #end=allFiles.find('.pdf',start)
       # tags=allFiles[start:end]
    return allFiles
      
allresume=preprocess_training_data_nlp(dir_cvs) 
alltags=getListOfFiles(dir_cvs)  

########################
#################
####Document Matching####

jd_cvs= '//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/New folder/JD'    
jd=preprocess_training_data_nlp(jd_cvs)                     
jdtags=getListOfFiles(jd_cvs)     


'''df =pd.read_csv('//nb\\corp\\Users\\NY\\NY16\\sbhardwa\\@nbcfg\\Desktop\\Python codes\\PDFextract\\data.csv')
jd = df['Job Description'].tolist()
companies = df['company'].tolist()
positions = df['position'].tolist()

jd=[item.lower() for item in jd]

def stemSentence(sentence):
    all_sentence=[]
    for i in range(len(jd)):
        token_words=word_tokenize(sentence[i])
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(ps.stem(word))
            stem_sentence.append(" ")
        stem_sentence="".join(stem_sentence)
        all_sentence.append(stem_sentence)
    return all_sentence

jd=stemSentence(jd)

def lemSentence(sentence):
    all_sentence=[]
    for i in range(len(jd)):
        lem_sentence=lemmatizer.lemmatize(jd[i])
        all_sentence.append(lem_sentence)
    return all_sentence
'''
import gensim
from gensim import models
docs = []
for i in range(len(allresume)):
    sent = models.doc2vec.TaggedDocument(words = ''.join(map(str,allresume[i])),tags = ['{}_{}'.format(alltags[i],i)])
    #sent = models.doc2vec.LabeledSentence(words = jd[i].split(),tags = ['jd'])
    docs.append(sent)
    
model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025, epochs=20)
model.save('//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/model')
#modeln=model.load('//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/model')
model.build_vocab(docs)

for epochs in range(10):
    model.train(docs, epochs=epochs, total_examples=model.corpus_count)
    model.alpha -= 0.002  # decrease the learning rate`[]
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    
data = []
for i in range(len(allresume)):
    data.append(model.docvecs[i])
    
data.append(model.infer_vector(jd))

mds = MDS(n_components=2, random_state=1)
pos = mds.fit_transform(data)
#print pos
xs,ys = pos[:,0], pos[:,1]
for x, y in zip(xs, ys):
    plt.scatter(x, y)
#    plt.text(x, y, name)
xs2,ys2 = xs[-1], ys[-1]
plt.scatter(xs2, ys2, c='Red', marker='+')
plt.text(xs2,ys2,'jd')
plt.savefig('distance.png')
plt.show()


#from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance
cos_dist =[]
for i in range(len(data)-1):
    cos_dist.append(float(distance.cosine(data[i],data[len(data)-1])))
  
key_list =[]

for j in allresume:
    key =''
    for word in keywords(j).split('\n'):
        key += '{} '.format(word)
    key_list.append(key)


summary = pd.DataFrame({
        'Resume_Tag': alltags,
        'Cosine Distances': cos_dist,
        'Keywords': key_list,
        'Skills':skills,
        'Education':education,
        'Experience':experience,
        'Experience Years':experience_years
    })
z =summary.sort_values('Cosine Distances', ascending=False)
z.to_csv('//nb/corp/Users/NY/NY16/sbhardwa/@nbcfg/Desktop/Python codes/PDFextract/Summary.csv',encoding="utf-8")

#json=z.to_json()

