import os
import sys
import pickle
import numpy as np 
import pandas as pd


from src.exception import CustomException
from src.logger import logging

import string
import spacy
from nltk.stem.porter import PorterStemmer
exclude=string.punctuation

try:
    tokenizer = spacy.load("en_core_web_sm")
except: # If not present, we download
    spacy.cli.download("en_core_web_sm")
    tokenizer = spacy.load("en_core_web_sm")


#tokenizer=spacy.load('en_core_web_sm')

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)



def text_preprocessing(text):
    try:
        # Lowercasing
        text=str(text)
        text=text.lower() 

        # Removing punctuations
        
        text=text.translate(str.maketrans('','',exclude))

        # Tokenization
        text_new=[]
        for i in text.split(): 
            text_new.append(i.strip())  
        token_list=list(tokenizer(" ".join(text_new)))  

        # Stemming
        for i in range(0,len(token_list)):
            token_list[i]=PorterStemmer().stem(str(token_list[i]))
        
        logging.info("Text preprocessing completed")
        return " ".join(token_list)
    
    except Exception as e:
        logging.info("Error occured while doing text preprocessing")
        raise CustomException(e,sys)
