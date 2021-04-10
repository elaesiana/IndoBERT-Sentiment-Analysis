import os, sys
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import nltk
from nltk.tokenize import TweetTokenizer
import itertools
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Stemmer.Stemmer import Stemmer

nltk.download('stopwords')
from nltk.corpus import stopwords

tokenizer = BertTokenizer.from_pretrained('model/indobert_smsa_finetuned')
config = BertConfig.from_pretrained('model/indobert_smsa_finetuned')
model = BertForSequenceClassification.from_pretrained('model/indobert_smsa_finetuned', config=config)

stopwords_id = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

replace_mentions = r'@[A-Za-z0-9_]+' 
replace_links = r'https?://[A-Za-z0-9./]+' 

def preprocess(text):
    letters_only = re.sub(r'[^\w\s#@-]','', text)
    mention_replaced = re.sub(replace_mentions, '[USERNAME]', letters_only)
    link_replaced = re.sub(replace_links, '[URL]', mention_replaced)
    stemming = stemmer.stem(link_replaced)

    return " ".join(stemming.split())

def predict(text):
    subwords = tokenizer.encode(text)
    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

    logits = model(subwords)[0]
    label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
    score = F.softmax(logits, dim=-1).squeeze()[label] * 100
    return i2w[label], score.item()

def sentiment_analysis(filename, output_filename):
    data = pd.read_csv(filename)
    data['tweet_preprocess'] = data['tweet'].apply(lambda x: preprocess(x))

    data['predict'] = '-'
    data['score'] = 0.0

    for i, tweet in tqdm(enumerate(data['tweet_preprocess'])):
    label, score = predict(tweet)
    data['predict'][i] = label
    data['score'][i] = score

    data.to_csv(output_filename, index=False)