import streamlit as st
import nltk
from nltk.tag import untag, RegexpTagger
import numpy as np
import pandas as pd
import random
import math 
from tbl import *

st.subheader ('Training')
f = open('Datasets.txt', "r")
sentences = f.read().split('\n')
data_original = [[nltk.tag.str2tuple(word) for word in sentence.split()] for sentence in sentences]
random.shuffle(data_original)

if 'training_data' not in st.session_state:
    st.session_state['training_data'] = []
if 'testing_data' not in st.session_state:
    st.session_state['testing_data'] = []
if 'gold_data' not in st.session_state:
    st.session_state['gold_data'] = []

k = math.floor (len(sentences)*0.8)
training_data = data_original[:k]
gold_data = data_original[k:]
testing_data = [untag(s) for s in gold_data]
st.session_state.training_data = training_data
st.session_state.testing_data = testing_data
st.session_state.gold_data = gold_data

st.image ('https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/efed16a3-1a38-4bc6-83cb-2ba597679b36')
dict_option = {'Random POS-tag': 0, 'Most probable POS-tag': 1, 'Overall POS distribution': 2, 
               'Hapax legomena': 3, 'Regex tagger': 4, 'Other': -1}
valid_or_not = st.selectbox('**Do you use validation?**',
    ('True', 'False'))
if valid_or_not == 'True':
   option = st.selectbox('**Choose option to handle unknown words**',
    ('Random POS-tag', 'Most probable POS-tag', 'Overall POS distribution', 'Hapax legomena', 'Regex tagger'))
else: option = 'Other'

pos_temp = st.multiselect(
    '**What are contextual templates do you choose?**',
    [1, 2, 3, 4, 5, 6])
dict_pos_temp = {1: POS([-1]), 2: POS([1]), 3: POS([-2, -1]), 4: POS([1,2]), 5: POS([-3, -2, -1]), 6: POS([1, 2, 3])}
word_temp = st.multiselect(
    '**What are lexicalized templates do you choose?**',
    [1, 2, 3, 4, 5, 6, 7])
dict_word_temp = {1: WORD([-1]), 2: WORD([1]), 3: WORD([-2, -1]), 4: WORD([1,2]), 5: WORD([-2]), 6: WORD([2]), 7: WORD([0])}
templates = []
if (len(pos_temp) > 0 or len(word_temp)>0 ):
   for i in range(1, len(pos_temp)+1):
      templates.append(dict_pos_temp[i])
   for j in range(1, len(word_temp)+1):
      templates.append(dict_word_temp[j])

if 'tbl' not in st.session_state:
   st.session_state['tbl'] = []
if 'data' not in st.session_state:
   st.session_state['data'] = []
if 'rules' not in st.session_state:
   st.session_state['rules'] = []
if 'pred_with_tbl' not in st.session_state:
   st.session_state['pred_with_tbl'] = []
if 'without_tbl' not in st.session_state:
   st.session_state['without_tbl'] = []

if 'with_tbl' not in st.session_state:
   st.session_state['with_tbl'] = []

if st.button('Start'):
    tbl = Transformation_Based_Learning()
    data = tbl.train_valid_split(training_data, valid=valid_or_not)
    rules = tbl.fit (training_data, templates, data, dict_option[option])
    pred_without_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
    without_tbl = pd.DataFrame({'Accuracy': list (pred_without_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T

    pred_with_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
    with_tbl = pd.DataFrame({'Accuracy': list (pred_with_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T
    st.session_state['tbl'] = tbl
    st.session_state['data'] = data
    st.session_state['rules'] = rules
    st.session_state['pred_with_tbl'] = pred_with_tbl
    st.session_state['without_tbl'] = without_tbl
    st.session_state['with_tbl'] = with_tbl
    st.subheader('Rules')
    for i in rules:
            best_instance, template = i
            word_pos, f_tag, t_tag = best_instance
            best_rule = "Change tag FROM :: '" + f_tag + "' TO :: '" + t_tag + "'" + " IF " + template[1] + ": " + "'" + word_pos + "'" + str(template[0])
            st.write (best_rule)
    

    


