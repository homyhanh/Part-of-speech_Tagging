import streamlit as st
import nltk
from nltk.tag import untag, RegexpTagger
import numpy as np
import pandas as pd
import random
from tbl import *

st.title('Training')
backoff = RegexpTagger([
  (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
  (r'(The|the|A|a|An|an)$', 'DT'),   # articles
  (r'.*able$', 'JJ'),                # adjectives
  (r'.*ness$', 'NN'),                # nouns formed from adjectives
  (r'.*ly$', 'RB'),                  # adverbs
  (r'.*s$', 'NNS'),                  # plural nouns
  (r'.*ing$', 'VBG'),                # gerunds
  (r'.*ed$', 'VBD'),                 # past tense verbs
  (r'.*', 'NN')                      # nouns (default)
])
baseline = backoff
f = open('Datasets.txt', "r")

sentences = f.read().split('\n')[:-1]
data = [[nltk.tag.str2tuple(word) for word in sentence.split()] for sentence in sentences]
random.shuffle(data)

training_data = data[:35]
gold_data = data[35:]
testing_data = [untag(s) for s in gold_data]

option = st.selectbox('**Choose option to handle unknown words**',
    ('Random POS-tag', 'Most probable POS-tag', 'Overall POS distribution', 'Hapax legomena', 'Regex tagger'))
dict_option = {'Random POS-tag': 0, 'Most probable POS-tag': 1, 'Overall POS distribution': 2, 
               'Hapax legomena': 3, 'Regex tagger': 4}

valid_or_not = st.selectbox('**Do you use validation?**',
    ('True', 'False'))

templates = []
number = st.number_input('**Insert a number template you want to use**', step = 1)
for i in range(number):
    temp = st.selectbox(
        'Template ' + str(i+1) + ' (Choose rules in lexical tagging or in contextual tagging)',
        ('Pos', 'Word'))

    coeff = st.text_input('Coefficient ' + str(i+1) + ' (Insert coefficient, _ex: -2, -1 or -1_)')
    if temp == 'Pos':
       templates.append(POS(list(map(int, coeff.split(',')))))
    else:
       templates.append(WORD(list(map(int, coeff.split(',')))))
if st.button('Start'):
    tbl = Transformation_Based_Learning()
    data = tbl.train_valid_split(training_data, valid=valid_or_not)
    rules = tbl.fit (training_data, templates, data, dict_option[option])
    pred_without_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
    without_tbl = pd.DataFrame({'Accuracy': list (pred_without_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T

    pred_with_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
    with_tbl = pd.DataFrame({'Accuracy': list (pred_with_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T
    num = random.randint(0, len(testing_data)-1)
    st.subheader('Rules')
    for i in rules:
            best_instance, template = i
            word_pos, f_tag, t_tag = best_instance
            best_rule = "Change tag FROM :: '" + f_tag + "' TO :: '" + t_tag + "'" + " IF " + template[1] + ": " + "'" + word_pos + "'" + str(template[0])
            st.write (best_rule)
    st.subheader('Results')
    st.write('Without TBL algorithm')
    st.dataframe(without_tbl)

    st.write('With TBL algorithm') 
    st.dataframe(with_tbl)

    st.subheader('Predictions')
    st.write (pd.DataFrame({'Sentence': testing_data[num], 'True tag': np.array((gold_data[num]))[:, 1].tolist(), 'Predicted_tag': pred_with_tbl[num]}).T)


# if "button1" not in st.session_state:
#     st.session_state["button1"] = False

# if "button2" not in st.session_state:
#     st.session_state["button2"] = False

# if "button3" not in st.session_state:
#     st.session_state["button3"] = False

# tbl = Transformation_Based_Learning()
# data = tbl.train_valid_split(training_data, valid=valid_or_not)
# rules = tbl.fit (training_data, templates, data, dict_option[option])
# best_rules = []
# for i in rules:
#     best_instance, template = i
#     word_pos, f_tag, t_tag = best_instance
#     best_rule = "Change tag FROM :: '" + f_tag + "' TO :: '" + t_tag + "'" + " IF " + template[1] + ": " + "'" + word_pos + "'" + str(template[0])
#     best_rules.append (best_rule)

# pred_without_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
# without_tbl = pd.DataFrame({'Accuracy': list (pred_without_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T

# pred_with_tbl = tbl.predict(gold_data, data, testing_data, training_data, rules, dict_option[option], tbl = False, accuracy = True)
# with_tbl = pd.DataFrame({'Accuracy': list (pred_with_tbl[-1])}, index = ['Known_tag', 'Unknown_tag', 'All_tag']).T
# num = random.randint(0, len(testing_data)-1)


# if st.button("Starting training"):
#     st.session_state["button1"] = not st.session_state["button1"]
# if st.session_state["button1"]:
#     st.subheader('Rules')
    
# if st.session_state["button1"]:
#     if st.button("Print result"):
#         st.session_state["button2"] = not st.session_state["button2"]
#     if st.session_state["button2"]:
#         #Without TBL algorithm
#         st.subheader('Results')
#         st.write('Without TBL algorithm')
#         st.dataframe(without_tbl)

#         # With TBL algorithm
#         st.write('With TBL algorithm')
        
#         st.dataframe(with_tbl)
# if st.session_state["button1"] and st.session_state["button2"]:
#         if st.button("Predictions"):
#             st.session_state["button3"] = not st.session_state["button3"]

#         if st.session_state["button3"]:
#             st.subheader('Predictions')
#             st.write('Predict sentence in test data')
#             st.write (pd.DataFrame({'Sentence': testing_data[num], 'True tag': np.array((gold_data[num]))[:, 1].tolist(), 'Predicted_tag': pred_with_tbl[num]}).T)

#             new_sentence = st.text_input('Predict new sentence')
#             new_word = new_sentence.split()
#             pred_tag = tbl.predict(gold_data, data, [new_word], training_data, rules, dict_option[option], tbl = True, accuracy = False)
#             sentence = []
#             for i in range (len (new_word)):
#                 sentence.append (new_word[i] + '/' + pred_tag[0][i])
#             st.write (' '.join(sentence))

#             if st.button ('Reload'):
#                 st.session_state['button1'] = st.session_state['button2'] = st.session_state['button3'] = False

            
            
# st.write(
#     f"""
#     ## Session state:
#     {st.session_state["button1"]=}

#     {st.session_state["button2"]=}

#     {st.session_state["button3"]=}
#     """
# )
   
