import streamlit as st
import random 
import numpy as np
import pandas as pd


training_data = st.session_state.training_data
testing_data = st.session_state.testing_data
gold_data = st.session_state.gold_data
tbl = st.session_state['tbl'] 
data = st.session_state['data'] 
rules = st.session_state['rules']
pred_with_tbl = st.session_state['pred_with_tbl'] 
dict_option = {'Random POS-tag': 0, 'Most probable POS-tag': 1, 'Overall POS distribution': 2, 
               'Hapax legomena': 3, 'Regex tagger': 4}
dict_sentence = {}
for sentence in range (len (testing_data)):
  dict_sentence[' '.join (testing_data[sentence])] = sentence

genre = st.radio(
    "Choose your option",
    ('Testing data', 'New sentence'))

if genre == 'Testing data':
    option = st.selectbox('**Choose option to handle unknown words**',
    ('Random POS-tag', 'Most probable POS-tag', 'Overall POS distribution', 'Hapax legomena', 'Regex tagger'))
    option_sentence = st.selectbox('**Choose sentence**', dict_sentence)

    st.write('You choose sentence ' + str(dict_sentence[option_sentence]+1))
    st.write (pd.DataFrame({'Sentence': testing_data[dict_sentence[option_sentence]], 'True tag': np.array((gold_data[dict_sentence[option_sentence]]))[:, 1].tolist(), 'Predicted_tag': pred_with_tbl[dict_sentence[option_sentence]]}).T)

else:
    new_sentence = st.text_input('Predict new sentence')
    option = st.selectbox('**Choose option to handle unknown words**',
    ('Random POS-tag', 'Most probable POS-tag', 'Overall POS distribution', 'Hapax legomena', 'Regex tagger'))
    new_word = new_sentence.split()
    pred_tag = tbl.predict(gold_data, data, [new_word], training_data, rules, dict_option[option], tbl = True, accuracy = False)
    sentence = []
    for i in range (len (new_word)):
        sentence.append (new_word[i] + '/' + pred_tag[0][i])
    st.write (' '.join(sentence))