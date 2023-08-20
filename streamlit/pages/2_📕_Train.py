import streamlit as st
import nltk
from nltk.tag import untag, RegexpTagger
import numpy as np
import pandas as pd
import random

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
# Define template
def POS (a):
  return a, 'POS'
def WORD  (a):
  return a, 'WORD'

def unigram(training_data):
    data_words = np.array (training_data[0])[:, 0]
    for i in range (1, len(training_data)):
        data_words = np.concatenate((data_words, np.array (training_data[i])[:, 0]))

    data_pos = np.array (training_data[0])[:, 1]
    for i in range (1, len(training_data)):
        data_pos = np.concatenate((data_pos, np.array (training_data[i])[:, 1]))

    df = pd.DataFrame({'Word': data_words, 'POS': data_pos })
    data = df.drop_duplicates(ignore_index = True).sort_values(by=['Word', 'POS'], ignore_index = True)
    data['Count'] = df.groupby(['Word', 'POS']).size().values
    return data

def create_corpus(training_data):
    corpus_tuple = []
    for i in range (len(training_data)):
        corpus_tuple.append (np.array((training_data[i]))[:, 0].tolist())
    correct_tag = []
    for i in range (len(training_data)):
        correct_tag.append (np.array((training_data[i]))[:, 1].tolist())
    return corpus_tuple, correct_tag

def create_gold_tag (gold_data):
    gold_tag = []
    for i in range (len(gold_data)):
        gold_tag.append (np.array((gold_data[i]))[:, 1].tolist())
    return gold_tag


def handle_unknown_word(word, corpus_tuple, all_tags, correct_tag, choice):
        if (choice == 0):
            return random.choice(all_tags)
        elif (choice == 1):
            tags = np.concatenate(correct_tag).tolist()
            return max(set (tags),key = tags.count)
        elif (choice ==2):
            return random.choice(np.concatenate(correct_tag))
        elif (choice == 3):
            tags = []
            unique_values, counts = np.unique(np.concatenate(corpus_tuple), return_counts=True)
            for i in range (len (unique_values)):
                if counts[i] == 1:
                  id = np.concatenate (corpus_tuple).tolist().index(unique_values[i])
                  tags.append(np.concatenate(correct_tag)[id])
            return random.choice(tags)
        else:
            return baseline.tag([word])[0][1]

class Transformation_Based_Learning():
  def __init__(self):
    pass

  def initialize_with_most_likely_tag(self, corpus_tuple, data, all_tags, correct_tag, choice):
    current_tag = []
    for sentence in corpus_tuple:
      current_word = []
      for word in sentence:
        if word in data['Word'].values:
          current_word.append (data[data['Word'] == word].sort_values('Count', ascending=False).iloc[0][1])
        else:
          current_word.append (handle_unknown_word(word, corpus_tuple, all_tags, correct_tag, choice))
      current_tag.append (current_word)
    return current_tag

  def get_best_instance (self, all_tags, all_words, correct_tag, current_tag, corpus_tuple, template):
      best_Z = {}
      best_instance = (0, 0)
      k = template[0]
      for f_tag in all_tags:
        for t_tag in all_tags:
          if f_tag!= t_tag:
            good_transform = {}
            bad_transform = {}
            difference = {}
            for i in range (len (correct_tag)):
              for pos in range (len(correct_tag[i])):
                for j in k:
                  if ((pos + j) >= 0 and (pos + j) < len(correct_tag[i])):
                    if (template[1] == 'POS'):
                      rule = (current_tag[i][pos+j], f_tag, t_tag)
                    else:
                      rule = (corpus_tuple[i][pos+j], f_tag, t_tag)

                    if (correct_tag[i][pos] == t_tag and current_tag[i][pos] == f_tag):
                      if rule in good_transform:
                        good_transform[rule] += 1
                      else:
                        good_transform[rule] = 1

                    elif (correct_tag[i][pos] == f_tag and current_tag[i][pos] == f_tag):
                      if rule in bad_transform:
                        bad_transform[rule] += 1
                      else:
                        bad_transform[rule] = 1

            if (len(good_transform) > 0):
              for key, value in good_transform.items() :
                if key in bad_transform:
                    difference[key] = good_transform[key] - bad_transform[key]
                else:
                    difference[key] = good_transform[key]

              max_difference = max(difference, key = difference.get)
              best_Z[max_difference] = difference[max_difference]
      if (len(best_Z)>0):
        max_ = max (best_Z, key = best_Z.get)
        best_instance  = (max_, best_Z[max_])
      return best_instance

  def get_best_transform(self, all_tags, all_words, correct_tag, current_tag, corpus_tuple, templates):
    best_score = 0
    best_transform = (0, 0, 0)
    for template in templates:
      (best_instance, score) =  self.get_best_instance (all_tags, all_words, correct_tag, current_tag, corpus_tuple, template)
      if (score >= best_score):
        best_score = score
        best_transform = (best_instance, template, best_score)
    return best_transform

  def apply_transform(self, f_tag, t_tag, template, word_pos, current_tag, corpus_tuple):
    k = template[0]
    for i in range (len(current_tag)):
      for pos in range (len(current_tag[i])):
        for j in k:
          if pos + j >=0 and pos + j < len(current_tag[i]):
            if template[1] == 'POS':
              if current_tag[i][pos] == f_tag and current_tag[i][pos+j] == word_pos:
                current_tag[i][pos] = t_tag
            else:
              if current_tag[i][pos] == f_tag and corpus_tuple[i][pos+j] == word_pos:
                current_tag[i][pos] = t_tag
    return current_tag

  def TBL(self, all_tags, all_words, correct_tag, corpus_tuple, templates, data, choice):
    transforms_queue = []
    current_tag = self.initialize_with_most_likely_tag(corpus_tuple, data, all_tags, correct_tag, choice)
    while current_tag != correct_tag:
      best_instance, template, best_score = self.get_best_transform(all_tags, all_words, correct_tag, current_tag, corpus_tuple, templates)
      if best_score > 0:

        word_pos, f_tag, t_tag = best_instance
        transforms_queue.append((best_instance, template))
        current_tag = self.apply_transform(f_tag, t_tag, template, word_pos, current_tag, corpus_tuple)
      else:
        break
    return transforms_queue

  def train_valid_split(self, training_data, valid):
    if not valid:
      data = unigram(training_data)
    else:
      data = unigram(training_data[:30])
    self.all_words = np.unique (data['Word'].values)
    self.all_tags = np.unique (data['POS'].values)
    return data

  def fit (self, training_data, templates, data, choice):
    corpus_tuple, correct_tag = create_corpus(training_data)
    rules = self.TBL(self.all_tags, self.all_words, correct_tag, corpus_tuple, templates, data, choice)
    return rules

  def predict (self, gold_data, data, corpus_tuple, training_data, rules, choice, tbl = False, accuracy = False):
    _ , correct_tag = create_corpus(training_data)
    predict_tag = self.initialize_with_most_likely_tag(corpus_tuple, data, self.all_tags, correct_tag, choice)
    if tbl:
      for rule in rules:
        f_tag, t_tag, template, word_pos = rule[0][1], rule [0][2], rule[1], rule[0][0]
        predict_tag = self.apply_transform(f_tag, t_tag, template, word_pos, predict_tag, corpus_tuple)

    if accuracy:
        gold_tag = create_gold_tag (gold_data)
        known_data = unknown_data = predict_true_known_data = predict_true_unknown_data = 0
        for i in range (len(gold_tag)):
          for j in range (len(gold_tag[i])):
            if corpus_tuple[i][j] in self.all_words:
              known_data += 1
              if (predict_tag[i][j]) == gold_tag[i][j]:
                predict_true_known_data+=1
            else:
              unknown_data += 1
              if (predict_tag[i][j]) == gold_tag[i][j]:
                predict_true_unknown_data+=1
        score = (predict_true_known_data/known_data, predict_true_unknown_data/ unknown_data, (predict_true_known_data + predict_true_unknown_data)/(known_data+unknown_data))
        predict_tag.append(score)
    return predict_tag
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
   
