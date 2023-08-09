import numpy as np

# Define template
def POS (a):
  return a, 'POS'
def WORD  (a):
  return a, 'WORD'

def index(A):
  return A[-1]

# Create new data with tags labeled with most Frequent Class baseline
def initialize_with_most_likely_tag(corpus_tuple, data):
  current_tag = []
  for sentence in corpus_tuple:
    current_word = []
    for word in sentence:
      current_word.append (data[data['Word'] == word].sort_values('Count', ascending=False).iloc[0][1])
    current_tag.append (current_word)
  return current_tag

# Find the best instance of template POS
def get_best_instance_pos (all_tags, correct_tag, current_tag, f_tag, t_tag, k):
  good_transform = np.zeros(len(all_tags)+2)
  bad_transform = np.zeros(len(all_tags)+2)
  for i in range (len (correct_tag)):
    for pos in range (len(correct_tag[i])):
      for j in k:
        if ((pos + j) >= -1 and (pos + j) <= len(correct_tag[i]) and correct_tag[i][pos] == t_tag and current_tag[i][pos] == f_tag):
          if (pos + j == -1): good_transform[0]+=1
          elif (pos + j == len(correct_tag[i])): good_transform[-1]+=1
          else:
              idx = all_tags.tolist().index(current_tag[i][pos + j])
              good_transform[idx+1]+=1

        elif ((pos + j) >= -1 and (pos + j) <= len(correct_tag[i]) and correct_tag[i][pos] == f_tag and current_tag[i][pos] == f_tag):
          if (pos + j == -1): bad_transform[0]+=1
          elif (pos + j == len(correct_tag[i])): bad_transform[-1]+=1
          else:
              idx = all_tags.tolist().index(current_tag[i][pos + j])
              bad_transform[idx+1]+=1
  return good_transform, bad_transform

# Find the best instance of template WORD
def get_best_instance_word (corpus_tuple, all_words, correct_tag, current_tag, f_tag, t_tag, k):
  good_transform = np.zeros(len(all_words)+2)
  bad_transform = np.zeros(len(all_words)+2)
  for i in range (len (correct_tag)):
    for pos in range (len(correct_tag[i])):
      for j in k:
        if ((pos + j) >= -1 and (pos + j) <= len(correct_tag[i]) and correct_tag[i][pos] == t_tag and current_tag[i][pos] == f_tag):
          if (pos + j == -1): good_transform[0]+=1
          elif (pos + j == len(correct_tag[i])): good_transform[-1]+=1
          else:
              idx = all_words.tolist().index(corpus_tuple[i][pos + j])
              good_transform[idx+1]+=1

        elif ((pos + j) >= -1 and (pos + j) <= len(correct_tag[i]) and correct_tag[i][pos] == f_tag and current_tag[i][pos] == f_tag):
          if (pos + j == -1): bad_transform[0]+=1
          elif (pos + j == len(correct_tag[i])): bad_transform[-1]+=1
          else:
              idx = all_words.tolist().index(corpus_tuple[i][pos + j])
              bad_transform[idx+1]+=1
  return good_transform, bad_transform

# Find the best instance 
def get_best_instance (all_tags, all_words, correct_tag, current_tag, corpus_tuple, template):
    best_Z = []
    k = template[0]
    for f_tag in all_tags:
      for t_tag in all_tags:
        if f_tag!= t_tag:
          if (template[1] == 'POS'):
            good_transform, bad_transform = get_best_instance_pos (all_tags, correct_tag, current_tag, f_tag, t_tag, k)
            difference = good_transform - bad_transform
            difference = np.delete(difference, [0, -1])
            max_score = max (difference)
            best_Z .append ([f_tag, t_tag, all_tags[difference.tolist().index (max_score)], max_score ])
          else:
            good_transform, bad_transform = get_best_instance_word (corpus_tuple, all_words, correct_tag, current_tag, f_tag, t_tag, k)
            difference = good_transform - bad_transform
            difference = np.delete(difference, [0, -1])
            max_score = max (difference)
            best_Z .append ([f_tag, t_tag, all_words[difference.tolist().index (max_score)], max_score ])
    best_instance  = max (best_Z, key = index)
    return best_instance

# Find the best transform in all templates
def get_best_transform(all_tags, all_words, correct_tag, current_tag, corpus_tuple, templates):
  best_score = 0
  for template in templates:
    (f_tag, t_tag, word_pos, score) =  get_best_instance (all_tags, all_words, correct_tag, current_tag, corpus_tuple, template)
    if (score >= best_score):
      best_score = score
      best_transform = (f_tag, t_tag, template, word_pos, score)
  return best_transform

# Apply rule learned to new data
def apply_transform(f_tag, t_tag, template, word_pos, current_tag, corpus_tuple):
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

# Function Transformation based learning algorithm
def TBL(all_tags, all_words, correct_tag, corpus_tuple, templates, data):
  transforms_queue = []
  current_tag = initialize_with_most_likely_tag(corpus_tuple, data)
  while current_tag != correct_tag:
    f_tag, t_tag, template, word_pos, score = get_best_transform(all_tags, all_words, correct_tag, current_tag, corpus_tuple, templates)
    if score > 0: 
      if template[1] == 'POS':
        best_rule = "Change tag FROM :: '" + f_tag + "' TO :: '" + t_tag + "'" + " IF POS: "  + "'" + word_pos + "'" + str(template[0])
      else:
        best_rule = "Change tag FROM :: '" + f_tag + "' TO :: '" + t_tag + "'" + " IF WORD: "  + "'" + word_pos + "'" + str(template[0])
      print (best_rule)
      transforms_queue.append((f_tag, t_tag, template, word_pos))
      current_tag = apply_transform(f_tag, t_tag, template, word_pos, current_tag, corpus_tuple)
    else: break
  return transforms_queue