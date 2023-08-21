import streamlit as st
import pandas as pd

st.set_page_config(
   page_title= "Part of speech Tagging",
   page_icon= ":bookmark_tabs:"
)

st.write("Check out [GitHub](https://github.com/homyhanh/Part-of-speech_Tagging)")
st.sidebar.success('Select a page above to')
st.header('Introduction')
pos_tagger = ''' Part-of-speech (POS) tagging is the process of labeling words in a text with their corresponding parts of speech in natural language processing (NLP). It helps algorithms understand the grammatical structure and meaning of a text. A part of speech is a category of words with similar grammatical properties. \n
'''

st.image ('https://www.shiksha.com/online-courses/articles/wp-content/uploads/sites/11/2022/12/POS-Tagging.jpg.webp')
st.write(pos_tagger)
st.write('''Let’s take an example, \n
Text: “The cat sat on the mat.”

POS tags:''')
df = pd.DataFrame({'Word': ['The', 'cat', 'sat', 'on', 'the', 'mat', '.'], 'Tag': ['DT', 'NN', 'VBD', 'IN', 'DT', 'NN', '.']} )
st.write(df)
st.write('Transformation Based Learning is algorithm used to tag POS in an English sentence. ')
st.header('Transformation-based Tagging')   
st.write('Transformation-based tagging (TBT) is a method of part-of-speech (POS) tagging that uses a series of rules to transform the tags of words in a text. In TBT, a set of rules is defined to transform the tags of words in a text based on the context in which they appear.')
st.write('TBT can be more accurate than rule-based tagging, especially for tasks with complex grammatical structures. However, it can be more computationally intensive and requires a larger set of rules to achieve good performance.')
st.header('Datasets')
f = open('C:/Users/hanhm/Part-of-speech_Tagging/Datasets.txt', "r")
sentences = f.read().split('\n')[:-1]
number = len(sentences) 
sentence = sentences[-1]
st.write(f'There are {number} sentences that are manually labeled based on the Penn Treebank tagset.')

st.image('https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/82110bba-204a-4f85-8e23-07732dc5fe4e')
st.write(f'Example: \n {sentence} ')