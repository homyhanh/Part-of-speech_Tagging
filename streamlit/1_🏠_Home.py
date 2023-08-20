import streamlit as st
import pandas as pd

st.set_page_config(
   page_title= "Part of speech Tagging",
   page_icon= ":bookmark_tabs:"
)
st.title('Part of Speech Tagging')
st.write("Check out [GitHub](https://github.com/homyhanh/Part-of-speech_Tagging)")
st.sidebar.success('Select a page above to')
st.header('Introduction')
pos_tagger = ''' Part-of-speech tagging (POS tagging) is the task of tagging a word in a text with its part of speech. A part of speech is a category of words with similar grammatical properties. \n
Example: '''
st.write(pos_tagger)
df = pd.DataFrame({'Word': ['John', 'lives', 'in', 'London', '.'], 'Tag': ['NNP', 'VBZ', 'IN', 'NNP', '.']} ).T
st.write(df)
st.write('Transformation Based Learning is algorithm used to tag POS in an English sentence. \n')

st.header('Datasets')
st.write('There are 45 sentences that are manually labeled based on the Penn Treebank tagset.')
st.image('https://github.com/homyhanh/Part-of-speech_Tagging/assets/79818022/82110bba-204a-4f85-8e23-07732dc5fe4e')
