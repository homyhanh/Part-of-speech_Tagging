import streamlit as st

without_tbl = st.session_state['without_tbl']  
with_tbl = st.session_state['with_tbl'] 

st.subheader('Results')
st.write('Without TBL algorithm')
st.dataframe(without_tbl)

st.write('With TBL algorithm') 
st.dataframe(with_tbl)