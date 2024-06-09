import streamlit as st
import requests

st.title("Chatbot")
user_input = st.text_input("You:")
if st.button("Send"):
    response = requests.get(f"http://localhost:5000/get?msg={user_input}")
    st.write("Bot:", response.text)