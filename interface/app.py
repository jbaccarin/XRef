#1.Import

import streamlit as st

import numpy as np
import pandas as pd
import requests



#2. Page Elements
st.markdown("""# Code Recognizer Φ
""")

#3. User Input
user_input = st.text_area("Enter the code below.")

result = st.button('Find out creator')
user_input

#Calling API
url = ('https://xref-tf3z57rlzq-ew.a.run.app')

json={'text':user_input}
response = requests.post(url, json=json)

response.text
 
 
 
#from PIL import Image

#opening the image

#image = Image.open('C:\Users\rafae\Downloads\code detective.png')

#displaying the image on streamlit app

#st.image(image, caption='Enter any caption here')


