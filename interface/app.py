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

#Calling API
url = ('https://xref-tf3z57rlzq-ew.a.run.app')

#json={'text':user_input}
#response = requests.post(url, json=json)
#response.text


# Find out the author for the given piece of code
if st.button('Find out code author'):
    predict_url = 'https://xref-tf3z57rlzq-ew.a.run.app/predict_author'

    #params = dict(
    #        key=[pickup_datetime],
    #        pickup_datetime=[pickup_datetime],
    #        pickup_longitude=[pickup_longitude],
    #        pickup_latitude=[pickup_latitude],
    #        dropoff_longitude=[dropoff_longitude],
    #        dropoff_latitude=[dropoff_latitude],
    #        passenger_count=[passenger_count]
    #    )

    print('Author is being identified')
    st.write('Author is being identified')

    response = requests.get(predict_url).json()
    res = response["author"]
    #proba = response["probabilities"]

    # Return result
    # TODO: exchange the answer after we have the link to theh api
    st.write(f"{res} is the author.")
    #st.write(f"There is a one {proba[res]} percent probability that {res} is the author .")

    #st.write(f"Top 5 probabilities under all programmers:")
    #st.write(proba)

#from PIL import Image

#opening the image

#image = Image.open('C:\Users\rafae\Downloads\code detective.png')

#displaying the image on streamlit app

#st.image(image, caption='Enter any caption here')
