#1.Import

import streamlit as st

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt



#2. Page Elements
st.markdown("""# Code Recognizer Φ
""")

#3. User Input
user_input = st.text_area("Enter the code below.")

#Calling API
url = 'https://taxi-instance-tf3z57rlzq-ew.a.run.app'
predict_url = "https://taxi-instance-tf3z57rlzq-ew.a.run.app/predict_author_with_NN?code=https%3A%2F%2Ftaxi-instance-tf3z57rlzq-ew.a.run.app"

#json={'text':user_input}
#response = requests.post(url, json=json)
#response.text




# Find out the author for the given piece of code
if st.button('Find out code author'):

    params = dict(
            code=[user_input]
        )

    print('Author is being identified')
    st.write('Author is being identified')

    response = requests.get(predict_url, params).json()
    res = response["author"]
    proba = response["probabilities"]

    # Return result
    # TODO: exchange the answer after we have the link to theh api
    #st.write(f"{res} is the author.")
    st.write(f"There is a probability of {round(proba[res],2)*100}% that {res} is the author.")
    data_ = pd.DataFrame.from_dict(proba, orient='index')
    data_.rename({0:'Probability'},inplace=True,axis=1)
    data_['Probability'] = (data_['Probability']*100).astype(int)
    st.write(f"Top 5 probabilities under all programmers:")
    st.bar_chart(data = data_)

#from PIL import Image

#opening the image

#image = Image.open('C:\Users\rafae\Downloads\code detective.png')

#displaying the image on streamlit app

#st.image(image, caption='Enter any caption here')
