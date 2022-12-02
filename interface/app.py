#1.Import

import streamlit as st

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px




#2. Page Elements
st.markdown("""# Code Recognizer Φ
""")

#3. User Input
user_input = st.text_area("Enter the code below.", )


#Calling API
url = 'https://taxi-instance-tf3z57rlzq-ew.a.run.app'
predict_url = "https://taxi-instance-tf3z57rlzq-ew.a.run.app/predict_author_with_NN?code=https%3A%2F%2Ftaxi-instance-tf3z57rlzq-ew.a.run.app"


# Find out the author for the given piece of code
if st.button('Find out code author'):
    with st.spinner('Wait for it...'):
        st.write('Please wait, the author is being identified...')
    params = dict(
            code=[user_input]
        )

    print('Author is being identified')
    st.write(" ")
    st.write(" ")

    response = requests.get(predict_url, params).json()
    res = response["author"]
    proba = response["probabilities"]
    st.success('Done!')
    # Return result
    st.subheader(f"The predicted author is {res}!")


    st.write(" ")
    st.write(" ")

    # Calculate and display probabilities
    #st.subheader(f"Top 5 probabilities among all programmers:")
    #st.bar_chart(data = data_)
    data_ = pd.DataFrame.from_dict(proba, orient='index')
    data_.rename({0:'Probability'},inplace=True,axis=1)
    data_['Probability'] = (data_['Probability']*100).astype(int)
    data_ = data_.reset_index()
    data_.rename({"index":'Author'},inplace=True,axis=1)
    fig=px.bar(data_,x='Author',y='Probability', orientation='v', title='<span style="font-size: 30px;">Top 5 probabilities among all programmers</span>')
    fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig)

    top_terms_dict =   {
    "1000000000000000000ll": 0.5719383955001831,
    "ct": 0.5303371548652649,
    "fixed": 0.4039384126663208,
    "bool": 0.20360404253005981,
    "false": 0.19606833159923553,
    "true": 0.19548967480659485,
    "cin": 0.1816319078207016,
    "long": 0.15945880115032196,
    "vector": 0.15294823050498962,
    "main": 0.10164063423871994,
    "for": 0.09548912197351456,
    "int": 0.09533978998661041,
    "pg": 0,
    "pi": 0,
    "pick": 0
  }

    st.write(" ")
    st.write(" ")

    # Calculate and show tfidf terms
    #st.subheader("Most important terms identified by tf-idf vectorizer:")
    topterms = pd.DataFrame.from_dict(top_terms_dict, orient='index')
    topterms.rename({0:'Tfidf'},inplace=True,axis=1)
    topterms = topterms.reset_index()
    topterms.rename({'index':'Term'},inplace=True,axis=1)
    #st.table(data = topterms)
    fig=px.bar(topterms,x='Term',y='Tfidf', orientation='v', title='<span style="font-size: 30px;">Most important terms identified by tf-idf vectorizer</span>')
    fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig)

    #st.bar_chart(data = topterms, horizontal = True)
