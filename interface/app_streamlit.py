#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
from urllib.parse import urljoin

st.set_page_config(page_title="Xref - Code Authorship Attribution", page_icon="⚙️", initial_sidebar_state="expanded")

##########################################
##  Load and Prep Data                  ##
##########################################

base_url = 'https://xref-cloud-app-tf3z57rlzq-ew.a.run.app/'

##########################################
##  Style and Formatting                ##
##########################################

# CSS for tables

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>   """

center_heading_text = """
    <style>
        .col_heading   {text-align: center !important}
    </style>          """

center_row_text = """
    <style>
        td  {text-align: center !important}
    </style>      """

# Inject CSS with Markdown

st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.markdown(center_heading_text, unsafe_allow_html=True)
st.markdown(center_row_text, unsafe_allow_html=True)

# More Table Styling

def color_surplusvalue(val):
    if str(val) == '0':
        color = 'azure'
    elif str(val)[0] == '-':
        color = 'lightpink'
    else:
        color = 'lightgreen'
    return 'background-color: %s' % color

heading_properties = [('font-size', '16px'),('text-align', 'center'),
                      ('color', 'black'),  ('font-weight', 'bold'),
                      ('background', 'mediumturquoise'),('border', '1.2px solid')]

cell_properties = [('font-size', '16px'),('text-align', 'center')]

dfstyle = [{"selector": "th", "props": heading_properties},
               {"selector": "td", "props": cell_properties}]

# Expander Styling

st.markdown(
    """
<style>
.streamlit-expanderHeader {
 #   font-weight: bold;
    background: aliceblue;
    font-size: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)



##########################################
##  Title, Tabs, and Sidebar            ##
##########################################
st.write("")
st.write(
    '<img width=200 src="https://cdn.dribbble.com/users/330915/screenshots/3587000/media/cf9c914d04e017ab821bab2ee0bb87cb.gif" style="margin-left:0px">',
    unsafe_allow_html=True,
)
st.title("XRef_")
st.markdown('''##### <span style="color:gray">Code Authorship Attribution</span>
            ''', unsafe_allow_html=True)

tab_nn, tab_cnn, tab_svc, tab_faq, tab_sample = st.tabs(["NN Model", "CNN Model", "SVC Model", "FAQ", "Sample Code"])

col1, col2, col3 = st.sidebar.columns([1,2,1])
with col1:
    st.write("")
with col2:
    st.write('<img width=150 src="https://i.insider.com/5696abd2e6183efa428b645b?width=750&format=png">', unsafe_allow_html=True)
    # st.image('figures\\x.png',  use_column_width=True)
with col3:
    st.write("")

st.sidebar.markdown(" # About")
st.sidebar.markdown("This aplication is intended to showcase how machine/deep learning models can be used to determine the true author of a sample of code based solely on its style (stylometry).")
st.sidebar.markdown("Our models were trained with source code extracted from a famous coding competition (Google Code Jam).")
st.sidebar.markdown(" # Created by")
col1, col2 = st.sidebar.columns([1,1])
with col1:
    st.write("João Baccarin")
    st.write("Tim Certa")
    st.write("Thomas Grumberg")
    st.write("Rafael Incao")
    st.write("")
with col2:
    st.write("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/in/joaopaulobaccarin/)
            """)
    st.write("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/in/timcerta/)
            """)
    st.write("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/in/thomasgrumberg/)
            """)
    st.write("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue)](https://www.linkedin.com/in/rafaelincao/)
            """)
    st.write("")

st.sidebar.info("ℹ️ Read more about our project and check the code at [Github](https://github.com/jbaccarin/xref).")

##########################################
## NN Tab                               ##
##########################################

with tab_nn:
    st.markdown('''**Model description:**
    Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. Their name and structure are inspired by the human brain, mimicking the way that biological neurons signal to one another.

    For this model we used:
        - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
        - 2 dense layers 1000 nodes deep, with ReLu activation functions
        - 2 subsequent drop-out layers with 20% keep rate
        - Final dense layer with Softmax activation
        - Adam Optimizer''')
    st.markdown('''Accuracy on test data: **89%**''')
    st.markdown("---")
    
    user_input = st.text_area("Enter the code below.", key="nn1")

    # Find out the author for the given piece of code
    if st.button('Find out code author', key="nn2"):
        with st.spinner('Wait for it...'):
            
            st.write('Please wait, the author is being identified...')
        params = dict(code=[user_input])
        
        
        print('Author is being identified')
        st.write(" ")
        st.write(" ")

        # Calling API
        # response = ""
        # res = ""
        # proba = ""
        
        path_nn = "predict_with_nn"
        predict_url_nn = urljoin(base_url, path_nn)
        
        response = requests.get(predict_url_nn, params).json()
        
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


        st.write(" ")
        st.write(" ")

        # Calculate and show tfidf terms
        #st.subheader("Most important terms identified by tf-idf vectorizer:")
        top_terms_dict = response["top_terms"]
        topterms = pd.DataFrame.from_dict(top_terms_dict, orient='index')
        topterms.rename({0:'Tfidf'},inplace=True,axis=1)
        topterms = topterms.reset_index()
        topterms.rename({'index':'Term'},inplace=True,axis=1)
        #st.table(data = topterms)
        fig=px.bar(topterms,x='Term',y='Tfidf', orientation='v', title='<span style="font-size: 30px;">Most important terms identified by tf-idf vectorizer</span>')
        fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig)

##########################################
## CNN Tab                              ##
##########################################

with tab_cnn:
    st.markdown('''**Model description:**
    Convolutional neural network (CNN), a class of artificial neural networks that has become dominant in various computer vision tasks, is attracting interest across a variety of domains, including radiology. CNN is designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers, pooling layers, and fully connected layers.

    For this model we used:
        - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
        - 3 1D convolutional layers with kernels of size [3,4,5] and ReLu activation
        - 3 subsequent drop-out layers with 50% keep rate
        - 1 Flatten layer
        - Final dense layer with Softmax activation
        - Adam Optimizer''')

    st.markdown('''Accuracy on test data: **92%**''')
    st.markdown("---")
    
    user_input = st.text_area("Enter the code below.", key="cnn1")    

    # Find out the author for the given piece of code
    if st.button('Find out code author', key="cnn2"):
        with st.spinner('Wait for it...'):
            st.write('Please wait, the author is being identified...')
        params = dict(code=[user_input])

        print(params)
        print('Author is being identified')
        st.write(" ")
        st.write(" ")
        
        #Calling API
        # response = None
        # res = None
        # proba = None
        
        path_cnn = "predict_with_cnn"
        predict_url_cnn = urljoin(base_url, path_cnn)
        
        response = requests.get(predict_url_cnn, params).json()
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


        st.write(" ")
        st.write(" ")

        # Calculate and show tfidf terms
        #st.subheader("Most important terms identified by tf-idf vectorizer:")
        top_terms_dict = response["top_terms"]
        topterms = pd.DataFrame.from_dict(top_terms_dict, orient='index')
        topterms.rename({0:'Tfidf'},inplace=True,axis=1)
        topterms = topterms.reset_index()
        topterms.rename({'index':'Term'},inplace=True,axis=1)
        #st.table(data = topterms)
        fig=px.bar(topterms,x='Term',y='Tfidf', orientation='v', title='<span style="font-size: 30px;">Most important terms identified by tf-idf vectorizer</span>')
        fig.update_layout(height=400, width = 770, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        st.plotly_chart(fig)

##########################################
## SVC Tab                              ##
##########################################

with tab_svc:
    st.markdown('''**Model description:**
    A support vector machine (SVM) is a supervised machine learning model that uses classification algorithms for two-group classification problems. After giving an SVM model sets of labeled training data for each category, they’re able to categorize new text.''')
    st.markdown('''Compared to newer algorithms like neural networks, they have two main advantages: higher speed and better performance with a limited number of samples (in the thousands). This makes the algorithm very suitable for text classification problems, where it’s common to have access to a dataset of at most a couple of thousands of tagged samples.


    For this model we used:
    - A TF-IDF vectorizer (with top 5000 features, including non-word characters)
    - LinearSVC - uses a linear kernel (scales better with larger samples)
    - Bayesian Optimization for hyper-parameter tuning:
        - Best n-gram sizes (2,3)
    - Suggested by Ellen, based on some of her previous projects''')
    st.markdown('''Accuracy on test data: **85%**''')
    st.markdown("---")

    user_input = st.text_area("Enter the code below.", key="svc1")

    # Find out the author for the given piece of code
    if st.button('Find out code author', key="svc2"):
        with st.spinner('Wait for it...'):
            st.write('Please wait, the author is being identified...')
        params = dict(code=[user_input])

        print(params)
        print('Author is being identified')
        st.write(" ")
        st.write(" ")

        #Calling API
        # response = None
        # res = None
        # proba = None
        
        path_svc = "predict_with_svc"
        predict_url_svc = urljoin(base_url, path_svc)
        
        response = requests.get(predict_url_svc, params).json()
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


        st.write(" ")
        st.write(" ")





##########################################
## FAQ Tab                              ##
##########################################

with tab_faq:
    st.markdown(" ### Frequently Asked Questions 🔎 ")

    ##########
    expand_faq1 = st.expander("Did you use any academic research as references for your project?")
    with expand_faq1:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''Yes, we did! Quite a few, actually. We've listed them below.
                 If you feel like doing some research on this topic too, try to search for terms like: "code autorship attribution" and "stylometry". ''', unsafe_allow_html=True)
        st.write('''- Abuhamad, Mohammed, et al. "Code authorship identification using convolutional neural networks." Future Generation Computer Systems 95 (2019): 104-115.''', unsafe_allow_html=True)
        st.write('''- Caliskan-Islam, Aylin, et al. "De-anonymizing programmers via code stylometry." 24th USENIX security symposium (USENIX Security 15). 2015.''', unsafe_allow_html=True)
        st.write('''- Frankel, Sophia F., and Krishnendu Ghosh. "Machine Learning Approaches for Authorship Attribution using Source Code Stylometry." 2021 IEEE International Conference on Big Data (Big Data). IEEE, 2021.''', unsafe_allow_html=True)
        st.write('''- Simko, Lucy, Luke Zettlemoyer, and Tadayoshi Kohno. "Recognizing and Imitating Programmer Style: Adversaries in Program Authorship Attribution." Proc. Priv. Enhancing Technol. 2018.1 (2018): 127-144.''', unsafe_allow_html=True)
        st.write('''- Kalgutkar, Vaibhavi, et al. "Code authorship attribution: Methods and challenges." ACM Computing Surveys (CSUR) 52.1 (2019): 1-36.''', unsafe_allow_html=True)
        st.write('''- Abuhamad, Mohammed, et al. "Large-scale and Robust Code Authorship Identification with Deep Feature Learning." ACM Transactions on Privacy and Security (TOPS) 24.4 (2021): 1-35.''', unsafe_allow_html=True)
        st.write('''- Abuhamad, Mohammed, et al. "Multi-χ: Identifying Multiple Authors from Source Code Files." Proc. Priv. Enhancing Technol. 2020.3 (2020): 25-41.''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)

    ##########
    expand_faq2 = st.expander("Why use TF-IDF instead of other vectorizers (ex: BOW)? ")
    with expand_faq2:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''TF-IDF reflects how important a word is to a document in a collection or corpus. Thus, our approach does not require any prior information on specific programming languages. As a result, our approach is more resilient to language specifics and to the number of code files available per author.''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)


    ##########
    expand_faq3 = st.expander("Which machine learning models did we use?")
    with expand_faq3:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''We have tried many different models. The most recommended ones were RNN and SVC. RNN is notorious for its ability to deal with sequences of text. However, in this challenge the meaning of code was not so important as the style and frequency of terms that were available in each document.''', unsafe_allow_html=True)
                 
        st.write('''After testing so many models we realised the most effective ones were: a simple Neural Network, a Convolutional Neural Network and SVC (Support Vector Machine Classifier).
                 It's actually quite interesting to note that such models are often applied to image processing challenges. In a sense, this very similar to what we are attempting here given that what matters most to us is the stylometry of its content, rather than its intrinsic meaning.''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)


    ##########
    expand_faq4= st.expander("How was the predictive model trained?", expanded=False)
    with expand_faq4:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''The models were trained with data collected from Google Code Jam submissions from 2008 to 2020 and is avaliable at: [GCJ Dataset by Jur1cek](https://github.com/Jur1cek/gcj-dataset)''', unsafe_allow_html=True)
        st.write('''In order to properly feed our models with de-noised data and avoid overfitting we made sure to use only the top-3 most common languages in the dataset (~85% of all code samples).
                 We also selected code submissions ranging from 500 to 12000 characters. Such limits correspond to the interquartile range of code length from competition finalists. The purpose is to only use code that is meaningful and actual attempts to solve a challenge in the competition.''', unsafe_allow_html=True)
        st.write('''The dataset also featured some two-stage challenges, where the second stage involved answering the same problem in a limited amount of time and for a larger input. However, such problems are not identifiable in the dataset. In order to avoid such duplicates we calculated the Levenshtein distance between subsequent samples of code of each competitor. Eventually, code samples which were more than 90% similar to a previous one were discarded.''', unsafe_allow_html=True)
        st.write('''Another filter we applied was to only consider code authors with AT LEAST 9 samples fo code. This limitation was chosen according to tests run by Abuhamad, M, et al. (2019) when attempting to train similar models (CNN and RNN).''', unsafe_allow_html=True)
        st.write('''At last, the resulting code samples were vectorized using the TF-IDF Vectorizer available at the sci-kit learn package. Abuhamad, M, et al. (2019) also discovered that using the 2500 top-most frequent terms produced models with best accuracy. We tried to replicate their tests and eventually arrived at the same value.''', unsafe_allow_html=True)
        st.write('''It is important to notice that no Embedding Layer was used as input to our Neural Networks as TF-IDF produces a vector of constant size for each code sample.''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)
    
        
    ##########
    expand_faq5 = st.expander('''So, how good is this model really?''')
    with expand_faq5:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''As mentioned previously, we used **accuracy** as our training optimization metric. By applying the holdout method to our dataset on a 80/20 ratio we identified the following rates of successful autorship attribution for the test set:''', unsafe_allow_html=True)
        st.write('''- SVC: 85%''', unsafe_allow_html=True)
        st.write('''- CNN: 92%''', unsafe_allow_html=True)
        st.write('''- NN: 89%''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)



    ##########
    expand_faq7 = st.expander("What happens if the model isn't sure about a prediction? ")
    with expand_faq7:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''For this Streamlit app, we also return the predicted probability for the multi-class classification task.
                 The idea is to showcase other possible authors for a given piece of test code. The model is then able to provide the probability of other authors, which is most commonly referred to Proba.''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)


    ##########
    expand_faq8 = st.expander("Where can I see the code for the model?")
    with expand_faq8:
        st.write(''' ''', unsafe_allow_html=True)
        st.write('''Glad you asked! 🤓 It's all on our [Github](https://github.com/jbaccarin/xref)''', unsafe_allow_html=True)
        st.write(''' ''', unsafe_allow_html=True)
        
        
        
##########################################
## Sample Code Tab                      ##
##########################################

with tab_sample:
    st.markdown(" ### Some code samples for testing 🛠️ ")
    st.markdown("""Click on the top right hand corner of each code sample to copy, then paste it into one of our models and test them.""")
    st.markdown("""*Note: Yes, we did make sure these samples are only available on our test dataset.""") 
                
        
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        code1 = """import java.io.*;
 import java.util.*;
 
 public class Solution implements Runnable {
 
 	private BufferedReader in;
 	private StringTokenizer st;
 	private PrintWriter out;
 	private int[] p;
 	
 	private void solve() throws IOException {
 		int tests = nextInt();
 		for (int test = 1; test <= tests; ++test) {
 			int k = nextInt();
 			char[] s = next().toCharArray();
 			int[][] diff = new int[k][k];
 			for (int i = 0; i < s.length; i += k) {
 				for (int u = 0; u < k; ++u) {
 					for (int v = 0; v < k; ++v) {
 						diff[u][v] += s[i + u] != s[i + v] ? 1 : 0;
 					}
 				}
 			}
 			int[][] diff2 = new int[k][k];
 			for (int i = 0; i < s.length - k; i += k) {
 				for (int u = 0; u < k; ++u) {
 					for (int v = 0; v < k; ++v) {
 						diff2[u][v] += s[i + u] != s[i + k + v] ? 1 : 0;
 					}
 				}
 			}
 			int[][][] d = new int[1 << k][k][k];
 			for (int[][] ar : d) {
 				for (int[] br : ar) {
 					Arrays.fill(br, s.length + 1);
 				}
 			}
 			for (int u = 0; u < k; ++u) {
 				d[1 << u][u][u] = 0;
 			}
 			for (int m = 0; m < 1 << k; ++m) {
 				for (int u = 0; u < k; ++u) {
 					for (int v = 0; v < k; ++v) {
 						if (d[m][u][v] == s.length + 1) {
 							continue;
 						}
 						for (int w = 0; w < k; ++w) {
 							if ((m & (1 << w)) != 0) {
 								continue;
 							}
 							d[m ^ (1 << w)][u][w] = Math.min(d[m ^ (1 << w)][u][w], d[m][u][v] + diff[v][w]);
 						}
 					}
 				}
 			}
 			int ans = s.length + 1;
 			for (int u = 0; u < k; ++u) {
 				for (int v = 0; v < k; ++v) {
 					ans = Math.min(ans, d[(1 << k) - 1][u][v] + diff2[v][u] + 1);
 				}
 			}
 			out.println("Case #" + test + ": " + ans);
 		}
 	}
 
 	private int rec(int i, int k, char[] s) {
 		if (i == k) {
 			char[] ss = new char[s.length];
 			for (int a = 0; a < s.length; a += k) {
 				for (int b = 0; b < k; ++b) {
 					ss[a + b] = s[a + p[b]];
 				}
 			}
 			int res = 1;
 			for (int a = 0; a < ss.length - 1; ++a) {
 				if (ss[a] != ss[a + 1]) {
 					res++;
 				}
 			}
 			return res;
 		}
 		int res = Integer.MAX_VALUE;
 		loop: for (p[i] = 0; p[i] < k; ++p[i]) {
 			for (int j = 0; j < i; ++j) {
 				if (p[i] == p[j]) {
 					continue loop;
 				}
 			}
 			res = Math.min(res, rec(i + 1, k, s));
 		}
 		return res;
 	}
 
 	public void run() {
 		try {
 			in = new BufferedReader(new InputStreamReader(System.in));
 			st = new StringTokenizer("");
 			out = new PrintWriter(System.out);
 			
 			solve();
 			
 			out.close();
 		} catch (Exception ex) {
 			ex.printStackTrace();
 			System.exit(-1);
 		}
 	}
 	
 	String next() throws IOException {
 		while (!st.hasMoreTokens()) {
 			String line = in.readLine();
 			if (line == null) {
 				return null;
 			}
 			st = new StringTokenizer(line);
 		}
 		return st.nextToken();
 	}
 	
 	int nextInt() throws IOException {
 		return Integer.parseInt(next());
 	}
 	
 	double nextDouble() throws IOException {
 		return Double.parseDouble(next());
 	}
 	
 	long nextLong() throws IOException {
 		return Long.parseLong(next());
 	}
 	
 	public static void main(String[] args) {
 		new Thread(new Solution()).start();
 	}
 
 	private void ass(boolean b) {
 		if (!b) {
 			throw new RuntimeException("Assertion failed");
 		}
 	}
 	
 	static final boolean Ъ = true;
 }"""
        st.markdown("##### **Code Author:** winger")
        st.code(code1)
    with col2:
        code2 = """#!/usr/local/bin/python
 import sys, string
 
 
 #solve case function
 def solve_case(nm, lawn, case_number):
     zipped_lawn = zip(*lawn)
     for n in range(0, nm[0]):
         max_lawn_n = max(lawn[n])
         for m in range(0, nm[1]):
             max_lawn_m = max(zipped_lawn[m])
             if lawn[n][m] < max_lawn_n and lawn[n][m] < max_lawn_m:
                 print "Case #%d: NO" % case_number
                 return
 
     print "Case #%d: YES" % case_number
 
 
 #main
 def main():
     r = sys.stdin
     if len(sys.argv) > 1:
         r = open(sys.argv[1], 'r')
 
     total_cases = r.readline()
     for case_number in range(1, int(total_cases) + 1):
         nm = map(int, r.readline().strip().split(' '))
         lawn = []
         for n in range(0, nm[0]):
             lawn.append(map(int, r.readline().strip().split(' ')))
         solve_case(nm, lawn, case_number)
 
 # invoke main
 if __name__ == "__main__":
     main()"""
        st.markdown("##### **Code Author:** addie9000")
        st.code(code2)
    with col3:
        code3 = """#include <iostream>
 #include <sstream>
 #include <fstream>
 #include <string>
 #include <vector>
 #include <list>
 #include <set>
 #include <map>
 #include <queue>
 #include <stack>
 #include <deque>
 #include <complex>
 #include <algorithm>
 #include <cstdio>
 #include <cmath>
 #include <cstring>
 
 #define REP(i,x) for(int i=0 ; i<(int)(x) ; i++)
 #define ALL(x) (x).begin(),(x).end()
 #define LL long long
 
 using namespace std;
 
 vector<int> ll_to_vll(LL N){
     vector<int> numbers;
     while(N){
         numbers.push_back(N%10);
         N /= 10;
     }
     return numbers;
 }
 
 LL vll_to_ll(vector<int> numbers){
     LL res = 0;
     while(!numbers.empty()){
         res *= 10LL;
         res += numbers.back();
         numbers.pop_back();
     }
     return res;
 }
 
 
 struct Node{
     LL n, c, p;
     Node(){}
     Node(LL _n,LL _c, LL _p):n(_n),c(_c),p(_p){}
     bool operator<(const Node &n)const{
         return c > n.c;
     }
 };
 LL solve(LL N){
     priority_queue<Node> que;
 
     que.push(Node(0,0, -2));
 
     vector<LL> used(10000000, -1);
 
     LL res = 1LL<<50;
     while(!que.empty()){
         Node node = que.top();que.pop();
         if(used[node.n]!=-1)continue;
         used[node.n] = node.p;
         if(node.n==N){
             res = node.c;
             LL now = node.n;
             while(now!=-2){
                 //if(abs(now-used[now])!=1)cout << now << " " << used[now] << endl;
                 now = used[now];
             }
             break;
         }
 
 
 
         LL next = node.n + 1;
         if(used[next]==-1)que.push(Node(next,node.c+1, node.n));
 
         vector<int> numbers = ll_to_vll(node.n);
         vector<int> reversed_numbers = numbers;
         reverse(ALL(reversed_numbers));
         LL reversed = vll_to_ll(reversed_numbers);
         if(used[reversed]==-1)que.push(Node(reversed,node.c+1, node.n));
     }
     return res;
 }
 
 int main(){
     int T;
     cin >> T;
     REP(test_case,T){
         LL N;
         cin >> N;
         cout << "Case #" << test_case + 1 << ": " << solve(N) << endl;
     }
     return 0;
 }"""
        st.markdown("##### **Code Author:** nel215")
        st.code(code3)