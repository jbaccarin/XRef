# XRef - a Code Authenticity Detector

Xref is an app created to analyze and demonstrate how code authorship attribution works. It provides 3 trained models (1 machine learning [SVC] and 2 deep learning [NN and CNN]) to determine who wrote a particular piece of code. You can clone it, apply it to the given dataset or even train it with your own dataset.

The project was set up as part of the Le Wagon Bootcamp. It includes:
- Exploratory notebooks we've used to find the most accurate models
- Functions to clean and preprocess the Google Code Jam dataset
- Different modeling approaches (Naive Bayes, NN, CNN, LinearSVC)
- Functions for model initalization, training, prediction and evaluation
- Interface and API code to replicate a simple interface in Streamlit

# Install

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
```

Clone the project and install it:

```bash
git clone git@github.com:jbaccarin/xref.git
cd xref
pip install -r requirements.txt
make clean install test                # install and test
```