# xref - a Code Authenticity Detector

Xref is a python tool to analyze code authorship attribution. It provides a model to determine who wrote a particular piece of code. You can clone it, apply it to the given dataset or even train it with you own dataset.

The project was set up as part of the Le Wagon Bootcamp. It includes
- functions to prepare the Google Code Jam dataset, which can be found here
- different modeling approaches (Naive Bayes, RNN, CNN, LinearSVC)
- functions for model initalization, training, prediction and evaluation

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for xref in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/xref
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "xref"
git remote add origin git@github.com:{group}/xref.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
xref-run
```

# Install

Go to `https://github.com/{group}/xref` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/xref.git
cd xref
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
xref-run
```
