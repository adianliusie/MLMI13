# Official MLMI13 Coursework Code

Handout Link: \
Contact Details: Dr. Kate Knill, Adian Liusie (kmk1001@cam.ac.uk, al826@cam.ac.uk) 

## Overview 
This repository will provide the code framework for the MLMI coursework. The code base has been partialy completed where code sections marked with `#TODO` are left for the student to compelte. The task considered for the coursework is sentiment classifcation on the IMDB dataset, where the methods proposed in the [original paper](https://arxiv.org/pdf/cs/0205070.pdf) will be considered.

#### The code base has a very modular set up
- practical.py is the main python file to run all the experiments (an identical ipython version is also provided)
- Corpora.py is a class that reads and processes the IMDB data.
- Evaluation.py is a class which many other classes inherits for automatic evaluation capabilities.
- There are several other files to be completed to assess different NLP techniques on IMDB sentiment classification
 
#### This repository contains all the data needed for the first section of the handout:
- The IMDB reviews (1000 positive and 1000 negative) are saved in data/reviews 
- The Lexicon sentiment file is saved in data/sent_lexicon

## Requirements 
This Coursework is based in python3 and is compatabile with any operating system. The required packages are:
- scipy
- scikit
- gensim

Which are easily available through pip

## Acknowledgements
This coursework was based on that of the previous [MLSALT13 course](https://bitbucket.org/mlsalt13/practical/src/master/). The python code closely follows the on the previous framework but has been updated to interface better with python3.
