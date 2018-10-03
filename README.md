# Automated English language error detection using Google Books NGram dataset

**General Assembly Data Science Immersive - final Capstone Project** 

The purpose of the project is to create a machine learning classifier to detect errors in English written by non-native speakers - in particular, by using ngram probabilities derived from the Google Books Ngram corpus. 


## Why choose this topic?

Automated error detection and correction has been studied for decades, yet commercially available editing software still performs quite poorly. Even state of the art apps like Grammarly still cannot detect obvious errors to native English speakers. This reflects not only the complexity of the problem, but also the ambiguity of language - reasonable people can disagree about language and there's not always a single right answer.

Moreover, I wanted to take a new slant on the problem. Most research covering this topic generally has fallen into two categories that I diverge from with this project:

- a focus on <i>error correction</i> as a single task, rather than separating out error detection
- a focus on building / tuning models to correct <i>specific error types</i> rather than errors in general.

The availability of massive web-scale corpora of natural language, accessible on the fly (e.g. search engines, Google Books etc.), presents new opportunities for statistical approaches to error detection that don't require storing huge amounts of data on a machine. Some previous research has used web-scale ngrams in error correction and detection, but nothing I'm aware of that combines ngram probabilities with a classifier.

Ultimately, the aim of the project is not to try and devise a standalone model. Rather, it is to look at a specific new approach to the problem that could potentially be incorporated within existing error detection / correction models or softwares to improve results.


## Project overview 

### CLC FCE Dataset

The project was built on the publically available Cmabridge Learner Corpus (CLC) FCE Dataset made up of 1,244 written exam scripts taken from the Cambridge ESOL First Certificate in English (FCE) exam 2000-2001 (Yannakoudakis et al., 2011). The version used in this project was one adapted specifically for error detection purposes as developed by Rei & Yannakoudakis (2016). 

The dataset was pre-divided into training (1141 scripts)and test sets (97 scripts). The scripts are separated first into tokens (words / punctuation) and then into sentences. Each token has a label classifying it as either correct 'c' or incorrect 'i'.

### Phrasefinder API and the Google Books NGram Viewer

[Phrasefinder API](http://phrasefinder.io/) is an open source search engine for accessing the Google Books NGram Viewer dataset without having to download fragments onto . 

The [Google Books Ngrams](https://books.google.com/ngrams) dataset provides n-gram counts from a huge corpus of books scanned by Google using OCR.


## Getting started

The project consists of three iPython notebooks, a Python module and .

You'll need to save the ErrorDetection.py module and iPython notebooks into the same folder:

**Error-Detection-Part-1-Data-wrangling** - taking the sentences / words from the exam scripts and querying through the API 
**Error-Detection-Part-2-feature-engineering** - merging the data and creating the features
**Error-Detection-Part-3-EDA-and-modelling** - exploring / visualising the data and then training / evaluating classifiers
**ErrorDetection.py** - contains all custom functions required within the notebooks
**fce-public.train.original.tsv** - the original FCE training dataset
**fce-public.test.original.tsv** - original FCE test dataset
**fce_train_final.csv** - FCE training set cleaned and with engineered features
**fce_test_final.csv** - FCE test set cleaned and with engineered features
**fce_pos_train_final.csv** - FCE training set cleaned and with POS tagged ngrams
**fce_pos_test_final.csv** - FCE test set cleaned and with POS tagged ngrams


### Prerequisites

To run the code, there are a number of dependencies. You may need to pip install the following:

```
Spacy
NLTK
Requests
Pandas
NumPy
Matplotlib
Seaborn
Pickle
```

The project was developed using Python 3 on an Anaconda distribution.

## License

This project makes use of the [CLC FCE dataset](https://ilexir.co.uk/datasets/index.html), [Phrasfinder API](http://phrasefinder.io/about) and [Google Books NGram Viewer](https://books.google.com/ngrams/info), all of which have their own licences. 

Please click through to the links to get the full details of these licences.

## References

> Compositional Sequence Labeling Models for Error Detection in Learner Writing
Marek Rei and Helen Yannakoudakis
In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)

> A New Dataset and Method for Automatically Grading ESOL Texts
Helen Yannakoudakis, Ted Briscoe and Ben Medlock
In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL-2011)


