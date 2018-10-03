"""This module is to be used as part of the GA Capstone - automated language error detection project. 
It contains all the custom functions to be used with the three iPython notebooks. It should be imported
into each notebook.
"""

import re
import nltk
from tqdm import tqdm
import string
import requests
import time
import json
import pickle
import requests
from tqdm import tqdm
import multiprocessing as mp
import sys
import threading
from scipy import stats
from multiprocessing.pool import ThreadPool


def find_sentence_indices(df, word_col):
    
    """Takes a pandas dataframe of sentences where each word is a row and 
    each sentence is separated by a row of NaN values. Returns a list of sentence indices."""
    
    mask = df[word_col].isnull()==True
    sentence_indices_lower = [i+1 for i in df[mask].index]
    sentence_indices_upper = [i for i in df[mask].index]
    sentence_indices_lower = [0] + sentence_indices_lower
    sentence_indices_upper = sentence_indices_upper + [df.index[-1]]

    sentence_indices = list(zip(sentence_indices_lower, sentence_indices_upper))
    sentence_indices.pop()
    
    
    return sentence_indices

def find_sentence_length(df, word_col):
    
    """Takes a pandas dataframe of sentences where each word is a row and each 
    sentence is separated by a row of NaN values. Returns a list of sentence lengths"""
    
     # Find the sentence lengths using the indices
    sentence_indices = find_sentence_indices(df, word_col)
    sentence_length = [j-i for i, j in sentence_indices]
    sentence_length = [0] + sentence_length
    
    return sentence_length

def create_save_sentences(df, word_col, filename):
    
    """Takes a pandas dataframe of sentences where each word is a row and each 
    sentence is separated by a row of NaN values. Returns a list of 
    sentences, saving them as a pickle file"""
    
    sentences = []
    for i in find_sentence_indices(df, word_col):
        sentences.append(df.iloc[i[0]:i[1], 0].str.cat(sep=" "))
    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(sentences, f)

    return sentences

def find_df_sentence_indices(df, word_col, filename):
    
    """Takes a pandas dataframe of sentences where each word is a row and each 
    sentence is separated by a row of NaN values. Returns a list of 
    sentence indices, saving them as a pickle file"""
    
    sentence_indices_2 = []
    sentence_length = find_sentence_length(df, word_col)
    sentence_indices_2.append([sentence_length[0], sentence_length[0+1]+sentence_length[0]])
    for i in range(2, len(sentence_length)-1):
        sentence_indices_2.append([sentence_indices_2[-1][1], sentence_indices_2[-1][1]+sentence_length[i]])

    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(sentence_indices_2, f)

    return sentence_indices_2
        
        
def make_ngram_list(sentences, ngram_length):
    
    """Take a list of sentences and the length of desired ngram as arguments. Returns 
    the ngram boundaries for each word in the sentences"""
    
    ngram_list = []
    
    for i in tqdm(range(len(sentences))):
        for j in range(len(sentences[i].split())):
            if len(sentences[i].split()) <= ngram_length:
                ngram_list.extend([" ".join(sentences[i].split())])
            elif j == 0:
                ngram_list.extend([" ".join(sentences[i].split()[0:ngram_length])])
            elif (j == 1) and (ngram_length > 1):
                ngram_list.extend([" ".join(sentences[i].split()[j-1:j+ngram_length])])
            elif (j == 2) and (ngram_length > 2):
                ngram_list.extend([" ".join(sentences[i].split()[j-2:j+ngram_length])])
            elif (j == 3) and (ngram_length > 3):
                ngram_list.extend([" ".join(sentences[i].split()[j-3:j+ngram_length])])
            elif (j == 4) and (ngram_length > 4):
                ngram_list.extend([" ".join(sentences[i].split()[j-(ngram_length-1):j+ngram_length])])
            elif (j == len(sentences[i].split())-1) and (ngram_length > 1):
                ngram_list.extend([" ".join(sentences[i].split()[len(sentences[i].split())-(ngram_length):])])
            elif (j == len(sentences[i].split())-2) and (ngram_length > 2):
                ngram_list.extend([" ".join(sentences[i].split()[j-(ngram_length-1):j+2])])
            elif (j == len(sentences[i].split())-3) and (ngram_length > 3):
                ngram_list.extend([" ".join(sentences[i].split()[j-(ngram_length-1):j+3])])
            elif (j == len(sentences[i].split())-4) and (ngram_length > 4):
                ngram_list.extend([" ".join(sentences[i].split()[j-(ngram_length-1):j+4])])
            elif (j == len(sentences[i].split())-5) and (ngram_length > 5):
                ngram_list.extend([" ".join(sentences[i].split()[j-(ngram_length-1):j+5])])
            else:
                ngram_list.append([" ".join(sentences[i].split()[j-(ngram_length-1):j+ngram_length])])
        
    ngram_list = [i[0] if type(i) == list else i for i in ngram_list]
    
    return ngram_list


def make_pos_ngram_list(sentences, parsed_sentences, ngram_length):
    
    """Take a list of sentences, Spacy parsed sentences and the length 
    of desired ngram as arguments. Returns the left and right ngram context
    for each word"""
    
    ngram_list = []
    
    for i in tqdm(range(len(parsed_sentences))):
        for j in range(len(sentences[i].split())):
            if len(parsed_sentences[i]) <= ngram_length:
                ngram_list.extend([parsed_sentences[i]])
            elif j == 0:
                ngram_list.extend([parsed_sentences[i][0:ngram_length]])
            elif (j == 1) and (ngram_length > 1):
                ngram_list.extend([parsed_sentences[i][j-1:j+ngram_length]])
            elif (j == 2) and (ngram_length > 2):
                ngram_list.extend([parsed_sentences[i][j-2:j+ngram_length]])
            elif (j == 3) and (ngram_length > 3):
                ngram_list.extend([parsed_sentences[i][j-3:j+ngram_length]])
            elif (j == 4) and (ngram_length > 4):
                ngram_list.extend([parsed_sentences[i][j-(ngram_length-1):j+ngram_length]])
            elif (j == len(parsed_sentences[i])-1) and (ngram_length > 1):
                ngram_list.extend([parsed_sentences[i][len(parsed_sentences[i])-ngram_length:]])
            elif (j == len(parsed_sentences[i])-2) and (ngram_length > 2):
                ngram_list.extend([parsed_sentences[i][j-(ngram_length-1):j+2]])
            elif (j == len(parsed_sentences[i])-3) and (ngram_length > 3):
                ngram_list.extend([parsed_sentences[i][j-(ngram_length-1):j+3]])
            elif (j == len(parsed_sentences[i])-4) and (ngram_length > 4):
                ngram_list.extend([parsed_sentences[i][j-(ngram_length-1):j+4]])
            elif (j == len(parsed_sentences[i])-5) and (ngram_length > 5):
                ngram_list.extend([parsed_sentences[i][j-(ngram_length-1):j+5]])
            else:
                ngram_list.append([parsed_sentences[i][j-(ngram_length-1):j+ngram_length]])
        
    ngram_list = [i[0] if type(i) == list else i for i in ngram_list]
    
    return ngram_list


def ngrammize_pos(ngram_boundaries, ngram_length):
    
    """Takes a list of left and right context words for a given word
    and a given ngram length, as well as a specified ngram length as
    an integer. Returns a list of part of speech tag ngrams of 
    the specified ngram length"""
    
    ngram_list = []
    for i in tqdm(range(len(ngram_boundaries))):
        ngram_list.append(list(nltk.ngrams(ngram_boundaries[i], ngram_length)))
    
    return ngram_list


def ngrammize(ngram_boundaries, ngram_length):
      
    """Takes a list of left and right context words for a given word
    and a given ngram length, as well as a specified ngram length as
    an integer. Returns a list of ngrams of 
    the specified ngram length"""
    
    ngram_list = []
    for i in tqdm(range(len(ngram_boundaries))):
        ngram_list.append(list(nltk.ngrams(ngram_boundaries[i].split(), ngram_length)))
    
    return ngram_list


def create_ngram_dicts(sentences, ngram_length, ngram_name):
    
    """Takes a list of sentences, an ngram length and an ngram name. 
    Returns two dictionaries: 
    1) a dict that maps ngrams to phrasfinder-ready queries; and 
    2) a master reference dict that maps individual words to ngrams."""
    
    ngram_boundaries = make_ngram_list(sentences, ngram_length)
    ngram_list = ngrammize(ngram_boundaries, ngram_length)
    ngram_score = {}
    for i in tqdm(ngram_list):
        for j in i:        
            query = " ".join(j)
            query = clean_query(query, ngram_length)
            ngram_score[j] = {"query": query}
    
    ngram_reference = {}
    for i in range(0,ngram_length):
        ngram_reference[ngram_name + "_" + str(i+1)] = [j[i] if len(j) >= i+1 else "NA" for j in ngram_list]

    return ngram_score, ngram_reference



def create_pos_ngram_dicts(sentences, parsed_sentences, ngram_length, ngram_name, train_test):
    
    """Takes a list of original sentences, parsed sentences, an ngram length and an 
    ngram name. Returns a dictionary that maps words to ngram part of speech tags."""
    
    ngram_boundaries = make_pos_ngram_list(sentences, parsed_sentences, ngram_length)
    tagged_ngram_boundaries = [[word.tag_ for word in sent] for sent in ngram_boundaries]
    ngram_list = ngrammize_pos(tagged_ngram_boundaries, ngram_length)
    ngram_reference = {}
    for i in range(0,ngram_length):
        ngram_reference[ngram_name + "_" + str(i+1)] = [j[i] if len(j) >= i+1 else "NA" for j in ngram_list]
    
    with open(f'tagged_{ngram_name}_{train_test}.pickle', 'wb') as f:
        pickle.dump(ngram_reference, f)
    
    with open(f'tagged_{ngram_name}_boundaries_{train_test}.pickle', 'wb') as f:
        pickle.dump(tagged_ngram_boundaries, f)
    
    return ngram_reference, tagged_ngram_boundaries


def parse(sentences, nlp):
    
    """take a list of sentences and parse / tokenize them using Spacy library.
    Return a list of parsed sentences"""
    parsed_sentences = []
    for i, parsed in tqdm(enumerate(nlp.pipe(sentences, batch_size=50, n_threads=4))):
        assert parsed.is_parsed
        parsed_sentences.append(parsed)
    
    return parsed_sentences


def create_query_list(ngram_score):
    
    """Takes a dictionary of ngrams and converts them
    to a list of queries."""
    
    queries = []
    for i in ngram_score:
        queries.append(ngram_score[i]["query"])
    
    return queries



def extract_scores(query_dict, score_type):
    
    """Takes a dict mapping ngrams to API queries and scores and
    a desired score metric (as defined by Phrasefinder.io, e.g. 'mc' 
    for match count). Returns a dict with scores inserted."""
    
    ngram_list = []
    ngram_dict = {}
    for i in query_dict:
        for j in query_dict[i]:   
            if i == j:
                try:
                    for k in range(len(query_dict[i][i]["phrases"])):
                        try:
                            ngram_list.append(query_dict[i][i]["phrases"][k][score_type])
                        except:
                            ngram_list.append(0)

                    ngram_dict[i] = sum(ngram_list)
                    ngram_list = []
                
                except:
                    ngram_dict[i] = query_dict[i]
                
            else:
                try:
                    for k in range(len(query_dict[i]["phrases"])):
                        try:
                            ngram_list.append(query_dict[i]["phrases"][k][score_type])
                        except:
                            ngram_list.append(0)

                    ngram_dict[i] = sum(ngram_list)
                    ngram_list = []
                except:
                    ngram_dict[i] = query_dict[i]

    return ngram_dict
    

def load_extract_query_data(ngrams, pickle_file_stem, pickle_final_file, batch_size, score_type):
    
    ngram_score = {}
    for i in range(1000, len(ngrams), batch_size):
        with open(f'./ngrams/{pickle_file_stem}_{i}.pickle', 'rb') as f:
            s = pickle.load(f)
        ngram_score.update(extract_scores(s, score_type))

    with open(f'./ngrams/{pickle_final_file}.pickle', 'rb') as f:
        s = pickle.load(f)
    ngram_score.update(extract_scores(s, score_type))
    
    return ngram_score


def create_score_dict(ngram_dict, score_dict):
    
    """Take a dictionary containing the match score for each query and match this to the 
    original ngram dictionary, returning the updated dictionary and a list capturing any errors"""
    
    ngram_errors = []
    for key, val in tqdm(ngram_dict.items()):
        try:
            ngram_dict[key]["score"] = score_dict["".join(list(val.values()))]
        except:
            ngram_dict[key]["score"] = "NA"
            ngram_errors.append(ngram_dict[key]["query"])

    return ngram_dict, ngram_errors


def assign_word_scores(ngram_dict, score_dict, gram_name, test_train):
    
    """return and pickle a new dictionary containing word scores in lists (values)
    mapped to each ngram length (keys)"""
    
    wordscore_dict = {}
    wordscore_list = []
    for i in ngram_dict:
        for j in ngram_dict[i]:
            try:
                wordscore_list.append(score_dict[j]["score"])
            except:
                wordscore_list.append("No score")

        wordscore_dict[i] = wordscore_list
        wordscore_list = []
    
    with open(f'{gram_name}_scores_col_{test_train}.pickle','wb') as f:
        pickle.dump(wordscore_dict, f)
    
    return wordscore_dict

def assign_context_word_scores(ngram_dict, comparison_dict, gram_name, train_test):
    
    """Takes a dict of ngram scores, a comparison dict of scores lower order ngram 
    scores, a name (string) of the desired ngram length and whether it is part of 
    a train or test set. Returns a dict of context scores for later use in creating
    ngram probabilities. Also saves dictionary as a pickle file."""
    
    
    wordscore_dict = {}
    lc_wordscore_list = []
    rc_wordscore_list = []
    for i in ngram_dict:
        for j in ngram_dict[i]:            
            if 'bi' in gram_name:
                try:
                    lc_wordscore_list.append(comparison_dict[j[0:1]]['score'])
                except:
                    lc_wordscore_list.append("No score")
                try:
                    rc_wordscore_list.append(comparison_dict[j[-1:]]['score'])
                except:
                    rc_wordscore_list.append("No score")                                                                                        
                    
            elif 'tri' in gram_name:
                try:
                    lc_wordscore_list.append(comparison_dict[j[0:2]]['score'])
                except:
                    lc_wordscore_list.append("No score")
                try:
                    rc_wordscore_list.append(comparison_dict[j[-2:]]['score'])
                except:
                    rc_wordscore_list.append("No score")
            
            elif 'four' in gram_name:
                try:
                    lc_wordscore_list.append(comparison_dict[j[0:3]]['score'])
                except:
                    lc_wordscore_list.append("No score")
                try:
                    rc_wordscore_list.append(comparison_dict[j[-3:]]['score'])
                except:
                    rc_wordscore_list.append("No score")
            
            elif 'five' in gram_name:
                try:
                    lc_wordscore_list.append(comparison_dict[j[0:4]]['score'])
                except:
                    lc_wordscore_list.append("No score")
                try:
                    rc_wordscore_list.append(comparison_dict[j[-4:]]['score'])
                except:
                    rc_wordscore_list.append("No score")
            
        wordscore_dict[f'{i}_left_context'] = lc_wordscore_list
        wordscore_dict[f'{i}_right_context'] = rc_wordscore_list
        lc_wordscore_list = []
        rc_wordscore_list = []

    with open(f'{gram_name}_{train_test}_context.pickle','wb') as f:
        pickle.dump(wordscore_dict, f)
    
    return wordscore_dict
                                             

def update_score_dict(final_ngram_scores, ngrams_dict, score_type):
    final_ngram_scores_dict = {}
    final_ngram_scores_dict.update(extract_scores(final_ngram_scores, score_type))
    
    for key, val in tqdm(ngrams_dict.items()):
        try:
            ngrams_dict[key]["score"] = final_ngram_scores_dict[val["query"]]
        except:
            pass
    
    return ngrams_dict


def clean_query(ngram, length):
    
    """Take an ngram and return it in a suitable string form for querying the Phrasefinder API"""
    
    cleaned = ngram.replace('"', '\\"').lower()
    cleaned = cleaned.replace('?', '\\?')
    cleaned = cleaned.replace('%', '%25')
    cleaned = cleaned.replace('/', "%2F")
    cleaned = cleaned.replace('&', "%26")
    
    if '-' in cleaned and ' - ' not in cleaned:
        replacement = "".join([i for i in cleaned.split() if "-" in i])
        cleaned = cleaned.replace(replacement, replacement + " / " + '%22' + replacement.replace("-", " - ")
                                 + '%22')
    
    if length == 1:
        
        if ngram != '.' and not re.match(r'[.]+', ngram):           
            cleaned = cleaned.replace('.', '')
        
    
    else:
        cleaned = cleaned[:-2] + cleaned[-2:].replace('.', '')
        
    contractions_b = ["n't", "'re", "'s", "'m"]
    replacements_b = ["n't / not", "'re / are", "is / 's", "'m / am"]
    for i in range(len(contractions_b)):
        cleaned = cleaned[:3].replace(contractions_b[i], replacements_b[i]) + cleaned[3:]

    contractions_d = ["would n't", "could n't", "should n't", "did n't", "wo n't", "ca n't", "they 're",
                     "we 're", "do n't", "does n't", "have n't", "were n't", "had n't", "are n't", 
                     "must n't", "was n't", "they 'll", "she 'll", "I 'll", "it 'll",
                     "sha n't", "we 'll"]
    for i in contractions_d:
        cleaned = cleaned.replace(i, '%22' + i + '%22' + " / " + i[:-4] + i[-3:]+ " ")
    
    if "she 'll" not in cleaned:
        cleaned = cleaned.replace("he 'll", "%22he 'll%22 / he'll ")
        
    
    contractions_e = ["it 's", "I 'm", "I 'd", "they 'd", "she 'd", "she 's", "it 'd", "we 'd"]
    for i in contractions_e:
        cleaned = cleaned.replace(i, '%22' + i + '%22' + " / " + i[:-3] + i[-2:]+ " ")
    

    contractions_h = ["she 'd", "she 's"]
    contractions_j = ["he 'd", "he 's"]
    for i in range(len(contractions_h)):
        if contractions_h[i] not in cleaned:
            cleaned = cleaned.replace(contractions_j[i], '%22' + contractions_j[i]
                                      + '%22' + " / " + contractions_j[i][:-3] + 
                                      contractions_j[i][-2:]+ " ")
    
    contractions_a = [' ca', ' wo']
    for i in contractions_a:
        cleaned = cleaned[:-3] + cleaned[-3:].replace(i, i+"n't / " + '%22' + i + " n't " + '%22')

    contractions_g = ["she's", "she' s", "he's", "he 's", "it's", "it 's", "is / 's"]
    checker = True
    for i in contractions_g:
        if i in cleaned:
            checker = False
    
    if checker == True:
        cleaned = cleaned.replace(" 's", "'s")       
    
    
    return cleaned



def get_score(query):
    url = f'http://phrasefinder.io/search?corpus=eng-us&query={query}'


    result_dict = {}
    try:
        r = requests.get(url)
    except:
        result_dict[query] = "connection failed"
    try:
        result_dict[query] = r.json()
    except:
        result_dict[query] = "error"
    
    return result_dict


def async_get(queries):
    """
    will return a list of dictionaries containing the relevant score information
    from each url passed to it
    """
    qr_dict = []                                    # set up a list to store the scores
    pool = ThreadPool(18)                           # Create a Threadpool with 18 threads maximum
    results = pool.map_async(get_score, queries)    # map the gget_score function asynchronously to all queries
    results.wait()                                  # wait for the results to come in
    qr_dict.append(results.get())                   # add the returned dictionary from get_reviews_2 to the ls_ list
    pool.close()                                    # close the pool once all threads have finished
    pool.join()                                     # close open threads
    
    return qr_dict                                  # return the list of dictionaries


def run_api_queries(ngram_length, train_test, queries, start_value=0, batch_size=1000):
    counter = start_value
    counter_2 = 0
    error_list = []
    error_list_2 = []
    start = time.time()
    for i in range(start_value + batch_size, len(queries), batch_size):    
        n_grams_checked = {}
        n_gram_scores = async_get(queries[counter:i])
        duplicates = 0
        print("responses from API: ", len(n_gram_scores[0]))
        counter = i
        for j in range(len(n_gram_scores[0])):
            if "error" in n_gram_scores[0][j].values(): 
                error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
            elif "connection failed" in n_gram_scores[0][j].values():
                error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
            else:
                try:
                    check_duplicate = n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))]
                    duplicates += 1
                    
                except:
                    n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j]
                    
        print(f'found {duplicates} duplicates') 
        print("ngrams successfully scored: ", len(n_grams_checked))
        counter_2 = 0
        while (error_list != []) and (counter_2 < 50):
            n_gram_scores = async_get(error_list)
            print("errors to requery: ", len(n_gram_scores[0]))
            error_list = []
            for j in range(len(n_gram_scores[0])):
                if "error" in n_gram_scores[0][j].values(): 
                    error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                elif "connection failed" in n_gram_scores[0][j].values():
                    error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                else:
                    n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j][
                        " ".join(list(n_gram_scores[0][j].keys()))]
                    
            counter_2 += 1
            if counter_2 == 50:
                print("some errors remaining - appended to a separate list")
                error_list_2.extend(error_list)

        with open(f'../project-capstone/ngrams/{ngram_length}_{train_test}_{i}.pickle', 'wb') as f:
            pickle.dump(n_grams_checked, f)
        
        filename = f'{ngram_length}_{train_test}_{i}.pickle'
        print(filename, "completed! :)")

        if (len(queries) - counter) < batch_size:
            duplicates = 0
            n_gram_scores = async_get(queries[i:])
            print("final file being evaluated...")
            print("responses from API: ", len(n_gram_scores[0]))
            for j in range(len(n_gram_scores[0])):                
                if "error" in n_gram_scores[0][j].values(): 
                    error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                elif "connection failed" in n_gram_scores[0][j].values():
                    error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                else:
                    try:
                        check_duplicate = n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))]
                        duplicates += 1
                    except:
                        n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j][
                            " ".join(list(n_gram_scores[0][j].keys()))]       
                    
            counter_2 = 0
            while (error_list != []) and (counter_2 < 50):
                n_gram_scores = async_get(error_list)
                print("errors to requery: ", len(n_gram_scores[0]))
                error_list = []
                for j in range(len(n_gram_scores[0])):
                    duplicates = 0
                    if "error" in n_gram_scores[0][j].values(): 
                        error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                    elif "connection failed" in n_gram_scores[0][j].values():
                        error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
                    else:
                        try:
                            check_duplicate = n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))]
                            duplicates += 1
                        except:
                            n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j][
                                " ".join(list(n_gram_scores[0][j].keys()))]
                        
                counter_2 += 1
                if counter_2 == 50:
                    print("some errors remaining - appended to a separate list and saved")
                    error_list_2.extend(error_list)
            
                    with open(f'../project-capstone/{ngram_length}_{train_test}_torescore.pickle', 'wb') as f:
                        pickle.dump(error_list_2, f)
                
            with open(f'../project-capstone/ngrams/{ngram_length}_{train_test}_final.pickle', 'wb') as f:
                pickle.dump(n_grams_checked, f)
                
            print("final file completed")

    end = time.time()
    print(end - start)

    
def run_error_queries(filename, queries):
    error_list = []
    start = time.time()
    n_grams_checked = {}
    n_gram_scores = async_get(queries)
    for j in range(len(n_gram_scores[0])):
        if "error" in n_gram_scores[0][j].values(): 
            error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
        elif "connection failed" in n_gram_scores[0][j].values():
            error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
        else:
            n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j]
    
    counter_2 = 0
    while (len(n_grams_checked) < len(queries)) and (counter_2 < 50):
        n_gram_scores = async_get(error_list)
        error_list = []
        for j in range(len(n_gram_scores[0])):
            if "error" in n_gram_scores[0][j].values(): 
                error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
            elif "connection failed" in n_gram_scores[0][j].values():
                error_list.append(" ".join(list(n_gram_scores[0][j].keys())))
            else:
                n_grams_checked[" ".join(list(n_gram_scores[0][j].keys()))] = n_gram_scores[0][j][
                    " ".join(list(n_gram_scores[0][j].keys()))]

            with open(f'../project-capstone/{filename}_gram_errors_torescore.pickle', 'wb') as f:
                pickle.dump(error_list, f)
        
        counter_2 += 1

                
    with open(f'../project-capstone/{filename}_gram_errors_scored.pickle', 'wb') as f:
        pickle.dump(n_grams_checked, f)
    
    return n_grams_checked


        
def tukey_outlier_bounds(col, threshold):
    
    """Takes a pandas dataframe column and an outlier threshold value.
    Returns the upper and lower bounds above / below which outliers are
    defined using the Tukey method.
    """
    
    pct25 = stats.scoreatpercentile(col, 25)
    pct75 = stats.scoreatpercentile(col, 75)
    iqr = (pct75 - pct25)
    upper_bound = pct75 + iqr * threshold
    lower_bound = pct25 - iqr * threshold
    
    return upper_bound, lower_bound

def manual_zscore(df):
    return ((df - df.mean())/df.std(ddof=0))
    