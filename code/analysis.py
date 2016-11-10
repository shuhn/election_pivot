from name_dict import parse_names, file_grab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from summa import summarizer
from textblob import TextBlob
import urllib, re, itertools, string, operator
from collections import Counter, OrderedDict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

def setup_buckets(file_name):
    """
    INPUT: Filename of transcript (string)
    OUTPUT:
    - Main Text (list)
    - List of Potential "Key-Players", i.e. moderator or participant (list)
    """
    with open(file_name, 'r') as f:
        printable = set(string.printable)
        key_players = []
        main_text = []
        for line in f:
            line = filter(lambda x: x in printable, line)
            line = re.sub("\[.*\]$", '', line) #remove [word here] annotations, i.e. [applause]
            words = line.split()
            main_text.append(words)
            for word in words:
                word = word.strip(":")
                if word.isupper() and len(word) > 2:
                    key_players.append(word)

    return main_text, set(key_players)

def parse_text(main_text, key_players_textdict, output_counter):
    """
    INPUT:
    - main_text (list of lists)
    - key_players_textdict (dict): values empty
    - output_counter (dict): values empty
    OUTPUT:
    - key_players_textdict (dict): values filled
    - output_counter (dict): values filled
    """
    merged_text = list(itertools.chain(*main_text))
    flag = merged_text[0]
    for word in merged_text:
        if word in key_players_textdict.iterkeys():
            key_players_textdict[flag].append("STOP")
            flag = word
        else:
            key_players_textdict[flag].append(word)

        if word in output_counter.iterkeys():
            output_counter[word] += 1
    return key_players_textdict, output_counter

def frequency(key_players_textdict, output_counter):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    - output_counter (dict): | candidate_name (string) : # of time spokens (int)
    OUTPUT:
    - prints intervention values
    """
    total_interventions_list = output_counter.values()
    total_interventions_sum = 0
    num_candidates = len(total_interventions_list)
    for intervention in total_interventions_list:
        total_interventions_sum += int(intervention)

    interventiion_list = []
    for candidate, intervention in output_counter.iteritems():
        percent_intervention = float(intervention) / total_interventions_sum
        output_counter[candidate] = percent_intervention
        interventiion_list.append(percent_intervention)

    avg_percent_intervention = sum(interventiion_list) / float(len(interventiion_list))

    sorted_x = sorted(output_counter.items(), key=operator.itemgetter(1), reverse = True)

    for tuple1 in sorted_x:
        print tuple1[0], round(float(tuple1[1]) - avg_percent_intervention, 2)
    print ""

def sentiment_analysis(key_players_textdict, keys):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    OUTPUT:
    - prints sentiment values for a given debate
    """
    list_words = key_players_textdict[keys[1]]
    string_words = (' '.join(list_words))
    textblob = TextBlob(string_words)
    print textblob.sentiment

def nltk_sentiment(key_players_textdict, keys):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    OUTPUT:
    - prints nltk sentiment values for a given debate
    """
    output_data = []

    list_words = key_players_textdict[keys[1]]
    string_words = (' '.join(list_words))

    dict_string = {"text" : string_words}
    data = urllib.urlencode(dict_string)
    u = urllib.urlopen("http://text-processing.com/api/sentiment/", data)
    the_page = u.read()

    items = the_page.split(':')
    imp_items = items[2:4]
    output_list = []
    for string1 in imp_items:
        output_list.append(float(string1.split(',')[0]))

    if output_list[1] < .5:
        #as long as neutral prob isn't > .5
        output_data.append(1-output_list[0])

    try:
        prob_positive = sum(output_data) / len(output_data)
    except ZeroDivisionError:
        prob_positive = 'Neutral'
    print "NLTK Prob of Positive: ", prob_positive
    print "Number of Speeches Included::", len(output_data)
    print ''

def summarize_speech(key_players_textdict, keys, words):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    - keys (tuple): tuple that identifies the candidate to analyze
    OUTPUT:
    - prints 200 word summary for a given debate
    """
    list_words = key_players_textdict[keys[1]]
    STOPWORDS = ['STOP']
    list_words = [w for w in list_words if not w in STOPWORDS]
    string_words = (' '.join(list_words))
    print summarizer.summarize(string_words, words = words)
    print "-----"

def lemmitize(text_string):
    """
    INPUT:
    - text_string (string)
    OUTPUT:
    - list of lemmitized words (list)
    """
    regex = re.compile('<.+?>|[^a-zA-Z]')
    clean_txt = regex.sub(' ', text_string)
    tokens = clean_txt.split()
    lowercased = [t.lower() for t in tokens]
    STOPWORDS = stopwords.words('english')
    #the following words where identified as irrelevant based on context from hierachical clustering (see below)
    new_stop = ['unidentified', 'male', 'applause', 'laughter', 'well', 'know', 'let', 'crosstalk', 'thanks', 'thank', 'you', 'cross', 'talk', 'booing', 'good', 'lot', 'point', 'going', 'say', 'want', 'year', 'inaudible', 'know', 'think', 'later', 'thing', 'york', 'new', 'said', 'people', 'cnn', 'jeb', 'florida', 'able', 'unidentifiable', 'need', 'ted', 'trump', 'ben', 'senator', 'sanders', 'make', 'flint', 'question', 'tell', 'come', 'like', 'wait', 'year', 'million', 'dollar', 'including', 'pretty', 'saying']
    for word in new_stop:
        STOPWORDS.append(word)
    no_stopwords = [w for w in lowercased if not w in STOPWORDS]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmed = [wordnet_lemmatizer.lemmatize(w) for w in no_stopwords]
    return [w for w in lemmed if w]

def setup_agg_df_lem(relevant_debates, keys):
    """
    **Lemitized Version**
    INPUT:
    - relevant_debates (list): list of dictionaries, each dictionary has a key (candidate) : value (list of words) pair
    - keys (tuple): tuple that identifies the candidate to analyze
    OUTPUT:
    - dictionary | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    """
    #aggregate dictionaries
    aggregate_dict = {}
    for d in relevant_debates:
        for k, v in d.iteritems():
            if aggregate_dict.has_key(k):
                aggregate_dict[k] = aggregate_dict[k] + v
            else:
                aggregate_dict[k] = v

    #setup df - column = candidate name, rows = times spoken
    df_dict = {}

    sentence_list = []
    list_words = aggregate_dict[keys[1]]
    string_words = (' '.join(list_words))
    speeches = string_words.split("STOP")

    for speech in speeches:
        sentence = str(speech)
        lem_sentence = lemmitize(sentence)
        sent_string = ' '.join(lem_sentence)
        if len(sent_string.split(" ")) > 7:
            sentence_list.append(sent_string)
    df_dict[keys[0]] = pd.DataFrame(sentence_list, columns = [keys[1]])
    return df_dict

def setup_agg_df(relevant_debates, keys):
    """
    **Non-lemitized Version**
    INPUT:
    - relevant_debates (list): list of dictionaries, each dictionary has a key (candidate) : value (list of words) pair
    - keys (tuple): tuple that identifies the candidate to analyze
    OUTPUT:
    - df_dict (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    """
    #aggregate dictionaries
    aggregate_dict = {}
    for d in relevant_debates:
        for k, v in d.iteritems():
            if aggregate_dict.has_key(k):
                aggregate_dict[k] = aggregate_dict[k] + v
            else:
                aggregate_dict[k] = v

    #setup df - column = candidate name, rows = times spoken
    df_dict = {}
    sentence_list = []
    list_words = aggregate_dict[keys[1]]
    string_words = (' '.join(list_words))
    speeches = string_words.split("STOP")
    for speech in speeches:
        sentence = str(speech)
        if len(sentence.split(" ")) > 7:
            sentence_list.append(sentence)
    df_dict[keys[0]] = pd.DataFrame(sentence_list, columns = [keys[1]])

    return df_dict

def k_means(df_dict, df_dict_nolem, keys, n_clusters_, ngram_range):
    """
    INPUT:
    - df_dict (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = non-lemitized speeches)
    - keys (tuple): tuple that identifies the candidate to analyze
    - n_clusters_ (int): number of clusters to use in k-means
    - ngram_range (tuple): range of n-grams to include
    OUTPUT:
    - print top features for each cluster, and 3 random speeches from each cluster
    """
    df = df_dict[keys[0]]
    df_nolem = df_dict_nolem[keys[0]]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(df[keys[1]])

    features = vectorizer.get_feature_names()
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters_, max_iter = 1000, n_init = 100)
    kmeans.fit(X)

    #LOOK INTO CLASS BALANCES
    value_counts = kmeans.labels_
    value_counter = Counter(value_counts)
    print value_counter

    # Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))

    print "--------"

    #print random sample from each cluster
    assigned_cluster = kmeans.transform(X).argmin(axis=1)
    for i in range(kmeans.n_clusters):
        cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
        sample_speeches = np.random.choice(cluster, 3, replace=False)
        print "cluster %d:" % i
        pd.set_option('display.max_colwidth', 400)
        for speech in sample_speeches:
            print "    %s" % df_nolem.ix[speech]
        print "\n"
    pd.reset_option('display.max_colwidth')

def hierarch_clust(df_dict, keys, ngram_range):
    """
    INPUT:
    - df_dict (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - keys (tuple): tuple that identifies the candidate to analyze
    - ngram_range (tuple): range of n-grams to include
    OUTPUT:
    - reveals dendrogram
    """
    X_df = df_dict[keys[0]]

    # Create vector
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    X = vectorizer.fit_transform(X_df[keys[1]])
    small_features = vectorizer.get_feature_names()

    # Compute Distances
    distxy = squareform(pdist(X.todense(), metric='cosine'))
    distxy = np.nan_to_num(distxy)

    # Pass this matrix into scipy's linkage function to compute
    # hierarchical clusters.
    link = linkage(distxy, method='complete')

    # Use scipy's dendrogram function plot the linkages as
    # a hierachical tree.
    dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9,
                labels=small_features)
    # Fix spacing to better view dendrogram and the labels
    plt.subplots_adjust(top=.99, bottom=0.5, left=0.05, right=0.99)
    plt.show()

def find_key(candidate):
    """
    INPUT:
    - candidate (string): 'CLINTON:' or 'TRUMP:'
    OUTPUT:
    - keys (tuple): tuple that identifies the candidate to analyze
    """
    if candidate == 'CLINTON:':
        return 'C', 'CLINTON:'
    elif candidate == 'TRUMP:':
        return 'T', 'TRUMP:'
    else:
        print "Sorry, we don't recognize that candidate name."
        exit()

def agg_summarizer(df_dict_nolem, keys, words):
    """
    **non-lemitized DF**
    INPUT:
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - keys (tuple): tuple that identifies the candidate to analyze
    - words (int) : word count of requested summary
    OUTPUT:
    - prints 300 word summary for key candidate
    """
    long_string = []
    for array1 in df_dict_nolem[keys[0]].values:
        long_string.append(' '.join(array1))

    long_string = ' '.join(long_string)
    print summarizer.summarize(long_string, words = words)

def agg_sentiment(df_dict_nolem, keys):
    """
    **non-lemitized DF**
    INPUT:
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - keys (tuple): tuple that identifies the candidate to analyze
    OUTPUT:
    - print sentiment and polarity of given candidate (TextBlob)
    """
    long_string = []
    for array1 in df_dict_nolem[keys[0]].values:
        long_string.append(' '.join(array1))
    long_string = ' '.join(long_string)
    textblob = TextBlob(long_string)
    print textblob.sentiment

def agg_nltk_sentiment(df_dict_nolem, keys, explore = 'Y'):
    """
    **non-lemitized DF**
    INPUT:
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - keys (tuple): tuple that identifies the candidate to analyze
    - explore (string): 'Y' or 'N' to display detailed output
    OUTPUT:
    - print agg. NLTK sentiment
    """
    output_data = []
    sent_speech_dict = {}

    for array1 in df_dict_nolem[keys[0]].values:
        long_string = ' '.join(array1)
        dict_string = {"text" : long_string}
        data = urllib.urlencode(dict_string)
        u = urllib.urlopen("http://text-processing.com/api/sentiment/", data)
        the_page = u.read()

        items = the_page.split(':')
        imp_items = items[2:4]
        output_list = []
        for string1 in imp_items:
            output_list.append(float(string1.split(',')[0]))

        if output_list[1] < .5:
            #as long as neutral prob isn't > .5
            output_data.append(1-output_list[0])
            sent_speech_dict[1-output_list[0]] = long_string

    print "NLTK Prob of Positive: ", sum(output_data) / len(output_data)
    print "Number of Speeches Included::", len(output_data)

    # EXPLORATORY WORK
    if explore == 'Y':
        print "\nBottom 5 Responses:\n"
        top_5 = OrderedDict(sorted(sent_speech_dict.items()))
        counter = 0
        for key, value in top_5.iteritems():
            if counter < 6:
                print "NLTK Prob of Positive: ", key
                print value
                print ""
            counter += 1
