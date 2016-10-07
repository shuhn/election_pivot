from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from name_dict import parse_names, file_grab
import re
import numpy as np
import itertools
import string
import operator
from textblob import TextBlob
from summa import summarizer
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from nltk.stem import WordNetLemmatizer
from collections import Counter

def setup_buckets(file_name):
    """
    INPUT: Filename of transcript (string)
    OUTPUT: 2 Lists
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
    print name
    for tuple1 in sorted_x:
        print tuple1[0], round(float(tuple1[1]) - avg_percent_intervention, 2)
    print ""

def sentiment_analysis(key_players_textdict):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    OUTPUT:
    - prints sentiment values for each candidate
    """
    for candidate, list_words in key_players_textdict.iteritems():
        string_words = (' '.join(list_words))
        textblob = TextBlob(string_words)
        print candidate, textblob.sentiment

def summarize_speech(key_players_textdict, keys):
    """
    INPUT:
    - key_players_textdict (dict) | candidate_name (string) : all words spoken (list)
    - (short_name, long_name)
    OUTPUT:
    - prints 200 word summary for key candidate
    """
    list_words = key_players_textdict[keys[1]]
    string_words = (' '.join(list_words))
    print candidate
    print summarizer.summarize(string_words, words = 200)
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
    new_stop = ['unidentified', 'male', 'applause', 'laughter', 'well', 'know', 'let', 'crosstalk', 'thanks', 'thank', 'you', 'cross', 'talk', 'booing', 'good', 'lot', 'point', 'going', 'say', 'want', 'year', 'inaudible', 'know', 'think', 'later']
    for word in new_stop:
        STOPWORDS.append(word)
    no_stopwords = [w for w in lowercased if not w in STOPWORDS]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmed = [wordnet_lemmatizer.lemmatize(w) for w in no_stopwords]
    return [w for w in lemmed if w]

def setup_agg_df_lem(relevant_debates, keys):
    """
    INPUT:
    - relevant_debates (list): list of dictionaries, each dictionary has a key (candidate) : value (list of words) pair
    - keys (tuple): tuple that identifies the candidate to analyze (TRUMP:) or (CLINTON:)
    OUTPUT:
    - dictionary | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)

    **Lemitized Version**
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
    INPUT:
    - relevant_debates (list): list of dictionaries, each dictionary has a key (candidate) : value (list of words) pair
    - keys (tuple): tuple that identifies the candidate to analyze (TRUMP:) or (CLINTON:)
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

def k_means(df_dict, df_dict_nolem, keys, n_clusters_):
    """
    INPUT:
    - df_dict (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = non-lemitized speeches)
    - keys (tuple): tuple that identifies the candidate to analyze (TRUMP:) or (CLINTON:)
    - n_clusters_ (int): number of clusters to use in k-means
    OUTPUT:
    - print top features for each cluster, and 3 random speeches from each cluster
    """
    df = df_dict[keys[0]]
    df_nolem = df_dict_nolem[keys[0]]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df[keys[1]])
    features = vectorizer.get_feature_names()
    kmeans = KMeans(n_clusters=n_clusters_)
    kmeans.fit(X)

    #LOOK INTO CLASS BALANCES
    value_counts = kmeans.labels_
    value_counter = Counter(value_counts)
    print value_counter

    # print "CENTERS: "
    # for center in kmeans.cluster_centers_:
    #     print center
    #     print center.shape

    # 3. Find the top 10 features for each cluster.
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print "top features for each cluster:"
    for num, centroid in enumerate(top_centroids):
        print "%d: %s" % (num, ", ".join(features[i] for i in centroid))

    print "--------"

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

def hierarch_clust(df_dict, keys):
    """
    INPUT:
    - df_dict (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    - keys (tuple): tuple that identifies the candidate to analyze (TRUMP:) or (CLINTON:)
    OUTPUT:
    - reveals dendrogram
    """

    df_small = df_dict[keys[0]]

    # first vectorize...
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    small_X = vectorizer.fit_transform(df_small[keys[1]])
    small_features = vectorizer.get_feature_names()

    # now get distances
    distxy = squareform(pdist(small_X.todense(), metric='cosine'))
    distxy = np.nan_to_num(distxy)

    # 4. Pass this matrix into scipy's linkage function to compute our
    # hierarchical clusters.
    link = linkage(distxy, method='complete')

    # 5. Using scipy's dendrogram function plot the linkages as
    # a hierachical tree.
    dendro = dendrogram(link, color_threshold=1.5, leaf_font_size=9,
                labels=small_features)
    # fix spacing to better view dendrogram and the labels
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

def agg_summarizer(df_dict_nolem):
    """
    INPUT:
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    OUTPUT:
    - prints 300 word summary for key candidate
    **non-lemitized DF**
    """
    long_string = []
    for array1 in df_dict_nolem[keys[0]].values:
        long_string.append(' '.join(array1))

    long_string = ' '.join(long_string)
    print summarizer.summarize(long_string, words = 300)

def agg_sentiment(df_dict_nolem):
    """
    INPUT:
    - df_dict_nolem (dictionary) | key ('C' or 'T' based on keys) : value (dataframe, column = full candidate name, rows = speeches)
    OUTPUT:
    - print sentiment and polarity of given candidate
    **non-lemitized DF**
    """
    long_string = []
    for array1 in df_dict_nolem[keys[0]].values:
        long_string.append(' '.join(array1))
    long_string = ' '.join(long_string)
    textblob = TextBlob(long_string)
    print textblob.sentiment

def nltk_sentiment(df_dict_nolem):
    import urllib

    output_data = []
    for array1 in df_dict_nolem[keys[0]].values:
        long_string = ' '.join(array1)
        dict_string = {"text" : long_string}
        data = urllib.urlencode(dict_string)
        u = urllib.urlopen("http://text-processing.com/api/sentiment/", data)
        the_page = u.read()
        output_data.append(the_page)

    pos_prob_list = []

    for point in output_data:
        negative_prob = float(point[25:41])
        neu_prob = float(point[55:72])
        if neu_prob < .5:
            pos_prob_list.append(1-negative_prob)

    print sum(pos_prob_list) / len(pos_prob_list)
    print "total length:", len(pos_prob_list)



if __name__ == '__main__':
    candidate = 'TRUMP:'
    keys = find_key(candidate)
    all_files = file_grab('G')
    relevant_debates = []
    for name, file_name in all_files.iteritems():
        m_t, k_p = setup_buckets(file_name)
        mods, parts, key_players, output_counter = parse_names(k_p)
        key_players_textdict, output_counter = parse_text(m_t, key_players, output_counter)

        #SUMMARIES
        # print name
        # summarize_speech(key_players_textdict, keys)
        # print "-----"

        #FREQUENCY
        # frequency(key_players_textdict, output_counter)
        # print "-----"

        #SENTIMENT ANALYSIS
        # print name
        # sentiment_analysis(key_players_textdict)
        # print "------"

        relevant_debates.append(key_players_textdict)

    #TOPIC MODELING
    df_dict_nolem = setup_agg_df(relevant_debates, keys)
    df_dict = setup_agg_df_lem(relevant_debates, keys)

    #AGG SUMMARIZER
    # agg_summarizer(df_dict_nolem)

    #AGG SENTIMENT ANALYSIS
    # print "Polarity and Sentiment for:", candidate
    # agg_sentiment(df_dict_nolem)
    nltk_sentiment(df_dict_nolem)

    #KMEANS
    # k_means(df_dict, df_dict_nolem, keys, 4)

    #HIER_ARCH
    # hierarch_clust(df_dict, keys)
