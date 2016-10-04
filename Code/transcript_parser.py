from name_dict import parse_names, file_grab
import re
import itertools
import string
import operator
from textblob import TextBlob
from summa import summarizer

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
                if word.isupper() and len(word) >= 4:
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
    for word in merged_text:
        if word in key_players_textdict.iterkeys():
            flag = word
        else:
            key_players_textdict[flag].append(word)

        if word in output_counter.iterkeys():
            output_counter[word] += 1
    return key_players_textdict, output_counter

def frequency(key_players_textdict, output_counter):
    total_interventions_list = output_counter.values()
    total_interventions_sum = 0
    num_candidates = len(total_interventions_list)
    for intervention in total_interventions_list:
        total_interventions_sum += int(intervention)

    sorted_x = sorted(output_counter.items(), key=operator.itemgetter(1), reverse = True)
    print name
    for tuple1 in sorted_x:
        print tuple1[0], tuple1[1], round(float(tuple1[1]) / (total_interventions_sum), 2)
    print ""

def sentiment_analysis(key_players_textdict):
    for candidate, list_words in key_players_textdict.iteritems():
        string_words = (' '.join(list_words))
        textblob = TextBlob(string_words)
        print candidate, textblob.sentiment

def summarize_speech(key_players_textdict, defaults = ['CLINTON:', 'TRUMP:']):
    # for candidate, list_words in key_players_textdict.iteritems():
    for candidate in defaults:
        list_words = key_players_textdict[candidate]
        string_words = (' '.join(list_words))
        print candidate
        print summarizer.summarize(string_words, words = 200)
        print "-----"

def topic_modeling(relevant_debates, defaults = ['CLINTON:', 'TRUMP:'])):
    #aggregate dictionaries
    aggregate_dict = {}
    for d in relevant_debates:
      for k, v in d.iteritems():
        if aggregate_dict.has_key(k):
          aggregate_dict[k] = aggregate_dict[k] + v
        else:
          aggregate_dict[k] = v

    #setup df - column = candidate name, rows = sentences


    for candidate in defaults:
        sentences = []
        list_words = key_players_textdict[candidate]
        string_words = (' '.join(list_words))
        textblob = TextBlob(string_words)
        for sentence in textblob.sentence:
            sentences.append(sentence)
        df = DataFrame(sentences, columns = candidate)




if __name__ == '__main__':
    all_files = file_grab('G')
    relevant_debates = []
    for name, file_name in all_files.iteritems():
        m_t, k_p = setup_buckets(file_name)
        mods, parts, key_players, output_counter = parse_names(k_p)
        key_players_textdict, output_counter = parse_text(m_t, key_players, output_counter)

        #SUMMARIES
        # print name
        # summarize_speech(key_players_textdict)
        # print "-----"

        #SENTIMENT ANALYSIS
        # print name
        # sentiment_analysis(key_players_textdict)
        # print "------"

        relevant_debates.append(key_players_textdict)

    #TOPIC MODELING
