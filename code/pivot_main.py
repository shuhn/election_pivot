from analysis import *

def main():
    menu1 = """
    Has there been a General Election pivot? Let's explore.

    ******************************************************

    Would you like to look at Clinton (C) or Trump (T)?
    \t>> """

    menu2 = """
    Would you like to look at the primaries (P) or the general election (GE)?
    \t>> """

    menu3 = """
    MENU - Please enter the number of your choice.

    Debate by Debate Analysis:
    1) Summaries
    2) Frequency
    3) Sentiment

    Aggregate Analysis:
    4) Summaries
    5) Sentiment
    6) K-Means
    7) Hierarchical Clustering

    \t>> """

    choice1 = raw_input(menu1)
    if choice1 == 'C':
        choice1 = 'CLINTON:'
    elif choice1 == 'T':
        choice1 = 'TRUMP:'
    else:
        print 'Invalid input, options are \'C\' or \'T\'. Exiting now.'
        exit()
    choice2 = raw_input(menu2)
    if choice2 == 'P' and choice1 == 'CLINTON:':
        choice2 = 'D'
    elif choice2 == 'P' and choice1 == 'TRUMP:':
        choice2 = 'R'
    elif choice2 == 'GE':
        choice2 = 'G'
    else:
        print 'Invalid input, options are \'P\' or \'GE\'. Exiting now.'
        exit()
    choice3 = raw_input(menu3)
    try:
        choice3 = int(choice3)
    except ValueError:
        print 'Invalid input, you may enter a number 1-7. Exiting now.'
    if choice3 < 1 or choice3 > 7:
        print 'Invalid input, you may enter a number 1-7. Exiting now.'

    if choice3 == 1 or choice3 == 4:
        word_count = int(raw_input("""
        How many words would you like?
        >>  """))

    if choice3 == 5:
        detailed_y_n = raw_input("""
        Would you like a detailed output (Y or N)?
        >>  """)

    if choice3 == 6:
        print ("""
        How many clusters would you like?
        I recommend the following combinations:
        Trump - 4
        Clinton - 8
        """)
        num_clusters = int(raw_input("\t>> "))
    if choice3 == 6 or choice3 == 7:
        print ("""
        How many n-grams would you like to consider?
        I recommend the following combinations:

        Trump -
        Primaries: (1,3)
        General Election: (1,2)

        Clinton -
        Primaries: (1,3)
        General Election: (1,1)
        """)
        start = int(raw_input("\tStart:\n\t>> "))
        end = int(raw_input("\tEnd:\n\t>> "))
        ngram_range = (start, end)

    print "\nOkay, starting now...\n"

    candidate = choice1
    keys = find_key(candidate)
    chosen_debates = file_grab(choice2)
    relevant_debates = []
    for name, file_name in chosen_debates.iteritems():
        m_t, k_p = setup_buckets(file_name)
        mods, parts, key_players, output_counter = parse_names(k_p)
        key_players_textdict, output_counter = parse_text(m_t, key_players, output_counter)

        if choice3 == 1:
            #SUMMARIES
            print name
            summarize_speech(key_players_textdict, keys, word_count)

        if choice3 == 2:
            # FREQUENCY
            print name
            frequency(key_players_textdict, output_counter)
            print "-----"

        if choice3 == 3:
            # SENTIMENT ANALYSIS
            print name
            sentiment_analysis(key_players_textdict, keys)
            nltk_sentiment(key_players_textdict, keys)

        relevant_debates.append(key_players_textdict)

    if choice3 < 4:
        print "Complete. Exiting now."
        exit()

    #SETUP AGG DICTS
    df_dict_nolem = setup_agg_df(relevant_debates, keys)
    df_dict = setup_agg_df_lem(relevant_debates, keys)

    print "Set-up complete..."

    if choice3 == 4:
        #AGG SUMMARIZER
        print "Starting aggregate summarizer...\n"
        print word_count, "word summary for", keys[1]
        agg_summarizer(df_dict_nolem, keys, word_count)

    if choice3 == 5:
        #AGG SENTIMENT ANALYSIS
        print "Starting aggregate sentiment analysis...\n"
        print "Polarity and Sentiment for:", candidate
        agg_sentiment(df_dict_nolem, keys)
        agg_nltk_sentiment(df_dict_nolem, keys, detailed_y_n)

    if choice3 == 6:
        #KMEANS
        print "Starting aggregate k-means analysis...\n"
        k_means(df_dict, df_dict_nolem, keys, num_clusters, ngram_range)

    if choice3 == 7:
        #HIER_ARCH
        print "Starting hierarchical clustering...\n"
        hierarch_clust(df_dict, keys, ngram_range)

    print """\nComplete. Exiting now.\n
    ******************************************************"""

if __name__ == '__main__':
    main()
