from name_dict import parse_names
import re
import itertools
import string
import operator

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

if __name__ == '__main__':
    d_files = ['D_10_13_15.txt', 'D_1_17_16.txt', 'D_4_14_16.txt']
    r_files = ['R_2_6_16.txt', 'R_3_10_16.txt','R_8_6_15.txt']
    g_files = ['G_9_26_16.txt']
    all_files = {'1st Democratic Debate: October 10th, 2015' : 'D_10_13_15.txt', '2nd Democratic Debate: January 17th, 2016' : 'D_1_17_16.txt', '3rd Democratic Debate: April 4th, 2016': 'D_4_14_16.txt', '2nd Republican Debate: February 6th, 2016' : 'R_2_6_16.txt', '3rd Republican Debate: March 10th, 2016' : 'R_3_10_16.txt','1st Republican Debate: August 8th, 2015' : 'R_8_6_15.txt', '1st General Election Debate: September 9th, 2016' : 'G_9_26_16.txt'}

    for name, file_name in all_files.iteritems():
        m_t, k_p = setup_buckets(file_name)
        mods, parts, key_players, output_counter = parse_names(k_p)
        key_players_textdict, output_counter = parse_text(m_t, key_players, output_counter)

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
