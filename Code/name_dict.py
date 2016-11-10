import itertools

def parse_names(key_players):
    """
    INPUT:
    - key_players (list): list of potential key-players

    OUTPUT:
    - output_mods (dict) | shortname (key) : longform name (value)
    - output_parts (dict) | shortname (key) : longform name (value)
    - output_keyplayers (dict) | shortname (key) : []
    - output_counter (dict) | shortname (key) : 0
    """

    moderators = {'Bret Baier (Fox News)' : 'BAIER:',
    'Megyn Kelly (Fox News)' : 'KELLY:',
    'Chris Wallace (Fox News)' : 'WALLACE:',
    'Lester Holt (NBC News)' : 'HOLT:',
    'Andrea Mitchell (NBC News)' : 'MITCHELL:',
    'Wolf Blitzer (CNN)' : 'BLITZER:',
    'Dana Bash (CNN)' : 'BASH:',
    'Errol Louis (NY1)': 'LOUIS:',
    'Jake Tapper (CNN)': 'TAPPER:',
    'Hugh Hewitt (Salem Radio Network)' : 'HEWITT:',
    'Stephen Dinan (Washington Times)' : 'DINAN:',
    'Anderson Cooper (CNN)' : 'COOPER:',
    'Don Lemon (CNN)' : 'LEMON:',
    'Juan Carlos Lopez (CNN en Espanol)' : 'LOPEZ:',
    'David Muir (ABC News)' : 'MUIR:',
    'Martha Raddatz (ABC News)' : 'RADDATZ:',
    'Mary Katherine Ham' : 'HAM:',
    'Josh McElveen (WMUR TV)' : 'MCELVEEN:',
    'Gwen Ifill (PBS)' : 'IFILL:',
    'Judy Woodruff (PBS)' : 'WOODRUFF:',
    'John Dickerson (CBS News)' : 'DICKERSON:',
    'Major Garrett (CBS News)' : 'GARRETT:',
    'Kimberly Strassel (WSJ)' : 'STRASSEL:',
    'Maria Bartiromo (Fox Business Network)' : 'BARTIROMO:',
    'Neil Cavuto (Fox Business Network)' : 'CAVUTO:',
    'Gerard Baker (WSJ)' : 'BAKER:',
    'John Harwood (CNBC)' : 'HARWOOD:',
    'Becky Quick (CNBC)' : 'QUICK:',
    'Carl Quintanilla (CNBC)' : 'QUINTANILLA:',
    'Maria Celeste Arraras (Telemundo)' : 'ARRARAS:',
    'Nancy Cordes (CBS News)' : 'CORDES:',
    'Kevin Cooney (CBS News)' : 'COONEY:',
    'Kathie Obradovich (Des Moines)' : 'OBRADOVICH:',
    'Chuck Todd (MSNBC)' : 'TODD:',
    'Rachel Maddow (MSNBC)' : 'MADDOW:',
    'Jorge Ramos (Univision)' : 'RAMOS:',
    'Maria Elena Salinas (Univision)' : 'SALINAS:',
    'Karen Tumulty (Washington Post)' : 'TUMULTY:',
    'TOWN HALL QUESTION' : 'QUESTION:'}

    participants = {'Former Secretary of State Hillary Clinton' : 'CLINTON:',
    'Former Governor Martin O\'Malley (MD)': 'O\'MALLEY:',
    'Senator Bernie Sanders (VT)': 'SANDERS:',
    'Former Governor Jeb Bush (FL)': 'BUSH:',
    'Ben Carson' : 'CARSON:',
    'Governor Chris Christie (NJ)': 'CHRISTIE:',
    'Senator Ted Cruz (TX)': 'CRUZ:',
    'Former Governor Mike Huckabee (AR)': 'HUCKABEE:',
    'Governor John Kasich (OH)' : 'KASICH:',
    'Senator Rand Paul (KY)': 'PAUL:',
    'Senator Marco Rubio (FL)' : 'RUBIO:',
    'Donald Trump' : 'TRUMP:',
    'Governor Scott Walker (WI)': 'WALKER:',
    'Former Governor Lincoln Chafee (RI)' : 'CHAFEE:',
    'Former Senator Jim Webb (VA)' : 'WEBB:',
    'Carly Fiorina':'FIORINA:'}

    output_mods = {}
    output_parts = {}
    output_keyplayers = {}
    output_counter = {}

    for key, value in moderators.iteritems():
        if value.strip(": ") in key_players:
            output_mods[value] = key
            output_keyplayers[value] = []

    for key, value in participants.iteritems():
        if value.strip(": ") in key_players:
            output_parts[value] = key
            output_keyplayers[value] = []
            output_counter[value] = 0

    return output_mods, output_parts, output_keyplayers, output_counter

def file_grab(name = 'All'):
    if name == 'D':
        return {'1st Democratic Debate: October 13th, 2015' : 'data/D_10_13_15.txt',
        '2nd Democratic Debate: November 14th, 2015' : 'data/D_11_14_15.txt',
        '3rd Democratic Debate: December 19th, 2015' : 'data/D_12_19_15.txt',
        '4th Democratic Debate: January 17th, 2016' : 'data/D_1_17_16.txt',
        '5th Democratic Debate: February 4th, 2016': 'data/D_2_4_16.txt',
        '6th Democratic Debate: February 11th, 2016' : 'data/D_2_11_16.txt',
        '7th Democratic Debate: March 6th, 2016': 'data/D_3_6_16.txt',
        '8th Democratic Debate: March 9th, 2016': 'data/D_3_9_16.txt',
        '9th Democratic Debate: April 4th, 2016': 'data/D_4_14_16.txt'}

    elif name == 'R':
        return {'1st Republican Debate: August 6th, 2015' : 'data/R_8_6_15.txt',
        '2nd Republican Debate: September 16th, 2015' : 'data/R_9_16_15.txt',
        '3rd Republican Debate: October 28th, 2015' : 'data/R_10_28_15.txt',
        '4th Republican Debate: November 10th, 2015' : 'data/R_11_10_15.txt',
        '5th Republican Debate: December 15th, 2015' : 'data/R_12_15_15.txt',
        '6th Republican Debate: January 14th, 2016' : 'data/R_1_14_16.txt',
        '7th Republican Debate: February 6th, 2016' : 'data/R_2_6_16.txt',
        '8th Republican Debate: February 13th, 2016' : 'data/R_2_13_16.txt',
        '9th Republican Debate: February 25th, 2016' : 'data/R_2_25_16.txt',
        '10th Republican Debate: March 3rd, 2016' : 'data/R_3_3_16.txt',
        '11th Republican Debate: March 10th, 2016' : 'data/R_3_10_16.txt'}

    elif name == 'G':
        return {'1st General Election Debate: September 26th, 2016' : 'data/G_9_26_16.txt',
        '2nd General Election Debate: October 9th, 2016' : 'data/G_10_9_16.txt',
        '3rd General Election Debate: October 19th, 2016' : 'data/G_10_19_16.txt'}

    elif name == 'R1':
        return {'1st Republican Debate: August 6th, 2015' : 'data/R_8_6_15.txt'}

    else:
        return {'1st Democratic Debate: October 13th, 2015' : 'data/D_10_13_15.txt',
        '2nd Democratic Debate: December 19th, 2015' : 'data/D_12_19_15.txt',
        '3rd Democratic Debate: January 17th, 2016' : 'data/D_1_17_16.txt',
        '4th Democratic Debate: February 11th, 2016' : 'data/D_2_11_16.txt',
        '5th Democratic Debate: April 4th, 2016': 'data/D_4_14_16.txt',
        '1st Republican Debate: August 6th, 2015' : 'data/R_8_6_15.txt',
        '2nd Republican Debate: November 10th, 2015' : 'data/R_11_10_15.txt',
        '3rd Republican Debate: December 15th, 2015' : 'data/R_12_15_15.txt',
        '4th Republican Debate: January 14th, 2016' : 'data/R_1_14_16.txt',
        '5th Republican Debate: February 6th, 2016' : 'data/R_2_6_16.txt',
        '6th Republican Debate: February 13th, 2016' : 'data/R_2_13_16.txt',
        '7th Republican Debate: March 3rd, 2016' : 'data/R_3_3_16.txt',
        '8th Republican Debate: March 10th, 2016' : 'data/R_3_10_16.txt'}
