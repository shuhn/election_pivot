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

    'Charles Gibson (ABC-News)' : 'GIBSON:',
    'Scott Spradling (WMUR-TV)' : 'SPRADLING:',
    'Brian Williams (NBC-News)' : 'WILLIAMS:',
    'Tim Russert (NBC-News)' : 'RUSSERT:',
    'Natalie Morales (NBC-News)' : 'MORALES:',
    'Suzanne Malveaux (CNN)' : 'MALVEAUX:',
    'Joe Johns (CNN)' : 'JOHNS:',
    'Doyle McManus (Los Angeles Times)' : 'MCMANUS:',
    'Jeanne Cummings (Politico.com)' : 'CUMMINGS:',
    'Campbell Brown (CNN)' : 'BROWN:',
    'Jorge Ramos (Univision)' : 'RAMOS:',
    'John King (CNN)' : 'KING:',
    'George Stephanopoulos (ABC News)' : 'STEPHANOPOULOS:',
    'John Roberts (CNN)' : 'ROBERTS:',
    'Unknown' : 'MODERATOR:',
    'Carolyn Washburn (Des Moines Register)' : 'WASHBURN:',
    'LEHRER' : 'LEHRER:',
    'Tom Brokaw (NBC News)' : 'BROKAW:',
    'TOWN HALL QUESTION' : 'QUESTION:',
    'Bob Schieffer (CBS News)' : 'SCHIEFFER:',
    'Brit Hume (Fox News)' : 'HUME:',
    'Carl Cameron (Fox News)' : 'CAMERON:',
    'Wendell Goler (Fox News)' : 'GOLER:',
    'Paul Tash (St. Petersburg Times)' : 'TASH:',
    'Janet Hook (Los Angeles Times)' : 'HOOK:',
    'Jim Vandehei (The Politico)' : 'VANDEHEI:',
    'David Yepsen (Des Moines Register)' : 'YEPSEN:',
    'UNKNOWN' : 'ANNOUNCER:',
    'Chris Matthews, MSNBC' : 'MATTHEWS:',
    'Gerald Seib, Wall Street Journal' : 'SEIB:',
    'Governor Charlie Crist (FL)' : 'CRIST:',
    'Jim Greer, Chairman, Republican Party of Florida' : 'GREER:',
    'Gloria Borger, CNN' : 'BORGER:'}

    participants = {'Senator Hillary Clinton (NY)' : 'CLINTON:',
    'Former Senator John Edwards (NC)': 'EDWARDS:',
    'Senator Barack Obama (IL)': 'OBAMA:',
    'Governor Bill Richardson (NM)': 'RICHARDSON:',
    'Fmr. Sen. Mike Gravel (AK)' : 'GRAVEL:',
    'Sen. Christopher Dodd (CT)' : 'DODD:',
    'Rep. Dennis Kucinich (OH)' : 'KUCINICH:',
    'Sen. Joseph Biden, Jr. (DE)' : 'BIDEN:',
    'Former Mayor Rudy Giuliani (New York City)': 'GIULIANI:',
    'Former Governor Mike Huckabee (AR)': 'HUCKABEE:',
    'Senator John McCain (AZ)' : 'MCCAIN:',
    'Representative Ron Paul (TX)' : 'PAUL:',
    'Former Governor Mitt Romney (MA)' : 'ROMNEY:',
    'Former Senator Fred Thompson' : 'THOMPSON:',
    'Senator Sam Brownback' : 'BROWNBACK:',
    'Virginia Gov. James Gilmore' : 'GILMORE:',
    'Rep. Duncan Hunter (CA)' : 'HUNTER:',
    'Rep. Tom Tancredo (CO)' : 'TANCREDO:',
    'Former Ambassador Alan Keyes' : 'KEYES:'}

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
        return {'1st Democratic Debate' : 'data/D_8_19_7.txt',
        '2nd Democratic Debate' : 'data/D_9_26_7.txt',
        '3rd Democratic Debate' : 'data/D_11_15_7.txt',
        '4th Democratic Debate' : 'data/D_12_4_7.txt',
        '5th Democratic Debate' : 'data/D_12_13_7.txt',
        '6th Democratic Debate' : 'data/D_1_5_8.txt',
        '7th Democratic Debate' : 'data/D_1_15_8.txt',
        '8th Democratic Debate' : 'data/D_1_21_8.txt',
        '9th Democratic Debate' : 'data/D_1_31_8.txt',
        '10th Democratic Debate' : 'data/D_2_21_8.txt',
        '11th Democratic Debate' : 'data/D_2_26_8.txt',
        '12th Democratic Debate' : 'data/D_4_16_8.txt'}

    elif name == 'R':
        return {'1st Republican Debate' : 'data/R_6_5_7.txt',
        '2nd Republican Debate' : 'data/R_8_5_7.txt',
        '3rd Republican Debate' : 'data/R_9_5_7.txt',
        '4th Republican Debate' : 'data/R_10_9_7.txt',
        '5th Republican Debate' : 'data/R_10_21_7.txt',
        '6th Republican Debate' : 'data/R_12_9_7.txt',
        '7th Republican Debate' : 'data/R_12_12_7.txt',
        '8th Republican Debate' : 'data/R_1_5_8.txt',
        '9th Republican Debate' : 'data/R_1_6_8.txt',
        '10th Republican Debate' : 'data/R_1_10_8.txt',
        '11th Republican Debate' : 'data/R_1_24_8.txt',
        '12th Republican Debate' : 'data/R_1_30_8.txt'}

    elif name == 'G':
        return {'1st General Election Debate' : 'data/G_9_26_8.txt',
        '2nd General Election Debate' : 'data/G_10_7_8.txt',
        '3rd General Election Debate' : 'data/G_10_15_8.txt'}

    else:
        return "None"
