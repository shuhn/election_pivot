# -*- coding: utf-8 -*-
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
    'Martha Raddatz (ABC News)' : 'RADDATZ:'}

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
    'Former Senator Jim Webb (VA)' : 'WEBB:'}

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
