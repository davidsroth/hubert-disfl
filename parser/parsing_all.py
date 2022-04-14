# Usage like: python parsing_all.py sw2005 > result/sw2005
import os
import xml.etree.ElementTree as ET
import sys

# get_IDdict will return built up IDdict and IDlist
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
SWB_NXT_ROOT = f"{PROJECT_ROOT}/switchboard/nxt_switchboard_ann/xml"

def get_IDdict(root, IDdict, IDlist):
    for child in root:
        if child.tag == 'word':
            wordid = child.get(namespaceIdentifier + 'id')
            IDdict[wordid] = []
            IDlist.append(wordid)
            IDdict[wordid].append(child.get(namespaceIdentifier+'start'))
            IDdict[wordid].append(child.get(namespaceIdentifier+'end'))
            IDdict[wordid].append(child.get('orth'))

        if child.tag == 'punc':
            wordid = child.get(namespaceIdentifier + 'id')
            IDdict[wordid] = []
            # attach the word
            IDlist.append(wordid)
            IDdict[wordid].append("None")
            IDdict[wordid].append("None")
            IDdict[wordid].append(child.text)
        if child.tag == 'sil':
            pass
        if child.tag == 'trace':
            pass
        else:
            continue
    return IDdict, IDlist


# print out sentence with word-level attributes
# print out with space between sentences
def pretty_print(AIDdict, AIDlist, BIDdict, BIDlist):
    indexA = 0
    indexB = 0
    inwhich = ''
    if AIDlist[0][1:].split('_')[0] == '1':
        inwhich = 'A'
    else:
        inwhich = 'B'

    while indexA < len(AIDlist) - 1 or indexB < len(BIDlist) - 1:
        if inwhich == 'A':
            # print(swnumb, end=' ')
            if indexA >= len(AIDlist) - 1 and indexB < len(BIDlist):
                print('A', AIDlist[indexA], end=' ')
                for element in AIDdict[AIDlist[indexA]]:
                    if type(element) is tuple:
                        for subele in element:
                            print(subele, end=' ')
                    elif type(element) is list:
                        for subele in element:
                            print(subele, end=' ')
                    else:
                        print(element, end=' ')
                print("")
                inwhich = 'B'
                print('')
                continue

            print('A', AIDlist[indexA], end=' ')
            for element in AIDdict[AIDlist[indexA]]:
                if type(element) is tuple:
                    for subele in element:
                        print(subele, end=' ')
                elif type(element) is list:
                    for subele in element:
                        print(subele, end=' ')
                else:
                    print(element, end=' ')
            print("")
            nextsentnum = int(AIDlist[indexA + 1].split('_')[0][1:])
            sentnum = int(AIDlist[indexA].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'B'
                print('')
            if nextsentnum - sentnum == 1:
                print('')
            indexA += 1

        if inwhich == 'B':
            # print(swnumb, end = ' ')
            if indexB >= len(BIDlist) - 1 and indexA < len(AIDlist):
                print('B', BIDlist[indexB], end=' ')
                for element in BIDdict[BIDlist[indexB]]:
                    if type(element) is tuple:
                        for subele in element:
                            print(subele, end=' ')
                    elif type(element) is list:
                        for subele in element:
                            print(subele, end=' ')
                    else:
                        print(element, end=' ')
                print("")
                inwhich = 'A'
                print('')
                continue

            print('B', BIDlist[indexB], end=' ')
            for element in BIDdict[BIDlist[indexB]]:
                if type(element) is tuple:
                    for subele in element:
                        print(subele, end=' ')
                elif type(element) is list:
                    for subele in element:
                        print(subele, end=' ')
                else:
                    print(element, end=' ')
            print("")
            nextsentnum = int(BIDlist[indexB + 1].split('_')[0][1:])
            sentnum = int(BIDlist[indexB].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'A'
                print('')
            if nextsentnum - sentnum == 1:
                print('')
            indexB += 1

def get_tokens(AIDdict, AIDlist, BIDdict, BIDlist, swnumb):
    indexA = 0
    indexB = 0
    inwhich = ''
    if AIDlist[0][1:].split('_')[0] == '1':
        inwhich = 'A'
    else:
        inwhich = 'B'

    tokens = []
    token = []
    while indexA < len(AIDlist) - 1 or indexB < len(BIDlist) - 1:
        if inwhich == 'A':
            # print(swnumb, end=' ')
            token.append(swnumb)
            if indexA >= len(AIDlist) - 1 and indexB < len(BIDlist):
                # print('A', AIDlist[indexA], end=' ')
                token.extend(['A', AIDlist[indexA]])
                for element in AIDdict[AIDlist[indexA]]:
                    if type(element) is tuple:
                        for subele in element:
                            # print(subele, end=' ')
                            token.append(subele)
                    elif type(element) is list:
                        for subele in element:
                            # print(subele, end=' ')
                            token.append(subele)
                    else:
                        # print(element, end=' ')
                        token.append(element)
                # print("")
                tokens.append(token)
                token = []
                inwhich = 'B'
                tokens.append([])
                continue

            # print('A', AIDlist[indexA], end=' ')
            token.extend(['A', AIDlist[indexA]])
            for element in AIDdict[AIDlist[indexA]]:
                if type(element) is tuple:
                    for subele in element:
                        token.append(subele)
                        # print(subele, end=' ')
                elif type(element) is list:
                    for subele in element:
                        token.append(subele)
                        # print(subele, end=' ')
                else:
                    token.append(element)
                    # print(element, end=' ')
            tokens.append(token)
            token = []
            nextsentnum = int(AIDlist[indexA + 1].split('_')[0][1:])
            sentnum = int(AIDlist[indexA].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'B'
                tokens.append([])
            if nextsentnum - sentnum == 1:
                tokens.append([])
                # print('')
            indexA += 1
            # if indexA >= len(AIDlist) and indexB >= len(BIDlist):
            #     break

        if inwhich == 'B':
            token.append(swnumb)
            # print(swnumb, end = ' ')
            if indexB >= len(BIDlist) - 1 and indexA < len(AIDlist):
                # print('B', BIDlist[indexB], end=' ')
                token.extend(['B', BIDlist[indexB]])
                for element in BIDdict[BIDlist[indexB]]:
                    if type(element) is tuple:
                        for subele in element:
                            token.append(subele)
                            # print(subele, end=' ')
                    elif type(element) is list:
                        for subele in element:
                            token.append(subele)
                            # print(subele, end=' ')
                    else:
                        token.append(element)
                        # print(element, end=' ')
                # print("")
                tokens.append(token)
                token = []
                inwhich = 'A'
                tokens.append([])
                continue

            # print('B', BIDlist[indexB], end=' ')
            token.extend(['B', BIDlist[indexB]])
            for element in BIDdict[BIDlist[indexB]]:
                if type(element) is tuple:
                    for subele in element:
                        token.append(subele)
                        # print(subele, end=' ')
                elif type(element) is list:
                    for subele in element:
                        token.append(subele)
                        # print(subele, end=' ')
                else:
                    token.append(element)
                    # print(element, end=' ')
            # print("")
            tokens.append(token)
            token = []
            nextsentnum = int(BIDlist[indexB + 1].split('_')[0][1:])
            sentnum = int(BIDlist[indexB].split('_')[0][1:])
            if nextsentnum - sentnum > 1:
                inwhich = 'A'
                tokens.append([])
            if nextsentnum - sentnum == 1:
                tokens.append([])
                # print('')
            indexB += 1
            # if indexA >= len(AIDlist) and indexB >= len(BIDlist):
            #     break
    return tokens

def attach(termi_attribute_dict, IDdict):
    for ID in IDdict:
        IDdict[ID].append(termi_attribute_dict[ID])


def None_dflfile_dict_builder(IDdict, reparandum_dict, repair_dict):
    termi_dfl_dict = {}
    for key in IDdict:
        if key not in reparandum_dict and key not in repair_dict:
            termi_dfl_dict[key] = None

    return termi_dfl_dict


# Simplifies nested disfluencies to only retain the outermost repair
def get_dfl_dict(root):
    reparandum_dict = {}
    repair_dict = {}
    for child in root:
        # since disfluency is in tree structrue, the depth are not decided
        # we use iter() to convert every disfluency child into a list.
        all_children = list(child.iter())
        reparandum_depth = 1

        for subchild in all_children:
            if subchild.tag == 'reparandum':
                if subchild.find(namespaceIdentifier + 'child') is None:
                    reparandum_depth += 1
                else:
                    words = []
                    termis = subchild.findall(namespaceIdentifier + 'child')
                    for word in termis:
                        words.append(word.get('href').split('#')[1][3:-1])
                    reparandum_dict[words[0]] = '+'
                    if len(words) > 1:
                        for i in range(1, len(words)):
                            reparandum_dict[words[i]] = '+'

            elif subchild.tag == 'repair':
                if subchild.find(namespaceIdentifier + 'child') is None:
                    continue
                else:
                    repair_words = []
                    termis = subchild.findall(namespaceIdentifier + 'child')
                    for word in termis:
                        repair_words.append(
                            word.get('href').split('#')[1][3:-1])
                    if (reparandum_depth > 1):
                        repair_dict[repair_words[-1]] = '+'
                    else:
                        repair_dict[repair_words[-1]] = '-'
                    if len(repair_words) > 1:
                        for i in range(len(repair_words) - 1):
                            if (reparandum_depth > 1):
                                repair_dict[repair_words[i]] = '+'
                            else:
                                repair_dict[repair_words[i]] = '-'
                    reparandum_depth -= 1
    return reparandum_dict, repair_dict


# create terminals disfluency dict
def terminal_dfl_dict_builder(reparandum_dict, repair_dict, IDdict):
    # termi_dfl_dict structure:
    # {termi_wordID: disfluency_label}
    termi_dfl_dict = {}
    for key in reparandum_dict:
        termi_dfl_dict[key] = reparandum_dict[key]
    for key in repair_dict:
        termi_dfl_dict[key] = repair_dict[key]

    for key in IDdict:
        if key not in reparandum_dict and key not in repair_dict:
            termi_dfl_dict[key] = None

    return termi_dfl_dict


namespaceIdentifier = '{http://nite.sourceforge.net/}'

# for iteration purpose, we split filename according to
# their name pattern, only the first part varies
# swnumb = sys.argv[1]

def parse(swnumb):
    # use ET package retrieve tree structure data for A and B speaker
    Afilepath = os.path.join(SWB_NXT_ROOT, 'terminals', swnumb + '.A.terminals.xml')
    Bfilepath = os.path.join(SWB_NXT_ROOT, 'terminals', swnumb + '.B.terminals.xml')
    try:
        Atree = ET.parse(Afilepath)
        Btree = ET.parse(Bfilepath)
    except:
        return []

    Aroot = Atree.getroot()
    Broot = Btree.getroot()

    # IDdict is a dictionary for quick checking attribute of each word
    # IDdict structure:
    # {terminal_wordID: ['word', 'pos', 'starttime', 'endtime', ]}
    AIDdict = {}
    BIDdict = {}

    AIDlist = []
    BIDlist = []

    Phoneword_dict = {}

    AIDdict, AIDlist = get_IDdict(Aroot, AIDdict, AIDlist)
    BIDdict, BIDlist = get_IDdict(Broot, BIDdict, BIDlist)

    try:
        Afilepath = os.path.join(SWB_NXT_ROOT, 'disfluency',
                                swnumb + '.A.disfluency.xml')
        Bfilepath = os.path.join(SWB_NXT_ROOT, 'disfluency',
                                swnumb + '.B.disfluency.xml')
        Atree = ET.parse(Afilepath)
        Aroot = Atree.getroot()
        Btree = ET.parse(Bfilepath)
        Broot = Btree.getroot()

        # create 2 list to record the position of reparandum and repair in
        # terminal

        # get reparandum_dict and repair_dict
        Areparandum_dict, Arepair_dict = get_dfl_dict(Aroot)
        Breparandum_dict, Brepair_dict = get_dfl_dict(Broot)

        # link termi_wordID to reparandum and repair
        Atermi_dfl_dict = terminal_dfl_dict_builder(
            Areparandum_dict, Arepair_dict, AIDdict)
        Btermi_dfl_dict = terminal_dfl_dict_builder(
            Breparandum_dict, Brepair_dict, BIDdict)

    except:
        Atermi_dfl_dict = None_dflfile_dict_builder(
            AIDdict, Areparandum_dict, Arepair_dict)
        Btermi_dfl_dict = None_dflfile_dict_builder(
            BIDdict, Breparandum_dict, Brepair_dict)

    attach(Atermi_dfl_dict, AIDdict)
    attach(Btermi_dfl_dict, BIDdict)

    return get_tokens(AIDdict, AIDlist, BIDdict, BIDlist, swnumb)