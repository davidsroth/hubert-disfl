import sys
import parsing_all

filler_words = ["Um", "um", "Uh", "uh"]

def extract(conversation_id, return_fluent=False):
    content = parsing_all.parse(conversation_id)
    # List of dicts
    sentences = []


    sentDict = {}
    curr_text = []
    # sentDict['labels'] = []
    running = "None"
    start_time = "None"
    end_time = "None"
    conversation_id = "None"
    sentence_id = "None"
    speaker = "None"
    for token in content:
        # line = line.strip()
        if len(token) == 0:
            if running != "None" and return_fluent == False:
                if running == '+':
                    curr_text.append('<ip>')
                    curr_text.append('<r>')
                if running == '-':
                    curr_text.append('<r>')
            # sentDict['labels'] = [conversation_id, speaker,  sentence_id, start_time, end_time]
            # sentDict['end_time'] = end_time
            sentDict['conversation_id'] = conversation_id
            sentDict.update(
                {
                    'conversation_id': conversation_id,
                    'speaker': speaker,
                    'sentence_id': sentence_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': " ".join(curr_text)
                }
            )
            sentences.append(sentDict)
            start_time = "None"
            end_time = "None"
            conversation_id = "None"
            sentence_id = "None"
            sentDict = {}
            curr_text = []
            running = "None"
            pass
        else:
            assert(len(token) == 7)
            if start_time == "None":
                if token[3] not in ["n/a", "non-aligned"]:
                    start_time = token[3]
                sentence_id = token[2]
                speaker = token[1]
                conversation_id = token[0]
            if return_fluent:
                if token[-1] != '+' and token[-2] not in filler_words:
                    curr_text.append(token[-2])
            else:
                curr_text.append(token[-2])
            # elif token[-1] != "None" or running != "None":
            #     if token[-1] == '+' and running == "None":
            #         running = '+'
            #         curr_text.append('<e>')
            #     elif token[-1] == '-' and running == '+':
            #         running = '-'
            #         curr_text.append('<ip>')
            #     elif token[-1] == "None" and running == '-':
            #         running = "None"
            #         curr_text.append('<r>')
            #     elif token[-1] == "None" and running == '+':
            #         running = "None"
            #         curr_text.append('<ip>')
            #         curr_text.append('<r>')
            #     curr_text.append(token[-2])
            if token[4] not in ["n/a", "None", "non-aligned"]:
                end_time = token[4]
            
    return sentences

swnumb = sys.argv[1]
if len(sys.argv) > 2:
    if sys.argv[2] == 'fluent':
        fluent = True
    elif sys.argv[2] == 'disfluent':
        fluent = False
    else:
        print("Invalid input for 'fluency', proceeding with full disfluent text")
        fluent = False

sentences = extract(swnumb, return_fluent=fluent)

for sentDict in sentences:
    print(sentDict)
