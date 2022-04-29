# usage like: python extract_all.py sw2005 > result_extracted/sw200
import sys
from .parsing_all import parse

filler_words = ["Um", "um", "Uh", "uh"]

def extract(conversation_id, return_fluent=False, filler_words=filler_words, chars_to_ignore=[], min_length=0, max_length=20, verbose=False):
    """
    Annotate disfluency using parsing of conversations and returning in the form of sequence
    Params:
        conversation_id: str - conversation id
        return_fluent: bool - if only fluent transcript required
        filler_words: List of tokens
        chars_to_ignore: List of characters to remove from the transcripts
    Returns:
        sentences: list - of extracted annotations of sequences in a conversation
    """
    content = parse(conversation_id, verbose=verbose)
    if verbose:
        print(content)
    # List of dicts
    sentences = []

    sentDict = {}
    curr_text = []
    running = "None"
    start_time = "None"
    end_time = "None"
    conversation_id = "None"
    sentence_id = "None"
    speaker = "None"
    for token in content:
        if len(token) == 0:
            # if running != "None" and return_fluent == False:
            #     if running == '+':
            #         curr_text.append('<ip>')
            #         curr_text.append('<r>')
            #     if running == '-':
            #         curr_text.append('<r>')
            if start_time == "None" or end_time == "None":
                start_time = "None"
                end_time = "None"
                conversation_id = "None"
                sentence_id = "None"
                sentDict = {}
                curr_text = []
                running = "None"
                continue
            if float(end_time) - float(start_time) < min_length or float(end_time) - float(start_time) > max_length:
                start_time = "None"
                end_time = "None"
                conversation_id = "None"
                sentence_id = "None"
                sentDict = {}
                curr_text = []
                running = "None"
                continue
            sentDict.update(
                {
                    'conversation_id': conversation_id,
                    'speaker': speaker,
                    # 'sentence_id': sentence_id,
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'text': " ".join(curr_text),
                    'duration': float(end_time) - float(start_time)
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
                if token[-1] != '+' and token[-2] not in filler_words and not token[-2].endswith("-"):
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

def extract_w_tags(conversation_id, verbose=False):
    """
    Annotate disfluency using parsing of conversations and returning in the form of sequence
    Params:
        conversation_id: str - conversation id
        chars_to_ignore: List of characters to remove from the transcripts
    Returns:
        sentences: list - of extracted annotations of sequences in a conversation
    """
    content = parse(conversation_id, verbose=verbose)
    if verbose:
        print(content)
    # List of dicts
    sentences = []

    sentDict = {}
    curr_text = []
    curr_tags = []
    conversation_id = "None"
    sentence_id = "None"
    speaker = "None"
    for token in content:
        if len(token) == 0:
            sentDict.update(
                {
                    'text': curr_text,
                    'tags': curr_tags
                }
            )
            sentences.append(sentDict)
            conversation_id = "None"
            sentence_id = "None"
            sentDict = {}
            curr_text = []
            curr_tags = []
            pass
        else:
            if token[-1] == '+' or token[-2] in filler_words:
                curr_tags.append(1)
            else:
                curr_tags.append(0)
            curr_text.append(token[-2])
    return sentences

if __name__ == "__main__":
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
