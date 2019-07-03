from string import punctuation
import re

#List of punctuations that usually end a sentence
end_sentence_punctuations = ["'",'"',"!","?",".",")"]
#Add what are considered as bad responses (e.g. banned or curse words)
bad_responses = ["banned_word1", "banned_word2", "curse_word1", "curse_word2"]

'''
Utility function to remove punctuations from a sentence
'''
def remove_punctuations(txt):
    for individual_punctuation in punctuation:
        txt = txt.replace(individual_punctuation,'')
    return txt

'''
Penalize if most of the answer contains repetitive words
'''
def answer_echo(index, question, answer):
    try:
        score_modifier = 0
        
        answer = remove_punctuations(answer)
        answer_tokenized = answer.split(' ')
    
        tokens_length = len(answer_tokenized)
        token_repeats = 0
        for token in answer_tokenized:
            if answer_tokenized.count(token) > 1:
                token_repeats += 1
    
        repeat_percentage = float(token_repeats)/ float(tokens_length)
        if repeat_percentage == 1.0:
            score_modifier = -5
        elif repeat_percentage >= 0.75:
            score_modifier = -4
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier

'''
Penalize if most of the answer contains words from question
'''
def answer_echo_question(index, question, answer):
    try:
        score_modifier = 0
        
        answer = remove_punctuations(answer)
        question = remove_punctuations(question)
        
        answer_tokenized = answer.split(' ')
        question_tokenized = question.split(' ')
    
        tokens_length = len(answer_tokenized)
    
        if tokens_length == 1:
            return score_modifier
        else:
            token_repeats = 0
            for token in answer_tokenized:
                if question_tokenized.count(token) > 0:
                    token_repeats += 1
    
            echo_percentage = float(token_repeats)/ float(tokens_length)
            if echo_percentage == 1.0:
                score_modifier = -5
            elif echo_percentage >= 0.75:
                score_modifier = -4
            return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier

'''
Penalize if answer and question are the same or are part of each other
'''
def is_answer_similar_to_question(index, question, answer):
    try:
        score_modifier = 0
        
        question = remove_punctuations(question)
        answer = remove_punctuations(answer)
            
        if question == answer:
            score_modifier = -4
        if answer in question or question in answer:
            score_modifier = -3
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier

'''
Credit if a sentence ends in a punctuation.
'''
def answer_end_in_punctuation(index, question, answer):
    try:
        score_modifier = 0
        
        sentence_end_with_punctuation = False
        for end_sentence_punctuation in end_sentence_punctuations:
            if answer[-1] == end_sentence_punctuation:
                sentence_end_with_punctuation = True

        if sentence_end_with_punctuation:
            score_modifier = 1
        else:
            score_modifier = 0
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier
    
'''
Penalize if link at the end of an answer ends with '='
'''
def answer_ends_in_equals(index, question, answer):
    try:
        score_modifier = 0
        
        if answer[-1] == "=":
            score_modifier = -3
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier

'''
Penalize if sentence includes <unk> or 'unk' token
'''
def unk_checker(index, question, answer):
    try:
        score_modifier = 0
        
        if '<unk>' in answer or '_unk' in answer:
            score_modifier = -4
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier
    
'''
Penalize if there are more badly formatted links than good links
'''
def messedup_link(index, question, answer):
    try:
        score_modifier = 0
        
        badlinks = re.findall(r'\[.*?\]\s?\(',answer)
        goodlinks = re.findall(r'\[.*?\]\s?\(.*?\)',answer)
        if len(badlinks) > len(goodlinks):
            score_modifier = -3
        else:
            score_modifier = 0
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier

'''
Penalize if answer contains any of the words considered as bad response
'''
def bad_response(index, question, answer):
    try:
        score_modifier = 0
        
        for bad_response in bad_responses:
            if bad_response in answer:
                score_modifier = -10
        return score_modifier
    except Exception as exception:
        print(str(exception))
        return score_modifier
     
'''
Calculate score for all answers based on question itself and heuristically calculated scoring algorithms
'''
def score_answers(question, answers):
    starting_score = 10 #same for all answers to start with
    
    # scoring functions to run
    functions = [
       answer_echo,
       answer_echo_question,
       is_answer_similar_to_question,
       answer_end_in_punctuation,
       answer_ends_in_equals,
       unk_checker,
       messedup_link,
       bad_response
    ]

    scores = {'score': [], 'score_modifiers': []}

    # Iterate thru answers, apply every scoring function
    for i, answer in enumerate(answers):
        score_modifiers = [function(i+1, question, answer) for function in functions]
        scores['score'].append(starting_score + sum(score_modifiers))
        scores['score_modifiers'].append(score_modifiers)

    # Return score
    return scores        