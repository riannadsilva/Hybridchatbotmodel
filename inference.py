import sys
import os
original_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/nmt")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/setup")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/core")
from nmt import nmt
import argparse as ap
from settings import hparams, out_dir, preprocessing, score as score_settings
sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/setup")
import tensorflow as tf
from tokenizer import tokenize, detokenize, apply_bpe, apply_bpe_load
from sentence import replace_in_answers, normalize_new_lines
from scoring import score_answers
sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/core")
import random

# Cross-platform colored terminal text
from colorama import init, Fore
init()

# For displaying only program related output
current_stdout = None

'''
Set global tuple with model, flags and hparams which is used in inference
'''
def setup_inference_parameters(out_dir, hparams):

    # Print output on stdout while temporarily sending other output to /dev/null
    global current_stdout
    current_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    nmt_parser = ap.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    # Get settingds from configuration file
    flags, unparsed = nmt_parser.parse_known_args(['--'+key+'='+str(value) for key,value in hparams.items()])

    # Add output (model) folder to flags
    flags.out_dir = out_dir

    ## Exit if model folder doesn't exist
    if not tf.gfile.Exists(flags.out_dir):
        nmt.utils.print_out("# Model folder (out_dir) doesn't exist")
        sys.exit()

    # Load hyper parameters (hparams) from model folder
    hparams = nmt.create_hparams(flags)
    hparams = nmt.create_or_load_hparams(flags.out_dir, hparams, flags.hparams_path, save_hparams=True)

    # Choose checkpoint (provided with hparams or last one)
    if not flags.ckpt:
        flags.ckpt = tf.train.latest_checkpoint(flags.out_dir)

    # Create model
    if not hparams.attention:
        model_creator = nmt.inference.nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = nmt.inference.attention_model.AttentionModel
    elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
        model_creator = nmt.inference.gnmt_model.GNMTModel
    else:
        raise ValueError("Unknown model architecture")
    infer_model = nmt.inference.model_helper.create_infer_model(model_creator, hparams, None)

    return (infer_model, flags, hparams)

'''
Do actual inference based on supplied data, model and hyper parameters returning answers
'''
def do_inference(infer_data, infer_model, flags, hparams):

    # Disable TF logs and set stdout to devnull
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout
    if not current_stdout:
        current_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    # Spawn new TF session
    with tf.Session(graph=infer_model.graph, config=nmt.utils.get_config_proto()) as sess:

        # Load model
        loaded_infer_model = nmt.inference.model_helper.load_model(infer_model.model, flags.ckpt, sess, "infer")

        # Run model (translate)
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })


        # Calculate number of translations to be returned
        num_translations_per_input = max(min(hparams.num_translations_per_input, hparams.beam_width), 1)

        answers = []
        while True:
            try:

                nmt_outputs, _ = loaded_infer_model.decode(sess)

                if hparams.beam_width == 0:
                    nmt_outputs = nmt.inference.nmt_model.np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]

                for sent_id in range(batch_size):

                    # Iterate through responses
                    translations = []
                    for beam_id in range(num_translations_per_input):

                        if hparams.eos: tgt_eos = hparams.eos.encode("utf-8")

                        # Select a sentence
                        output = nmt_outputs[beam_id][sent_id, :].tolist()

                        # If there is an eos symbol in outputs, cut them at that point
                        if tgt_eos and tgt_eos in output:
                            output = output[:output.index(tgt_eos)]
                        print(output)

                        # Format response
                        if hparams.subword_option == "bpe":  # BPE
                            translation = nmt.utils.format_bpe_text(output)
                        elif hparams.subword_option == "spm":  # SPM
                            translation = nmt.utils.format_spm_text(output)
                        else:
                            translation = nmt.utils.format_text(output)

                        # Add response to the list
                        translations.append(translation.decode('utf-8'))

                    answers.append(translations)

            except tf.errors.OutOfRangeError:
                print("end")
                break

        # Reset back stdout and log level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        sys.stdout.close()
        sys.stdout = current_stdout
        current_stdout = None

        return answers

'''
Setup inference parameters and then invoke inference engine passing question and inference parameters
'''
def start_inference(question):

    global inference_helper, inference_object

    # Set global tuple with model, flags and hparams
    inference_object = setup_inference_parameters(out_dir, hparams)

    # Update inference_helper with actual inference function as we have completed inference parameter settings
    inference_helper = lambda question: do_inference(question, *inference_object)

    # Load BPE join pairs
    if preprocessing['use_bpe']: apply_bpe_load()

    # Finally start inference
    return inference_helper(question)

# Model, flags and hparams
inference_object = None

# Function call helper
inference_helper = start_inference

'''
Main inference function which returns answer list for the given set of one or more questions
'''
def inference(questions):

    # Change current working directory (needed to load relative paths properly)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Process questions
    answers_list = process_questions(questions)

    # Change directory back to original directory from where we started
    os.chdir(original_cwd)

    # Return answer_list
    if not isinstance(questions, list):
        return answers_list[0]
    else:
        return answers_list

'''
Get index and score for the best answer based on one of the following three settings
- default : pick first available best scored response
- random best score : pick random best scored response
- random above threshold : pick random response with score above threshold
'''
def get_best_score(answers_score):

    # Return first available best scored response
    if score_settings['pick_random'] is None:
        max_score = max(answers_score)
        if max_score >= score_settings['bad_response_threshold']:
            return (answers_score.index(max_score), max_score)
        else:
            return (-1, None)

    # Return random best scored response
    elif score_settings['pick_random'] == 'best_score':
        indexes = [index for index, score in enumerate(answers_score) if score == max(answers_score) and score >= score_settings['bad_response_threshold']]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    # Return random response with score above threshold
    elif score_settings['pick_random'] == 'above_threshold':
        indexes = [index for index, score in enumerate(answers_score) if score > (score_settings['bad_response_threshold'] if score_settings['bad_response_threshold'] >= 0 else max(score)+score_settings['bad_response_threshold'])]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    # Else return starting score by default
    return (0, score_settings['starting_score'])

'''
Process question or list of questions and provide answers with scores
It does it in three steps:
- prepare list of questions (can be one or more)
- run inference to obtain answers list (each question can have one or more answers)
- return answers along with scores and best score answer
'''
def process_questions(questions, return_score_modifiers = False):

    # Make sure questions is list (convert to list if only one question)
    if not isinstance(questions, list):
        questions = [questions]

    # Clean and tokenize
    prepared_questions = []
    for question in questions:
        question = question.strip()
        if question:
            prepared_questions.append(apply_bpe(tokenize(question)))
        else:
            prepared_questions.append('missing_question')
            
    # Run inference function
    answers_list = inference_helper(prepared_questions)

    # Process answers to return list along with score
    prepared_answers_list = []
    for i, answers in enumerate(answers_list):
        answers = detokenize(answers)
        answers = replace_in_answers(answers)
        answers = normalize_new_lines(answers)
        answers_score = score_answers(questions[i], answers)
        best_index, best_score = get_best_score(answers_score['score'])

        if prepared_questions[i] == 'missing_question':
            prepared_answers_list.append(None)
        elif return_score_modifiers:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score, 'score_modifiers': answers_score['score_modifiers']})
        else:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score})

    return prepared_answers_list

'''
Two ways to invoke inference function
- by giving input file of questions
- interactive mode and feeding each question
'''
if __name__ == "__main__":

    # Specify input file
    if sys.stdin.isatty() == False:

        # Process questions
        answers_list = process_questions(sys.stdin.readlines())

        # Print answers
        for answers in answers_list:
            print(answers['answers'][answers['best_index']])

        # Exit
        sys.exit()

    # If no file is specified, enter in interactive mode
    print('\n\n{}Starting interactive mode:{}\n'.format(Fore.GREEN, Fore.RESET))

    # In loop, get user question and print answer along with score
    while True:
        question = input("\n> ")
        answers = process_questions(question, True)[0]

        if answers is None:
            print("\n{}Question can't be empty!{}\n".format(Fore.RED, Fore.RESET))
        else:
            for i, answer in enumerate(answers['scores']):
                if answers['scores'][i] == max(answers['scores']) and answers['scores'][i] >= score_settings['bad_response_threshold']:
                    fgcolor = Fore.GREEN
                elif answers['scores'][i] >= score_settings['bad_response_threshold']:
                    fgcolor = Fore.YELLOW
                else:
                    fgcolor = Fore.RED
                print("\n{}{}[{}]{}\n".format(fgcolor, answers['answers'][i], answers['scores'][i], Fore.RESET))

os.chdir(original_cwd) # Change back to original directory
