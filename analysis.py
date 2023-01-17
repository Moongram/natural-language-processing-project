
import math
import random
import numpy
from collections import *
import matplotlib.pyplot as plt
import pylab

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence


#####################  STUDENT CODE BELOW THIS LINE  #####################


def compute_counts(training_data: list, order: int) -> tuple:
    """
    Given training_data and order, return 4 or 5 different types of counts
    Inputs:
      training_data --- a list of (word, POS-tag) pairs returned by the function read_pos_file
      order ---  the order of the HMM
    Returns:
       num_token --- the number of tokens in training_data
       word_tag --- a dictionary that contains counts of all unique words tagged with unique tags (keys correspond to tags)
       times_tag --- a dictionary contains counts of number of times all unique tags appear
       times_seq --- a dictionary contains counts of number of times of all tag sequences of length 2 appears
       times_seq2(when order equals 3) --- a dictionary contains counts of number of times of all tag sequences of length 3 appears
    """

    #compute number of tokens
    num_token = len(training_data)

    #initialize defaultdict for word_tag, times_tag, times_seq
    word_tag = defaultdict(lambda: defaultdict(int))
    times_tag = defaultdict(int)
    times_seq = defaultdict(lambda: defaultdict(int))

    #compute word_tag and times_tag
    for tup in training_data:
        tag = tup[1]
        word = tup[0]
        times_tag[tag] += 1
        word_tag[tag][word] += 1

    #computr timrd_seq
    for tup in range(num_token - 1):
        word_1 = training_data[tup][1]
        word_2 = training_data[tup + 1][1]
        times_seq[word_1][word_2] += 1

    #if order equals 2, return the dictionaries computed
    if order == 2:
        return num_token, word_tag, times_tag, times_seq

    #if order equals 3, compute time_seq2, which is a 3d dictionary containing counts of tag sequence of length 3
    if order == 3:
        times_seq2 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for tup in range(num_token - 2):
            word_1 = training_data[tup][1]
            word_2 = training_data[tup + 1][1]
            word_3 = training_data[tup + 2][1]
            times_seq2[word_1][word_2][word_3] +=1
        return num_token, word_tag, times_tag, times_seq, times_seq2


def compute_initial_distribution(training_data: list, order: int) -> dict:
    """
    Given training_data and order, returns initial probability
    Inputs:
      training_data --- a list of (word, POS-tag) pairs returned by the function read_pos_file
      order ---  the order of the HMM
    Returns:
        pi_prob --- (order equals 2) a dictionary contains the probability of all unique tags 
                    appear at the beginning of a sentence;
                    (order equals 3) a dictionary contains the probability of all unique tag 
                    sequence of length 2 appear at the beginning of a sentence
    """
    # compute number of tokens
    num_token = len(training_data)

    # order 2 initial distribution
    if order == 2:
        count = 1
        pi_prob = defaultdict(int)
        pi_prob[training_data[0][1]] = 1

        # first compute count
        for tup in range(num_token - 1):

            # beggining of new sentence follows period of last sentence
            if training_data[tup][1] == ".":
                begin = training_data[tup + 1][1]
                pi_prob[begin] += 1
                count += 1
        
        # use count number to compute probability
        for start in pi_prob:
            pi_prob[start] = pi_prob[start]/count

    # order 3 initiual distribution
    if order == 3:
        count = 1
        pi_prob = defaultdict(lambda: defaultdict(int))
        pi_prob[training_data[0][1]][training_data[1][1]] = 1

        #compute count
        for tup in range(num_token - 2):

            # beggining of new sentence follows period of last sentence
            if training_data[tup][1] == ".":
                pi_prob[training_data[tup + 1][1]][training_data[tup + 2][1]] += 1
                count += 1

        #compute probability
        for start in pi_prob:
            for second in pi_prob[start]:
                pi_prob[start][second] = pi_prob[start][second]/count
    return pi_prob


def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
    """
    Compute emission probabilities as a dictionary whose keys are the tags, based on input lists and dictionaries
    Inputs:
      unique_words --- a list of unique words returned by read_pos_file
      unique_tags --- a list of unique tags returned by read_pos_file
      W --- the dictionary word_tag returned by compute_counts
      C --- the dictionary times_tag returned by compute_counts
    Returns:
      emmi_prob --- a dictionary contains the emission probabilities of words by different tags
    """

    #initialize defaultdict emmi_prob
    emmi_prob = defaultdict(lambda: defaultdict(int))

    #for all tags and words, we compute emmision probability by W[tag][word]/C[tag]
    for tag in unique_tags:
        for word in unique_words:
            emmi_prob[tag][word] = W[tag][word]/C[tag]
    return emmi_prob


def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
    """ 
    Compute lambdas that will be used in calculating transition probabilities if use smoothing
    Inputs:
        unique_tags --- a list of unique tags returned by read_pos_file
        num_tokens --- the number of tokens in training_data
        C1 --- the dictionaries with number of times of all tags appear
        C2 --- the dictionaries with number of times all tag sequences of length 2 appear
        C3 --- the dictionaries with number of times all tag sequences of length 3 appear
        order ---  the order of the HMM		
    Returns:
        lambda_lst --- a list that contains lambda0, lambda1, lambda2 respectivelys
    """

    # lambda for order 2
    if order == 2:

        #initialize lambda_lst and alpha_lst with all 0
        lambda_lst = [0, 0, 0]
        alpha_lst = [0, 0]

        # iterate over all tag sequences of length 2
        for first in C2:
            for second in C2[first]:

                # only consider when C2[first][second] > 0
                if C2[first][second] > 0 and first in unique_tags and second in unique_tags:
                    
                    # when compute alpha, if denominator equals 0, set alpha to 0
                    if num_tokens == 0:
                        alpha_lst[0] = 0
                    if num_tokens != 0:
                        alpha_lst[0] = (C1[second] - 1) / num_tokens
                    if (C1[first] - 1) == 0:
                        alpha_lst[1] = 0
                    if (C1[first] - 1) != 0:
                        alpha_lst[1] = (C2[first][second] - 1) / (C1[first] - 1)
                    
                    #compute lambda
                    idx = numpy.argmax(alpha_lst)
                    lambda_lst[idx] = lambda_lst[idx] + C2[first][second]
    
    # lambda for order 3
    if order == 3:
        #initialize lambda_lst and alpha_lst with all 0
        lambda_lst = [0, 0, 0]
        alpha_lst = [0, 0, 0]

        # iterate over all tag sequences of length 3
        for first in C3:
            for second in C3[first]:
                for third in C3[first][second]:
                    
                    # when compute alpha, if denominator equals 0, set alpha to 0
                    if C3[first][second][third] > 0 and first in unique_tags and second in unique_tags and third in unique_tags:
                        if num_tokens == 0:
                            alpha_lst[0] = 0
                        if num_tokens != 0:
                            alpha_lst[0] = (C1[third] - 1) / num_tokens
                        if (C1[second] - 1) == 0:
                            alpha_lst[1] = 0
                        if (C1[second] - 1) != 0:
                            alpha_lst[1] = (C2[second][third] - 1) / (C1[second] - 1)
                        if (C2[first][second] - 1) == 0:
                            alpha_lst[2] = 0
                        if (C2[first][second] - 1) != 0:
                            alpha_lst[2] = (C3[first][second][third] - 1) / (C2[first][second] - 1)

                        # compute lambda
                        idx = numpy.argmax(alpha_lst)
                        lambda_lst[idx] = lambda_lst[idx] + C3[first][second][third]	
    
    # normalize lambda
    lambda_sum = sum(lambda_lst)
    for ele in range(len(lambda_lst)):
        lambda_lst[ele] = lambda_lst[ele] / lambda_sum		
    return lambda_lst
    

def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
    """
    Based on inputs, returns a fully trained HMM
    Inputs:
      training_data --- a list of (word, POS-tag) pairs returned by the function read_pos_file
      unique_tags --- a list of unique tags returned by read_pos_file
      unique_words --- a list of unique words returned by read_pos_file
      order ---  the order of the HMM
      use_smoothing --- a boolean parameter indicating using smoothing or not 
    Returns:
       hmm --- a fully trained hmm model
    """

    # use previous functions to compute ini_dis, num_tokens, W, C1, C2, emmi_mat
    ini_dis = compute_initial_distribution(training_data, order)
    num_tokens = compute_counts(training_data, order)[0]
    W = compute_counts(training_data, order)[1]
    C1 = compute_counts(training_data, order)[2]
    C2 = compute_counts(training_data, order)[3]
    emmi_mat = compute_emission_probabilities(unique_words, unique_tags, W, C1)
    
    # compute order 2 hmm
    if order == 2:

        #with smoothing hmm
        if use_smoothing:

            # get all lambdas
            lambda_lst = compute_lambdas(unique_tags,num_tokens, C1, C2, {}, order)
            lambda0 = lambda_lst[0]
            lambda1 = lambda_lst[1]

            # compute trans_mat using equation provided
            trans_mat = defaultdict(lambda: defaultdict(int))
            for tag1 in unique_tags:
                for tag2 in unique_tags:
                    if C1[tag1] != 0:
                        trans_mat[tag1][tag2] = (lambda1 * (C2[tag1][tag2] / C1[tag1])) + (lambda0 * (C1[tag2] / num_tokens))
        
        #without smoothing hmm
        else:

            # compute trans_mat with know lambda
            trans_mat = defaultdict(lambda: defaultdict(int))
            for tag1 in unique_tags:
                for tag2 in unique_tags:
                    if C1[tag1] != 0:
                        trans_mat[tag1][tag2] = C2[tag1][tag2] / C1[tag1]
    
    # compute order 3 hmm
    else:
        C3 = compute_counts(training_data, order)[4]

        #with smoothing hmm
        if use_smoothing:

            #get all lambda
            lambda_lst = compute_lambdas(unique_tags,num_tokens, C1, C2, C3, order)
            lambda0 = lambda_lst[0]
            lambda1 = lambda_lst[1]
            lambda2 = lambda_lst[2]

            # compute trans_mat with provided equation
            trans_mat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for tag1 in unique_tags:
                for tag2 in unique_tags:
                    for tag3 in unique_tags:
                        if C1[tag2] != 0 and C2[tag1][tag2] != 0:
                            trans_mat[tag1][tag2][tag3] = (lambda1 * (C2[tag2][tag3] / C1[tag2])) +  \
                             (lambda0 * (C1[tag3] / num_tokens)) + (lambda2 * (C3[tag1][tag2][tag3] / C2[tag1][tag2]))		

        #without smoothing
        else:

            #compute trans_mat with known lambdas
            trans_mat = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for tag1 in unique_tags:
                for tag2 in unique_tags:
                    for tag3 in unique_tags:
                        if C2[tag1][tag2] != 0:
                            trans_mat[tag1][tag2][tag3] = C3[tag1][tag2][tag3] / C2[tag1][tag2]
    
    #build a hmm object
    hmm = HMM(order, ini_dis, emmi_mat, trans_mat)
    return hmm


def trigram_viterbi(hmm, sentence: list) -> list:
    """
    Implements the Viterbi algorithm for the trigram model on an input HMM and sentence.
    Returns the sentence with each word tagged with its part-of-speech.

    :param hmm: input Hidden Markov Model
    :param sentence: a list of words and the period at the end
    :return: a list of (word, tag) pairs
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for outer_tag in hmm.transition_matrix.keys():
        unique_tags.update(set(hmm.transition_matrix[outer_tag].keys()))
        for inner_tag in hmm.transition_matrix[outer_tag].keys():
            unique_tags.update(set(hmm.transition_matrix[outer_tag][inner_tag].keys()))
    # print('trigram_viterbi() unique_tags:', unique_tags)  # Debugging

    # print('trigram_viterbi(): past initialization')  # Debugging

    # Setting initial distributions
    # print('trigram_viterbi(): emission probability: ', hmm.emission_matrix['PN'][sentence[0]])  # Debugging
    # print(sentence[0])  # Debugging
    # print('trigram_viterbi(): emission probability: ', hmm.emission_matrix['DA'][sentence[1]])  # Debugging
    # print('trigram_viterbi(): initial distribution probability: ', hmm.initial_distribution['PN']['DA'])  # Debugging
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            init_dist = hmm.initial_distribution[tag1][tag2]
            tag1_emit = hmm.emission_matrix[tag1][sentence[0]]
            tag2_emit = hmm.emission_matrix[tag2][sentence[1]]
            # print('init_dist', init_dist)  # Debugging
            # print('tag1:', tag1_emit, tag1, sentence[0])  # Debugging
            # print('tag2:', tag2_emit)  # Debugging

            if init_dist != 0 and tag1_emit != 0 and tag2_emit != 0:
                # print('tag1:', tag1_emit)  # Debugging
                # print('tag2:', tag2_emit)  # Debugging
                viterbi[tag1][tag2][1] = math.log(init_dist) + math.log(tag1_emit) + math.log(tag2_emit)
            else:
                viterbi[tag1][tag2][1] = -1 * float('inf')

    # print('trigram_viterbi(): initial distribution probability: ', viterbi['PN']['DA'])  # Debugging

    # print('trigram_viterbi(): past initial distributions')  # Debugging

    # Dynamic programming
    for t in range(2, len(sentence)):
        backpointer['No_Path']['No_Path'][t] = 'No_Path'
        for tag1 in unique_tags:
            for tag2 in unique_tags:

                # Finding previous state with max probability
                max_value = -1 * float('inf')
                max_state = None
                for tag0 in unique_tags:
                    past_vit_val = viterbi[tag0][tag1][t - 1]
                    trans_val = -1 * float('inf')
                    if hmm.transition_matrix[tag0][tag1][tag2] != 0:
                        trans_val = math.log(hmm.transition_matrix[tag0][tag1][tag2])
                    curr_value = past_vit_val + trans_val
                    if curr_value > max_value:
                        max_value = curr_value
                        max_state = tag0

                # Adding emission probability
                emit_val = -1 * float('inf')
                if hmm.emission_matrix[tag2][sentence[t]] != 0:
                    emit_val = math.log(hmm.emission_matrix[tag2][sentence[t]])
                viterbi[tag1][tag2][t] = max_value + emit_val

                # Setting backpointer
                if max_state == None:
                    backpointer[tag1][tag2][t] = "No_Path"
                else:
                    backpointer[tag1][tag2][t] = max_state

    # print('trigram_viterbi(): past dynamic programming')  # Debugging

    # Termination
    max_value = -1 * float('inf')
    next_to_last_state = None
    last_state = None
    final_time = len(sentence) - 1
    for tag1 in unique_tags:
        for tag2 in unique_tags:
            if viterbi[tag1][tag2][final_time] > max_value:
                max_value = viterbi[tag1][tag2][final_time]
                next_to_last_state = tag1
                last_state = tag2
    if last_state == None:
        next_to_last_state = "No_Path"
        last_state = "No_Path"

    # print('trigram_viterbi(): past termination')  # Debugging
    # print('trigram_viterbi(): viterbi matrix:', viterbi)  # Debugging
    # print('first backpoint:',backpointer['No_Path']['No_Path'][5 + 2])  # Debugging

    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence) - 1], last_state))
    tagged_sentence.append((sentence[len(sentence) - 2], next_to_last_state))
    for i in range(len(sentence) - 3, -1, -1):
        next_tag = tagged_sentence[-1][1]
        next_next_tag = tagged_sentence[-2][1]
        curr_tag = backpointer[next_tag][next_next_tag][i + 2]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

def update_hmm(hmm, test_word, training_word):
    """
    Update HMM model by assigning a very small emission probability of 0.00001 to words
    that is not in training data, also adjust emission probabilities of other words by adding
    the same 0.00001. Finally normalize the probabilities.
    Inputs:
        hmm --- the HMM to use to predict the POS of the words in the sentence.
        test_word --- a list of unique words in test corpus
        training_word --- a list of unique words in training data
    Returns:
        hmm --- the updated hmm
    """

    # build update_list to store all words in test_word that are not in training_word
    update_list = []
    for word in test_word:
        if word not in training_word:
           update_list.append(word)
    
    if len(update_list) != 0:
        #update emission_matrix
        for tag in hmm.emission_matrix:
            for word in hmm.emission_matrix[tag]:

                # add 0.00001 to all word that are not in update_list
                if word not in update_list and hmm.emission_matrix[tag][word] != 0:
                    hmm.emission_matrix[tag][word] += 0.00001

        # set hmm.emission_matrix[tag][word] = 0.00001 for words in update_list
        for word in update_list:
            for tag in hmm.emission_matrix:
                hmm.emission_matrix[tag][word] = 0.00001

    # normalize the emission values for each state so that they add up to 1
    for tag in hmm.emission_matrix:
        sum = 0
        for word in hmm.emission_matrix[tag]:
            sum += hmm.emission_matrix[tag][word]
        for word in hmm.emission_matrix[tag]:
            hmm.emission_matrix[tag][word] = hmm.emission_matrix[tag][word] / sum
    return hmm


def experiment(order, percentage, smoothing):
    """
    Conduct experiments to test out accuracy of second order and third order hmm, with smoothing or without smoothing
    also with different percentage of test data when building the hmm
    Inputs:
        order --- the order of HMM
        percentage --- a float indicating amount of training corpus to use to train hmm
        smoothing --- a boolean parameter indicating use smoothing or not
    Returns:
        accuracy --- a float representing the accuracy of hmm
    """

    # read all training data
    data_base = read_pos_file("training.txt")
    training_data = data_base[0]

    # get first x percent training data corresponding to input percentage
    training_data = training_data[0 : int(len(training_data) * percentage)]
    
    # calculate unique_words and unique_tags based on updated training_data
    unique_words = set()
    unique_tags = set()
    for tup in training_data:
        unique_words.add(tup[0])
        unique_tags.add(tup[1])

    # build a hmm
    hmm = build_hmm(training_data, unique_tags, unique_words, order, smoothing)
    
    # get test data
    test_result = read_pos_file("testdata_tagged.txt")
    test_data = test_result[0]
    test_word = test_result[1]

    # get test sentence by splitting (word, tag) pairs
    sentence = []
    for tup in test_data:
        sentence.append(tup[0])

    # get a list of index indicating beginning of sentences
    sentence_end_idx = []
    for idx in range(len(sentence)):
        if sentence[idx] == ".":
            sentence_end_idx.append(idx + 1)

    # split the text into individual sentences based on sentence_end idx
    sentence_seg = []
    sentence_seg.append(sentence[0:sentence_end_idx[0]])
    for idx in range(len(sentence_end_idx) - 1):
        segment = sentence[sentence_end_idx[idx]:sentence_end_idx[idx+1]]
        sentence_seg.append(segment)

    # update hmm
    hmm = update_hmm(hmm, test_word, unique_words)

    #initialize a list to store word tag pairs computed by viterbi
    tagged_sentence = []

    # trigram viterbi
    if order == 3:

        # run viterbi over each individual sentences
        for sen in sentence_seg:
            tagged_sen = trigram_viterbi(hmm, sen)
            tagged_sentence.extend(tagged_sen)

    #bigram viterbi
    if order == 2:

        # run viterbi over each individual sentences
        for sen in sentence_seg:
            tagged_sen = bigram_viterbi(hmm, sen)
            tagged_sentence.extend(tagged_sen)
    
    # compute accuracy
    count = 0
    for idx in range(len(tagged_sentence)):
        if tagged_sentence[idx][1] == test_data[idx][1]:
            count += 1
    accuracy = count / len(sentence) * 100
    return accuracy


def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

# experiment 1
exp1_1 = experiment(2, 0.01, False)
exp1_5 = experiment(2, 0.05, False)
exp1_10 = experiment(2, 0.1, False)
exp1_25 = experiment(2, 0.25, False)
exp1_50 = experiment(2, 0.5, False)
exp1_75 = experiment(2, 0.75, False)
exp1_100 = experiment(2, 1, False)
experiment1 = {1: exp1_1, 5: exp1_5, 10: exp1_10, 25: exp1_25, 50: exp1_50, 75: exp1_75, 100: exp1_100}
print(experiment1)

# experiment 2
exp2_1 = experiment(3, 0.01, False)
exp2_5 = experiment(3, 0.05, False)
exp2_10 = experiment(3, 0.1, False)
exp2_25 = experiment(3, 0.25, False)
exp2_50 = experiment(3, 0.5, False)
exp2_75 = experiment(3, 0.75, False)
exp2_100 = experiment(3, 1, False)
experiment2 = {1: exp2_1, 5: exp2_5, 10: exp2_10, 25: exp2_25, 50: exp2_50, 75: exp2_75, 100: exp2_100}
print(experiment2)

# experiment 3
exp3_1 = experiment(2, 0.01, True)
exp3_5 = experiment(2, 0.05, True)
exp3_10 = experiment(2, 0.1, True)
exp3_25 = experiment(2, 0.25, True)
exp3_50 = experiment(2, 0.5, True)
exp3_75 = experiment(2, 0.75, True)
exp3_100 = experiment(2, 1, True)
experiment3 = {1: exp3_1, 5: exp3_5, 10: exp3_10, 25: exp3_25, 50: exp3_50, 75: exp3_75, 100: exp3_100}
print(experiment3)

# experiment 4
exp4_1 = experiment(3, 0.01, True)
exp4_5 = experiment(3, 0.05, True)
exp4_10 = experiment(3, 0.1, True)
exp4_25 = experiment(3, 0.25, True)
exp4_50 = experiment(3, 0.5, True)
exp4_75 = experiment(3, 0.75, True)
exp4_100 = experiment(3, 1, True)
experiment4 = {1: exp4_1, 5: exp4_5, 10: exp4_10, 25: exp4_25, 50: exp4_50, 75: exp4_75, 100: exp4_100}
print(experiment4)

# plot 4 experiment
plot_lines([experiment1, experiment2, experiment3, experiment4], "HMM Accuracy", "Percentage of Training Data", "Accuracy", ["Bigram without Smoothing", "Trigram without Smoothing", "Bigram with Smoothing", "Trigram with Smoothing"], filename='Accuracy')
show()