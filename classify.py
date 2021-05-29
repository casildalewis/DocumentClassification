import os
import math

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset
    
#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding='UTF-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}

    # Opening the file
    with open(filepath,'r', encoding='UTF-8') as doc:

        # For each word in the file,
        for word in doc:
            # we strip the whitespace around it,
            word = word.strip()

            if len(word) > 0:

                # and make sure it is in vocab before adding it to bow.
                # We increment OOV if it isn't.

                if word in vocab:

                    if word in bow:
                        bow[word] += 1
                    else:
                        bow[word] = 1

                else:

                    if None in bow:
                        bow[None] += 1
                    else:
                        bow[None] = 1

    return bow

def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}

    # Total number of files
    totalFiles = len(training_data)

    # Number of files per label
    filesPerLabel = {}

    # For each label (2020 and 2016),
    for givenLabel in label_list:
        # we find all corresponding bags of words for each file,
        for element in training_data:
            if element['label'] == givenLabel:
                # and update the number of bags of words (aka the number of files) there are.
                if givenLabel in filesPerLabel:
                    filesPerLabel[givenLabel] += 1
                else:
                    filesPerLabel[givenLabel] = 1

    # Calculating logprob for each label (2020 and 2016) using the given formula
    for label in filesPerLabel:
        logprob[label] = math.log(float(filesPerLabel[label] + smooth)) - math.log(float(totalFiles + 2*smooth))
    
    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}

    # Total word count (including OOV) for a given label
    n = 0
    # Word counts of each word (including OOV as None) for a given label
    nw = {}
    
    # Initialising nw
    for word in vocab:
            nw[word] = 0
    nw[None] = 0

    # Updating count for each word that is in the vocab
    for word in vocab:
        nw[word] = 0
        for element in training_data:
            if element['label'] == label:
                for eachWord in element['bow']:
                    if word == eachWord:
                            nw[word] += element['bow'][eachWord]
    
    # Updating count of total words and each OOV word
    for element in training_data:
            if element['label'] == label:
                for eachWord in element['bow']:
                    n += element['bow'][eachWord]
                    if eachWord not in vocab:
                        nw[None] += element['bow'][eachWord]

    #Calculating word_prob for each word using the given formula
    for word in nw:
        word_prob[word] = math.log(nw[word] + (smooth*1)) - math.log(n + (smooth*(len(vocab) + 1)))

    return word_prob


def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)

    retval['vocabulary'] = create_vocabulary(training_directory, cutoff)

    training_data = load_training_data(retval['vocabulary'],training_directory)
    retval['log prior'] = prior(training_data, label_list)

    retval['log p(w|y=2016)'] = p_word_given_label(retval['vocabulary'], training_data, '2016')

    retval['log p(w|y=2020)'] = p_word_given_label(retval['vocabulary'], training_data, '2020')

    return retval

def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    # Number of times each word in the vocab is used in given text file
    bow = create_bow(model['vocabulary'], filepath)

    # Initialising the label probabilities with corresponding log of prior probabilities
    retval['log p(y=2016|x)'] = model['log prior']['2016']
    retval['log p(y=2020|x)'] = model['log prior']['2020']

    # Adding the log conditional probabilities of each word in the text file
    for word in bow:
        retval['log p(y=2016|x)'] += (bow[word] * model['log p(w|y=2016)'][word])
        retval['log p(y=2020|x)'] += (bow[word] * model['log p(w|y=2020)'][word])

    # Predicted label based on label probabilities
    retval['predicted y'] = '2016' if (retval['log p(y=2016|x)'] > retval['log p(y=2020|x)']) else '2020'

    return retval
