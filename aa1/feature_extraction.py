
#basics
import pandas as pd

# own imports
import nltk
nltk.download('averaged_perceptron_tagger')
import torch
import torch.nn as nn
from nltk.corpus import stopwords

# Feel free to add any new code to this script

def pos_tag(sent):
    """
    Takes a sentence and gives each word a POS tag. Returns a list of POS tags.
    """
    tagged = nltk.pos_tag(sent)
    tags_only = []

    for every_word, pos_tag in tagged:
        tags_only.append(pos_tag)
    
    return tags_only

def is_upper(sent):
    """
    Takes a sentence, checks if a word is capitalised (ie. Ibuprofen vs ibuprofen). 
    Returns a tensor of filled with 1 or 0 for each word. 
    1 for yes the word is capitalised, 0 for not capitalised.
    """
    upper = []

    for every_word in sent:
        if every_word[0].isupper() == True:
            feature = 1
            upper.append(feature)
        else: # False
            feature = 0
            upper.append(feature)
    
    upper_tensor = torch.FloatTensor(upper)
    return upper_tensor


def get_features(split_df, id2word, max_sample_length, pos_dict):
    """
    Takes a split dataframe and returns a tensor of tensor of tensors. (split, sentence, word)
    """    

    sentences = list(split_df['sentence_id'].unique())
    token_ids = split_df['token_id'].tolist()
    char_start = split_df['char_start_id'].tolist()
    char_end = split_df['char_end_id'].tolist()
    
    split_features = []

    stop_words = stopwords.words('english')
    
    for every_sentence in sentences:

        sent_words = [] # list of strings, each element is token in the sentence
        word_lengths = []
        stop_or_not = []  
        
        # create smaller df with just sentence being examined
        sent_df = split_df.loc[split_df['sentence_id'] == every_sentence]

        token_ids = sent_df['token_id'].tolist()
        
        for every_token in token_ids:
            word = id2word[every_token]
            sent_words.append(word)
            
            # word lengths for features
            word_length = len(word)
            word_lengths.append(word_length)

            # check if word is in NLTK stop words list (1 for yes, 0 for no)
            if word in stop_words:
                stop_label = 1
                stop_or_not.append(stop_label)
            else: # not a stop word
                stop_label = 0
                stop_or_not.append(stop_label)
        
        # word lengths tensor
        length_features = torch.FloatTensor(word_lengths)

        # stop words tensor
        stop_features = torch.FloatTensor(stop_or_not)
             
        # get pos tags
        pos_tagged = pos_tag(sent_words)
        pos_feats = []

        for every_pos in pos_tagged:
            if every_pos in pos_dict.keys():
                pos_feature = pos_dict[every_pos]
                pos_feats.append(pos_feature)
            else: # not yet in pos dict
                pos_feat_counter = len(pos_dict) +1
                pos_dict[every_pos] = pos_feat_counter
                pos_feature = pos_dict[every_pos]
                pos_feats.append(pos_feature)
        pos_features = torch.FloatTensor(pos_feats)
        
        # check if word is capitalised
        upper_feats = is_upper(sent_words)
        
        # put together all of the features
        final_features = torch.stack((pos_features, upper_feats, length_features, stop_features), dim=1)

        # pad
        num_words = final_features.shape[0]
        num_features = final_features.shape[1]
        pad_to_add = max_sample_length - num_words
        padding = torch.zeros([pad_to_add, num_features])
        final = torch.cat((final_features, padding), dim=0)

        split_features.append(final)
    
    final_split = torch.stack(split_features)

    # .size [# of sentences, words, features]
    # print(final_split.shape)
    
    return final_split, pos_dict
        

def extract_features(data:pd.DataFrame, max_sample_length:int, id2word):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb

    device = torch.device('cuda:1')
    
    # create smaller dataframes by split 
    train_rows = data.loc[data['split'] == 'train'] 
    test_rows = data.loc[data['split'] == 'test'] 
    dev_rows = data.loc[data['split'] == 'val'] 
    
    # initialise POS tag dicts (keep consistent throughout split dataframes)
    pos_dict = {}
    
    train = get_features(train_rows, id2word, max_sample_length, pos_dict)
    print('train features')
    train_features = train[0]
    print('train shape', train_features.shape)
    train_dict = train[1] # to pass to next split to have the same POS tags
    
    test = get_features(test_rows, id2word, max_sample_length, train_dict)
    print('test features')
    test_features = test[0]
    print('test shape', test_features.shape)
    test_dict = test[1]
    
    dev = get_features(dev_rows, id2word, max_sample_length, test_dict)
    print('dev features')
    dev_features = dev[0]
    print('dev shape', dev_features.shape)
    dev_dict = dev[1] 
    
    train_features = train_features.to(device)
    test_features = test_features.to(device)
    dev_features = dev_features.to(device)
    
    # pass
    return train_features, dev_features, test_features

