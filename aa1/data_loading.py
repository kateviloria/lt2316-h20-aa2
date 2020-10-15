
#basics
import random
import pandas as pd
import torch

# own imports
import os
import xml.etree.ElementTree as ET
import re
from nltk.tokenize import WhitespaceTokenizer 
from nltk.tokenize import WordPunctTokenizer 
import string
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
from collections import Counter
import numpy as np 
from venn import venn

# device = torch.device('cpu')

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):

    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    def get_subdir(self, path_to_folder):
        subfolders = [f.path for f in os.scandir(path_to_folder) if f.is_dir()]
        return subfolders

    def get_xmls(self, path):
        xml_list = []
        for filename in os.listdir(path):
            if filename.endswith(".xml"):
                xml_list.append(os.path.join(path, filename))
        return xml_list

    def process_text(self, sentence):
        # tokenizer !
        new_list = []
        word_list = WhitespaceTokenizer().tokenize(sentence)
        all_punct = list(string.punctuation)
        
        for every_word_idx in range(len(word_list)):
            word = word_list[every_word_idx]
            last_idx = len(word) -1
            start_char = word[0]
            end_char = word[last_idx]
            
            # using a second tokenizer for punctuation (ie. :ibuprofen -> ':' , 'ibuprofen')
            if start_char in all_punct:
                split_punct = WordPunctTokenizer().tokenize(word)
                add_split = [new_list.append(every_newtoken) for every_newtoken in split_punct]

            elif end_char in all_punct:
                if len(word) > 1:
                    before_last = last_idx -1
                    if word[before_last] in all_punct: # '):'
                        new_list.append(word)     
                    else: # word + punct
                        split_punct = WordPunctTokenizer().tokenize(word)
                        add_split = [new_list.append(every_newtoken) for every_newtoken in split_punct]
                
                else: # len == 1
                    new_list.append(word)        
                    
            else: # no punct
                new_list.append(word)
                
        return new_list

    def char_offset(self, original_string, tokens):
        """
        Takes a string and returns a list of tuples. 
        Each tuple represents a word with it's start and end character.
        """
        char_list = []
        start = 0
        for every_token in tokens:
            # start parameter is starting where to look if word is in middle of sentence
            beg = original_string.find(every_token, start) 
            char_length = len(every_token)
            end = beg + char_length -1   
            chars = (beg, end)
            char_list.append(chars)
            start = end
            
        # for checking 
        word2char = zip(tokens, char_list)

        return char_list

    def open_xmls(self, file_list):
        
        all_vocab = []
        token_dict = {}
        counter = 1 # 0 reserved for padding, token dict will also start from 1
        id2ner = {1:'not_ner', 2:'brand', 3:'drug', 4:'drug_n', 5:'group'}
        
        # list of lists, each list is a token with data for every column
        all_tokens = [['sentence_id', 'token_id', 'char_start_id', 'char_end_id', 'split']] 
        ner_data =[['sentence_id', 'ner_id', 'char_start_id', 'char_end_id']]
                    
        for every_file in file_list:
            
            split = every_file[1]
            file_path = every_file[0]
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # sentence
            root_tag = root[0].tag

            for every_sentence in root:

                sent_id = every_sentence.attrib['id']

                text = every_sentence.attrib['text']
                
                tokenized = self.process_text(text)

                # get char offset for each token
                char_list = self.char_offset(text, tokenized)

                assert len(tokenized) == len(char_list),"Token list and character list do not have the same amount of values."

                for every_word_index in range(len(tokenized)):
                    word = tokenized[every_word_index]
                    all_vocab.append(word)

                    # make token dict
                    if word not in token_dict.keys():
                        token_dict[word] = counter
                        token_id = token_dict[word]
                        counter += 1
                    else: 
                        token_id = token_dict[word]

                    char_start_id = char_list[every_word_index][0]
                    char_end_id = char_list[every_word_index][1]

                    row = [sent_id, token_id, char_start_id, char_end_id, split]
                    all_tokens.append(row)

                # create ner_data
                for every_item in every_sentence.findall('entity'):
                    entity_id = every_item.attrib['id']

                    entity_name = every_item.attrib['text']

                    entity_type = every_item.attrib['type']
                    for ner_label, ner_type in id2ner.items():
                        if entity_type == ner_type:
                            ner_id = ner_label

                    entity_charoffset = every_item.attrib['charOffset']

                    # entity have two char starts and ends
                    if ';' in entity_charoffset:
                        split_semi = entity_charoffset.split(';')

                        first = split_semi[0].split('-')
                        first_start = int(first[0])
                        first_end = int(first[1])

                        second = split_semi[1].split('-')
                        second_start = int(second[0])
                        second_end = int(second[1])
                        
                        ner_data.append([sent_id, ner_id, first_start, first_end])
                        ner_data.append([sent_id, ner_id, second_start, second_end])

                    else: # only one char start and end
                        char_list = entity_charoffset.split('-')

                        start = int(char_list[0])
                        end = int(char_list[1])
                        
                        ner_data.append([sent_id, ner_id, start, end])

        # turn list of lists into dataframes
        data_df = pd.DataFrame(all_tokens[1:], columns=all_tokens[0])
        ner_df = pd.DataFrame(ner_data[1:], columns=ner_data[0])
      
        vocab = list(token_dict.keys())
        # invert keys and values (outcome is {int : 'word'})
        id2word = {v: k for k, v in token_dict.items()}
        
        return data_df, ner_df, vocab, id2ner, id2word

    def find_max_sample_length(self, data_df): 
    
        most_tokens = data_df["sentence_id"].value_counts()
        max_sample_length = most_tokens.max()
            
        return max_sample_length

    def _parse_data(self, data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.

        all_test_files = []
        all_train_files = []

        # get subfolders' paths of DDICorpus
        DDI_sub = self.get_subdir(data_dir)
        DDI_testdir = DDI_sub[0]
        DDI_traindir = DDI_sub[1]

        # folder paths under test directory
        test_sub = self.get_subdir(DDI_testdir)

        test_NER = test_sub[0] 
        test_NER_sub = self.get_subdir(test_NER)
        test_NER_MedLine = test_NER_sub[0]
        test_NER_DrugBank = test_NER_sub[1]

        # will be added to training
        test_DDI = test_sub[1]
        test_DDI_sub = self.get_subdir(test_DDI)
        test_DDI_MedLine = test_DDI_sub[0]
        test_DDI_DrugBank = test_DDI_sub[1]

        # folder paths under train directory
        train_sub = self.get_subdir(DDI_traindir)
        train_DrugBank = train_sub[0]
        train_MedLine = train_sub[1]

        # test xmls
        test_NER_MedLine_files = self.get_xmls(test_NER_MedLine)
        test_NER_DrugBank_files = self.get_xmls(test_NER_DrugBank)

        # for training
        test_DDI_MedLine_files = self.get_xmls(test_DDI_MedLine)
        test_DDI_DrugBank_files = self.get_xmls(test_DDI_DrugBank)

        test_files = test_NER_MedLine_files + test_NER_DrugBank_files 

        [all_test_files.append((every_file, 'test'))for every_file in test_files]

        # train xmls
        train_MedLine_files = self.get_xmls(train_MedLine)
        train_DrugBank_files = self.get_xmls(train_DrugBank)

        train_files = train_MedLine_files + train_DrugBank_files + test_DDI_MedLine_files + test_DDI_DrugBank_files

        # taking same amount of test files
        val_count = len(test_files)
        val = sample(train_files, val_count)
        all_val_files = []

        for every_file in val:
            all_val_files.append((every_file, 'val'))
            train_files.remove(every_file)

        [all_train_files.append((every_file, 'train'))for every_file in train_files]

        # list of tuples that has filepath and tag whether it's test, train, or val
        # ie. (filepath, 'val')
        all_data = all_val_files + all_test_files + all_train_files

        make_dataframes = self.open_xmls(all_data)
        self.data_df = make_dataframes[0]
        self.ner_df = make_dataframes[1]
        self.vocab = make_dataframes[2]
        self.id2ner = make_dataframes[3]
        self.id2word = make_dataframes[4]

        self.max_sample_length = self.find_max_sample_length(self.data_df)

        #return 

    
    def pad(self, labelled_sentence, label_length):
    
        original_length = len(labelled_sentence)
        to_add = label_length - original_length
        # 0 to designate padding
        pads = [0] * to_add 

        labelled_sentence.extend(pads)
        
        return labelled_sentence
    
    def make_sentlabels(self, split_df, ner_df, label_length):
    
        # get all sentence_id's
        sentences = list(split_df["sentence_id"].unique())
        
        not_ner = 1 # not_ner = id2ner.get('not_ner')
        sent_labels = []
        
        for every_sentence in sentences:
            sent = [] # for ner labels
            sent_tokens = split_df.loc[split_df['sentence_id'] == every_sentence]
            
            num_tokens = sent_tokens.shape[0]
            
            data_char_start = sent_tokens['char_start_id'].tolist()
            data_char_end = sent_tokens['char_end_id'].tolist()
            
            # list of tuples (each tuple represents a token's char start and char end)
            data_char_offset = list(zip(data_char_start, data_char_end))
            
            ner_entities = ner_df.loc[ner_df['sentence_id'] == every_sentence]
            ner_char_start = ner_entities['char_start_id'].tolist()
            ner_char_end = ner_entities['char_end_id'].tolist()
            ner_labels = ner_entities['ner_id'].tolist()
            
            # [(char start, char end, ner id)]
            ner_char_offset = list(zip(ner_char_start, ner_char_end, ner_labels))

            if len(ner_char_offset) == 0: # no entity in sentence

                for every_token in data_char_offset:
                    sent.append(not_ner)
                    
                # padding    
                if len(sent) < label_length:
                    padded = self.pad(sent, label_length)
                    sent_labels.append(padded)
                else: # same length as max
                    sent_labels.append(sent)
                
            else: # has entity
                for every_token in data_char_offset:
                    for every_entity in ner_char_offset:

                        entity_start = int(every_entity[0])
                        entity_end = int(every_entity[1])
                        ner_id = every_entity[2]

                        token_start = every_token[0]
                        token_end = every_token[1]

                        # check if each entity char offset is within each token in dataframe
                        if ((token_start >= entity_start) and (token_end <= entity_end)):
                            label = ner_id

                        else: # token not an entity
                            label = not_ner
                    sent.append(label)
 
                if len(sent) < label_length:
                    padded = self.pad(sent, label_length)
                    sent_labels.append(padded)
                else: 
                    sent_labels.append(sent)
                
        return sent_labels

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU

        device = torch.device('cuda:1')

        # get split df's
        train_rows = self.data_df.loc[self.data_df['split'] == 'train'] 
        test_rows = self.data_df.loc[self.data_df['split'] == 'test'] 
        dev_rows = self.data_df.loc[self.data_df['split'] == 'val'] 
        
        # get labels for each set
        self.train_labelled = self.make_sentlabels(train_rows, self.ner_df, self.max_sample_length)
        print('train done')
        self.test_labelled = self.make_sentlabels(test_rows, self.ner_df, self.max_sample_length)
        print('test done')
        self.dev_labelled = self.make_sentlabels(dev_rows, self.ner_df, self.max_sample_length)
        print('dev done')
        
        # make tensors and save in gpu
        self.train_tensor = torch.tensor(self.train_labelled)
        self.train_tensor = self.train_tensor.to(device)

        self.test_tensor = torch.tensor(self.test_labelled)
        self.test_tensor = self.test_tensor.to(device)
        
        self.dev_tensor = torch.tensor(self.dev_labelled)
        self.dev_tensor = self.dev_tensor.to(device)
        
        return self.train_tensor, self.dev_tensor, self.test_tensor

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        
        # get labels
        self.get_y()

        # flatten label lists and take out padding label 0 and not_ner (too many)
        flatten = lambda l: [item for sublist in l for item in sublist if item != 0 if item != 1]

        train_flattened = flatten(self.train_labelled)
        dev_flattened = flatten(self.dev_labelled)
        test_flattened = flatten(self.test_labelled)

        # count tokens
        train_counts = Counter(train_flattened)
        dev_counts = Counter(dev_flattened)
        test_counts = Counter(test_flattened)

        all_counts = [train_counts, dev_counts, test_counts]

        for_plotting = pd.DataFrame(all_counts, index=['train', 'dev', 'test'])
        for_plotting.plot.bar(figsize=(10,15))
        print('id2ner', self.id2ner)
        plt.show()

        #pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        
        all_sentences = list(self.data_df["sentence_id"].unique())
        sentence_lengths = []
        
        for every_sentence in all_sentences:
            sent_df = self.data_df.loc[self.data_df['sentence_id'] == every_sentence]
            length = sent_df.shape[0]
            sentence_lengths.append(length)
        
        # for bar graph
        #counted = Counter(sentence_lengths)
        #plt.bar(range(len(counted)), counted.values())
        
        sent_array = np.array(sentence_lengths)
        plt.hist(sent_array);
        plt.show()
        #pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        
        all_sentences = list(self.ner_df["sentence_id"].unique())
        ner_counts = []
        print('Based on ner_df.')
        print('*will have more counts than basing it off of the xml files since an entity with two character offsets are counted as 2 entities in ner_df.')
        for every_sentence in all_sentences:
            sent_df = self.ner_df.loc[self.ner_df['sentence_id'] == every_sentence]
            num_ners = sent_df.shape[0]
            ner_counts.append(num_ners)
        print('Bar Graph')
        counted = Counter(ner_counts)
        plt.bar(range(len(counted)), counted.values())
        plt.show()

        print('Histogram')
        ner_array = np.array(ner_counts)
        plt.hist(ner_array, bins=30);
        plt.show()
        #pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur

        ner_sentences = self.ner_df["sentence_id"].tolist()
        ner_ids = self.ner_df["ner_id"].tolist()
        
        # lists consisting of each sentence_id that has an entity of that group
        brand = [] #ner_id = 2
        drug = [] # 3
        drug_n = [] # 4
        group = [] # 5

        sent_to_ner = zip(ner_sentences, ner_ids)
        
        for sent_id, ner_id in sent_to_ner: 
            if ner_id == 2:
                brand.append(sent_id)
            elif ner_id == 3:
                drug.append(sent_id)
            elif ner_id == 4:
                drug_n.append(sent_id)
            elif ner_id == 5:
                group.append(sent_id)   
        
        # make sure ner_ids are only ner_ids 
        assert list(set(ner_ids))  == [2,3,4,5], 'There are unwanted ner_id\'s!'

        brand = set(brand)
        drug = set(drug)
        drug_n = set(drug_n)
        group = set(group)

        venn({'brand': brand, 'drug': drug, 'drug_n': drug_n, 'group': group})

        pass
