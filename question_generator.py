# -*- coding: utf-8 -*-
"""
@author: Clive Gomes <cliveg@andrew.cmu.edu>
@description: Simple Question Generator
"""

# Import Libraries
from textblob import TextBlob
import neattext.functions as nxf
import contractions
from flair.data import Sentence
from flair.models import MultiTagger
import regex as re
import numpy as np

# Question Generator Class
class QuesGen:
    
    # No initialization is required here
    def __init__(self):
        pass
        
    # Returns an ordered list of chunk tags (like NP) for the sentence
    def __chunk_tags(self, sentence):
        return [x.tag for x in sentence.get_spans('chunk-fast')]
    
    # Returns an ordered list of NER tags (like DATE) for the sentence    
    def __ner_tags(self, sentence):
        return [x.tag for x in sentence.get_spans('ner-ontonotes-fast')]
    
    # Returns an ordered list of chunks in the sentence
    def __chunks(self, sentence):
        return [x.text for x in sentence.get_spans('chunk-fast')]
    
    # Returns an ordered list of NER entities in the sentence
    def __ners(self, sentence):
        return [x.text for x in sentence.get_spans('ner-ontonotes-fast')]
    
    # Returns a list of indices for a specific chunk in a sentence
    def __chunk_idxs(self, sentence, chunk):
        idxs = []
        phrases = self.__chunks(sentence)
        
        for i in range(len(phrases)):
            if chunk in phrases[i]:
                idxs.append(i)
                
        return idxs
    
    # Main function to generate questions
    def ask(self, passage, N, verbose=True):
        
        # Perform Sentence Tokenization
        if verbose:
            print('Tokenizing Sentences...')
            
        passage = TextBlob(passage) 
        sentences = [sentence.raw for sentence in passage.sentences]
        
        # Clean Text
        if verbose:
            print('Cleaning Text...')
            
        for i in range(len(sentences)):
            # Replace contractions with complete phrases
            sentences[i] = contractions.fix(sentences[i])
            
            # Remove problematic characters
            sentences[i] = nxf.remove_non_ascii(sentences[i])
            sentences[i] = nxf.remove_multiple_spaces(sentences[i])
            sentences[i] = nxf.remove_custom_pattern(sentences[i], r'[.!?]')
            
        # Perform POS Chunking & NER Tagging
        if verbose:
            print('Loading Taggers...')
            print('\n----------')
            
        tagger = MultiTagger.load(['chunk-fast', 'ner-ontonotes-fast'], )
        if verbose:
            print('----------\n')
            print('Tagging Sentences...')

        sentences = [Sentence(sentence) for sentence in sentences]
        for sentence in sentences:
            tagger.predict(sentence)
            
            
        # Generate Questions
        if verbose:
            print('Generating Questions')
            
        questions = []
        for sentence in sentences: 
            # Get tags and entities
            tags = self.__chunk_tags(sentence)
            tags_ner = self.__ner_tags(sentence)
            phrases = self.__chunks(sentence)
            entities = self.__ners(sentence)
                   
              
            ## Word-based Patterns
            # Generate questions from Original Sentence
            
            # 'because'
            bec_split = sentence.to_original_text().split('because')
            if len(bec_split) > 1:
                questions.append('Why ' + bec_split[0][0].lower() + bec_split[0][1:] + '?')
            
            # ' is in '
            if ' is in ' in sentence.tokenized:
                x,y = re.split(r' is ', sentence.tokenized)
                questions.append('Where is ' + x[0].lower() + x[1:] + '?')
                questions.append('What can be found in ' + y + '?')
            
            # ' has '
            if ' has ' in sentence.tokenized:
                x,y = re.split(r' has ', sentence.tokenized)
                questions.append('Who has ' + y + '?')
        
            # ' have '
            if ' have ' in sentence.tokenized:
                x,y = re.split(r' have ', sentence.tokenized)
                questions.append('Who have ' + y + '?')
            
            
            ## Tag-based Rules (Chunks, NER)
            # Generate questions from Tagged Sentence
            
            v_idxs = list(np.where(np.array(tags) == 'VP')[0]) # Indices of Verb Phrases
            
            # "<NP> <VP> <NP>." or "<NP> <VP> <NP> <NP> ..." 
            v_idx = v_idxs[0]
            if tags[v_idx-1] == 'NP' and tags[v_idx+1] == 'NP':
                
                if v_idx+2 < len(tags) and (tags[v_idx+2] == 'NP'):
                    questions.append('What ' + phrases[v_idx] + ' ' + phrases[v_idx+1] + '?')
                elif v_idx+2 == len(tags): 
                    questions.append('What ' + phrases[v_idx] + ' ' + phrases[v_idx+1] + '?')
                    
            # <NP> ... <VP> ... <DATE>
            if 'DATE' in tags_ner:
                date_nidx = tags_ner.index('DATE')
                date_cidx = self.__chunk_idxs(sentence, entities[date_nidx])[-1]
                v_idx = 0
                for idx in reversed(v_idxs):
                    if idx < date_cidx:
                        v_idx = idx
                        break
                
                question = 'What'
                for phrase in phrases[v_idx:date_cidx+1]:
                    question += ' ' + str(phrase)
                questions.append(question + '?')
                
                
        # Return Top N Questions
        if len(questions) >= N:
            if verbose:
                print('Successfully Generated', N, 'Questions!')
            return questions[:N]
        else:
            if verbose:
                print('Successfully Generated', len(questions), 'Questions!')
            return questions
        
    # Display list of questions as a numbered list
    def display(self, questions):
        print('Top', len(questions), 'Questions-->')
        for i, question in enumerate(questions):
            print(str(i+1) + ') ' + question)
        
        
# Main Routine
if __name__=='__main__':
    
    # Example Input Passage
    passage = '''The Yukon Quest is a 1,000-mile dog sled race that takes place 
    every February. The race is run between the cities of Fairbanks, Alaska and 
    Whitehorse in Canada’s Yukon Territory. Sled dogs and their mushers, or 
    drivers, race through extreme weather conditions. They also race over 
    mountains and frozen rivers. The Yukon Quest is considered the toughest sled 
    dog race in the world. Teams consist of a musher and up to 14 dogs. The 
    musher guides and controls the sled. The dogs closest to the sled are called 
    wheel dogs. These dogs must have a calm nature because they run close to a 
    moving sled. Team dogs are next on the towline. They provide power to the 
    dog sled team. Swing dogs are next in position after team dogs. They help 
    guide the dogs around curves on a trail. Lead dogs are at the front of the 
    team. They help steer the entire dog team and keep the pace. There can be a 
    single lead dog or double lead dogs.'''
    
    # Number of questions to generate
    N = 5
    
    # Generate N Questions
    qgen = QuesGen()
    questions = qgen.ask(passage, N)
    qgen.display(questions)

