import math
import re
import string
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import math


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

nltk.download('stopwords')


class Bayes_Classifier:
    def __init__(self):
        self.word_probs = {}
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.class_word_totals = defaultdict(int)

    def preprocess_text(self, text):

        text = text.lower()
        

        text = re.sub(f"[{string.punctuation}]", "", text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        

        bigrams = list(nltk.bigrams(tokens))
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]
        
        return tokens + bigram_tokens

    def train(self, lines):
        word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        total_docs = 0
        
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) < 3:
                continue
            label = parts[0] 
            self.class_counts[label] += 1
            total_docs += 1
            
            words = self.preprocess_text(parts[2])
            for word in words:
                word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)
        print("Done preproc")


        vocab_size = len(self.vocab)
        self.word_probs = {
            label: {
                word: (word_counts[label][word] + 1) / (self.class_word_totals[label] + vocab_size)
                for word in self.vocab
            }
            for label in self.class_counts
        }
        

        for label in self.class_counts:
            self.class_counts[label] /= total_docs
    
    def classify(self, lines):
        predictions = []
        for line in lines:
            parts = line.strip().split('|')
            words = self.preprocess_text( parts[2])
            scores = {}
            
            for label in self.class_counts:
                scores[label] = math.log(self.class_counts[label])
                for word in words:
                    if word in self.word_probs[label]:
                        scores[label] += math.log(self.word_probs[label][word])
                    else:
                        scores[label] += math.log(1 / (self.class_word_totals[label] + len(self.vocab)))

            predictions.append(max(scores, key=scores.get))
        print("PREDICTIONS: ")
        print(predictions)
        return predictions  
# class Bayes_Classifier:

#     def __init__(self):
#         pass


#     def train(self, lines):
#         for line in lines:
#             [rating, id, text] = line.split('|')

#             text = re.sub(f"[{string.punctuation}]", "", text)


#             print(f"{id} with rating {rating}:  {text}")
#         return


#     def classify(self, lines):
#         pass
