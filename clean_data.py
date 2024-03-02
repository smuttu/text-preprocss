import csv
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from nltk.stem.porter import PorterStemmer

class Process_Text():
    def __init__(self):
        dictionary_path = "models/dictionary.dict"
        self.dictionary = corpora.Dictionary.load(dictionary_path)
  
    def load_stopwords(self):
        stopwords = {}
        with open('stopwords.txt', 'rU') as stop_words_list:
            for line in stop_words_list:
                stopwords[line.strip()] = 1
        return stopwords 

    def extract_lemmatized_words(self, answer):
        stopwords = self.load_stopwords()
        words = []
        # break a given text into lines using sent_tokenize.
        sentences = nltk.sent_tokenize(answer.lower())

        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)

            # removing stop words from wordList

            text = [word for word in tokens if word not in stopwords]
            stemmed = [PorterStemmer().stem(word) for word in text]
            lem = WordNetLemmatizer()
            lemmatized_words = [lem.lemmatize(word) for word in stemmed]

            # Using a Tagger. part-of-speech tagging as nouns, adjective etc

            tagged_text = nltk.pos_tag(lemmatized_words)

            for word, tag in tagged_text:
                words.append({"word": word, "pos": tag})

        nouns = []
        for word in words:
            if word["pos"] in ["NN", "NNS"]:
                nouns.append(word["word"])


        return nouns

    def run(self, answer):
        words = self.extract_lemmatized_words(answer)
        return words

def main(csv_file_path):
    process = Process_Text()

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            answer = row['Answer']
            result = process.run(answer)
            print(f"\n\nResult for '{answer}': {result}\n")

if __name__ == '__main__':
    csv_file_path = "chats_data.csv"  # Replace with the actual path to your CSV file
    main(csv_file_path)
