import os
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


print('Successfully imported required packages.')

def generateDocumentDictionary():
    
    corpusroot = 'search-engine/US_Inaugural_Addresses'

    documentsDictionary = {}
    for filename in os.listdir(corpusroot):
        correct_file_name = filename.startswith(('0', '1', '2', '3', '4'))  # More concise check
        
        if correct_file_name:
            column_name = filename[3:-4]  # Extract column name
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                doc = file.read().lower()  # Read and lower case the document
            documentsDictionary[column_name] = doc  # Add to dictionary

    print('Successfully generated document dictionary.')
    return documentsDictionary

def tokenizeDocuments(documentsDictionary):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

    tokens = {}
    for key, value in documentsDictionary.items():
        tokens[key] = tokenizer.tokenize(value)

    print('Successfully tokenized documents.')
    return tokens

def removeStopwords(documentTokens):
    stop_words = stopwords.words('english')

    # print(f"No. of Stop words in english: {len(stop_words)}")

    total_words_b = sum(len(value) for key, value in documentTokens.items())
    print(f"Total words before stop word removal is {total_words_b}.")

    stop_words_count = 0
    for key, value in documentTokens.items():
        l1 = len(value)
        for token in value:
            if token in stop_words:
                value.remove(token)
                
        stop_words_count += l1 - len(value)

    total_words_a = sum(len(value) for key, value in documentTokens.items())
    print(f"Total words after stop word removal is {total_words_a}.")

    # check if the count matches
    if stop_words_count == (total_words_b - total_words_a):
        print("Successfully removed stopwords.")
    return documentTokens

def stemmingTokens(cleanTokens):

    stemmer = PorterStemmer()

    stemmedTokens = {}
    for key, value in cleanTokens.items():
        tokens = []
        for token in value:
            tokens.append(stemmer.stem(token))
        stemmedTokens[key] = tokens
        
    print("Successfully completed stemming.")
    return stemmedTokens

def getUniqueTokens(stemmedTokens):

    uniqueTokens = {}
    for key, value in stemmedTokens.items():
        for token in value:
            if token in uniqueTokens:
                continue
            else:
                uniqueTokens[token] = 1

    print(f"Unique Tokens(no. of rows): {len(uniqueTokens.keys())}")
    return uniqueTokens


def getCountMatrix(uniqueTokens, stemmedTokens):

    countMatrix = pd.DataFrame(index=uniqueTokens.keys(), columns=stemmedTokens.keys())

    for token in uniqueTokens.keys():
        for column in stemmedTokens.keys():
            # count occurrences of token in column and assign to count_matrix[token][column]
            countMatrix.loc[token, column] = stemmedTokens[column].count(token)

    # print(len(countMatrix.columns))
    print('Successfully Generated Term-frequency.')
    return countMatrix

def getDFMatrix(countMatrix):

    df = (countMatrix > 0).sum(axis=1)

    dfMatrix = pd.DataFrame(data=df, columns=['df'])
    print('Successfully generated DF values.')
    return dfMatrix

def getIDFMatrix(dfMatrix, stemmedTokens):
    N = len(stemmedTokens.items())

    for token in dfMatrix.index:
        df = dfMatrix.loc[token, 'df']
        dfMatrix.loc[token, 'idf'] = np.log10(N / df)

    print('Successfully generated IDF values.')
    return dfMatrix

def getWeightedTFMatrix(countMatrix):
    weightedTFMatrix = countMatrix.copy()
    for token in countMatrix.index:
        for column in countMatrix.columns:
            if countMatrix.loc[token, column] == 0:
                continue
            else:
                weightedTFMatrix.loc[token, column] = 1 + np.log10(countMatrix.loc[token, column])

    print('Successfully generated Weighted TF Matrix.') 
    return weightedTFMatrix

def getTFIDFMatrix(countMatrix, weightedTFMatrix, dfMatrix):
    tfidfMatrix = pd.DataFrame(index=countMatrix.index, columns=countMatrix.columns)
    for token in countMatrix.index:
        for column in countMatrix.columns:
            tfidfMatrix.loc[token, column] = (
                weightedTFMatrix.loc[token, column] * dfMatrix.loc[token, 'idf']
            )

    print('Successfullt generated TF-IDF Matrix.')
    return tfidfMatrix

def normalizeTFIDFMatrix(tfidfMatrix):

    # calculate row sum
    rowSum = {}
    for token in tfidfMatrix.index:
        tempSum = 0
        for column in tfidfMatrix.columns:
            tempSum += tfidfMatrix.loc[token, column]
        rowSum[token] = tempSum

    # divide tf-idf value b row sum
    for token in tfidfMatrix.index:
        for column in tfidfMatrix.columns:
            tfidfMatrix.loc[token, column] /= rowSum[token]

    return tfidfMatrix

def processQuery(query):

    query = query.lower()
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    query = tokenizer.tokenize(query)

    # remove stop words from query
    stop_words = stopwords.words('english')
    for token in query:
        if token in stop_words:
            query.remove(token)

    # stemming the query
    stemmer = PorterStemmer()
    for token in query:
        token = stemmer.stem(token)

    # query term frequency
    count = Counter(query)
    print(count)

    for key, value in count.items:
        count[key] = 1 + np.log10(value)

    # normalize 
    countSum= sum(count.values())
    for key, value in count:
        count[key] = value / countSum
    
    return count


if __name__ == "__main__":

    # generate document dictionary
    documentsDictionary = generateDocumentDictionary()

    # tokenize documents
    documentTokens = tokenizeDocuments(documentsDictionary)

    # remove stopwords
    cleanTokens = removeStopwords(documentTokens)

    # stemming using PorterStemmer
    stemmedTokens = stemmingTokens(cleanTokens)

    # generate unique tokens
    uniqueTokens = getUniqueTokens(stemmedTokens)

    # generate count matrix [term-frequency matrix]
    countMatrix = getCountMatrix(uniqueTokens, stemmedTokens)

    # document frequency matrix
    dfMatrix = getDFMatrix(countMatrix)

    # IDF matrix
    dfMatrix = getIDFMatrix(dfMatrix, stemmedTokens)

    # weighted tf matrix
    # tf = 1 + log10(tf)
    weightedTFMatrix = getWeightedTFMatrix(countMatrix)

    # generate TF-IDF matrix
    tfidfMatrix = getTFIDFMatrix(countMatrix, weightedTFMatrix, dfMatrix)

    # normalize TF-IDF Matrix.
    tfidfMatrix = normalizeTFIDFMatrix(tfidfMatrix)

    # get query
    query = input('Enter query:')
    print(f'Query: {query}')

    queryVector = processQuery(query)
    print(f'Query vector: {queryVector}')

    # find cosine similarity
