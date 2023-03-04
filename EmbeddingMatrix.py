import gensim.downloader
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from tqdm import tqdm
import pickle

model_num = 1   #DEFINE THIS PER RUN (1,2,3)
def form_embedding_matrix(clean_train_df, columns, word2vec):
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    stop_words = set(stopwords.words('english'))

    def text_to_vec(df, w2v, vocabulary, inverse_vocabulary):
        numb_represantations = []
        for index, row in tqdm(df.iterrows()):
            questions = []
            for question in columns:
                q2n = []
                for word in row.loc[question].split():
                    # Stopwords have not yet been removed since they might be included in the pretrained word2vec
                    if word in stop_words and word not in w2v.key_to_index:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])
                questions.append(q2n)
            numb_represantations.append(questions)

        return numb_represantations, vocabulary, inverse_vocabulary

    numb_representation_train, vocabulary, inverse_vocabulary = text_to_vec(clean_train_df, word2vec, vocabulary,
                                                                            inverse_vocabulary)
    # May also need to add the test set?

    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
    embeddings[0] = 0

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in word2vec.key_to_index:
            embeddings[index] = word2vec.get_vector(word)

    return embeddings, numb_representation_train

# Load the vec model
if model_num == 1:
    word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
elif model_num == 2:
    pass
    #word2vec = KeyedVectors.load_word2vec_format("crawl-300d-2M.vec", binary=False)
elif model_num == 3:
    pass
    #word2vec = KeyedVectors.load_word2vec_format("crawl-300d-2M-subword.vec", binary=False)


# Create a train test split
#from sklearn.model_selection import train_test_split
#train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
#test_df.to_csv("data/Test set processed.csv", index=False)
#train_df.to_csv("data/Train set processed.csv", index=False)

train_df = pd.read_csv('data/Train set processed.csv').dropna()


embeddings, numb_representation_train = form_embedding_matrix(train_df, ['question1', "question2"], word2vec)

# Save the embeddings to disk
np.save('data/embeddings_matrix_model_{}.npy'.format(model_num), embeddings)

# Save the representations
with open('data/model{}_train.pkl'.format(model_num), 'wb') as fp:
    pickle.dump(numb_representation_train, fp)


