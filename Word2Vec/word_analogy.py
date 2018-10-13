import os
import pickle
import numpy as np

# ./score_maxdiff.pl word_analogy_dev_mturk_answers.txt word_analogy_nce.txt output_nce.txt


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce_loss'
from scipy import spatial

model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
input_file = open('word_analogy_test.txt', 'r')
output_file = open('word_analogy_nce.txt', 'w')
result = ""

for line in input_file:
    line.strip()
    given_words, word_pairs = line.split("||")[0], line.split("||")[1]
    cosine_scores_of_given_words = []
    cosine_scores_of_predicted_words = []
    given_word_pairs = given_words.strip().split(",")
    predict_word_pairs = word_pairs.strip().split(",")
    similarity = 0
    for pair in given_word_pairs:
        first_word, second_word = pair.strip().split(":")[0][1:], pair.strip().split(":")[1][:-1]
        embedding_word1 = embeddings[dictionary[first_word]]
        embedding_word2 = embeddings[dictionary[second_word]]
        cosine_scores_of_given_words.append(spatial.distance.cosine(embedding_word1, embedding_word2))
    average_of_given_words = np.mean(cosine_scores_of_given_words)

    for pair in predict_word_pairs:
        pair.strip()
        first_word, second_word = pair.split(":")[0][1:], pair.split(":")[1][:-1]
        embedding_word1 = embeddings[dictionary[first_word]]
        embedding_word2 = embeddings[dictionary[second_word]]
        cosine_scores_of_predicted_words.append(
            abs(average_of_given_words - spatial.distance.cosine(embedding_word1, embedding_word2)))
    max_index = cosine_scores_of_predicted_words.index(max(cosine_scores_of_predicted_words))
    min_index = cosine_scores_of_predicted_words.index(min(cosine_scores_of_predicted_words))
    result += word_pairs.strip().replace(",", " ") + " " + predict_word_pairs[max_index].strip() + " " + \
              predict_word_pairs[min_index] + "\n"

output_file.write(result)
output_file.close()
