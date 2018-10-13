import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    A = tf.log(tf.exp(tf.reduce_sum(tf.multiply(inputs, true_w), axis=1)) + 1e-10)
    reduced_sum = tf.reduce_sum(tf.exp(tf.matmul(inputs, tf.transpose(true_w))), axis=1)
    B = tf.log(reduced_sum + 1e-10)
    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    sample_shape = len(sample)
    batch_size = labels.shape

    unigram_prob = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    unigram_prob_labels = tf.nn.embedding_lookup(unigram_prob, labels)
    unigram_prob_noise_words = tf.tile(tf.transpose(tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), [-1, 1])),
                                       [batch_size[0], 1])

    embeddings_label = tf.reshape(tf.nn.embedding_lookup(weights, labels), [-1, inputs.shape[1]])
    embeddings_noise_words = tf.transpose(tf.nn.embedding_lookup(weights, sample))

    bias_true_word_sampling = tf.nn.embedding_lookup(biases, labels)
    bias_negative_sampling = tf.transpose(tf.reshape(tf.nn.embedding_lookup(biases, sample), [sample_shape, 1]))
    bias_negative_sampling = tf.tile(bias_negative_sampling, [batch_size[0], 1])

    product_of_context_target = tf.reshape(tf.reduce_sum(tf.multiply(inputs, embeddings_label), axis=1),
                                           [-1, 1])
    product_of_noise_distribution_target = tf.matmul(inputs, embeddings_noise_words)

    temp_A1 = tf.add(product_of_context_target, bias_true_word_sampling)
    temp_A2 = tf.log(tf.scalar_mul(sample_shape, unigram_prob_labels) + 1e-10)
    A = tf.log(tf.sigmoid(tf.subtract(temp_A1, temp_A2)) + 1e-10)

    temp_B1 = tf.add(product_of_noise_distribution_target, bias_negative_sampling)
    temp_B2 = tf.log(tf.scalar_mul(sample_shape, unigram_prob_noise_words) + 1e-10)
    B = tf.reshape(
        tf.reduce_sum(tf.log(1 - tf.sigmoid(tf.subtract(temp_B1, temp_B2)) + 1e-10), axis=1),
        [-1, 1])

    return tf.scalar_mul(-1, tf.add(A, B))
