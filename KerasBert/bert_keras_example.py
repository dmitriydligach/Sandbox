# source: https://github.com/clulab/transformers/blob/master/src/main/python/bert_keras_example.py

import numpy as np
import tensorflow as tf
import transformers


# required according to https://github.com/huggingface/transformers/issues/1350
class BERT(transformers.TFBertModel):
    def __init__(self, config, *inputs, **kwargs):
        super(BERT, self).__init__(config, *inputs, **kwargs)
        self.bert.call = tf.function(self.bert.call)


# converts texts into input matrices required by BERT
def to_inputs(tokenizer, texts, pad_token=0):
    rows = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    shape = (len(rows), max(len(row) for row in rows))
    token_ids = np.full(shape=shape, fill_value=pad_token)
    is_token = np.zeros(shape=shape)
    for i, row in enumerate(rows):
        token_ids[i, :len(row)] = row
        is_token[i, :len(row)] = 1
    return dict(word_inputs=token_ids, mask_inputs=is_token, segment_inputs=np.zeros(shape=shape))


# Define some script parameters
N_OUTPUTS = 2
LEARNING_RATE = 3e-5
N_EPOCHS = 10

# Define inputs (token_ids, mask_ids, segment_ids)
token_inputs = tf.keras.Input(shape=(None,), name='word_inputs', dtype='int32')
mask_inputs = tf.keras.Input(shape=(None,), name='mask_inputs', dtype='int32')
segment_inputs = tf.keras.Input(shape=(None,), name='segment_inputs', dtype='int32')

# Load model and collect encodings
bert = BERT.from_pretrained('bert-base-cased')
token_encodings = bert([token_inputs, mask_inputs, segment_inputs])[0]

# Keep only [CLS] token encoding
sentence_encoding = tf.squeeze(token_encodings[:, 0:1, :], axis=1)

# Apply dropout
sentence_encoding = tf.keras.layers.Dropout(0.1)(sentence_encoding)

# Final output layer
outputs = tf.keras.layers.Dense(N_OUTPUTS, activation='sigmoid', name='outputs')(sentence_encoding)

# Define model
model = tf.keras.Model(inputs=[token_inputs, mask_inputs, segment_inputs], outputs=[outputs])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08, clipnorm=1.0),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Train model on two made-up examples
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
x = to_inputs(tokenizer, [
    "I love that!",
    "This is my favorite.",
    "I hate this.",
    "That is disgusting!"])
y = np.array([1, 1, 0, 0])
model.fit(x=x, y=y, epochs=N_EPOCHS)

# Test model on two slightly changed examples
x = to_inputs(tokenizer, [
    "I like this.",
    "I do not like that!"])
y = np.array([1, 0])
model.evaluate(x=x, y=y)
