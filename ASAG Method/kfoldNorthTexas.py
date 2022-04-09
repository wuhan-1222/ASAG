from bert4keras.models import build_transformer_model
from bert4keras.backend import set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Dropout, Dense, Bidirectional, \
    LSTM, MaxPooling1D, LayerNormalization, BatchNormalization
from tensorflow import keras
import numpy as np
import tensorflow as tf
from bert4keras.layers import MultiHeadAttention
from tensorflow.keras import backend as K
from capsule import Capsule
import math
from sklearn.model_selection import KFold

set_gelu('tanh')

config_path = 'dataset/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'dataset/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'dataset/uncased_L-12_H-768_A-12/vocab.txt'

dataset = './dataset/NorthTexasDataset/data.txt'

maxlen = 128
batch_size = 32
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

classes = {'0':0,'0.5':1,'1':2,'1.5':3,'2':4,'2.5':5,'3':6,
           '3.5':7,'4':8,'4.5':9,'5':10}

tokenizer = Tokenizer(dict_path, do_lower_case=True)
kfold = KFold(n_splits=5,shuffle=True,random_state=seed)

def split_valid(data,valid_percentage,test_percentage):

    length = len(data)
    np.random.shuffle(data)

    train = data[:int(length * (1 - valid_percentage - test_percentage))]
    valid = data[int(length * (1 - valid_percentage - test_percentage)):int(length * (1 - test_percentage))]
    test = data[int(length * (1 - test_percentage)):]
    return train,valid, test

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            q, r, a, label = l.strip().split('\t')
            if label in classes.keys():
                D.append((q, r, a, label))
    return np.asarray(D)

dataset = load_data(dataset)

train, valid, test = split_valid(dataset,0,0.2)

class data_generator(DataGenerator):
    """data generator
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q, r, a, label) in self.sample(random):
            # qa = tokenizer.concat(q, a)
            token_ids, segment_ids = tokenizer.encode(
                r, a, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([int(classes[label])])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
cvscores = []
for train, test in kfold.split(dataset[:,0:3], dataset[:, 3]):
    train = dataset[train].tolist()
    test = dataset[test].tolist()
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        return_keras_model=False,
    )

    lstm = Bidirectional(LSTM(units=200,
                              activation='relu',
                              dropout=0.1,
                              return_sequences=True,
                              kernel_initializer=bert.initializer
                              ))(bert.model.output)
    lstm = LayerNormalization()(lstm)
    cnn = Capsule(num_capsule=maxlen,
                  dim_capsule=400,
                  routings=3,
                  kernel_initializer=bert.initializer)(bert.model.output)
    cnn = LayerNormalization()(cnn)
   
    feature = tf.concat([lstm, cnn], axis=-1)
    att = MultiHeadAttention(heads=2, head_size=400, kernel_initializer=bert.initializer)([feature,feature,feature])
    seq = LayerNormalization()(att)
    output = tf.reduce_max(seq, axis=1)
    output = Dropout(rate=0.1, seed=seed)(output)
    output = Dense(
        units=len(classes), activation='softmax', kernel_initializer=bert.initializer
    )(output)
    model = keras.models.Model(bert.model.input, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(2e-5), 
        # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
        # metrics=['accuracy'],
    )

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    train_generator = data_generator(data=train,batch_size=batch_size)
    valid_generator = data_generator(data=valid,batch_size=batch_size)
    test_generator = data_generator(data=test,batch_size=batch_size)


    def pearson(vector1, vector2):
        n = len(vector1)
        # simple sums
        sum1 = sum(float(vector1[i]) for i in range(n))
        sum2 = sum(float(vector2[i]) for i in range(n))
        # sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in vector1])
        sum2_pow = sum([pow(v, 2.0) for v in vector2])
        # sum up the products
        p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
        num = p_sum - (sum1 * sum2 / n)
        den = math.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if den == 0:
            return 0.0
        return num / den


    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            y_true = y_true[:, 0]
            total += len(y_true)
            right += (y_true == y_pred).sum()
        p = pearson(y_pred, y_true)
        return right / total, math.fabs(p)
        # return right / total, p


    class Evaluator(keras.callbacks.Callback):
        def __init__(self, test):
            self.test = test
            self.best_val_acc = 0.
            self.pear = 0.

        def on_epoch_end(self, epoch, logs=None): 
            val_acc, p = evaluate(self.test)
            print('\n pearson value:', p)
            # if val_acc > self.best_val_acc:
            if p > self.pear:
                # self.best_val_acc = val_acc
                self.pear = p
                model.save_weights('./result/best_model.weights.h5')


    model.fit(train_generator.forfit(),
              steps_per_epoch=len(train_generator),
              epochs=5,
              callbacks=[Evaluator(test_generator)])

    model.load_weights('./result/best_model.weights.h5')
    print('final test acc:')
    acc, pearson_value = evaluate(test_generator)
    print(u'acc: %.5f\n' % (acc))
    print(u'pearson value:',pearson_value)
    cvscores.append(pearson_value)
print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))