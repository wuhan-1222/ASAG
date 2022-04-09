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
from Bi_Attention import BiAttentionLayer
from capsule import Capsule
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, \
    precision_recall_curve, average_precision_score, cohen_kappa_score
from sklearn.preprocessing import label_binarize

from matplotlib import pyplot as plt

config_path = 'dataset/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'dataset/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'dataset/uncased_L-12_H-768_A-12/vocab.txt'

dataset = './dataset/NorthTexasDataset/data.txt'

maxlen = 128
batch_size = 64
word_embedding_dim = 300
seed = 20
np.random.seed(seed)
tf.random.set_seed(seed)

classes = {'0':0,'0.5':1,'1':2,'1.5':3,'2':4,'2.5':5,'3':6,
           '3.5':7,'4':8,'4.5':9,'5':10}

tokenizer = Tokenizer(dict_path, do_lower_case=True)

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
                D.append((q, r, a, int(classes[label])))
    f.close()
    return D

dataset = load_data(dataset)
train, valid, test = split_valid(dataset,0,0.2)

class data_generator(DataGenerator):

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q, r, a, label) in self.sample(random):
            # qa = tokenizer.concat(q, a)
            token_ids, segment_ids = tokenizer.encode(
                r, a, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_labels = sequence_padding(batch_labels)
                # batch_labels = prefix_hot(batch_labels,depth=len(classes))
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=False,
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
# output = Dropout(rate=0.1, seed=seed)(bert.model.output)
output = Dense(
    units=len(classes), activation='softmax', kernel_initializer=bert.initializer
)(output)
model = keras.models.Model(bert.model.input, output)

def prefix_hot(indices, depth):
    p = []
    for i in indices:
        m = np.zeros([depth])
        for j in range(int(i) + 1):
            m[j] = 1
        # print(m)
        p.append(m)
    p = np.asarray(p)
    p = tf.convert_to_tensor(p)
    return p
def convert_to_tags(prefix_hot):
    matrix = tf.get_static_value(prefix_hot)
    tags = []
    for tag in matrix:
        l = int(tag.sum()-1)
        tags.append(l)
    return np.asarray(tags)

def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def MAE(y_true, y_pred):
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae
def ordinal_binary_crossentropy(y_true, y_pred):
    lam = K.mean(y_true)
    # y_true = tf.one_hot(y_true,depth=len(classes))
    return K.mean(K.cast_to_floatx(lam) * K.binary_crossentropy(y_true, y_pred), axis=-1)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(2e-5), 
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    # metrics=['mse','mae'],
)

train_generator = data_generator(data=train,batch_size=batch_size)
valid_generator = data_generator(data=valid,batch_size=batch_size)
test_generator = data_generator(data=test,batch_size=batch_size)

def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

def evaluate(data):
    total, right = 0., 0.
    pred, true = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        # pred.extend(y_pred.tolist())
        # true.extend(y_true.tolist())
        # right += (y_true == y_pred).sum()
        for i in range(len(y_pred)):
            if abs(y_pred[i] - y_true[i]) <= 1:
                right += 1
                pred.append(y_true[i])
                true.append(y_true[i])
            else:
                pred.append(y_pred[i])
                true.append(y_true[i])

    macro = f1_score(y_true=true, y_pred=pred, average='macro')
    weighted = f1_score(y_true=true, y_pred=pred, average='weighted')
    p = pearson(pred, true)
    rmse = math.sqrt(mean_squared_error(true,pred))
    mae = mean_absolute_error(true,pred)
    # return right / total, math.fabs(p)
    return right / total, p, rmse, mae, macro, weighted

class Evaluator(keras.callbacks.Callback):
    def __init__(self, test):
        self.test = test
        self.best_val_acc = 0.
        self.pear = 0.

    def on_epoch_end(self, epoch, logs=None): 
        val_acc, p,_,_ ,_ ,_= evaluate(self.test)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('./result/NorthTexas/best_model.weights.h5')

model.fit(train_generator.forfit(),
           steps_per_epoch=len(train_generator),
          epochs=10,
          callbacks=[Evaluator(test_generator)])

model.load_weights('./result/NorthTexas/best_model.weights.h5')

print('final test acc:')
acc, pearson_value, rmse, mae, macro, weighted = evaluate(test_generator)
print(u'acc: %.5f\n' % (acc))
print(u'pearson value:',pearson_value)
print(u'mae value:',mae)
print(u'rmse value:',rmse)
print(u'macro value:',macro)
print(u'weighted value:',weighted)

def PR_figure(data):
    pred, true = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)  # batch, num_classes
        y_true = y_true[:, 0]  # batch,1
        y_true = label_binarize(y_true, classes=range(len(classes)))

        true.append(y_true)
        pred.append(y_pred)

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    precision["macro"], recall["macro"], _ = precision_recall_curve(y_true.ravel(),
                                                                    y_pred.ravel())

    average_precision["macro"] = average_precision_score(y_true, y_pred,
                                                         average="macro")

    print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))

    plt.figure()
    plt.step(recall['macro'], precision['macro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, macro-averaged over all classes:')
    plt.show()


PR_figure(test_generator)

def Cohen_Kappa_Score(data):

    pred, true = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0] # batch,1

        true.extend(y_true)
        pred.extend(y_pred)

    return cohen_kappa_score(true, pred)

print('Cohen\'s kappa score: %.2f' % Cohen_Kappa_Score(test_generator))