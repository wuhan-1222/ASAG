from bert4keras.models import build_transformer_model
from bert4keras.backend import set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from tensorflow.keras.layers import Dropout, Dense, Bidirectional, \
    LSTM, LayerNormalization
from tensorflow import keras
import numpy as np
import tensorflow as tf
from bert4keras.layers import MultiHeadAttention
from capsule import Capsule
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, cohen_kappa_score

from sklearn.preprocessing import label_binarize

from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

set_gelu('tanh')

config_path = 'dataset/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'dataset/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'dataset/uncased_L-12_H-768_A-12/vocab.txt'

train_data = './dataset/ScientsBank_3way/train.txt'
test_ua_data = './dataset/ScientsBank_3way/test_ua.txt'
test_uq_data = './dataset/ScientsBank_3way/test_uq.txt'
test_ud_data = './dataset/ScientsBank_3way/test_ud.txt'

# train_data = './dataset/ScientsBank_5way/train.txt'
# test_ua_data = './dataset/ScientsBank_5way/test_ua.txt'
# test_uq_data = './dataset/ScientsBank_5way/test_uq.txt'
# test_ud_data = './dataset/ScientsBank_5way/test_ud.txt'

maxlen = 128
batch_size = 64
seed = 10
np.random.seed(seed)
tf.random.set_seed(seed)

classes = {'correct':0,'incorrect':1,'contradictory':2}

tokenizer = Tokenizer(dict_path, do_lower_case=True)

def split_valid(data,valid_percentage):

    length = len(data)
    np.random.shuffle(data)
    train = data[:int(length * (1 - valid_percentage))]
    test = data[int(length * (1 - valid_percentage)):]
    return train,test

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            q, r, a, label = l.strip().split('\t')
            D.append((q, r, a, int(classes[label])))
    f.close()
    return D

train = load_data(train_data)
test_ua = load_data(test_ua_data)
test_uq = load_data(test_uq_data)
test_ud = load_data(test_ud_data)
train, valid = split_valid(train, 0.1)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q, r, a, label) in self.sample(random):
            # qa = tokenizer.concat(q, a)
            # qr = tokenizer.concat(q, r)
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
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

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
output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=len(classes), activation='softmax', kernel_initializer=bert.initializer
)(output)
model = keras.models.Model(bert.model.input, output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(2e-5), 
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)

train_generator = data_generator(data=train,batch_size=batch_size)
test_ua_generator = data_generator(data=test_ua,batch_size=batch_size)
test_uq_generator = data_generator(data=test_uq,batch_size=batch_size)
test_ud_generator = data_generator(data=test_ud,batch_size=batch_size)
valid_generator = data_generator(data=valid,batch_size=batch_size)


def evaluate(data):
    total, right = 0., 0.
    pred, true = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        pred.extend(y_pred.tolist())
        true.extend(y_true.tolist())
        right += (y_true == y_pred).sum()

    macro = f1_score(y_true=true, y_pred=pred, average='macro')
    weighted = f1_score(y_true=true, y_pred=pred, average='weighted')
    return right / total, macro, weighted

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.
    def on_epoch_end(self, epoch, logs=None):
        acc,_,_ = evaluate(valid_generator)
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            model.save_weights('./result/best_model.weights.h5')

model.fit(train_generator.forfit(),
          steps_per_epoch=len(train_generator),
          epochs=15,
          callbacks=[Evaluator()])
model.load_weights('./result/best_model.weights.h5')

print('final test acc:')
acc_1, macro_1, weighted_1 = evaluate(test_ua_generator)
acc_2, macro_2, weighted_2 = evaluate(test_uq_generator)
acc_3, macro_3, weighted_3 = evaluate(test_ud_generator)
print(u'ua_acc: %.5f, weighted_ua: %.5f, macro_ua: %.5f\n' % (acc_1,weighted_1,macro_1))
print(u'uq_acc: %.5f, weighted_uq: %.5f, macro_uq: %.5f\n' % (acc_2,weighted_2,macro_2))
print(u'ud_acc: %.5f, weighted_ud: %.5f, macro_ud: %.5f\n' % (acc_3,weighted_3,macro_3))

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
    plt.title('Average precision score, macro-averaged over all classes:')
    plt.show()

PR_figure(test_ua_generator)
PR_figure(test_uq_generator)
PR_figure(test_ud_generator)

def Cohen_Kappa_Score(data):

    pred, true = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0] # batch,1

        true.extend(y_true)
        pred.extend(y_pred)

    return cohen_kappa_score(true, pred)

print('ua Cohen\'s kappa score: %.2f' % Cohen_Kappa_Score(test_ua_generator))
print('uq Cohen\'s kappa score: %.2f' % Cohen_Kappa_Score(test_uq_generator))
print('ud Cohen\'s kappa score: %.2f' % Cohen_Kappa_Score(test_ud_generator))