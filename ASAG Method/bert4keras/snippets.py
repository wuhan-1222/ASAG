import os, sys, six, re, json
import unicodedata
import logging
import numpy as np
from collections import defaultdict
from bert4keras.backend import K, keras, tf

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


def to_array(*args):
    
    
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def is_string(s):
    
    
    return isinstance(s, basestring)


def strQ2B(ustring):
    
    
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        
        if inside_code == 12288:
            inside_code = 32
        
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def string_matching(s, keywords):
    
    
    for k in keywords:
        if re.search(k, s):
            return True
    return False


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    
    
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    
    
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def lowercase_and_normalize(text):
    
    
    if is_py2:
        text = unicode(text)
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


class open:
    def __init__(
        self, name, mode='r', encoding=None, errors='strict', indexable=False
    ):
        self.name = name
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors
        self.iterator = None
        if indexable:
            if is_string(indexable) and os.path.exists(indexable):
                self.offsets = json.load(_open_(indexable))
            else:
                self.create_indexes()
                if is_string(indexable):
                    json.dump(self.offsets, _open_(indexable, 'w'))

    def create_indexes(self):
        print('creating indexes ...')
        self.offsets, offset = [], 0
        pbar = keras.utils.Progbar(os.path.getsize(self.name))
        while self.readline():
            self.offsets.append(offset)
            offset = self.tell()
            pbar.update(offset)
        self.seek(0)
        print('indexes created.')

    def __getitem__(self, key):
        self.seek(self.offsets[key])
        l = self.readline()
        if self.encoding:
            l = convert_to_unicode(l, self.encoding, self.errors)
        return l

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        if hasattr(self, 'offsets'):
            for i in range(len(self)):
                yield self[i]
        else:
            for l in self.file:
                if self.encoding:
                    l = convert_to_unicode(l, self.encoding, self.errors)
                yield l

    def next(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        return next(self.iterator)

    def __next__(self):
        return self.next()

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readline(self):
        text = self.file.readline()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readlines(self):
        if self.encoding:
            return [
                convert_to_unicode(text, self.encoding, self.errors)
                for text in self.file.readlines()
            ]
        else:
            return self.file.readlines()

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def tell(self):
        return self.file.tell()

    def seek(self, offset=0):
        return self.file.seek(offset)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


def parallel_apply_generator(
    func, iterable, workers, max_queue_size, dummy=False, random_seeds=True
):
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        
        
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                while out_queue.qsize() > max_queue_size:
                    yield out_queue.get()
                    out_count += 1
        if out_queue.qsize() > 0:
            yield out_queue.get()
            out_count += 1

    while out_count != in_count:
        yield out_queue.get()
        out_count += 1

    pool.terminate()


def parallel_apply(
    func,
    iterable,
    workers,
    max_queue_size,
    callback=None,
    dummy=False,
    random_seeds=True,
    unordered=True
):
    generator = parallel_apply_generator(
        func, iterable, workers, max_queue_size, dummy, random_seeds
    )

    if callback is None:
        if unordered:
            return [d for i, d in generator]
        else:
            results = sorted(generator, key=lambda d: d[0])
            return [d for i, d in results]
    else:
        for i, d in generator:
            callback(d)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    
    
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


def truncate_sequences(maxlen, indices, *sequences):
    
    
    sequences = [s for s in sequences if s]
    if not isinstance(indices, (list, tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences


def text_segmentate(text, maxlen, seps='\n', strips=None):
    
    
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def is_one_of(x, ys):
    for y in ys:
        if x is y:
            return True
    return False


class DataGenerator(object):
    
    
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        
        
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    for i in np.random.permutation(len(self.data)):
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d

    def fortest(self, random=False):
        while True:
            for d in self.__iter__(random):
                yield d[0]

    def to_dataset(self, types, shapes, names=None, padded_batch=False):

        if names is None:

            generator = self.forfit

        else:

            if is_string(names):
                warps = lambda k, v: {k: v}
            elif is_string(names[0]):
                warps = lambda k, v: dict(zip(k, v))
            else:
                warps = lambda k, v: tuple(
                    dict(zip(i, j)) for i, j in zip(k, v)
                )

            def generator():
                for d in self.forfit():
                    yield warps(names, d)

            types = warps(names, types)
            shapes = warps(names, shapes)

        if padded_batch:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types
            )
            dataset = dataset.padded_batch(self.batch_size, shapes)
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_types=types, output_shapes=shapes
            )
            dataset = dataset.batch(self.batch_size)

        return dataset


class ViterbiDecoder(object):
    
    
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        
        
        
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)

        
        return paths[:, scores[:, 0].argmax()]


def softmax(x, axis=-1):
    
    
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)


class AutoRegressiveDecoder(object):
    def __init__(self, start_id, end_id, maxlen, minlen=1):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
        else:
            self.first_output_ids = np.array([[self.start_id]])

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        def actual_decorator(predict):
            def new_predict(
                self,
                inputs,
                output_ids,
                states,
                temperature=1,
                rtype=default_rtype
            ):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (
                        softmax(prediction[0] / temperature), prediction[1]
                    )
                elif temperature != 1:
                    probas = np.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return np.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    def last_token(self, model):
        
        
        if model not in self.models:
            outputs = [
                keras.layers.Lambda(lambda x: x[:, -1])(output)
                for output in model.outputs
            ]
            self.models[model] = keras.models.Model(model.inputs, outputs)

        return self.models[model]

    def predict(self, inputs, output_ids, states=None):
        raise NotImplementedError

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):
            scores, states = self.predict(
                inputs, output_ids, states, temperature, 'logits'
            )  
            if step == 0:  
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  
            indices = scores.argpartition(-topk, axis=None)[-topk:]  
            indices_1 = indices // scores.shape[1]  
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  
            output_ids = np.concatenate([output_ids[indices_1], indices_2],
                                        1)  
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  
            is_end = output_ids[:, -1] == self.end_id  
            end_counts = (output_ids == self.end_id).sum(1)  
            if output_ids.shape[1] >= self.minlen:  
                best = output_scores.argmax()  
                if is_end[best] and end_counts[best] >= min_ends:  
                    return output_ids[best]  
                else:  
                    flag = ~is_end | (end_counts < min_ends)  
                    if not flag.all():  
                        inputs = [i[flag] for i in inputs]  
                        output_ids = output_ids[flag]  
                        output_scores = output_scores[flag]  
                        end_counts = end_counts[flag]  
                        topk = flag.sum()  
        
        return output_ids[output_scores.argmax()]

    def random_sample(
        self,
        inputs,
        n,
        topk=None,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1
    ):
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(
                inputs, output_ids, states, temperature, 'probas'
            )  
            probas /= probas.sum(axis=1, keepdims=True)  
            if step == 0:  
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk is not None:
                k_indices = probas.argpartition(-topk,
                                                axis=1)[:, -topk:]  
                probas = np.take_along_axis(probas, k_indices, axis=1)  
                probas /= probas.sum(axis=1, keepdims=True)  
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]  
                probas = np.take_along_axis(probas, p_indices, axis=1)  
                cumsum_probas = np.cumsum(probas, axis=1)  
                flag = np.roll(cumsum_probas >= topp, 1, axis=1)  
                flag[:, 0] = False  
                probas[flag] = 0  
                probas /= probas.sum(axis=1, keepdims=True)  
            sample_func = lambda p: np.random.choice(len(p), p=p)  
            sample_ids = np.apply_along_axis(sample_func, 1, probas)  
            sample_ids = sample_ids.reshape((-1, 1))  
            if topp is not None:
                sample_ids = np.take_along_axis(
                    p_indices, sample_ids, axis=1
                )  
            if topk is not None:
                sample_ids = np.take_along_axis(
                    k_indices, sample_ids, axis=1
                )  
            output_ids = np.concatenate([output_ids, sample_ids], 1)  
            is_end = output_ids[:, -1] == self.end_id  
            end_counts = (output_ids == self.end_id).sum(1)  
            if output_ids.shape[1] >= self.minlen:  
                flag = is_end & (end_counts >= min_ends)  
                if flag.any():  
                    for ids in output_ids[flag]:  
                        results.append(ids)
                    flag = (flag == False)  
                    inputs = [i[flag] for i in inputs]  
                    output_ids = output_ids[flag]  
                    end_counts = end_counts[flag]  
                    if len(output_ids) == 0:
                        break
        
        for ids in output_ids:
            results.append(ids)
        
        return results


def insert_arguments(**arguments):
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def longest_common_substring(source, target):
    c, l, span = defaultdict(int), 0, (0, 0, 0, 0)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
                if c[i, j] > l:
                    l = c[i, j]
                    span = (i - l, i, j - l, j)
    return l, span


def longest_common_subsequence(source, target):
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


def orthogonally_resize(a, new_shape, window=2):
    
    
    assert a.ndim == len(new_shape)
    slices, a_norm, w = [], np.linalg.norm(a), window
    for i, (d1, d2) in enumerate(zip(a.shape, new_shape)):
        if d1 != d2:
            k = d2 // d1 + int(d2 % d1 != 0)
            if k > 1:
                assert d1 % w == 0
                a = a.reshape(a.shape[:i] + (d1 // w, w) + a.shape[i + 1:])
                a = np.repeat(a, k, axis=i)
                a = a.reshape(a.shape[:i] + (d1 * k,) + a.shape[i + 2:])
        slices.append(np.s_[:d2])
    a = a[tuple(slices)]
    return a / np.linalg.norm(a) * a_norm


class WebServing(object):
    
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.graph = tf.get_default_graph()
        self.sess = K.get_session()
        self.set_session = K.set_session
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                with self.graph.as_default():
                    self.set_session(self.sess)
                    outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        
        
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        
        
        self.bottle.run(host=self.host, port=self.port, server=self.server)


class Hook:
    
    
    def __init__(self, module):
        self.module = module

    def __getattr__(self, attr):
        if attr == 'uniout':
            if is_py2:
                import uniout
        else:
            return getattr(self.module, attr)


Hook.__name__ = __name__
sys.modules[__name__] = Hook(sys.modules[__name__])
del Hook
