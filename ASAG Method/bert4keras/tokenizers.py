import unicodedata, re
from bert4keras.snippets import is_string, is_py2
from bert4keras.snippets import open
from bert4keras.snippets import convert_to_unicode
from bert4keras.snippets import truncate_sequences
from bert4keras.snippets import lowercase_and_normalize


def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    
    
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict and not Tokenizer._is_redundant(t):
                new_token_dict[t] = len(new_token_dict)
                keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


def save_vocab(dict_path, token_dict, encoding='utf-8'):
    
    
    with open(dict_path, 'w', encoding=encoding) as writer:
        for k, v in sorted(token_dict.items(), key=lambda s: s[1]):
            writer.write(k + '\n')


class TokenizerBase(object):
    
    
    def __init__(
        self,
        token_start='[CLS]',
        token_end='[SEP]',
        pre_tokenize=None,
        token_translate=None
    ):
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        self._pre_tokenize = pre_tokenize
        self._token_translate = token_translate or {}
        self._token_translate_inv = {
            v: k
            for k, v in self._token_translate.items()
        }

    def tokenize(self, text, maxlen=None):
        
        
        tokens = [
            self._token_translate.get(token) or token
            for token in self._tokenize(text)
        ]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            truncate_sequences(maxlen, -index, tokens)

        return tokens

    def token_to_id(self, token):
        
        
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        
        
        return [self.token_to_id(token) for token in tokens]

    def encode(
        self,
        first_text,
        second_text=None,
        maxlen=None,
        pattern='S*E*E',
        truncate_from='right'
    ):
        
        
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen is not None:
            if truncate_from == 'right':
                index = -int(self._token_end is not None) - 1
            elif truncate_from == 'left':
                index = int(self._token_start is not None)
            else:
                index = truncate_from
            if second_text is not None and pattern == 'S*E*E':
                maxlen += 1
            truncate_sequences(maxlen, index, first_tokens, second_tokens)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = second_tokens[idx:]
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        
        
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        
        
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        
        
        raise NotImplementedError

    def _tokenize(self, text):
        
        
        raise NotImplementedError


class Tokenizer(TokenizerBase):
    def __init__(
        self, token_dict, do_lower_case=False, word_maxlen=200, **kwargs
    ):
        super(Tokenizer, self).__init__(**kwargs)
        if is_string(token_dict):
            token_dict = load_vocab(token_dict)

        self._do_lower_case = do_lower_case
        self._token_dict = token_dict
        self._token_dict_inv = {v: k for k, v in token_dict.items()}
        self._vocab_size = len(token_dict)
        self._word_maxlen = word_maxlen

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def token_to_id(self, token):
        
        
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        
        
        return self._token_dict_inv[i]

    def decode(self, ids, tokens=None):
        
        
        tokens = tokens or self.ids_to_tokens(ids)
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = '', False
        for i, token in enumerate(tokens):
            if token[:2] == '
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += ' '
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += ' '
                text += token

        text = re.sub(' +', ' ', text)
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
        punctuation = self._cjk_punctuation() + '+-/={(<['
        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex
        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)

        return text.strip()

    def _tokenize(self, text, pre_tokenize=True):
        
        
        if self._do_lower_case:
            text = lowercase_and_normalize(text)

        if pre_tokenize and self._pre_tokenize is not None:
            tokens = []
            for token in self._pre_tokenize(text):
                if token in self._token_dict:
                    tokens.append(token)
                else:
                    tokens.extend(self._tokenize(token, False))
            return tokens

        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        
        
        if len(word) > self._word_maxlen:
            return [word]

        tokens, start, end = [], 0, 0
        while start < len(word):
            end = len(word)
            while end > start:
                sub = word[start:end]
                if start > 0:
                    sub = '
                if sub in self._token_dict:
                    break
                end -= 1
            if start == end:
                return [word]
            else:
                tokens.append(sub)
                start = end

        return tokens

    @staticmethod
    def stem(token):
        
        
        if token[:2] == '
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_space(ch):
        
        
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'

    @staticmethod
    def _is_punctuation(ch):
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        
        
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        
        
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def _is_redundant(token):
        
        
        if len(token) > 1:
            for ch in Tokenizer.stem(token):
                if (
                    Tokenizer._is_cjk_character(ch) or
                    Tokenizer._is_punctuation(ch)
                ):
                    return True

    def rematch(self, text, tokens):
        
        
        if is_py2:
            text = unicode(text)

        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping


class SpTokenizer(TokenizerBase):
    
    
    def __init__(self, sp_model_path, **kwargs):
        super(SpTokenizer, self).__init__(**kwargs)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        self._token_pad = self.sp_model.id_to_piece(self.sp_model.pad_id())
        self._token_unk = self.sp_model.id_to_piece(self.sp_model.unk_id())
        self._vocab_size = self.sp_model.get_piece_size()

        for token in ['pad', 'unk', 'mask', 'start', 'end']:
            try:
                _token = getattr(self, '_token_%s' % token)
                _token_id = self.sp_model.piece_to_id(_token)
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass

    def token_to_id(self, token):
        
        
        return self.sp_model.piece_to_id(token)

    def id_to_token(self, i):
        
        
        if i < self._vocab_size:
            return self.sp_model.id_to_piece(i)
        else:
            return ''

    def decode(self, ids):
        
        
        tokens = [
            self._token_translate_inv.get(token) or token
            for token in self.ids_to_tokens(ids)
        ]
        text = self.sp_model.decode_pieces(tokens)
        return convert_to_unicode(text)

    def _tokenize(self, text):
        
        
        if self._pre_tokenize is not None:
            text = ' '.join(self._pre_tokenize(text))

        tokens = self.sp_model.encode_as_pieces(text)
        return tokens

    def _is_special(self, i):
        
        
        return self.sp_model.is_control(i) or \
            self.sp_model.is_unknown(i) or \
            self.sp_model.is_unused(i)

    def _is_decodable(self, i):
        
        
        return (i < self._vocab_size) and not self._is_special(i)
