import os
import re

def re_text(text):
    text = str(text).strip().split()
    text = " ".join(text)
    text = re.sub(r"<STOP>", "", text)
    text = re.sub(r"  ", " ", text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

# with open('./data/sent/1.1', encoding='utf-8') as f:
#     for l in f:
#         list = l.strip().split(' ')
#         question_id = list[0]
#         answer = ' '.join(list[1:])
#         print(question_id)
#         print(re_text(answer).strip())

question_dict = {}
reference_dict = {}
with open('./dataset/questions', encoding='utf-8') as q:
    for l in q:
        list = l.strip().split(' ')
        question_id = list[0]
        question = ' '.join(list[1:])
        question_dict[question_id] = re_text(question).strip()
    q.close()

with open('./dataset/answers', encoding='utf-8') as r:
    for l in r:
        list = l.strip().split(' ')
        question_id = list[0]
        reference = ' '.join(list[1:])
        reference_dict[question_id] = re_text(reference).strip()
    r.close()

# question reference answer score
def read_file(file_id, text, scoredir_dict):
    answers = []
    scores = []
    with open(answerdir_dict[file_id],encoding='utf-8') as a: 
        with open(scoredir_dict[file_id], encoding='utf-8') as s: 
            for l in a:
                list = l.strip().split(' ')
                answer = re_text(' '.join(list[1:])).strip() 
                answers.append(answer)
            for l in s:
                score = l.strip()
                scores.append(score)
            if len(answers) != len(scores):
                raise ValueError('File Error!')
            for i in range(len(answers)):
                text.writelines([question_dict[file_id], '\t', reference_dict[file_id], '\t', answers[i], '\t', scores[i], '\n'])
            s.close()
        a.close()
    return



scoredir_dict = {}
answerdir_dict = {}
# with open('data.txt', mode='a', encoding='utf-8') as f:
# with open('train.txt', mode='a', encoding='utf-8') as f:
with open('test.txt', mode='a', encoding='utf-8') as f:
    root_path = os.getcwd()
    # dataset_path = os.path.join(root_path, 'dataset\\sent')
    # dataset_path = os.path.join(root_path, 'dataset\\train\\sent')
    dataset_path = os.path.join(root_path, 'dataset\\test\\sent')

    dataset_listdir_name = os.listdir(dataset_path)
    # dataset_listdir_path = [os.path.join(dataset_path, name) for name in dataset_listdir_name]
    for name in dataset_listdir_name:
        answerdir_dict[name] = os.path.join(dataset_path, name)

    # scoredata_path = os.path.join(root_path, 'dataset\\scores')
    # scoredata_path = os.path.join(root_path, 'dataset\\train\\scores')
    scoredata_path = os.path.join(root_path, 'dataset\\test\\scores')
    scoredata_listdir_name = os.listdir(scoredata_path)
    # scoreset_listdir_path = [os.path.join(os.path.join(scoredata_path, name), 'ave') for name in scoredata_listdir_name]
    for name in scoredata_listdir_name:
        scoredir_dict[name] = os.path.join(os.path.join(scoredata_path, name), 'ave')

    for file_id in answerdir_dict:
        read_file(file_id=file_id, text=f, scoredir_dict=scoredir_dict)
    f.close()