import xml.etree.ElementTree as ET
import os

def read_file(xml_file, text):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    references, answer, answer_accuracy = [], [], []
    question_node = root.find('questionText')
    reference_nodes = root.findall('referenceAnswers')
    for reference_node in reference_nodes:
        reference_sub_node = reference_node.findall('referenceAnswer')
        for reference_three_node in reference_sub_node:
            references.append(reference_three_node.text)

    answer_nodes = root.findall('studentAnswers')
    for answer_node in answer_nodes:
        answer_sub_node = answer_node.findall('studentAnswer')
        for answer_three_node in answer_sub_node:
            answer_accuracy.append(answer_three_node.attrib['accuracy'])
            answer.append(answer_three_node.text)

    for i in range(len(references)):
        for j in range(len(answer)):
            text.writelines(
                [question_node.text, '\t', references[i], '\t', answer[j], '\t', answer_accuracy[j], '\n'])

    return

# with open('./dataset/test_3way_SciEntsBank_ua.txt',encoding='utf-8',mode='a') as f:
# with open('./dataset/test_3way_SciEntsBank_uq.txt',encoding='utf-8',mode='a') as f:
# with open('./dataset/test_3way_SciEntsBank_ud.txt',encoding='utf-8',mode='a') as f:
# with open('./dataset/train_3way_SciEntsBank.txt', encoding='utf-8', mode='a') as f

# with open('./dataset/train_5way_scientsbank.txt',encoding='utf-8',mode='a') as f:
# with open('./dataset/test_5way_ua_scientsbank.txt', encoding='utf-8', mode='a') as f:
# with open('./dataset/test_5way_uq_scientsbank.txt', encoding='utf-8', mode='a') as f:
with open('./dataset/test_5way_ud_scientsbank.txt', encoding='utf-8', mode='a') as f:
    root_path = os.getcwd()
    # dataset_path = os.path.join(root_path,'dataset/semeval2013-Task7-2and3way/test/3way/sciEntsBank/test-unseen-answers')
    # dataset_path = os.path.join(root_path,'dataset/semeval2013-Task7-2and3way/test/3way/sciEntsBank/test-unseen-domains')
    # dataset_path = os.path.join(root_path,'dataset/semeval2013-Task7-2and3way/test/3way/sciEntsBank/test-unseen-questions')
    # dataset_path = os.path.join(root_path, 'dataset/semeval2013-Task7-2and3way/training/3way/sciEntsBank')

    # dataset_path = os.path.join(root_path,'./dataset/semeval2013-Task7-5way/sciEntsBank/train/Core')
    # dataset_path = os.path.join(root_path,'./dataset/semeval2013-Task7-5way/sciEntsBank/test-unseen-answers/Core')
    # dataset_path = os.path.join(root_path,'./dataset/semeval2013-Task7-5way/sciEntsBank/test-unseen-questions/Core')
    dataset_path = os.path.join(root_path, './dataset/semeval2013-Task7-5way/sciEntsBank/test-unseen-domains/Core')
    dataset_listdir_name = os.listdir(dataset_path)
    dataset_listdir_path = [os.path.join(dataset_path,name) for name in dataset_listdir_name]
    for file in dataset_listdir_path:
        read_file(xml_file=file,text = f)
    f.close()