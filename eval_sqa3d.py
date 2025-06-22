import os
import json


def answer_match(pred, gts):
    # return EM and refined EM
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0

def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return "what"
    elif question[:2].lower() == 'is':
        return "is"
    elif question[:3].lower() == 'how':
        return "how"
    elif question[:3].lower() == 'can':
        return "can"
    elif question[:5].lower() == 'which':
        return "which"
    else:
        return 5   # others

save_path = ''
sqa_qa_file = "/nas-ssd2/yuezhang/Spartun3D/leo_data/annotations/instruction/sqa3d/v1_balanced_questions_test_scannetv2.json"
files = os.listdir(save_path)
print(len(files))


acc = 0
fuzzy_acc = 0
question_type_dic = {"what":0, "is":0, "how":0, "can":0, "which":0, "others":0}

with open(sqa_qa_file) as sqa_in:
    sqa_data = json.load(sqa_in)

sqa_question_dict = {str(question['question_id']):question["question"] for question in sqa_data['questions']}

for file in files:
    if 'json' not in file:
        continue
    result = json.load(open(os.path.join(save_path, file)))
    question = sqa_question_dict[result['q_uid']]
    question_type = get_sqa_question_type(question.lower())
   
    try:
        question_type_dic[question_type] += 1
    except:
        question_type_dic["others"] += 1
  
    gt = result['truth'][0]['answer'].lower()
#     print(gt, result['response'])

    pred = result['response'].lower()
    
    em, fem = answer_match(pred, [gt])
    #if fem == 1:
    try:
        question_type_dic[question_type] += 1
    except:
        question_type_dic["others"] += 1
    
    print(pred, '|' ,gt, fem)
    
    acc += em
    fuzzy_acc += fem
