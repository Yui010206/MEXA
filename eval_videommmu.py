import os
import json

def get_cache_dir(subject):
    if subject in ["Art", "Art_Theory", "Design", "Music"]:
        return "Art"
    elif subject in ["Biology", "Chemistry", "Geography", "Math", "Physics"]:
        return "Science"
    elif subject in ["History", "Literature", "Sociology", "Psychology"]:
        return "Humanities"
    elif subject in ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"]:
        return "Engineering"
    elif subject in ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"]:
        return "Medicine"
    elif subject in ["Accounting", "Economics", "Finance", "Manage", "Marketing"]:
        return "Business"
    else:
        raise ValueError(f"Subject {subject} not recognized.")
    


save_path = 'mexa_videommmu/logs/'
files = os.listdir(save_path)
# print(len(files))

acc = 0
result_dict_count = {
    'Overall' : 0,
    'Adaptation' : 0,
    'Perception' : 0,
    'Comprehension' : 0,
    'Art' : 0,
    'Science' : 0,
    'Humanities' : 0,
    'Engineering' : 0,
    'Medicine': 0,
    'Business' : 0,
}

result_dict_acc = {
    'Overall' : 0,
    'Adaptation' : 0,
    'Perception' : 0,
    'Comprehension' : 0,
    'Art' : 0,
    'Science' : 0,
    'Humanities' : 0,
    'Engineering' : 0,
    'Medicine': 0,
    'Business' : 0,
}


for file in files:
    if 'json' not in file:
        continue
    result = json.load(open(os.path.join(save_path, file)))
    gt = result['truth']
    
    pred = result['response'][0]

    subtype1 = result['q_uid'].split('_')[0]

    subtype2 = get_cache_dir('_'.join(result['id'].split('_')[1:-1]))
    result_dict_count['Overall'] += 1
    result_dict_count[subtype1] += 1
    result_dict_count[subtype2] += 1
    
    if gt==pred:
        acc += 1
        result_dict_acc['Overall'] += 1
        result_dict_acc[subtype1] += 1
        result_dict_acc[subtype2] += 1
        
        
for tp in result_dict_acc:
    print(tp, result_dict_acc[tp] / result_dict_count[tp] * 100)