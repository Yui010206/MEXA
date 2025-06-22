import json
import os


with open("mmau-test.json") as f_in1:
    mmau_test = json.load(f_in1)

task_dict = {item["id"]: item["task"] for item in mmau_test}
gt_answer_dict = {}
for item in mmau_test:
    gt_choice = chr(ord('A') + item["choices"].index(item["answer"]))
    gt_answer_dict[item["id"]] = gt_choice


save_path = './mexa_mmmau/logs'

files = os.listdir(save_path)
type_eval_pred = {"sound":0, "music":0, "speech":0}
type_eval = {"sound":0, "music":0, "speech":0}
for file in files:
    if 'json' not in file:
        continue
    result = json.load(open(os.path.join(save_path, file)))
    q_id = result["q_uid"]
    predict_answer = result["response"]
    gt_answer = gt_answer_dict[q_id]
    q_type = task_dict[q_id]
    if predict_answer == gt_answer:
        type_eval_pred[q_type] += 1
    type_eval[q_type] += 1
result = {}
for each_type in ['sound', 'music', 'speech']:
    result[each_type] = type_eval_pred[each_type]/type_eval[each_type]
avg = sum(list(result.values()))/3