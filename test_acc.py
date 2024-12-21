import json
spth = '/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/eval/1b_mistral/nextqa_8f.sh_20240614_110009/val_global_step0_eval_results.json'

with open(spth, 'r') as file:
    # Load the JSON data into a Python object
    data = json.load(file)
    
total = len(data)
acc = 0
for item in data:
    try:
        gt = item['original_answer'][len('Answer: (')]
        answer = item['answer'][0]
        if gt == answer:
            acc += 1
        else:
            print(gt, answer)
    except:
        continue

print('total: ', total, 'right:', acc , 'Accuracy: ', acc / total)