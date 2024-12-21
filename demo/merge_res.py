import os
import json
import sys

root_dir = sys.argv[1]
all_jsons = sorted(os.listdir(root_dir))
save_path = root_dir + '.json'

all_dat = []
for json_file in all_jsons:
    with open(os.path.join(root_dir, json_file), 'r') as f:
        all_dat.extend(json.load(f))

with open(save_path, 'w') as f:
    json.dump(all_dat, f)

if 'correct' in all_dat[0].keys():
    correct = sum([x['correct'] for x in all_dat])
    total = len(all_dat)
    print(f'Acc: {correct/total}, {correct}/{total}')

for json_file in all_jsons:
    os.remove(os.path.join(root_dir, json_file))