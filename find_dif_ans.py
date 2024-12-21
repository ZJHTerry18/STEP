import json

baseline_file = 'results/mvbench/internvideo2_1b_mistral7b_stage4_hd_post_f16.json'
ours_file = 'results/mvbench/stage4_16f_glfgtc_hd.json'
output_file = 'diff_ans.json'

def check_ans(pred, gt):
    pred_option = pred[1]
    gt_option = gt[1]
    
    return pred_option == gt_option

with open(baseline_file, 'r') as f:
    bsl_res = json.load(f)['res_list']

with open(ours_file, 'r') as f:
    ours_res = json.load(f)['res_list']

selected_examples = []
for bsl_ans, ours_ans in zip(bsl_res, ours_res):
    bsl_correct = check_ans(bsl_ans['pred'], bsl_ans['gt'])
    ours_correct = check_ans(ours_ans['pred'], ours_ans['gt'])
    
    if ours_correct and not bsl_correct:
        selected_examples.append(bsl_ans)

with open(output_file, 'w') as f:
    json.dump(selected_examples, f)