import json
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
import os

import argparse
def main(eval_dir):
    # 初始化 ArgumentParser
    import re
    path_root = eval_dir
    json_data = []
    for p in os.listdir(path_root):
        if re.match(r'^\d+\.json$', p):
            json_data += read_jsonl(os.path.join(path_root, p))

    import re

    st = set()
    for data in json_data:
        if data['index'] in st: continue
        st.add(data['index'])

    assert len(st) == 2700, f'not all. {len(st)=}'

    def extract_characters_regex(s):
        s = s.strip()
        answer_prefixes = [
            'The best answer is',
            'The correct answer is',
            'The answer is',
            'The answer',
            'The best option is'
            'The correct option is',
            'Best answer:'
            'Best option:',
            'Answer:',
            'Option:',
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, '')

        if len(s.split()) > 10 and not re.search('[ABCD]', s):
            return ''
        matches = re.search(r'[ABCD]', s)
        if matches is None:
            return ''
        return matches[0]
    for data in json_data:
        if int(extract_characters_regex(data['prediction']) == data['answer']):
            data['score'] = 1
        else:
            data['score'] = 0

    DURATIONS = [
        'short',
        'medium',
        'long',
    ]

    DOMAINS = [
        'Knowledge',
        'Film & Television',
        'Sports Competition',
        'Artistic Performance',
        'Life Record',
        'Multilingual'
    ]

    SUB_CATEGORIES = [
        'Humanity & History',
        'Literature & Art',
        'Biology & Medicine',
        'Finance & Commerce',
        'Astronomy',
        'Geography',
        'Law',
        'Life Tip',
        'Technology',
        'Animation',
        'Movie & TV Show',
        'Documentary',
        'News Report',
        'Esports',
        'Basketball',
        'Football',
        'Athletics',
        'Other Sports',
        'Stage Play',
        'Magic Show',
        'Variety Show',
        'Acrobatics',
        'Handicraft',
        'Food',
        'Fashion',
        'Daily Life',
        'Travel',
        'Pet & Animal',
        'Exercise',
        'Multilingual'
    ]

    TASK_CATEGORIES = [
        'Temporal Perception',
        'Spatial Perception',
        'Attribute Perception',
        'Action Recognition',
        'Object Recognition',
        'OCR Problems',
        'Counting Problem',
        'Temporal Reasoning',
        'Spatial Reasoning',
        'Action Reasoning',
        'Object Reasoning',
        'Information Synopsis',
    ]
    duration_rating = {k: {} for k in DURATIONS}
    for duration in DURATIONS + ['overall']:
        duration_rating[duration] = {
            'overall': '',
            'domain': {k: [] for k in DOMAINS},
            'sub_category': {k: [] for k in SUB_CATEGORIES},
            'task_type': {k: [] for k in TASK_CATEGORIES}
        }


    import numpy as np
    for data in json_data:
        domain, sub_ctg, task_ctg, duration = (
            data['domain'],
            data['sub_category'],
            data['task_type'],
            data['duration']
        )
        duration_rating[duration]['domain'][domain].append(data['score'])
        duration_rating[duration]['sub_category'][sub_ctg].append(data['score'])
        duration_rating[duration]['task_type'][task_ctg].append(data['score'])

        duration_rating['overall']['domain'][domain].append(data['score'])
        duration_rating['overall']['sub_category'][sub_ctg].append(data['score'])
        duration_rating['overall']['task_type'][task_ctg].append(data['score'])

    for duration in DURATIONS + ['overall']:
        overall_res_dur = f'{np.mean([x for x in sum(duration_rating[duration]["domain"].values(), []) if x >= 0]):.2f}'
        duration_rating[duration]['overall'] = overall_res_dur

        for domain in DOMAINS:
            domain_res_dur = f'{np.mean([x for x in duration_rating[duration]["domain"][domain] if x >= 0]):.2f}'
            duration_rating[duration]['domain'][domain] = domain_res_dur

        for sub_ctg in SUB_CATEGORIES:
            sub_res_dur = f'{np.mean([x for x in duration_rating[duration]["sub_category"][sub_ctg] if x >= 0]):.2f}'
            duration_rating[duration]['sub_category'][sub_ctg] = sub_res_dur

        for task_ctg in TASK_CATEGORIES:
            task_res_dur = f'{np.mean([x for x in duration_rating[duration]["task_type"][task_ctg] if x >= 0]):.2f}'
            duration_rating[duration]['task_type'][task_ctg] = task_res_dur

    print(duration_rating, duration_rating['short']['overall'], duration_rating['medium']['overall'], duration_rating['long']['overall'], duration_rating['overall']['overall'])

if __name__ == '__main__':
    import sys
    main(sys.argv[1])