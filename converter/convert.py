import json
import os
import pdb

def convert(input_path, output_path):
    x = json.load(open('./converter/prediction.json'))
    t = []
    with open(input_path, 'r') as f:
        for i in f.readlines():
            item = i.split('\t')[1].split('\n')[0]
            t.append(item)
    f.close()

    for i in range(len(t)):
        x[i]['answer'] = t[i]

    with open(os.path.join(output_path, 'prediction.json'), 'w') as f:
        json.dump(x, f)
    f.close()
