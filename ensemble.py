import json

# prediction1, prediction2, prediction3 are the preditions 
# of three different models trained by you using different seeds.
# You can run test code to obtain prediction.json file.

x1 = json.load(open('prediction1.json'))
x2 = json.load(open('prediction2.json'))
x3 = json.load(open('prediction3.json'))


for i in range(len(x2)):
    answers = [x2[i]['answer'], x1[i]['answer'], x3[i]['answer']] 
    x2[i]['answer'] = max(set(answers), key = answers.count)


with open('prediction_ensemble.json', 'w') as f:
    json.dump(x2, f)
f.close()
