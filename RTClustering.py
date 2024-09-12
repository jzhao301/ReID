import pickle
import torch
from DenStream import DenStream
from FeatureAnalysis import evaluate
import json
import os.path


stream = DenStream()
if os.path.isfile('denstream.pkl'):
    with open('denstream.pkl', 'rb') as fr:
        stream = pickle.load(fr)

with open('final_detections.json') as fr:
    final_detections = json.load(fr)

final_detections.sort(key= lambda x: x['detection_id'])
data = []
for i in len(final_detections):
    data.append(final_detections[i]['feature'])

data = torch.tensor(data)

for d in data:
    print(stream.fit_predict([evaluate(d)]))
    stream.partial_fit([evaluate(d)])