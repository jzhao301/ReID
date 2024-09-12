import pickle
import torch
from DenStream import DenStream
from FeatureAnalysis import evaluate
import json
import numpy as np
from sklearn.cluster import DBSCAN
import os.path


# stream = DenStream()
# if os.path.isfile('denstream.pkl'):           Alternative streaming based clustering algorithm for real time prediction
#     with open('denstream.pkl', 'rb') as fr:
#         stream = pickle.load(fr)

stream = DBSCAN(eps=0.2) # If number of unique persons in detections is known, K-Means may be better since DBSCAN may ommit detections as noise

with open('detections.json') as fr:
    detections = json.load(fr)

# detections.sort(key= lambda x: x['detection_id'])
data = []
for i in len(detections):
    data.append(evaluate(torch.tensor(detections[i]['feature'])))

data = torch.tensor(data)

clustering = stream.fit(data)
num_unique_people = max(clustering.labels_)
predictions = [[]]*num_unique_people
for i in range(len(clustering.labels_)):
    if clustering.labels_[i] > -1:
        predictions[clustering.labels_[i]].append(detections[i]['detection_id'])

with open('predictions.json', 'w+') as fw:
    json.dump(predictions, fw)

