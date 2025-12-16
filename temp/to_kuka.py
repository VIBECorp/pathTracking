import json
import os
import numpy as np

data = json.load(open('temp/trajectory.json'))

joints = data['joint']

joints = np.array(joints)

last = np.zeros((len(joints), 1))

joints = np.concatenate([joints, last], axis=1)
data['joint'] = joints.tolist()

with open('temp/trajectory_kuka.json', 'w') as f:
    json.dump(data, f)
    
