import pickle
import csv

A =  ['kitchen_cabinet','sofa','chair','toilet','table', 'sink','wardrobe_cabinet','bed',
      'shelving','desk','television','household_appliance','dresser','vehicle','pool']
      #'table_and_chair']

D = dict({'sofa': 9, 'desk': 17, 'sink': 13, 'wardrobe_cabinet': 14, 'bed': 15,
          'kitchen_cabinet': 8, 'shelving': 16, 'dresser': 20, 'chair': 10, 'television': 18,
          'toilet': 11, 'vehicle': 21, 'table': 12, 'pool': 22, 'household_appliance': 19,
          'table_and_chair': (10, 12), 'dressing_table': 17, 'hanging_kitchen_cabinet': 8})

filename = 'ModelCategoryMapping.csv'

H = dict()

with open(filename,'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        id, cat = row['model_id'], row['coarse_grained_class']
        if cat in D:
            H[id] = cat

import json
with open('map_modelid_to_targetcat.json','w') as f:
    json.dump(H, f)

