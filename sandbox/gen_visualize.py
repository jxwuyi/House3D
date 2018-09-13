import json,pickle
import seaborn as sns
import matplotlib.pyplot as plt
import progressbar

flag_env_set = 'train'
house_ids_dict = json.load(open('all_house_ids.json','r'))
all_houseIDs = house_ids_dict[flag_env_set]

n_house = len(all_houseIDs) #10

bar = progressbar.ProgressBar()
for i in bar(range(n_house)):
    h = all_houseIDs[i]
    pkl_file = './storage/' + h + '_data.pkl'
    with open(pkl_file,'rb') as f:
        m, d = pickle.load(f)
    sns_plot = sns.heatmap(d)
    pic_file = './storage/' + h + '_vis.png'
    plt.savefig(pic_file)
    plt.clf()