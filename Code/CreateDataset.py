import pandas as pd
import csv
import glob
import os
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from matplotlib import pyplot as plt

def fits2png(file_path, save_path, name):
    data = fits.getdata(file_path)
    zscale = ZScaleInterval()
    data = zscale(data).squeeze()
    plt.imshow(data)
    plt.axis('off')
    plt.savefig('./data/train/' + name + '.png', bbox_inches='tight', pad_inches=0)

train_positives = {os.path.splitext(f)[0] : 1 for f in os.listdir('./data/train/positives')}
train_negatives = {os.path.splitext(f)[0] : 0 for f in os.listdir('./data/train/negatives')}
eval_positives = {os.path.splitext(f)[0] : 1 for f in os.listdir('./data/eval/positives')}
eval_negatives = {os.path.splitext(f)[0] : 0 for f in os.listdir('./data/eval/negatives')}
train_dt = {**train_positives, **train_negatives}
eval_dt = {**eval_positives, **eval_negatives}

#augmentera här, lägg till .fits kopior i train_dt


#Spara alla .fits filer som png för inmatninig i CNN
for name, label in train_dt.items():
    if (label): fits2png('./data/train/positives/' + name + '.fits', './data/train/', name)
    else: fits2png('./data/train/negatives/' + name + '.fits', './data/train/', name)
    

for name, label in eval_dt.items():
    if (label): fits2png('./data/eval/positives/' + name + '.fits', './data/eval/', name)
    else: fits2png('./data/eval/negatives/' + name + '.fits', './data/eval/', name)

#Spara annotations för false och positives
with open('./data/train/annotations.csv', 'w') as f:
    for key in train_dt.keys():
        f.write("%s %s\n"%(key,train_dt[key]))

with open('./data/eval/annotations.csv', 'w') as f:
    for key in eval_dt.keys():
        f.write("%s %s\n"%(key,train_dt[key]))


