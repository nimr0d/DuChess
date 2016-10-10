import cv2
from sklearn import linear_model
import numpy as np
import random
import os
import math

# Assume Standard Setup
def coordinates_to_peices():
    normal_board = {}
    for i in range(8):
        for j in range(8):
            if j ==1:
                normal_board[(i,j)] = "WPawn"
            elif j==0 and (i==0 or i==7):
                normal_board[(i,j)] = "WRook"
            elif j==0 and (i==1 or i==6):
                normal_board[(i,j)] = "WKnight"
            elif j==0 and (i==2 or i==5):
                normal_board[(i,j)] = "WBishop"
            elif j==0 and (i==3):
                normal_board[(i,j)] = "WQueen"
            elif j==0 and (i==4):
                normal_board[(i,j)] = "WKing"
            elif j== 6:
                normal_board[(i,j)] = "BPawn"
            elif j==7 and (i==0 or i==7):
                normal_board[(i,j)] = "BRook"
            elif j==7 and (i==1 or i==6):
                normal_board[(i,j)] = "BKnight"
            elif j==7 and (i==2 or i==5):
                normal_board[(i,j)] = "BBishop"
            elif j==7 and (i==3):
                normal_board[(i,j)] = "BQueen"
            elif j==7 and (i==4):
                normal_board[(i,j)] = "BKing"
            else:
                normal_board[(i,j)] = "Empty"
    return normal_board

# Preprocess data
coord_to_label = coordinates_to_peices()
pixels_to_label = []
# Read in images and locations
def cap_to_images():
    os.chdir("/home/jesse/Documents/Data/")
    files = os.listdir()
    strs = [file[2:4] for file in files]
    l = []    
    for s in strs:
        if '_' in s:
            s = s.replace('_','') 
        l.append(int(s))
    for i in range(len(l)):
        
        temp = cv2.imread(files[i])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        im = []
        for g in temp:
            im.extend(g)
        pixels_to_label.append([im, coord_to_label[divmod(l[i], 8)]])
    return pixels_to_label
cap_to_images()    
# Split into training and testing data
shuffled = list(range(len(pixels_to_label)))
random.shuffle(shuffled)
all_squares = []
for i in shuffled:
    all_squares.append(pixels_to_label[i][0])
all_labels = []
for i in shuffled:
    all_labels.append(pixels_to_label[i][1])


chunk = math.floor(len(all_squares)*0.7)

xTrain = all_squares[:chunk]
yTrain = all_labels[:chunk]

xTest = all_squares[chunk:]
yTest = all_labels[chunk:]

# Perform training
min_size1 = min([len(i) for i in xTrain])
min_size2 = min([len(i) for i in xTest])
min_size = min(min_size1,min_size2)
logistic_reg = linear_model.LogisticRegression()
for i in range(len(xTrain)):
    xTrain[i] = xTrain[i][:min_size]
logistic_reg.fit(xTrain, yTrain)

for i in range(len(xTest)):
    xTest[i] = xTest[i][:min_size]

# Test classifier
label = logistic_reg.predict(xTest)
count = 0
correct = 0
cnt = {}
corr = {}
    
for i in range(len(label)):
    if yTest[i] not in cnt:
        cnt[yTest[i]] = 0
        corr[yTest[i]] = 0      
    cnt[yTest[i]] += 1
    if label[i] == yTest[i]:
        correct+=1
        corr[yTest[i]] += 1
    count += 1
print(correct)
print(cnt)
print(corr)
print(count)