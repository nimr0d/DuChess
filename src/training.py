import cv2
from sklearn import linear_model
import numpy as np
import random
import os
import math

whitePieces = ["WRook", "WKnight", "WBishop", "WQueen", "WKing"]
blackPieces = ["BRook", "BKnight", "BBishop", "BQueen", "BKing"]

def coord_to_label(x):
    i, j = divmod(x, 8)
    if i == 1:
        return "WPawn"
    elif i == 6:
        return "BPawn"
    elif i == 0:
        return whitePieces[j] if j < 5 else whitePieces[7 - j]
    elif i == 7:
         return blackPieces[j] if j < 5 else blackPieces[7 - j]
    else:
        return "BSq" if (i + j) % 2 == 0 else "WSq"

def get_labeled_data():
    os.chdir("../out/squares/")
    files = os.listdir()
    labeled_pairs = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
        label = coord_to_label(int(file[2:].split('_')[0]))

        labeled_pairs.append([img, label])
    return labeled_pairs
  
# Read images and assign labels
labeled_pairs = get_labeled_data()

# Split into training and testing data
shuffled = list(range(len(labeled_pairs)))
random.shuffle(shuffled)

all_squares = []
for i in shuffled:
    all_squares.append(labeled_pairs[i][0])
all_labels = []
for i in shuffled:
    all_labels.append(labeled_pairs[i][1])


chunk = math.floor(len(all_squares) * 0.7)

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
        correct += 1
        corr[yTest[i]] += 1
    count += 1

print(correct)
print(cnt)
print(corr)
print(count)