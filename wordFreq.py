import json
import numpy as np

def parse(description):
	s = []
	temp = ""
	for i in range (len(description)):
		if (description[i] == ' ' or description[i] == "\n"):
			s.append(temp)
			temp = ""
		else:
			if description[i] != ",":
				temp += description[i]
	return s
		

#Load data for listing points
with open('test.json') as f:
    js = json.load(f)
coordKey = list(js['latitude'].keys())

def parseCount(file):
	f = open(file,"r")
	dict = {}
	for line in f:
		id = ""
		count = ""
		control = False
		for i in range (len(line)):
			if control == True:
				count += line[i]
			if line[i] != "," and control == False:
				id += line[i]
			else:
				control = True
		dict[id] = int(count)
	return dict
	
print(parseCount("wordCountTesting.csv"))
'''
f = open("wordCountTesting.csv", "w")
wordCount = []

for i in range (len(coordKey)):
	x = 1
	for j in range(len(js['description'][coordKey[i]])):
		if js['description'][coordKey[i]][j] == " " or js['description'][coordKey[i]][j] == "\n":
			x += 1
	f.write(js['building_id'][coordKey[i]] + "," + str(x) + "\n")

wordCount = {}
for i in range (len(coordKey)):
	if js['interest_level'][coordKey[i]] == "high":
		s = parse(js['description'][coordKey[i]])
		if (s != []):
			for j in range(len(s)):
				if str(s[j]) in wordCount:
					wordCount[str(s[j])] += 1
				if str(s[j]) not in wordCount and str(s[j]) != []:
					wordCount[str(s[j])] = 1

f = open("wordFreqHigh.csv", 'w')
l = len(wordCount)
for i in range(l):
	k = max(wordCount, key=wordCount.get)
	f.write((k) + "," + str(wordCount[k]) + "\n")
	del wordCount[k]
f.close()
'''