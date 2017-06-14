import numpy as np

def readIn(file):
	return 0

def parse(file):
	f = open(file, "r")
	list = []
	for line in f:
		word = ""
		num = ""
		control = 0
		for i in range(len(line)):
			if control == 0 and line[i] != ",":
				word += line[i]
			if control == 1 and line[i] != "\n":
				num += line[i]
			if line[i] == ',':
				control += 1
		list.append(word)
	return list

def findUnique(f, f1, f2, top, threshold):
	list = []
	top1 = f1[:threshold]
	top2 = f2[:threshold]
	for i in range(top):
		if f[i] not in f1 and f[i] not in f2:
			list.append(f[i])
	return list
	
def dist(file):
	f = open(file, "r")
	district = {}
	i = 1
	for line in f:
		control = 0
		buildID = ""
		schoolID = ""
		for i in range (len(line)):
			if control == 1 and line[i] != ",":
				buildID += line[i]
			if control == 2 and line[i] != "\n":
				schoolID += line[i]
			if line[i] == ',':
				control += 1
		district[buildID] = schoolID
	return district

pHigh = parse("wordFreqHigh.csv")
pMed = parse("wordFreqMed.csv")
pLow = parse("wordFreqLow.csv")

x = ["a", "bad"]
if "bad" in x:
	print("Test")

print(findUnique(pHigh, pMed, pLow, 2000, 3000))

