import matplotlib.pyplot as plt
import numpy as np

def getData(name, count):
    f = open(name, 'r')
    f.readline()
    myData = []
    for line in f:
        if(len(line.split()) >= count):
            l = list(map(float, (line.split())[:count]))
            myData.append(l)
    f.close()
    return myData

def barDiagrams(x, y1, y2, y3, y4):
	plt.figure(1)
	plt.subplot(221)
	plt.bar(x, y1)
	plt.title('male')
	plt.subplot(222)
	plt.bar(x, y2, color = 'red')
	plt.title('female')
	plt.subplot(223)
	plt.plot(x, y3)
	plt.subplot(224)
	plt.plot(x, y4, color = 'red')
	plt.show()

def posteriorDiagram(length, male, test):
	female = [1 - m for m in male]
	position = length.index(test)
	plt.plot(length, male)
	plt.plot(length, female, color = 'red')
	if male[position] > female[position]:
		print(male[position])
		plt.scatter([test], [male[position]])
	else:
		print(female[position])
		plt.scatter([test], [female[position]], color = 'red')
	plt.title('posterior')
	plt.show()

def just(x, l):
	return[a[x] for a in l]

def computeCumulative(y):
	total = sum(y)
	prefix = 0
	probability = []
	for a in y:
		prefix += a
		probability.append(prefix / total)
	return probability

def computePosterior(y1, y2):
	probability = []
	for i in range(0, len(y1)):
		probability.append(y1[i] / (y1[i] + y2[i]))
	return probability

def run():
	data =  getData('fish.txt', 3)
	length = just(0, data)
	male = just(1, data)
	female = just(2, data)
	cumulative_male = computeCumulative(male)
	cumulative_female = computeCumulative(female)
	#barDiagrams(length, male, female, cumulative_male, cumulative_female)
	posteriorDiagram(length, computePosterior(male, female), 10)
run()
