import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np
from scipy.stats import ortho_group

def just(l, n):
	return [x[n] for x in l]

def getData(name, sep, count):
	f = open(name, 'r')
	f.readline()
	myData = []
	for line in f:
		if(len(line.split(',')) >= count):
			l = list(map(float, (line.split(sep))[:count]))
			myData.append(l)
	f.close()
	return myData

def distance(x, y):
	res = 0.0
	for i in range(0, len(x)):
		res += (x[i] - y[i]) * (x[i] - y[i])
	return res

def computeClosest(lemon, clusters):
	best = [distance(lemon, clusters[0]), 0]
	for i in range(0, len(clusters)):
		best = min (best, [distance(lemon, clusters[i]), i])
	return best[1]

def shiftClusters(clusters, points, closest):
	for i in range(0, len(clusters)):
		n = 0
		clusters[i] = [0, 0]
		for j in range(len(points)):
			if closest[j] == i:
				n += 1
				clusters[i] = [clusters[i][0] + points[j][0], clusters[i][1] + points[j][1]]
		if n == 0:
			n = 1
		clusters[i] = [clusters[i][0] / n, clusters[i][1] / n]

def boundries(data2D):
	w1 = min(just(data2D, 0))
	w2 = max(just(data2D, 0))
	h1 = min(just(data2D, 1))
	h2 = max(just(data2D, 1))
	return [[w1, w2], [h1, h2]]

def computeQuality(data2D, clusters, closest):
	res = 0.0
	for i in range(0, len(data2D)):
		res += distance(data2D[i], clusters[closest[i]])
	[[w1, w2], [h1, h2]] = boundries(data2D)
	return res / (len(data2D) * ((w2-w1)*(w2-w1)+(h2-h1)*(h2-h1)))

def kMeans(data2D, n):
	[[w1, w2],[h1, h2]] = boundries(data2D)
	clusters = []
	closest = []
	for i in range(0, n):
		clusters.append([w1 + (w2 - w1)*(i/n), h1 + (h2 - h1)*(i/n)])
	for i in range(0, 10):
		plt.show()
		closest.clear()
		for l in data2D:
			closest.append(computeClosest(l, clusters))
		shiftClusters(clusters, data2D, closest)
	return [closest, computeQuality(data2D, clusters, closest)]

def majorityVote(votes):
	votes.sort()
	votes.append(-1)
	count = 0
	best = [-1, -1]
	for i in range(0, len(votes) - 1):
		count += 1
		if votes[i] != votes[i + 1]:
			if count > best[0]:
				best = [count, votes[i]]
			count = 0
	return best[1]

def countErrors(pred):
	x = majorityVote(pred)
	err = 0
	for p in pred:
		if p != x:
			err += 1
	return err

def chooseColors(col, cm):
	return [cm(c/10) for c in col]

def kNN(test, points, labels, n):
	neighbours = []
	for i in range(0, len(points)):
		neighbours.append([distance(test, points[i]), labels[i]])
	neighbours.sort()
	neighbours = neighbours[:n]
	return majorityVote(just(neighbours, 1))

def f(X, Y, points, labels):
	res = []
	for i in range(0, len(X)):
		line = []
		for j in range(0, len(X[i])):
			line.append(kNN([X[i][j], Y[i][j]], points, labels, 1) / 10)
		res.append(line)
	return res	

def trainAndClassify(data2D, number_of_labels, number_of_neighbours, test_data, bound):
	[color, quality] = kMeans(data2D, number_of_labels)
	if quality > bound:
		return quality
	print (quality)
	errA = countErrors(color[:50])
	errB = countErrors(color[50:100])
	errC = countErrors(color[100:150])
	print ('errors:', errA, errB, errC, errA + errB + errC)
	prediction = []
	for t in test_data:
		prediction.append(kNN(t, data2D, color, number_of_neighbours))
	n = 100
	[[w1, w2],[h1, h2]] = boundries(data2D)
	x = np.linspace(w1, w2, n)
	y = np.linspace(h1, h2, n)
	X,Y = np.meshgrid(x,y)
	cm = plt.get_cmap('tab20') 
	plt.contourf(X, Y, f(X, Y, data2D, color), cmap = plt.get_cmap('ocean'))
	plt.scatter(just(test_data, 0), just(test_data, 1), c = chooseColors(prediction, cm), edgecolors = 'red')
	plt.scatter(just(data2D, 0), just(data2D, 1), c = chooseColors(color, cm), edgecolors = 'black')
	#entries = [mpatches.Patch(color=cm(i/10), label=str(i)) for i in range(0, 10)]
	#plt.legend(handles=entries)
	plt.xlabel('width')
	plt.ylabel('height')
	plt.show()
	return quality

def randomInvertibe(dim):
	return ortho_group.rvs(dim)

def transform(vector, basis):
	return np.matmul(np.matmul(np.linalg.inv(basis), vector), basis)

def applyPCA(_data, basis):
	data = np.copy(_data)
	return [transform(d, basis)[:2] for d in data]

def run():
	data = getData('iris.txt', ',', 4)
	test_data = [[6.5, 7.6], [7.5, 8.7], [8.5, 9]]
	test_data2 = []
	best = 1
	for i in range(0,1000000):
		basis = randomInvertibe(len(data[0]))
		boundry = best
		if (i < 0):
			boundry = 0
		q = trainAndClassify(applyPCA(data, basis), 3, 1, applyPCA(test_data2, basis), boundry)
		best = min(best, q)
run()
