import matplotlib.pyplot as plt
import random
import numpy

eruptions = []
wait = []
closest = []
clusterPoints = []
clusterColors = []
unassigned = 'blue'

def getData():
	f = open('faithful.txt', 'r')
	f.readline()
	for line in f:
		e, w = map(float, line.split())
		eruptions.append(e)
		wait.append(w)
		closest.append(-1)
	f.close()

def graph():
	color = []
	for x in closest:
		if x == -1:
			color.append(unassigned)
		else:
			color.append(clusterColors[x])

	plt.scatter(eruptions, wait, c = color)
	plt.scatter([a[0] for a in clusterPoints],[a[1] for a in clusterPoints], c = 'red')
	plt.xlabel('eruptions')
	plt.ylabel('wait')
	plt.show()

def pickRandomPoints(n):
	clusterPoints.clear()
	clusterColors.clear()
	[l1, r1] = [min(eruptions), max(eruptions)]
	[l2, r2] = [min(wait), max(wait)]
	for i in range(0, n):
		clusterPoints.append([random.uniform(l1, r1), random.uniform(l2, r2)])
		clusterColors.append(i)	

def computeClosest():
	closest.clear()
	for i in range(0, len(eruptions)):
		score = []
		for j in range(0, len(clusterPoints)):
			v = [eruptions[i] - clusterPoints[j][0], wait[i] - clusterPoints[j][1]]
			score.append([numpy.linalg.norm(v), j])
		best = min(score)
		closest.append(best[1])

def moveClusters():
	numberAssigned = []
	for i in range(0, len(clusterPoints)):
		clusterPoints[i] = [0.0, 0.0]
		numberAssigned.append(0)
	for i in range(0, len(eruptions)):
		numberAssigned[closest[i]] += 1
		clusterPoints[closest[i]][0] += eruptions[i]
		clusterPoints[closest[i]][1] += wait[i]
	for i in range(0, len(clusterPoints)):
		clusterPoints[i][0] /= numberAssigned[i]
		clusterPoints[i][1] /= numberAssigned[i]

def computeError():
	error = 0.0
	for i in range(0, len(eruptions)):
		x = clusterPoints[closest[i]]
		v = [eruptions[i] - x[0], wait[i] - x[1]]
		dist = numpy.linalg.norm(v)
		error += dist * dist
	return error

def kMeans():
	getData()
	pickRandomPoints(2)
	lastErr = 1000000.0
	error = 900000.0
	while lastErr - error > 10.0:
		computeClosest()
		moveClusters()
		lastErr = error
		error = computeError()
		print ("Error", error)
		graph()
kMeans()
