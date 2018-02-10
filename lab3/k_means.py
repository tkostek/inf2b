eruptions = []
wait = []
def getData():
	f = open('faithful.txt', 'r')
	f.readline()
	for line in f:
		e, w = map(float, line.split())
		eruptions.append(e)
		wait.append(w)
	f.close()

getData()

