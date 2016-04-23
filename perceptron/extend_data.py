import random

with open('data.csv', 'a') as f:
    for i in xrange(4500):
        f.write(str(random.random()) + ", " + str(random.random()) + ",\n")

