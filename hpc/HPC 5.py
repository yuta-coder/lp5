import math
import time
from multiprocessing import Pool, cpu_count


# -------- DATASET --------
data = [
    (1,2,0),(2,3,0),(3,3,0),
    (8,8,1),(9,10,1),(10,9,1)
]

query = (5,5)
k = 3


# -------- DISTANCE --------
def dist(p, q):
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)


# -------- SEQUENTIAL --------
def knn_seq(data, query, k):
    d = [(dist((x,y), query), label) for x,y,label in data]
    d.sort()
    labels = [l for _,l in d[:k]]
    return max(set(labels), key=labels.count)


# -------- PARALLEL --------
def compute(args):
    point, query = args
    return (dist((point[0],point[1]), query), point[2])


def knn_par(data, query, k):
    with Pool(cpu_count()) as p:
        d = p.map(compute, [(pt,query) for pt in data])
    d.sort()
    labels = [l for _,l in d[:k]]
    return max(set(labels), key=labels.count)


# -------- MAIN --------
if __name__ == "__main__":

    print("Dataset:", data)
    print("Query Point:", query)
    print("K =", k)

    # Sequential
    t = time.time()
    r1 = knn_seq(data, query, k)
    t1 = time.time() - t

    print("\nSequential Result:", r1)
    print("Time:", round(t1,5))

    # Parallel
    t = time.time()
    r2 = knn_par(data, query, k)
    t2 = time.time() - t

    print("\nParallel Result:", r2)
    print("Time:", round(t2,5))

    print("\nSpeedup:", round(t1/t2,2))