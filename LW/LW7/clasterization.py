import copy
import random

from matplotlib import pyplot as plt

class clasterization:

    def __init__(self, arr, k):
        print(f"Количество центров: {k}")
        print(f"Массив?: {arr}")
        self.arr = arr
        self.k = k
        self.n = len(self.arr)
        self.dim = len(self.arr[0])
    def data_distr(self, cluster):
        cluster_content = [[] for i in range(self.k)]

        for i in range(self.n):
            min_dist = float('inf')
            sit_clust = -1
            for j in range(self.k):
                dist = 0
                for q in range(self.dim):
                    dist += (self.arr[i][q] - cluster[j][q])**2

                dist = dist**(1/2)
                if dist < min_dist:
                    min_dist = dist
                    sit_clust = j

            cluster_content[sit_clust].append(self.arr[i])

        return cluster_content

    def cluster_update(self, cluster, cluster_content):
        k = len(cluster)
        for i in range(self.k):
            for q in range(self.dim):
                updated_param = 0
                for j in range(len(cluster_content[i])):
                    updated_param += cluster_content[i][j][q]
                if len(cluster_content[i]) != 0:
                    updated_param = updated_param / len(cluster_content[i])
                cluster[i][q] = updated_param
            return cluster
    def visualisation_2d(self, cluster_content):

        k = len(cluster_content)
        plt.grid()
        plt.xlabel("Transaction")
        plt.ylabel("Values")

        for i in range(k):
            x_coordinates = []
            y_coordinates = []
            for q in range(len(cluster_content[i])):
                x_coordinates.append(cluster_content[i][q][0])
                y_coordinates.append(cluster_content[i][q][1])
            plt.scatter(x_coordinates, y_coordinates)
    def clast(self):
        cluster = [[0 for i in range(self.dim)] for q in range(self.k)]
        cluster_content = [[] for i in range(self.k)]

        for i in range(self.dim):
            for q in range(self.k):
                cluster[q][i] = random.randint(0, 2)

        cluster_content = self.data_distr(cluster)
        prev_clust = copy.deepcopy(cluster)

        while 1:
            cluster = self.cluster_update(cluster, cluster_content)
            cluster_content = self.data_distr(cluster)
            if cluster == prev_clust:
                break
            prev_clust = copy.deepcopy(cluster)
        self.visualisation_2d(cluster_content)