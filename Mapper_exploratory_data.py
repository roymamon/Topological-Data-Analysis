import pandas as pd
import numpy as np
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx

data = pd.read_csv("exploratory_data.csv")
X = data.values

distance_matrix = pairwise_distances(X)
eccentricity = np.max(distance_matrix, axis=1)

eccentricity = MinMaxScaler().fit_transform(eccentricity.reshape(-1, 1)).flatten()

mapper = km.KeplerMapper(verbose=1)

lens = eccentricity

cover = km.Cover(n_cubes=10, perc_overlap=0.6)

clusterer = DBSCAN(eps=0.3, min_samples=3)

graph = mapper.map(lens, X, cover=cover, clusterer=clusterer)

mapper.visualize(graph,
                 path_html="mapper_exploratory_data.html",
                 title="Mapper Graph: Exploratory Data (Eccentricity)")

G = km.to_networkx(graph)

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=False, node_size=40, node_color='skyblue', edge_color='gray')
plt.title("Mapper Graph (static view)")
plt.savefig("mapper_exploratory_data.png", dpi=300)
plt.show()