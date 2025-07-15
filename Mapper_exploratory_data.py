import pandas as pd
import numpy as np
import kmapper as km
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

df = pd.read_csv("exploratory_data.csv")
data = df.values

dists = pairwise_distances(data)
eccentricity = np.mean(dists, axis=1).reshape(-1, 1)

mapper = km.KeplerMapper()

cover = km.Cover(n_cubes=10, perc_overlap=0.6)

clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage="single")

graph = mapper.map(eccentricity, data, cover=cover, clusterer=clusterer)

mapper.visualize(graph,
                 path_html="mapper_exploratory_data_threshold_0.5.html",
                 title="Mapper Graph with Eccentricity Filter",
                 custom_tooltips=df.values)

plt.figure(figsize=(8, 6))
km.draw_matplotlib(graph)
plt.title("Static Mapper Graph (Threshold 0.5)")
plt.savefig("mapper_exploratory_data_threshold_0.5.png", dpi=300)
