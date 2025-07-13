import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

points = pd.read_csv("data_A.csv") 

diagrams = ripser(points.values)['dgms']

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_diagrams(diagrams[0], ax=ax[0], show=False)
plot_diagrams(diagrams[1], ax=ax[1], show=False)
ax[0].set_title("H₀ Persistence Diagram (Connected Components)")
ax[1].set_title("H₁ Persistence Diagram (1D Holes/Loops)")
plt.tight_layout()
plt.savefig("data_A_persistence_diagram.png", dpi=300)
plt.show()