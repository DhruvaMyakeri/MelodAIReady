import json
import numpy as np
import matplotlib.pyplot as plt
from dataset import INSTRUMENTS

# Load JSON
with open("output2.json") as f:
    data = json.load(f)

segments = data["segments"]

# Build matrix
time_steps = len(segments)
num_classes = len(INSTRUMENTS)

matrix = np.zeros((num_classes, time_steps))

for t, seg in enumerate(segments):
    for inst in seg["instruments"]:
        idx = INSTRUMENTS.index(inst)
        matrix[idx, t] = 1

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.imshow(matrix, aspect='auto')

plt.yticks(range(num_classes), INSTRUMENTS)
plt.xlabel("Time segment")
plt.ylabel("Instruments")
plt.title("Instrument Timeline Heatmap")

plt.colorbar(label="Presence")
plt.show()