import matplotlib.pyplot as plt
import numpy as np
import sys

from setuptools.command.rotate import rotate

if len(sys.argv) != 2:
    print("Usage: python3 show_data.py <1-14>")
    exit(1)
power = int(sys.argv[1])
if power < 1 or power > 14:
    print("Usage: python3 show_data.py <1-14>")
    exit(1)

# Load data from file
with open("benchmark_results.csv", "r") as f:
    lines = f.readlines()

# Parse each line
lines = lines[1:]
data = []
for line in lines:
    parts = line.strip().split()
    x = int(parts[0])
    y = int(parts[1])
    timeA = float(parts[2])
    memA = float(parts[3].replace("MB", ""))
    timeB = float(parts[4])
    memB = float(parts[5].replace("MB", ""))
    data.append([x, y, timeA, memA, timeB, memB])

data = np.array(data)
data = data[np.argsort(data[:, 0])]

dic = {}
for item in data:
    part = int(item[0])
    numberOfCells = item[1]
    timeA = item[2]
    memA = item[3]
    timeB = item[4]
    memB = item[5]
    dic.setdefault(part, []).append((numberOfCells, timeA, memA, timeB, memB))

selected = 2 ** power
data = {selected: dic[selected]}






# Set up subplots
fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(17, 8), sharex=True)

plt.sca(ax_time)
plt.xticks(rotation=60)
plt.sca(ax_mem)
plt.xticks(rotation=60)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, (typ, entries) in enumerate(sorted(data.items())):
    entries.sort()
    x = [e[0] for e in entries]
    timeA = [e[1] for e in entries]
    memA = [e[2] for e in entries]
    timeB = [e[3] for e in entries]
    memB = [e[4] for e in entries]

    color = colors[i % len(colors)]

    # Time plot (top)
    ax_time.plot(x, timeA, label=f'Number of parts {typ} - Time Domain', marker='o', color=color)
    ax_time.plot(x, timeB, label=f'Number of parts {typ} - Time AltDomain', marker='x', linestyle='--', color=color)

    # Memory plot (bottom)
    ax_mem.plot(x, memA, label=f'Number of parts {typ} - Mem Domain', marker='s', linestyle=':', color=color)
    ax_mem.plot(x, memB, label=f'Number of parts {typ} - Mem AltDomain', marker='^', linestyle='-.', color=color)

# Customize time axis
ax_time.set_ylabel("Time (s)")
ax_time.set_title("Time")
ax_time.legend(loc='upper left')
ax_time.grid(True)

# Customize memory axis

ax_mem.set_ylabel("RAM (MB)")
ax_mem.set_xlabel("Number of Cells")
ax_mem.set_title("Memory")
ax_mem.legend(loc='upper left')
ax_mem.grid(True)


plt.tight_layout()
plt.show()