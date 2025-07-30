import matplotlib.pyplot as plt
import numpy as np
import sys

from setuptools.command.rotate import rotate

if len(sys.argv) != 2:
    print("Usage: python3 show_data.py <1-14>")
    exit(1)
power = int(sys.argv[1])
if power < 1 or power > 15:
    print("Usage: python3 show_data.py <1-14>")
    exit(1)

# Load data from file
with open("new_domain_benchamrks.csv", "r") as f:
    lines = f.readlines()

# Parse each line
lines = lines[1:]
data = []
for line in lines:
    parts = line.strip().split()
    x = int(parts[0])
    y = int(parts[1])
    time = float(parts[2])
    mem = float(parts[3].replace("MB", ""))
    data.append([x, y, time, mem])

data = np.array(data)
# data = data[np.argsort(data[:, 0])]

dic = {}
for item in data:
    part = int(item[0])
    numberOfCells = item[1]
    time = item[2]
    mem = item[3]

    selected = 2 ** power
    if part == selected:
        dic.setdefault(part, []).append((part, numberOfCells, time, mem))
        #dic.setdefault(numberOfCells, []).append((part, numberOfCells, time, mem))

# selected = 2 ** power
data = dic






# Set up subplots
fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(17, 8), sharex=True)

plt.sca(ax_time)
plt.xticks(rotation=60)
plt.sca(ax_mem)
plt.xticks(rotation=60)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, (typ, entries) in enumerate(sorted(data.items())):
    entries.sort()
    x = [e[1] for e in entries]
    time = [e[2] for e in entries]
    mem = [e[3] for e in entries]

    color = colors[i % len(colors)]

    # Time plot (top)
    ax_time.plot(x, time, label=f'Number of parts {typ} - Time Domain', marker='o', color=color)

    # Memory plot (bottom)
    ax_mem.plot(x, mem, label=f'Number of parts {typ} - Mem Domain', marker='s', linestyle=':', color=color)

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