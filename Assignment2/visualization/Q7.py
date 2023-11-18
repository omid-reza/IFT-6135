import pandas as pd
from matplotlib import pyplot as plt

plt.figure(figsize=(10, 7))
for config_num in range(1, 12+1):
    data=pd.read_csv(f"../LoggedData/GPU/{config_num}.txt", header=None)
    label = data[0].tolist()[0]
    data = data[4].tolist()
    data = [float(d.replace(" Used Memory: ", "")) for d in data]
    plt.plot(data, label=label)
plt.title("Memory Usage")
plt.xlabel("Epochs")
plt.ylabel("Memory Usage (mb)")
plt.legend(loc="lower right")
plt.savefig("GPURAM.png")
#%%
