import pandas as pd
from matplotlib import pyplot as plt
executed_configs = [1, 2, 3, 4, 5, 6, 9, 10, 11]

for config_num in executed_configs:
    data=pd.read_csv(f"../LoggedData/GPU/{config_num}.txt", header=None)
    label = data[0].tolist()[0]
    data = data[4].tolist()
    data = [float(d.replace(" Used Memory: ", "")) for d in data]
    plt.plot(data, label=label)
plt.title("Memory Usage")
plt.xlabel("Epochs")
plt.ylabel("Memory Usage (mb)")
plt.legend()
plt.savefig("GPURAM.png")
plt.show()
#%%
