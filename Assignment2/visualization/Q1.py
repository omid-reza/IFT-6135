import json
from matplotlib import pyplot as plt

executed_configs = {
    1:"01-lstm_layer_1_btch_16_adam",
    2:"02-lstm_layer_1_btch_16_adamw",
    3:"03-lstm_layer_1_btch_16_sgd",
    4:"04-lstm_layer_1_btch_16_momentum",
    9:"09-lstm_layer_2_btch_16_adamw",
    10:"10-lstm_layer_4_btch_16_adamw"
}

for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/train_time.txt", "r") as train_file:
        data = l = [float(line) for line in train_file]
        plt.plot(data, label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Train")
plt.ylabel("Time (Second)")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.savefig("EpochTime(Tr).png")

plt.cla()
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/valid_time.txt", "r") as validation_file:
        data = l = [float(line) for line in validation_file]
        plt.plot(data, label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Validation")
plt.ylabel("Time (Second)")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.savefig("EpochTime(Vl).png")
#%%
