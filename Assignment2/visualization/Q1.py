import json
from matplotlib import pyplot as plt
from statistics import mean
import numpy as np

executed_configs = {
    1:"01-lstm_layer_1_btch_16_adam",
    2:"02-lstm_layer_1_btch_16_adamw",
    3:"03-lstm_layer_1_btch_16_sgd",
    4:"04-lstm_layer_1_btch_16_momentum",
    5:"05-gpt1_layer_1_btch_16_adam",
    6:"06-gpt1_layer_1_btch_16_adamw",
    9:"09-lstm_layer_2_btch_16_adamw",
    10:"10-lstm_layer_4_btch_16_adamw",
    11:"11-gpt1_layer_2_btch_16_adamw",
}

for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/train_ppl.txt", "r") as train_file:
        data = l = [float(line) for line in train_file]
        plt.plot(np.cumsum(data), label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Train")
plt.ylabel("PPL")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.savefig("EpochPPL(Tr).png")

plt.cla()
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/valid_ppl.txt", "r") as validation_file:
        data = l = [float(line) for line in validation_file]
        plt.plot(np.cumsum(data), label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Validation")
plt.ylabel("PPL")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.savefig("EpochPPL(Vl).png")

plt.cla()
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/train_ppl.txt", "r") as train_ppl_file:
        with open(f"../LoggedData/TrVlTs/{folder_name}/train_time.txt", "r") as train_time_file:
            data = [float(line) for line in train_ppl_file]
            times = [float(line) for line in train_time_file]
            plt.plot(np.cumsum(times), data, label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Train")
plt.ylabel("PPL")
plt.xlabel("Time (Second)")
plt.legend(loc="upper right")
plt.savefig("TrainTime.png")

plt.cla()
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/valid_ppl.txt", "r") as valid_ppl_file:
        with open(f"../LoggedData/TrVlTs/{folder_name}/valid_time.txt", "r") as valid_time_file:
            data = [float(line) for line in valid_ppl_file]
            times = [float(line) for line in valid_time_file]
            plt.plot(np.cumsum(times), data, label=f"{args['model']} | layers:{args['layers']} | btch:{args['batch_size']} | optimizer:{args['optimizer']}")
plt.title("Validation")
plt.ylabel("PPL")
plt.xlabel("Time (Second)")
plt.legend(loc="upper right")
plt.savefig("ValidationTime.png")