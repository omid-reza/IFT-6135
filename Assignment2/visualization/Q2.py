from statistics import mean
import json
import tabulate

executed_configs = {
    1:"01-lstm_layer_1_btch_16_adam",
    2:"02-lstm_layer_1_btch_16_adamw",
    3:"03-lstm_layer_1_btch_16_sgd",
    4:"04-lstm_layer_1_btch_16_momentum",
    9:"09-lstm_layer_2_btch_16_adamw",
    10:"10-lstm_layer_4_btch_16_adamw"
}
train_accuracies = []
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/train_loss.txt", "r") as train_file:
        data = l = [100-float(line) for line in train_file]
        train_accuracies.append({
            "Accuracy" : mean(data),
            "Model":args["model"],
            "Num of Layers":args["layers"],
            "Optimizer":args["optimizer"]
        })

def draw_table(dataset):
    header = dataset[0].keys()
    rows =  [x.values() for x in dataset]
    print(tabulate.tabulate(rows, header))

train_accuracies = sorted(train_accuracies, key=lambda x: x['Model'])
print("")
print("Sorted By the Architecture of the Model:")
print("")
draw_table(train_accuracies)
print("-----------------------------------------------------------------------")

print("Sorted By the Number of the Layers:")
print("")
train_accuracies = sorted(train_accuracies, key=lambda x: x['Num of Layers'])
draw_table(train_accuracies)
print("-----------------------------------------------------------------------")

print("Sorted By the Optimizer:")
print("")
train_accuracies = sorted(train_accuracies, key=lambda x: x['Optimizer'])
draw_table(train_accuracies)
print("-----------------------------------------------------------------------")