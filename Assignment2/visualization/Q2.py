from statistics import mean
import json
import tabulate

executed_configs = {
    1:"01-lstm_layer_1_btch_16_adam",
    2:"02-lstm_layer_1_btch_16_adamw",
    3:"03-lstm_layer_1_btch_16_sgd",
    4:"04-lstm_layer_1_btch_16_momentum",
    5:"05-gpt1_layer_1_btch_16_adam",
    6:"06-gpt1_layer_1_btch_16_adamw",
    7:"07-gp1_layer_1_btch_16_sgd",
    8:"08-gpt1_layer_1_btch_16_momentum",
    9:"09-lstm_layer_2_btch_16_adamw",
    10:"10-lstm_layer_4_btch_16_adamw",
    11:"11-gpt1_layer_2_btch_16_adamw",
    12:"12-gpt1_layer_4_btch_16_adamw"
}
data = []
for x in executed_configs:
    folder_name = executed_configs[x]
    args = json.load(open(f"../LoggedData/TrVlTs/{folder_name}/args.json", "r"))
    with open(f"../LoggedData/TrVlTs/{folder_name}/train_ppl.txt", "r") as train_file:
        with open(f"../LoggedData/TrVlTs/{folder_name}/valid_ppl.txt", "r") as valid_file:
            with open(f"../LoggedData/TrVlTs/{folder_name}/test_ppl.txt", "r") as test_file:
                data.append({
                    "Train PPL" : mean([float(line) for line in train_file]),
                    "Validation PPL": mean([float(line) for line in valid_file]),
                    "Test PPL": mean([float(line) for line in test_file]),
                    "Model":args["model"],
                    "Num of Layers":args["layers"],
                    "Optimizer":args["optimizer"]
                })

def draw_table(dataset):
    header = dataset[0].keys()
    rows =  [x.values() for x in dataset]
    print(tabulate.tabulate(rows, header))

data = sorted(data, key=lambda x: x['Model'])
print("")
print("Sorted By the Architecture of the Model:")
print("")
draw_table(data)
print("-----------------------------------------------------------------------")

print("Sorted By the Number of the Layers:")
print("")
data = sorted(data, key=lambda x: x['Num of Layers'])
draw_table(data)
print("-----------------------------------------------------------------------")

print("Sorted By the Optimizer:")
print("")
data = sorted(data, key=lambda x: x['Optimizer'])
draw_table(data)
print("-----------------------------------------------------------------------")
#%%
