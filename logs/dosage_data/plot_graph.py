import os
import pandas as pd
import matplotlib.pyplot as plt

x = []
acc = []
loss = []
for file in os.listdir("."):
    if "csv" in file:
        muons = int(float(file.replace("_history.csv", "").replace("model_", "")) * 46000)
        x.append(muons)

        df = pd.read_csv(file)
        acc.append(max(df["val_accuracy"]))
        loss.append(min(df["val_loss"]))

plt.scatter(x, acc)
plt.xlabel("accuracy")
plt.ylabel("dosage (no. of muons)")
plt.show()

plt.scatter(x, loss)
plt.xlabel("loss")
plt.ylabel("dosage (no. of muons)")
plt.show()