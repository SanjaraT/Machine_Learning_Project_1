import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

cols = ["fLength","fwidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df = pd.read_csv("magic04.data",names=cols)
#print(df.head())

df["class"] =  (df["class"]=="g").astype(int)
#print(df.head(10))

#Data Visualization
for label in cols[:-1]:
    plt.hist(df[df["class"]==1][label],color='blue',label='gamma',alpha =0.7,density=True)
    plt.hist(df[df["class"]==0][label],color='red',label='lamda',alpha =0.7,density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel("Label")
    plt.legend()
    #plt.show()

# Correlation matrix
corr = df.corr()

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")


plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8
        )

plt.title("Correlation Heatmap of MAGIC Dataset")
plt.tight_layout()
plt.show()