import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

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

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

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
#plt.show()

#CLASS DISTRIBUTION
plt.figure(figsize=(5,4))
df["class"].value_counts().plot(kind="bar")
plt.xticks([0,1],["Hadron (0)","Gammma (1)"],rotation = 0)
plt.ylabel("Count")
plt.title("Class Distribution")
#plt.show()


#TRAIN-VALIDATION-TEST
X = df.drop("class", axis = 1)
y = df["class"]

#train(70%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

#validation + test (15% + 15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# print("Train shape: ", X_train.shape)
# print("Validation shape: ", X_val.shape)
# print("Test shape: ", X_test.shape)

#NORMALIZATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns= X_train.columns)
# print(X_train_scaled_df)

#DATA BALANACING
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled,y_train)

plt.figure(figsize=(5,4))
pd.Series(y_train_bal).value_counts().plot(kind ="bar")
plt.xticks([0,1], ["Hadron (0)", "Gamma (1)"], rotation=0)
plt.ylabel("Count")
plt.title("Class Distribution After Balancing (Training Set)")
#plt.show()

#KNN MODEL
#training
knn = KNeighborsClassifier(n_neighbors=9,metric="euclidean")
knn.fit(X_train_bal,y_train_bal)

#validation
y_val_pred = knn.predict(X_val_scaled)
# print("Validation Accuracy: ",accuracy_score(y_val,y_val_pred))
# print("\nValidation Classification Report: \n")
# print(classification_report(y_val, y_val_pred))

#test
y_test_pred = knn.predict(X_test_scaled)
# print("Test Accuracy: ",accuracy_score(y_test,y_test_pred))
# print("\n Test Classification Report: \n")
# print(classification_report(y_test,y_test_pred))

#NAIVE BAYES

nb = GaussianNB()
nb.fit(X_train_bal,y_train_bal)

y_val_pred_nb = nb.predict(X_val_scaled)
# print("Validation Accuracy: ",accuracy_score(y_val,y_val_pred_nb))
# print(classification_report(y_val,y_val_pred_nb))

y_test_pred_nb = nb.predict(X_test_scaled)
# print("Test accuracy : ",accuracy_score(y_test, y_test_pred_nb))
# print(classification_report(y_test, y_test_pred_nb))

#LOGIISTIC REGRESSION
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_bal,y_train_bal)

y_val_pred_lr = lr.predict(X_val_scaled)
#print(classification_report(y_val, y_val_pred_lr))

y_test_pred_lr = lr.predict(X_test_scaled)
#print(classification_report(y_test, y_test_pred_lr))

#SUPPORT VECTOR MACHINE
svm = SVC(kernel="rbf", C = 1.0, random_state=42)
svm.fit(X_train_bal, y_train_bal)

y_val_pred_svm = svm.predict(X_val_scaled)
print(classification_report(y_val, y_val_pred_svm))

y_test_pred_svm = svm.predict(X_test_scaled)
print(classification_report(y_test, y_test_pred_svm))