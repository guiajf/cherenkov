---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#f8f2ca72-f467-43ae-a073-05235e162bcd .cell .markdown}
### Importamos as bilbiotecas
:::

::: {#de2021c3-ec76-4f03-a299-5b5b3d4f4908 .cell .code}
``` python
                                                                                import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
```
:::

::: {#165e53ee-8517-42f4-9eb3-1c9bdb745669 .cell .markdown}
### Carregamos o dataset

**Renomeamos as colunas**
:::

::: {#7a0d624e-720f-434f-af63-5b3d47a78105 .cell .code}
``` python
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
```
:::

::: {#39e6ca56-323e-4176-8835-fba5da88fec7 .cell .markdown}
### Transformamos a variável categórica em numérica
:::

::: {#44fdeb68-fa8d-4d68-bb2f-b8295c980061 .cell .code}
``` python
df["class"] = (df["class"] == "g").astype(int)
```
:::

::: {#ec096d25-4f81-4453-b981-f49b9fc5ebd5 .cell .code}
``` python
df.head()
```
:::

::: {#1190a57e-34a5-4402-834f-dac2c1e7c893 .cell .markdown}
### Visualizamos as distribuições
:::

::: {#65289c79-d9be-4d49-9e57-dfdf6023efe8 .cell .markdown}
**Criamos uma matriz de gráficos de dispersão:**
:::

::: {#599246db-3f56-4225-a8cc-fabe1f32e834 .cell .code}
``` python
import seaborn as sns
dados = df.iloc[:, :-1].copy()
sns.pairplot(data=dados)
```
:::

::: {#1c20c72c-b7af-4f67-bb69-017262a4a114 .cell .markdown}
**Calculamos a matriz de correlação:**
:::

::: {#bcddde59-3d2f-4692-aa94-6271fcf33c31 .cell .code}
``` python
corr = df.drop(labels='class',axis=1).corr()
sns.heatmap(data=corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=0.5,annot=True)
```
:::

::: {#be271374-ca5c-4c07-b0df-b7f2bbb14e26 .cell .markdown}
### Separação do dataset em treinamento, validação e teste
:::

::: {#19f773c0-d9a5-4613-9722-cf917e5c3da8 .cell .code}
``` python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
```
:::

::: {#c660a39b-79f3-4873-b568-161098ba6730 .cell .markdown}
### Conferimos o balanceamento dos dados
:::

::: {#b9dd5192-7423-41cf-9d6f-7e64830cc096 .cell .code}
``` python
print(len(train[train["class"]==1]))
```
:::

::: {#6d9dec7c-4fbe-453d-a00d-c08736e9b4b2 .cell .code}
``` python
print(len(train[train["class"]==0]))
```
:::

::: {#b2ba3a91-1286-4ca0-9ebe-f1b059c4225a .cell .markdown}
### Padronização e balanceamento dos dados
:::

::: {#98e64b19-a8d9-4bbf-bf6d-50d08804d519 .cell .markdown}
Além da padronização dos dados, utilizamos o método *RandomOverSampler*
para reamostragem dos dados desbalanceados.
:::

::: {#962e0d72-99d5-4933-a8c9-a2da343e4bcb .cell .code}
``` python
    def scale_dataset(dataframe, oversample=False):
      X = dataframe[dataframe.columns[:-1]].values
      y = dataframe[dataframe.columns[-1]].values
    
      scaler = StandardScaler()
      X = scaler.fit_transform(X)
    
      if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    
      data = np.hstack((X, np.reshape(y, (-1, 1))))
    
      return data, X, y
```
:::

::: {#d58e2ad7-596d-41cd-bf41-6a8777a4d15a .cell .code}
``` python
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
```
:::

::: {#326a5c3d-a10b-452a-954f-abc1a2ab4e13 .cell .markdown}
## KNN
:::

::: {#0e3ac17b-022c-4b0c-bf96-0f7a4520bf63 .cell .code}
``` python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
```
:::

::: {#01ec929d-9a44-4786-b8be-8d85b0a5043e .cell .code}
``` python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```
:::

::: {#bd53b898-2f74-4750-b962-e72ab7e4e3ff .cell .code}
``` python
y_valid_pred = knn_model.predict(X_valid)
```
:::

::: {#753e721b-68b0-4000-94cb-e108d28ca7e2 .cell .code}
``` python
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#5d79655c-a7eb-46c0-b7b1-b70014e16be6 .cell .markdown}
## Naive Bayes
:::

::: {#5071cf78-b7fd-424c-bfe3-bcea071a0dbe .cell .code}
``` python
from sklearn.naive_bayes import GaussianNB
```
:::

::: {#fd0ebf28-26c6-4a0e-b912-472069e65f45 .cell .code}
``` python
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
```
:::

::: {#97e135e1-c1b2-413d-82ec-b33a295e8f6e .cell .code}
``` python
y_valid_pred = nb_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#7c182468-36e0-423a-ad9f-5aaf534061b5 .cell .code}
``` python
# Log Regression
```
:::

::: {#33bb14e8-a01f-448d-9506-d6538d6c0a52 .cell .code}
``` python
from sklearn.linear_model import LogisticRegression
```
:::

::: {#c4ac585d-f725-4447-8495-fc780c912e58 .cell .code}
``` python
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
```
:::

::: {#7b9f2071-8c90-4444-9c05-cd955187c89a .cell .code}
``` python
y_valid_pred = lg_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#f3b98d23-65df-4f93-a13a-b07ac4b313de .cell .markdown}
## SVM
:::

::: {#d6ecdc3d-bc7c-4846-9d1f-28d1f105008a .cell .code}
``` python
from sklearn.svm import SVC
```
:::

::: {#a89fb912-9de4-4729-907b-3e16422706df .cell .code}
``` python
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
```
:::

::: {#eb5bb443-cd98-4fd2-ba17-83545e5b716b .cell .code}
``` python
y_valid_pred = svm_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#843b2a1c-8b59-4638-a7a0-69e7e9f05ba0 .cell .markdown}
## Logistic Regression
:::

::: {#af014e04-e362-4ad0-87c8-9b515d767739 .cell .code}
``` python
from sklearn.linear_model import LogisticRegression
```
:::

::: {#76f5e578-2e51-4d65-af34-70d9d4ee64e5 .cell .code}
``` python
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
```
:::

::: {#a2fa405c-2291-463d-a06e-d2f219249d10 .cell .code}
``` python
y_valid_pred = logreg_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#874a5f2b-d97e-4ae0-8814-607a3b93ae7e .cell .markdown}
## Random Forest
:::

::: {#6ab3ed27-2164-4d97-93ce-e707c5715fe9 .cell .code}
``` python
from sklearn.ensemble import RandomForestClassifier
```
:::

::: {#785a4011-de47-48ea-bd12-3093f2af6c79 .cell .code}
``` python
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
```
:::

::: {#689f9e70-786a-46db-888b-14bc66ea20c6 .cell .code}
``` python
y_valid_pred = rf_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#9a3e603d-e8e9-46d7-bea9-6430b5a8984b .cell .markdown}
## AdaBoost
:::

::: {#6385d43d-52b8-43c7-8088-1f71b3628fb1 .cell .code}
``` python
from sklearn.ensemble import AdaBoostClassifier
```
:::

::: {#d054fb07-ba6e-4ded-be8f-d7cf73e7f048 .cell .code}
``` python
ada_model = AdaBoostClassifier(n_estimators=50)
ada_model.fit(X_train, y_train)
```
:::

::: {#35ed53bb-b012-4368-9f5f-3c2a8be04157 .cell .code}
``` python
y_valid_pred = ada_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```
:::

::: {#bbe36eaa-c3c3-42ef-834b-6123e39c7722 .cell .markdown}
### Escolha do modelo para teste
:::

::: {#a5349798-19d3-457b-815f-0ec9c9a37d0c .cell .code}
``` python
y_test_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_test_pred))
```
:::
