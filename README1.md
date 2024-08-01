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

::: {#bb51866a-5ffb-43fd-9784-be85a50a2987 .cell .markdown}
# Machine Learning
:::

::: {#ec96b34f-1138-42f1-8920-5d0c200656be .cell .markdown}
A plataforma web de aprendizagem interativa
[freeCodeCamp](https://www.freecodecamp.org/) publicou o excelente curso
desenvolvido pela engenheira e cientista graduada pelo MIT, [Kylie
Ying](https://www.kylieying.com/), intitulado **Machine Learning for
Everybody** -- [Full
Course](https://www.youtube.com/watch?v=i_LwzRVP7bg).
:::

::: {#2be7c028-88c9-41ea-80ca-76b3916306e9 .cell .markdown}
## Classificação binária
:::

::: {#8ccffac5-84df-48a5-86d0-845a800f700e .cell .markdown}
No início do curso, foi abordado um problema de aprendizado
supervisionado, para classificação binária, em que o modelo aprende com
dados de treinamento devidamente rotulados, para no fim ser colocado à
prova com os dados de teste, após o ajuste com dados de validação. O
modelo deve classificar as partículas, a partir das características
fornecidas pelas *features*, em raios gama(*sinal*) ou hadron(*fundo*).

Foram utilizados diversos algoritmos de classificação.
:::

::: {#d6f6a74a-3b3e-4c34-b048-7d469bd52984 .cell .markdown}
### Dataset:
:::

::: {#20610b7d-3d37-4174-aeb7-e110fc092af3 .cell .markdown}
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
\[<http://archive.ics.uci.edu/ml>\]. Irvine, CA: University of
California, School of Information and Computer Science.

Donated by: P. Savicky Institute of Computer Science, AS of CR Czech
Republic savicky \'@\' cs.cas.cz
:::

::: {#c4dc7065-d1a9-43fb-9599-89b3bbafef00 .cell .markdown}
O dataset *MAGIC Gamma Telescope* foi gerado com o método de simulação
denominado *Monte Carlo*, utilizado para reproduzir indiretamente as
características dos raios gama que teriam provocado as chuvas
eletromagnéticas na atmosfera. Esse método é particularmente eficaz para
modelar sistemas complexos e estocásticos, como as interações de
partículas de alta energia na atmosfera, permitindo a simulação
detalhada dos processos físicos envolvidos e a coleta de dados sobre as
características esperadas das chuvas eletromagnéticas geradas pelos
raios gama.

O artigo original que relata a simulação pode ser obtido em:
<https://inspirehep.net/literature/469835>
:::

::: {#70e99956-6a8e-4cce-bc87-433aaabc4106 .cell .markdown}
## Radiação Cherenkov
:::

::: {#457c5f2a-3876-4001-b39c-4ba255b2294b .cell .markdown}
Os telescópios Cherenkov são baseados no solo e os raios gama são
absorvidos pela atmosfera da Terra antes de alcançarem a superfície.
Esses telescópios detectam a radiação Cherenkov produzida quando os
raios gama interagem com a atmosfera, criando chuvas de partículas
secundárias. Esse método indireto permite estudar os raios gama de alta
energia de forma segura e eficaz.

**Radiação Cherenkov**, em homenagem físico russo e soviético, Nobel de
Física em 1958, *Pavel Alexeevitch Tcherenkov*, é a luz emitida quando
uma partícula carregada, como um elétron, viaja através de um meio (como
água ou ar) a uma velocidade superior à velocidade da luz. Este fenômeno
é análogo ao *boom* sônico produzido por um objeto que viaja mais rápido
que a velocidade do som no ar. A radiação Cherenkov é emitida em um
ângulo característico em relação à direção da partícula, formando um
cone de luz azulada, que pode ser detectado por dispositivos
especializados, como telescópios Cherenkov atmosféricos, para estudar
partículas de alta energia e suas interações.

Alguns dispositivos foram desenvolvidos para a observação direta dos
raios gama na atmosfera e além, incluindo telescópios de raios gama
espaciais, colocados em órbita acima da atmosfera terrestre, onde podem
detectar diretamente os raios gama sem interferência atmosférica.
Citamos alguns exemplos:

**Telescópio Espacial de Raios Gama Fermi**: Lançado pela NASA, este
telescópio detecta raios gama de alta energia.

**Observatório de Raios Gama Compton**: Um observatório de raios gama
que operou de 1991 a 2000.

**Balões Estratosféricos**: Equipados com detectores de raios gama,
esses balões são lançados até a estratosfera, onde a densidade
atmosférica é muito menor, permitindo a observação direta dos raios
gama. Exemplos de missões incluem:

**Observações com Balões de Alta Altitude**: Programas de balões da NASA
e outras agências espaciais que transportam detectores para altitudes
onde a interferência atmosférica é mínima.

Esses instrumentos foram projetados para operar fora da influência da
densa atmosfera terrestre, permitindo a detecção direta e o estudo dos
raios gama provenientes de fontes cósmicas.
:::

::: {#a56de4cc-30ae-4d6d-976c-1ceacd945c8e .cell .markdown}
## Processo de modelagem padrão
:::

::: {#372bb491-b5b6-45ce-bf96-60ad58dbf0b4 .cell .markdown}
Em um processo de modelagem padrão, utilizamos dados de validação para
ajustar o modelo, especialmente quando estamos testando diferentes
algoritmos ou ajustando hiperparâmetros. Somente após escolher o melhor
modelo com base nos dados de validação é que utilizamos os dados de
teste para avaliar a performance final.

Um fluxo de trabalho adequado deve garantir que os dados de validação
sejam usados adequadamente para ajustar o modelo.
:::

::: {#a997e5c0-a9ee-4cea-a67a-fe8c5b0ab95f .cell .markdown}
Ao usar os dados de validação, você pode ajustar hiperparâmetros e
selecionar o melhor modelo sem contaminar os dados de teste, que devem
ser usados apenas para a avaliação final. Se você estiver testando
múltiplos modelos, pode repetir o processo de treinamento e validação
para cada modelo, comparando seus desempenhos nos dados de validação
para decidir qual modelo será avaliado nos dados de teste.
:::

::: {#ce11b42d-bfe8-4926-8d9e-21df2a7f8dd3 .cell .markdown}
De outra sorte, para que subdividir o dataset em *treinamento*,
*validação* e *teste*?
:::

::: {#f8f2ca72-f467-43ae-a073-05235e162bcd .cell .markdown}
## Importamos as bilbiotecas
:::

::: {#de2021c3-ec76-4f03-a299-5b5b3d4f4908 .cell .code execution_count="1"}
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
## Carregamos o dataset

### Renomeamos as colunas
:::

::: {#7a0d624e-720f-434f-af63-5b3d47a78105 .cell .code execution_count="2"}
``` python
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
```

::: {.output .execute_result execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fLength</th>
      <th>fWidth</th>
      <th>fSize</th>
      <th>fConc</th>
      <th>fConc1</th>
      <th>fAsym</th>
      <th>fM3Long</th>
      <th>fM3Trans</th>
      <th>fAlpha</th>
      <th>fDist</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.7967</td>
      <td>16.0021</td>
      <td>2.6449</td>
      <td>0.3918</td>
      <td>0.1982</td>
      <td>27.7004</td>
      <td>22.0110</td>
      <td>-8.2027</td>
      <td>40.0920</td>
      <td>81.8828</td>
      <td>g</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.6036</td>
      <td>11.7235</td>
      <td>2.5185</td>
      <td>0.5303</td>
      <td>0.3773</td>
      <td>26.2722</td>
      <td>23.8238</td>
      <td>-9.9574</td>
      <td>6.3609</td>
      <td>205.2610</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>162.0520</td>
      <td>136.0310</td>
      <td>4.0612</td>
      <td>0.0374</td>
      <td>0.0187</td>
      <td>116.7410</td>
      <td>-64.8580</td>
      <td>-45.2160</td>
      <td>76.9600</td>
      <td>256.7880</td>
      <td>g</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.8172</td>
      <td>9.5728</td>
      <td>2.3385</td>
      <td>0.6147</td>
      <td>0.3922</td>
      <td>27.2107</td>
      <td>-6.4633</td>
      <td>-7.1513</td>
      <td>10.4490</td>
      <td>116.7370</td>
      <td>g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.1362</td>
      <td>30.9205</td>
      <td>3.1611</td>
      <td>0.3168</td>
      <td>0.1832</td>
      <td>-5.5277</td>
      <td>28.5525</td>
      <td>21.8393</td>
      <td>4.6480</td>
      <td>356.4620</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#39e6ca56-323e-4176-8835-fba5da88fec7 .cell .markdown}
## Transformamos a variável categórica em numérica
:::

::: {#44fdeb68-fa8d-4d68-bb2f-b8295c980061 .cell .code execution_count="3"}
``` python
df["class"] = (df["class"] == "g").astype(int)
```
:::

::: {#ec096d25-4f81-4453-b981-f49b9fc5ebd5 .cell .code execution_count="4"}
``` python
df.head()
```

::: {.output .execute_result execution_count="4"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fLength</th>
      <th>fWidth</th>
      <th>fSize</th>
      <th>fConc</th>
      <th>fConc1</th>
      <th>fAsym</th>
      <th>fM3Long</th>
      <th>fM3Trans</th>
      <th>fAlpha</th>
      <th>fDist</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.7967</td>
      <td>16.0021</td>
      <td>2.6449</td>
      <td>0.3918</td>
      <td>0.1982</td>
      <td>27.7004</td>
      <td>22.0110</td>
      <td>-8.2027</td>
      <td>40.0920</td>
      <td>81.8828</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.6036</td>
      <td>11.7235</td>
      <td>2.5185</td>
      <td>0.5303</td>
      <td>0.3773</td>
      <td>26.2722</td>
      <td>23.8238</td>
      <td>-9.9574</td>
      <td>6.3609</td>
      <td>205.2610</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>162.0520</td>
      <td>136.0310</td>
      <td>4.0612</td>
      <td>0.0374</td>
      <td>0.0187</td>
      <td>116.7410</td>
      <td>-64.8580</td>
      <td>-45.2160</td>
      <td>76.9600</td>
      <td>256.7880</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.8172</td>
      <td>9.5728</td>
      <td>2.3385</td>
      <td>0.6147</td>
      <td>0.3922</td>
      <td>27.2107</td>
      <td>-6.4633</td>
      <td>-7.1513</td>
      <td>10.4490</td>
      <td>116.7370</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.1362</td>
      <td>30.9205</td>
      <td>3.1611</td>
      <td>0.3168</td>
      <td>0.1832</td>
      <td>-5.5277</td>
      <td>28.5525</td>
      <td>21.8393</td>
      <td>4.6480</td>
      <td>356.4620</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#1190a57e-34a5-4402-834f-dac2c1e7c893 .cell .markdown}
## Visualizamos as distribuições
:::

::: {#65289c79-d9be-4d49-9e57-dfdf6023efe8 .cell .markdown}
**Criamos uma matriz de gráficos de dispersão:**
:::

::: {#599246db-3f56-4225-a8cc-fabe1f32e834 .cell .code execution_count="5"}
``` python
import seaborn as sns
dados = df.iloc[:, :-1].copy()
sns.pairplot(data=dados)
```

::: {.output .execute_result execution_count="5"}
    <seaborn.axisgrid.PairGrid at 0x778b3eea8d30>
:::

::: {.output .display_data}
![](vertopal_b542f81007a544e69d706e1b961baf8d/b231f66556e3a8ed62130f3371cc491f9711eb34.png)
:::
:::

::: {#1c20c72c-b7af-4f67-bb69-017262a4a114 .cell .markdown}
**Calculamos a matriz de correlação:**
:::

::: {#bcddde59-3d2f-4692-aa94-6271fcf33c31 .cell .code execution_count="6"}
``` python
corr = df.drop(labels='class',axis=1).corr()
sns.heatmap(data=corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=0.5,annot=True)
```

::: {.output .execute_result execution_count="6"}
    <Axes: >
:::

::: {.output .display_data}
![](vertopal_b542f81007a544e69d706e1b961baf8d/af5a587f19485d9611425279a4826f6aab7b12e3.png)
:::
:::

::: {#be271374-ca5c-4c07-b0df-b7f2bbb14e26 .cell .markdown}
### Separação do dataset em treinamento, validação e teste
:::

::: {#19f773c0-d9a5-4613-9722-cf917e5c3da8 .cell .code execution_count="7"}
``` python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
```
:::

::: {#c660a39b-79f3-4873-b568-161098ba6730 .cell .markdown}
### Conferimos o balanceamento dos dados
:::

::: {#b9dd5192-7423-41cf-9d6f-7e64830cc096 .cell .code execution_count="8"}
``` python
print(len(train[train["class"]==1]))
```

::: {.output .stream .stdout}
    7415
:::
:::

::: {#6d9dec7c-4fbe-453d-a00d-c08736e9b4b2 .cell .code execution_count="9"}
``` python
print(len(train[train["class"]==0]))
```

::: {.output .stream .stdout}
    3997
:::
:::

::: {#b2ba3a91-1286-4ca0-9ebe-f1b059c4225a .cell .markdown}
## Padronização e balanceamento dos dados
:::

::: {#98e64b19-a8d9-4bbf-bf6d-50d08804d519 .cell .markdown}
Além da padronização dos dados, utilizamos o método *RandomOverSampler*
para reamostragem dos dados desbalanceados.
:::

::: {#962e0d72-99d5-4933-a8c9-a2da343e4bcb .cell .code execution_count="10"}
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

::: {#d58e2ad7-596d-41cd-bf41-6a8777a4d15a .cell .code execution_count="11"}
``` python
train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
```
:::

::: {#832b5114-c739-4060-b441-b5294987d207 .cell .markdown}
# Algoritmos de classificação
:::

::: {#326a5c3d-a10b-452a-954f-abc1a2ab4e13 .cell .markdown}
## KNN
:::

::: {#6583be2a-bcbd-4449-b9d2-2c6e91592c3e .cell .markdown}
O KNN é um algoritmo de aprendizado supervisionado que classifica uma
nova amostra com base na maioria dos \"vizinhos\" mais próximos. Neste
caso, utilizamos `k=5`, o que significa que a classificação de uma nova
amostra será baseada nas 5 amostras mais próximas no espaço dos
recursos.
:::

::: {#0e3ac17b-022c-4b0c-bf96-0f7a4520bf63 .cell .code execution_count="12"}
``` python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
```
:::

::: {#01ec929d-9a44-4786-b8be-8d85b0a5043e .cell .code execution_count="13"}
``` python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="13"}
```{=html}
<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>
```
:::
:::

::: {#bd53b898-2f74-4750-b962-e72ab7e4e3ff .cell .code execution_count="14"}
``` python
y_valid_pred = knn_model.predict(X_valid)
```
:::

::: {#753e721b-68b0-4000-94cb-e108d28ca7e2 .cell .code execution_count="15"}
``` python
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.74      0.72      0.73      1334
               1       0.85      0.86      0.86      2470

        accuracy                           0.81      3804
       macro avg       0.80      0.79      0.79      3804
    weighted avg       0.81      0.81      0.81      3804
:::
:::

::: {#5d79655c-a7eb-46c0-b7b1-b70014e16be6 .cell .markdown}
# Naive Bayes
:::

::: {#4ba0c84f-6df5-4740-b1cb-3869e8291598 .cell .markdown}
O Naive Bayes é baseado no teorema de Bayes, que assume independência
entre os predictores. Este modelo é útil para problemas de classificação
binária e multiclasse.
:::

::: {#5071cf78-b7fd-424c-bfe3-bcea071a0dbe .cell .code execution_count="16"}
``` python
from sklearn.naive_bayes import GaussianNB
```
:::

::: {#fd0ebf28-26c6-4a0e-b912-472069e65f45 .cell .code execution_count="17"}
``` python
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
```
:::

::: {#97e135e1-c1b2-413d-82ec-b33a295e8f6e .cell .code execution_count="18"}
``` python
y_valid_pred = nb_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.69      0.40      0.50      1334
               1       0.74      0.91      0.81      2470

        accuracy                           0.73      3804
       macro avg       0.71      0.65      0.66      3804
    weighted avg       0.72      0.73      0.70      3804
:::
:::

::: {#7c182468-36e0-423a-ad9f-5aaf534061b5 .cell .code execution_count="19"}
``` python
# Log Regression
```
:::

::: {#33bb14e8-a01f-448d-9506-d6538d6c0a52 .cell .code execution_count="20"}
``` python
from sklearn.linear_model import LogisticRegression
```
:::

::: {#c4ac585d-f725-4447-8495-fc780c912e58 .cell .code execution_count="21"}
``` python
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
```
:::

::: {#7b9f2071-8c90-4444-9c05-cd955187c89a .cell .code execution_count="22"}
``` python
y_valid_pred = lg_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.69      0.72      0.71      1334
               1       0.85      0.83      0.84      2470

        accuracy                           0.79      3804
       macro avg       0.77      0.77      0.77      3804
    weighted avg       0.79      0.79      0.79      3804
:::
:::

::: {#f3b98d23-65df-4f93-a13a-b07ac4b313de .cell .markdown}
# SVM
:::

::: {#39b956f0-da5b-4110-821f-d4cf9b26af02 .cell .markdown}
O SVM é um algoritmo que encontra um hiperplano que melhor separa as
classes de dados. Utilizamos o SVM para garantir uma separação máxima
entre as classes.
:::

::: {#d6ecdc3d-bc7c-4846-9d1f-28d1f105008a .cell .code execution_count="23"}
``` python
from sklearn.svm import SVC
```
:::

::: {#a89fb912-9de4-4729-907b-3e16422706df .cell .code execution_count="24"}
``` python
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
```
:::

::: {#eb5bb443-cd98-4fd2-ba17-83545e5b716b .cell .code execution_count="25"}
``` python
y_valid_pred = svm_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.81      0.80      0.81      1334
               1       0.89      0.90      0.90      2470

        accuracy                           0.87      3804
       macro avg       0.85      0.85      0.85      3804
    weighted avg       0.87      0.87      0.87      3804
:::
:::

::: {#843b2a1c-8b59-4638-a7a0-69e7e9f05ba0 .cell .markdown}
# Logistic Regression
:::

::: {#81eb4b0e-4099-4644-8179-a8340fc394d3 .cell .markdown}
A Regressão Logística é um modelo estatístico utilizado para problemas
de classificação binária. Ela estima a probabilidade de uma variável
dependente pertencer a uma determinada classe com base em uma ou mais
variáveis independentes.
:::

::: {#af014e04-e362-4ad0-87c8-9b515d767739 .cell .code execution_count="26"}
``` python
from sklearn.linear_model import LogisticRegression
```
:::

::: {#76f5e578-2e51-4d65-af34-70d9d4ee64e5 .cell .code execution_count="27"}
``` python
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="27"}
```{=html}
<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>
```
:::
:::

::: {#a2fa405c-2291-463d-a06e-d2f219249d10 .cell .code execution_count="28"}
``` python
y_valid_pred = logreg_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.69      0.72      0.71      1334
               1       0.85      0.83      0.84      2470

        accuracy                           0.79      3804
       macro avg       0.77      0.77      0.77      3804
    weighted avg       0.79      0.79      0.79      3804
:::
:::

::: {#874a5f2b-d97e-4ae0-8814-607a3b93ae7e .cell .markdown}
# Random Forest
:::

::: {#5019fa05-0bd5-4307-9e36-48e6764bb77a .cell .markdown}
O Random Forest é um conjunto de múltiplas árvores de decisão, onde cada
árvore é treinada com uma amostra diferente do dataset. Ele é conhecido
por melhorar a precisão e reduzir o overfitting ao combinar as previsões
de várias árvores.
:::

::: {#6ab3ed27-2164-4d97-93ce-e707c5715fe9 .cell .code execution_count="29"}
``` python
from sklearn.ensemble import RandomForestClassifier
```
:::

::: {#785a4011-de47-48ea-bd12-3093f2af6c79 .cell .code execution_count="30"}
``` python
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="30"}
```{=html}
<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier()</pre></div></div></div></div></div>
```
:::
:::

::: {#689f9e70-786a-46db-888b-14bc66ea20c6 .cell .code execution_count="31"}
``` python
y_valid_pred = rf_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.86      0.80      0.83      1334
               1       0.90      0.93      0.91      2470

        accuracy                           0.88      3804
       macro avg       0.88      0.86      0.87      3804
    weighted avg       0.88      0.88      0.88      3804
:::
:::

::: {#9a3e603d-e8e9-46d7-bea9-6430b5a8984b .cell .markdown}
# AdaBoost
:::

::: {#104f2484-5f77-470b-91d4-1ce31ea815af .cell .markdown}
O AdaBoost é um algoritmo de ensemble que combina a performance de
múltiplos classificadores fracos para formar um classificador forte. Ele
ajusta iterativamente os pesos das instâncias para focar nos erros mais
difíceis.
:::

::: {#6385d43d-52b8-43c7-8088-1f71b3628fb1 .cell .code execution_count="32"}
``` python
from sklearn.ensemble import AdaBoostClassifier
```
:::

::: {#d054fb07-ba6e-4ded-be8f-d7cf73e7f048 .cell .code execution_count="33"}
``` python
ada_model = AdaBoostClassifier(n_estimators=50)
ada_model.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="33"}
```{=html}
<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>AdaBoostClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier()</pre></div></div></div></div></div>
```
:::
:::

::: {#35ed53bb-b012-4368-9f5f-3c2a8be04157 .cell .code execution_count="34"}
``` python
y_valid_pred = ada_model.predict(X_valid)
print(classification_report(y_valid, y_valid_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.72      0.82      0.77      1334
               1       0.90      0.83      0.86      2470

        accuracy                           0.83      3804
       macro avg       0.81      0.83      0.82      3804
    weighted avg       0.84      0.83      0.83      3804
:::
:::

::: {#bbe36eaa-c3c3-42ef-834b-6123e39c7722 .cell .markdown}
## Escolha do modelo para teste
:::

::: {#c90a6999-ca69-4a0e-b8be-b9bd7ae8d412 .cell .markdown}
Escolhemos o modelo que apresentou melhor desempenho durante a fase de
validação, para submeter os dados de teste.
:::

::: {#a5349798-19d3-457b-815f-0ec9c9a37d0c .cell .code execution_count="35"}
``` python
y_test_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_test_pred))
```

::: {.output .stream .stdout}
                  precision    recall  f1-score   support

               0       0.84      0.78      0.81      1357
               1       0.88      0.92      0.90      2447

        accuracy                           0.87      3804
       macro avg       0.86      0.85      0.86      3804
    weighted avg       0.87      0.87      0.87      3804
:::
:::

::: {#847df0da-a551-43de-8465-9597c9d3691a .cell .markdown}
## Principais métricas
:::

::: {#b3b3d924-d402-4ae9-8375-25cb2062e045 .cell .markdown}
A seguir, explicamos os principais indicadores fornecidos pelo
*classification_report* do *sklearn*:

1.  **Precision (Precisão)**: A precisão é a proporção de verdadeiros
    positivos entre as previsões positivas feitas pelo modelo. Ou seja,
    ela mede a exatidão das previsões positivas do modelo.

$$ \text{Precisão} = \frac{\text{Verdadeiros Positivos (VP)}}{\text{Verdadeiros Positivos (VP)} + \text{Falsos Positivos (FP)}} $$

1.  **Recall (Recall/Sensibilidade)**: O recall é a proporção de
    verdadeiros positivos entre todas as amostras que realmente
    pertencem à classe positiva. Ele mede a capacidade do modelo de
    encontrar todas as instâncias positivas.

$$ \text{Recall} = \frac{\text{Verdadeiros Positivos (VP)}}{\text{Verdadeiros Positivos (VP)} + \text{Falsos Negativos (FN)}} $$

1.  **F1-score**: O F1-score é a média harmônica entre precisão e
    recall. Ele fornece uma única métrica que balanceia ambos, sendo
    útil especialmente quando há um desequilíbrio entre classes.

$$ \text{F1-score} = 2 \times \frac{\text{Precisão} \times \text{Recall}}{\text{Precisão} + \text{Recall}} $$

1.  **Support**: O suporte é o número de ocorrências reais de cada
    classe no dataset. Ele indica a quantidade de instâncias que cada
    classe possui no conjunto de dados de teste.

$$ \text{Support} = \text{Número de exemplos da classe} $$

1.  **Accuracy (Acurácia)**: A acurácia é a proporção de previsões
    corretas (tanto verdadeiros positivos quanto verdadeiros negativos)
    em relação ao total de previsões feitas. Ela fornece uma visão geral
    do desempenho do modelo.

$$ \text{Acurácia} = \frac{\text{Verdadeiros Positivos (VP)} + \text{Verdadeiros Negativos (VN)}}{\text{Total de Exemplos}} $$

​
:::

::: {#65bd3845-8f4c-498e-9dcd-308325be0e48 .cell .markdown}
### Indicadores do modelo Random Forest, na fase de teste
:::

::: {#a59676b7-d273-4d21-ae51-b819e3ad8627 .cell .markdown}
**Classe 0:**

Precisão: 0.84 (84% das previsões para a classe 0 estavam
corretas)`<br>`{=html} Recall: 0.78 (78% das instâncias reais da classe
0 foram corretamente identificadas)`<br>`{=html} F1-score: 0.81 (média
harmônica de precisão e recall para a classe 0)`<br>`{=html} Support:
1357 (há 1357 instâncias reais da classe 0 no dataset de teste)

**Classe 1:**

Precisão: 0.88 (88% das previsões para a classe 1 estavam
corretas)`<br>`{=html} Recall: 0.92 (92% das instâncias reais da classe
1 foram corretamente identificadas)`<br>`{=html} F1-score: 0.90 (média
harmônica de precisão e recall para a classe 1)`<br>`{=html} Support:
2447 (há 2499 instâncias reais da classe 1 no dataset de
teste)`<br>`{=html}

Acurácia Geral: 0.87 (87% das previsões totais estavam corretas)

**Macro Average (média das métricas por classe):**

Precisão: 0.86`<br>`{=html} Recall: 0.85`<br>`{=html} F1-score: 0.86

**Weighted Average** (média ponderada das métricas, considerando o
suporte de cada classe):`<br>`{=html}

Precisão: 0.87`<br>`{=html} Recall: 0.87`<br>`{=html} F1-score: 0.87

Esses indicadores mostram que o modelo RandomForestClassifier teve um
bom desempenho, com uma acurácia de 87% e valores altos de precisão,
recall e F1-score para ambas as classes.
:::
