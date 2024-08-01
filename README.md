# Cherenkov
Algoritmos de *machine learning* para classificação binária de partículas.

## Machine Learning
A plataforma web de aprendizagem interativa [freeCodeCamp](https://www.freecodecamp.org/) publicou o excelente curso desenvolvido pela engenheira e cientista graduada pelo MIT, [Kylie Ying](https://www.kylieying.com/), intitulado **Machine Learning for Everybody** – [Full Course](https://www.youtube.com/watch?v=i_LwzRVP7bg). 

## Classificação binária
No início do curso, foi abordado um problema de aprendizado supervisionado, para classificação binária, em que o modelo aprende com dados de treinamento devidamente rotulados, para no fim ser colocado à prova com os dados de teste, após o ajuste com dados de validação. O modelo deve classificar as partículas, a partir das características fornecidas pelas features, em raios gama(sinal) ou hadron(fundo).

Foram utilizados diversos algoritmos de classificação.

## Dataset:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Donated by:
P. Savicky
Institute of Computer Science, AS of CR
Czech Republic
savicky '@' cs.cas.cz

O dataset MAGIC Gamma Telescope foi gerado com o método de simulação denominado Monte Carlo, utilizado para reproduzir indiretamente as características dos raios gama que teriam provocado as chuvas eletromagnéticas na atmosfera. Esse método é particularmente eficaz para modelar sistemas complexos e estocásticos, como as interações de partículas de alta energia na atmosfera, permitindo a simulação detalhada dos processos físicos envolvidos e a coleta de dados sobre as características esperadas das chuvas eletromagnéticas geradas pelos raios gama.

O artigo original que relata a simulação pode ser obtido em: https://inspirehep.net/literature/469835

## Radiação Cherenkov
Os telescópios Cherenkov são baseados no solo e os raios gama são absorvidos pela atmosfera da Terra antes de alcançarem a superfície. Esses telescópios detectam a radiação Cherenkov produzida quando os raios gama interagem com a atmosfera, criando chuvas de partículas secundárias. Esse método indireto permite estudar os raios gama de alta energia de forma segura e eficaz.

Radiação Cherenkov, em homenagem físico russo e soviético, Nobel de Física em 1958, Pavel Alexeevitch Tcherenkov, é a luz emitida quando uma partícula carregada, como um elétron, viaja através de um meio (como água ou ar) a uma velocidade superior à velocidade da luz. Este fenômeno é análogo ao boom sônico produzido por um objeto que viaja mais rápido que a velocidade do som no ar. A radiação Cherenkov é emitida em um ângulo característico em relação à direção da partícula, formando um cone de luz azulada, que pode ser detectado por dispositivos especializados, como telescópios Cherenkov atmosféricos, para estudar partículas de alta energia e suas interações.

Alguns dispositivos foram desenvolvidos para a observação direta dos raios gama na atmosfera e além, incluindo telescópios de raios gama espaciais, colocados em órbita acima da atmosfera terrestre, onde podem detectar diretamente os raios gama sem interferência atmosférica. Citamos alguns exemplos:

**Telescópio Espacial de Raios Gama Fermi:** Lançado pela NASA, este telescópio detecta raios gama de alta energia.

Observatório de Raios Gama Compton: Um observatório de raios gama que operou de 1991 a 2000.

**Balões Estratosféricos:** Equipados com detectores de raios gama, esses balões são lançados até a estratosfera, onde a densidade atmosférica é muito menor, permitindo a observação direta dos raios gama. Exemplos de missões incluem:

**Observações com Balões de Alta Altitude:** Programas de balões da NASA e outras agências espaciais que transportam detectores para altitudes onde a interferência atmosférica é mínima.

Esses instrumentos foram projetados para operar fora da influência da densa atmosfera terrestre, permitindo a detecção direta e o estudo dos raios gama provenientes de fontes cósmicas.

## Processo de modelagem padrão
Em um processo de modelagem padrão, utilizamos dados de validação para ajustar o modelo, especialmente quando estamos testando diferentes algoritmos ou ajustando hiperparâmetros. Somente após escolher o melhor modelo com base nos dados de validação é que utilizamos os dados de teste para avaliar a performance final.

Um fluxo de trabalho adequado deve garantir que os dados de validação sejam usados adequadamente para ajustar o modelo.

Ao usar os dados de validação, você pode ajustar hiperparâmetros e selecionar o melhor modelo sem contaminar os dados de teste, que devem ser usados apenas para a avaliação final. Se você estiver testando múltiplos modelos, pode repetir o processo de treinamento e validação para cada modelo, comparando seus desempenhos nos dados de validação para decidir qual modelo será avaliado nos dados de teste.

De outra sorte, para que subdividir o dataset em treinamento, validação e teste?

## Importamos as bilbiotecas

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
``

## Carregamos o dataset
### Renomeamos as colunas

```
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()
```



