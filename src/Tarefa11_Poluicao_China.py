# Rede neural de multiplos previsores ou seja mais de uma entrada para prever a pouluição na china
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

base = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/poluicao.csv')
base = base.dropna()

# Dropa colunas que nao serão utilizadas
base = base.drop('No', axis=1)
base = base.drop('year', axis=1)
base = base.drop('month', axis=1)
base = base.drop('day', axis=1)
base = base.drop('hour', axis=1)
base = base.drop('cbwd', axis=1)

# todos menos o indice 0, que é o que queremos prever
base_treinamento = base.iloc[:, 1:7].values
# Normaliza
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
# Pega a poluição para fazermos a previsao
poluicao = base.iloc[:, 0].values
# Transformando o Array de 1D para 2D, para aplicar o transform
poluicao = poluicao.reshape(-1, 1)
poluicao_normalizado = normalizador.fit_transform(poluicao)

previsores = []
poluicao_real = []
for i in range(10, len(base_treinamento_normalizada)):
    # pega as colunas 0 ate a 6 pois aqui temos tudo o que precisamos
    previsores.append(base_treinamento_normalizada[i-10:i, 0:6])
    poluicao_real.append(poluicao_normalizado[i, 0])
# Transforma de lista para numpy array, para aparecer certinho em coluna tudo
previsores, poluicao_real = np.array(previsores), np.array(poluicao_real)

# Quebrando a base de dados para realizarmos testes posterioires.
previsores_treinamento, previsores_testes, classe_treinamento, classe_teste = train_test_split(
    previsores, poluicao_real, test_size=0.25)

# Comeca a montar a rede neural, o valor 6 é a 3 dimensao como se fosse é a quantidade de atributos para previsao
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
              input_shape=(previsores_treinamento.shape[1], 6)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=90, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))
regressor.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
# parar o processamento se ocorrer um overfiting dos dados, ou seja se nao melhorar para.
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=15, verbose=1)
# Reduz a taxa de aprendizado quando um valor para de melhorar
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# Salva os melhores pesos a cada epoca.
mcp = ModelCheckpoint(filepath='/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/pesos.h5',
                      monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores_treinamento, classe_treinamento, epochs=200,
              batch_size=64, callbacks=[es, rlr, mcp])

# como nao tinha base de testes baixei o exemplo do professor para verificar como tinha feito essa parte
# e verifiquei que ele utilizou previsores, com isso pensei em fazer a quebra da base de dados quenem faziamos
# de pegar 25% da base para testes e com isso obtive resultados bem diferentes.
previsoes = regressor.predict(previsores_testes)
# volta para a escala de numeração normal para facilitar a comparação
previsoes = normalizador.inverse_transform(previsoes)

# Monta o grafico
plt.plot(poluicao, color='red', label='Poluição real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão poluição(previsores_testes)')
plt.xlabel('Horas')
plt.ylabel('Valor poluição')
plt.legend()
plt.show()
