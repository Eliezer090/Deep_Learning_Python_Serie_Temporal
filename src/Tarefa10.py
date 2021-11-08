# Rede neural para prever a queda das acoes da petrobras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_treinamento_ex.csv')
base = base.dropna()

base_teste = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_teste_ex.csv')


base_treinamento = base.iloc[:, 1:2].values

normalizador = MinMaxScaler(feature_range=(0, 1))

base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []

for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(
    previsores, (previsores.shape[0], previsores.shape[1], 1))

# Comeca a montar a rede neural
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
              input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=90, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs=100, batch_size=32)

# Teste
preco_real_teste = base_teste.iloc[:, 1:2].values
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste)-90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

x_teste = []

for i in range(90, 109):
    x_teste.append(entradas[i-90:i, 0])
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))
previsoes = regressor.predict(x_teste)
previsoes = normalizador.inverse_transform(previsoes)

# Montando um grafico para visualizar as previsoes
plt.plot(preco_real_teste, color='red', label='Preco Real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsoes precos das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()
