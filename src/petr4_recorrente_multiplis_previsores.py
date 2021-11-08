# Rede neural de multiplos previsores ou seja mais de uma entrada
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

base = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_treinamento.csv')
base = base.dropna()

# CARREGANDO  a base de testes
base_teste = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_teste.csv')


# todas menos a data
base_treinamento = base.iloc[:, 1:7].values

# Normalizando os dados, serve para transformar para 0 e 1 para a rede neural interpretar mais rapido
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# realiza essa nova atribuição pois na hora da previsao precisamos ter um valor só nao 6
normalizador_previsao = MinMaxScaler(feature_range=(0, 1))
base_tmp = base_treinamento.iloc[:, 0:1].values
normalizador_previsao.fit_transform(base_tmp)

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 1:6])
    preco_real.append(base_treinamento_normalizada[i, 0])
# Transforma de lista para numpy array, para aparecer certinho em coluna tudo
previsores, preco_real = np.array(previsores), np.array(preco_real)

# Comeca a montar a rede neural
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True,
              input_shape=(previsores.shape[1], 5)))
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
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
# Reduz a taxa de aprendizado quando um valor para de melhorar
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# Salva os melhores pesos a cada epoca.
mcp = ModelCheckpoint(filepath='/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/pesos.h5',
                      monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs=100,
              batch_size=32, callbacks=[es, rlr, mcp])

# Base de teste
preco_real_teste = base_teste.iloc[:, 1:2].values
frames = [base, base_teste]
# junta as duas
base_completa = pd.concat(frames)
# dropa a date pois nao é mais interessante.
base_completa = base_completa.drop('Date', axis=1)

entradas = base_completa[len(base_completa) - len(base_teste)-90:].values
# transforma para 0 e 1
entradas = normalizador.transform(entradas)

x_teste = []
for i in range(90, 112):
    x_teste.append(entradas[i-90:i, 0:5])
# Converte de um array para o NumpyArray
x_teste = np.array(x_teste)
# Faz a predicção conforme a rede neural
previsoes = regressor.predict(x_teste)
# volta para a escala de numeração normal para facilitar a comparação
previsoes = normalizador_previsao.inverse_transform(previsoes)

# Montando um grafico para visualizar as previsoes
plt.plot(preco_real_teste, color='red', label='Preco Real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsoes precos das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
