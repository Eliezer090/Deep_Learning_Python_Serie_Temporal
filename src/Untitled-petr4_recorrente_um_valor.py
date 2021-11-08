# Rede neural para prever o valor de abertura de uma acao
#
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

base = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_treinamento.csv')
# Dropando os vazios
base = base.dropna()
# Pega a coluna Open que seria a abertura, isso que vamos prever
base_treinamento = base.iloc[:, 1:2].values
# Importacao para realizar a scala/transform dos dados.
normalizador = MinMaxScaler(feature_range=(0, 1))
# Normaliza/Transforma os dados para uma escala menor, com isso a rede neural consegue trabalhar melhor
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []
# For para fazer o tratamento dos dados pegar os valores corretos
for i in range(90, 1242):
    # Realizado para pegar os 90 primieros valores
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    # Ignora os 90 primeiros registros e pega os seguintes
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

# CARREGANDO  a base de testes
base_teste = pd.read_csv(
    '/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste)-90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

x_teste = []

for i in range(90, 112):
    x_teste.append(entradas[i-90:i, 0])
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))
previsoes = regressor.predict(x_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

# Montando um grafico para visualizar as previsoes
plt.plot(preco_real_teste, color='red', label='Preco Real')
plt.plot(previsoes, color='blue', label='Previsoes')
plt.title('Previsoes precos das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()


# Salvando a rede neural KERAS, site: https://www.dobitaobyte.com.br/como-salvar-um-model-treinado-com-keras/
model_json = regressor.to_json()
with open("regressor.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
