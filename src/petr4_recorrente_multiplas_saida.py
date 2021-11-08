# Rede neural com multiplas saidas.
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


# vamos prever o Open e o Higth
# Pegando o Open
base_treinamento = base.iloc[:, 1:2].values
# Pegando o Higth
base_valor_maximo = base.iloc[:, 2:3].values

# Normalização das bases transformando para 0 e 1
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizado = normalizador.fit_transform(base_treinamento)

base_valor_maximo_normalizado = normalizador.fit_transform(base_valor_maximo)

previsores = []
preco_real1 = []
preco_real2 = []

for i in range(90, 1242):
    previsores.append(base_treinamento_normalizado[i-90:i, 0])
    preco_real1.append(base_treinamento_normalizado[i, 0])
    preco_real2.append(base_valor_maximo_normalizado[i, 0])

previsores, preco_real1, preco_real2 = np.array(
    previsores), np.array(preco_real1), np.array(preco_real2)
# Adiciona mais uma dimensão
previsores = np.reshape(
    previsores, (previsores.shape[0], previsores.shape[1], 1))
# juntar os preços reais para passar por parametro, aqui teremos o Open e o Higth
preco_real = np.column_stack((preco_real1, preco_real2))

# Comeca a montar a rede neural
regressor = Sequential()
regressor.add(LSTM(units=160, return_sequences=True,
              input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=120, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=90, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=80))
regressor.add(Dropout(0.3))

# units alterado para 2 pois temos 2 saidas
regressor.add(Dense(units=2, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
# parar o processamento se ocorrer um overfiting dos dados, ou seja se nao melhorar para.
es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=15, verbose=1)
# Se em 5 dodadas nao melhorar reduz a taxa de aprendizado em 0.2 para o modelo conseguir continuar a melhorar
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
# Salva os melhores pesos a cada epoca.
mcp = ModelCheckpoint(filepath='/Users/es19237/Desktop/Deep Learning/Redes neurais recorrentes/files/pesos_nultpl_saidas.h5',
                      monitor='loss', save_best_only=True, verbose=1)

regressor.fit(previsores, preco_real, epochs=300,
              batch_size=32, callbacks=[es, rlr, mcp])

# Tratando a base de TESTES
# open
preco_real_open = base_teste.iloc[:, 1:2].values
# high
preco_real_high = base_teste.iloc[:, 2:3].values

# concatena os 2 open da base normal e da base de testes
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
# Vai pegar as posições nao intendi mto bem como ele faz isso
entradas = base_completa[len(base_completa) - len(base_teste)-90:].values
# Transforma cada valor em um array dentro de outro aray para transformar para termos uma coluna
entradas = entradas.reshape(-1, 1)
# Normaliza para 0 e 1 os valores.
entradas = normalizador.transform(entradas)

x_teste = []
for i in range(90, 112):
    # passa as ultimas 90 acoes
    x_teste.append(entradas[i-90:i, 0])
# Converte de um array([]) para array simples[[[valores],[valores]]]
x_teste = np.array(x_teste)
x_teste = np.reshape(x_teste, (x_teste.shape[0], x_teste.shape[1], 1))

# faz a predicção, aqui ele já cria as 2 colunas open e higth na previsoes,
# caso tiver mais saidas ele já cria as outras colunas uma para cada saida.
previsoes = regressor.predict(x_teste)
# inverte para conseguirmos montar o grafico com os valores reais
previsoes = normalizador.inverse_transform(previsoes)

# Montando um grafico para visualizar as previsoes
plt.plot(preco_real_open, color='red', label='Preco Real Open')
plt.plot(preco_real_high, color='black', label='Preco Real higth')
plt.plot(previsoes[:, 0], color='blue', label='Previsoes Open')
plt.plot(previsoes[:, 1], color='orange', label='Previsoes Higth')
plt.title('Previsoes precos das acoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()
