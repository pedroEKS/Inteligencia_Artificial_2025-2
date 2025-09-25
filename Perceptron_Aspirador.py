"""
Aluno: Pedro Henrique Esperidião Aureliano RA: 124221061
Aluno: Rafael Araújo Pace RA: 12410152
Aluno: Humberto Freitas RA: 124114353
Aluno: Matheus Rocha Nogueira RA: 124221544
"""

import numpy as np

# Funções aux

def funcao_ativacao_degrau(soma_ponderada):
    """Função de ativação do tipo degrau. Retorna 1 se a soma ponderada for maior ou igual a 0, caso contrário, retorna 0."""
    if soma_ponderada >= 0:
        return 1
    else:
        return 0

def normalizar(valor, min_val, max_val):
    """Func que normaliza um valor entre um mínimo e máximo"""
    return (valor - min_val) / (max_val - min_val)

def desnormalizar(valor_normalizado, min_val, max_val):
    """Func para desnormalizar o valor de volta para o original"""
    return valor_normalizado * (max_val - min_val) + min_val

def mapear_tipo_piso(piso):
    """
    O tipo de piso precisa ser convertido para um valor numérico
    A melhor forma é usar "one-hot encoding" para evitar que o modelo
    entenda uma ordem (ex: 0<1<2)
    Madeira = [1, 0, 0]
    Cerâmica = [0, 1, 0]
    Carpete = [0, 0, 1]
    """
    if piso == 'madeira':
        return [1, 0, 0]
    elif piso == 'ceramica':
        return [0, 1, 0]
    elif piso == 'carpete':
        return [0, 0, 1]
    else:
        raise ValueError("Tipo de piso inválido")

# Definição dos dados

# Entradas == tipo de piso, nivel de sujeira, distancia
# Saídas == potencia, velocidade

# Ex1 Piso de madeira, pouca sujeira, sem obstáculo
input_exemplo1 = mapear_tipo_piso('madeira') + [normalizar(1, 0, 10), normalizar(5, 0, 5)] # esperado - potência + velocidade
output_potencia1 = normalizar(1, 1, 3) # Potência baixa
output_velocidade1 = normalizar(5, 1, 5) # Velocidade alta

# Ex2 Carpete, muita sujeira com obstáculo perto
input_exemplo2 = mapear_tipo_piso('carpete') + [normalizar(9, 0, 10), normalizar(0.5, 0, 5)] # esperado + potência - velocidade
output_potencia2 = normalizar(3, 1, 3) # Potência alta
output_velocidade2 = normalizar(1, 1, 5) # Velocidade baixa

# Conjunto de dados de entrada
inputs = np.array([
    np.array(input_exemplo1),
    np.array(input_exemplo2)
])

# lista de saída
outputs_potencia = np.array([output_potencia1, output_potencia2])
outputs_velocidade = np.array([output_velocidade1, output_velocidade2])

# Configuração do perceptron

#Configuração
learning_rate = 0.1
epochs = 100
np.random.seed(42)  # Para garantir a reprodutibilidade dos resultados

# P1 p/potência de sucção
num_inputs_potencia = inputs.shape[1]
weights_potencia = np.random.uniform(low=-1.0, high=1.0, size=num_inputs_potencia)
bias_potencia = np.random.uniform(low=-1.0, high=1.0)

# P2 p/ velocidade de movimento
num_inputs_velocidade = inputs.shape[1]
weights_velocidade = np.random.uniform(low=-1.0, high=1.0, size=num_inputs_velocidade)
bias_velocidade = np.random.uniform(low=-1.0, high=1.0)

# Treinamento

# Treinamento do Perceptron de Potência
print("TREINAMENTO DO PERCEPTRON DE POTÊNCIA")
for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
        weighted_sum = np.dot(inputs[i], weights_potencia) + bias_potencia
        prediction = funcao_ativacao_degrau(weighted_sum)
        error = outputs_potencia[i] - prediction
        weights_potencia += error * inputs[i] * learning_rate
        bias_potencia += error * learning_rate
        total_error += abs(error)

    if total_error == 0:
        print(f"Perceptron de potência aprendeu! Concluímos em {epoch + 1} épocas.")
        break
print(f"Pesos finais (Potência): {weights_potencia}")
print(f"Viés final (Potência): {bias_potencia}\n")

# Treinamento do Perceptron de Velocidade
print("TREINAMENTO DO PERCEPTRON DE VELOCIDADE")
for epoch in range(epochs):
    total_error = 0
    for i in range(len(inputs)):
        weighted_sum = np.dot(inputs[i], weights_velocidade) + bias_velocidade
        prediction = funcao_ativacao_degrau(weighted_sum)
        error = outputs_velocidade[i] - prediction
        weights_velocidade += error * inputs[i] * learning_rate
        bias_velocidade += error * learning_rate
        total_error += abs(error)

    if total_error == 0:
        print(f"Perceptron de velocidade aprendeu! Concluímos em {epoch + 1} épocas.")
        break
print(f"Pesos finais (Velocidade): {weights_velocidade}")
print(f"Viés final (Velocidade): {bias_velocidade}\n")

#Test
print("TESTE FINAL")
for i in range(len(inputs)):
    # Prev potência
    weighted_sum_potencia = np.dot(inputs[i], weights_potencia) + bias_potencia
    prediction_potencia = funcao_ativacao_degrau(weighted_sum_potencia)
    final_potencia = desnormalizar(prediction_potencia, 1, 3)

    # Prev velocidade
    weighted_sum_velocidade = np.dot(inputs[i], weights_velocidade) + bias_velocidade
    prediction_velocidade = funcao_ativacao_degrau(weighted_sum_velocidade)
    final_velocidade = desnormalizar(prediction_velocidade, 1, 5)

    print(f"Entrada: {inputs[i]}")
    print(f"Previsão de Potência: {final_potencia:.2f} (Esperado: {desnormalizar(outputs_potencia[i], 1, 3):.2f})")
    print(f"Previsão de Velocidade: {final_velocidade:.2f} (Esperado: {desnormalizar(outputs_velocidade[i], 1, 5):.2f})\n")
