# **Relatório do Projeto: Perceptron para Controle de Aspirador de Pó Inteligente**

**Integrantes:**

  * [Pedro Henrique Esperidião Aureliano](https://github.com/pedroEKS) 
  * [Matheus Rocha Nogueira](https://github.com/Mathrochaa) 
  * [Rafael Araújo Pace](https://github.com/RafaelPace) 
  * [Humberto Freitas](https://github.com/FreitasCyberSec)
-----

## **1. Introdução**

Este projeto tem como objetivo o desenvolvimento de um sistema de controle para um aspirador de pó inteligente utilizando o modelo de neurônio artificial Perceptron. O sistema foi projetado para tomar decisões autônomas com base em informações do ambiente, ajustando sua operação para uma limpeza eficiente e segura.

As entradas consideradas para o sistema são:

  * **Tipo de Piso:** Carpete, Cerâmica ou Madeira.
  * **Quantidade de Sujeira:** Em uma escala de 1 a 5.
  * **Distância de Obstáculos:** Medida em metros, de 0 a 5.

Com base nesses dados, o sistema determina duas saídas:

  * **Potência de Aspiração:** Nível de 1 a 3.
  * **Velocidade de Movimento:** Nível de 1 a 5.

O desafio consiste em modelar, treinar e validar um ou mais Perceptrons capazes de mapear as condições de entrada para as ações de saída desejadas, criando um comportamento inteligente e adaptativo para o aspirador.

## **2. Desenvolvimento**

A implementação foi realizada em Python, utilizando a biblioteca `NumPy` para operações matemáticas e de vetores, que são fundamentais na manipulação de pesos e cálculos de um Perceptron. O processo de desenvolvimento foi dividido nas seguintes etapas:

### **2.1. Pré-processamento dos Dados**

Para que o Perceptron pudesse processar os dados de entrada, foi necessário convertê-los para um formato numérico e padronizado.

1.  **Conversão do Tipo de Piso (One-Hot Encoding):** O tipo de piso é uma variável categórica. Usar uma simples atribuição numérica (ex: Madeira=1, Cerâmica=2, Carpete=3) poderia introduzir uma relação de ordem inexistente, levando o modelo a interpretar que "Carpete \> Cerâmica", o que não faz sentido. Para evitar isso, optamos pela técnica **One-Hot Encoding**, que transforma a categoria em um vetor binário:

      * **Madeira:** `[1, 0, 0]`
      * **Cerâmica:** `[0, 1, 0]`
      * **Carpete:** `[0, 0, 1]`

2.  **Normalização:** As entradas numéricas (sujeira, distância) e as saídas (potência, velocidade) possuem escalas diferentes. Para garantir que nenhuma variável domine o aprendizado apenas por ter uma magnitude maior, normalizamos todos esses valores para um intervalo entre 0 e 1. Isso melhora a estabilidade e a velocidade de convergência do treinamento. Funções de `normalizar()` e `desnormalizar()` foram criadas para este fim.

### **2.2. Arquitetura do Modelo**

O problema exige duas saídas distintas: potência e velocidade. Um único Perceptron com uma função de ativação degrau só consegue produzir uma única saída binária (0 ou 1). Portanto, a solução mais direta e eficaz foi projetar **dois Perceptrons independentes**:

  * **Perceptron 1:** Responsável por determinar a **potência** de aspiração.
  * **Perceptron 2:** Responsável por determinar a **velocidade** de movimento.

Ambos os Perceptrons recebem o mesmo vetor de entrada (piso, sujeira, distância), mas possuem seus próprios pesos e bias, que são treinados de forma independente para aprender a mapear as entradas para sua respectiva saída.

### **2.3. Escolha da Função de Ativação**

O projeto pedia a escolha entre a função de ativação Sigmoide e a Degrau (Step Function).

**Escolha: Função Degrau (Step Function)**

**Justificativa:**
A função Degrau foi escolhida por sua simplicidade e adequação ao modelo Perceptron clássico. Ela classifica a saída em duas categorias distintas (0 ou 1), o que é ideal para problemas de classificação binária. Embora nossas saídas (potência e velocidade) não sejam estritamente binárias, o uso da normalização nos permitiu mapear os níveis mínimos e máximos para 0 e 1, respectivamente. Por exemplo, para a potência, 0 representa o nível 1 e 1 representa o nível 3.

A função Sigmoide, por outro lado, retorna um valor contínuo entre 0 e 1, sendo mais adequada para problemas onde a saída representa uma probabilidade ou para uso em redes neurais mais complexas (Multi-Layer Perceptrons), onde sua característica de ser derivável é essencial para o algoritmo de backpropagation. Para este problema específico, com um conjunto de dados simples e uma arquitetura de Perceptron único, a função Degrau é suficiente, computacionalmente mais leve e cumpre o objetivo de forma eficaz.

### **2.4. Treinamento**

O treinamento foi realizado utilizando a regra de aprendizagem do Perceptron. Para cada exemplo no conjunto de dados:

1.  Calcula-se a soma ponderada das entradas com os pesos e adiciona-se o bias.
2.  O resultado é passado pela função de ativação (Degrau) para gerar uma predição.
3.  O erro é calculado subtraindo a predição da saída esperada.
4.  Os pesos e o bias são ajustados proporcionalmente ao erro, à entrada e a uma taxa de aprendizado (`learning_rate`).

Este processo é repetido por um número definido de épocas (`epochs`) ou até que o erro total em uma época seja zero, indicando que o modelo aprendeu a classificar corretamente todos os exemplos de treinamento.

## **3. Análise do Processo de Treinamento (Bônus)**

Para analisar a eficácia do treinamento, é possível monitorar o erro do modelo ao longo das épocas. Embora o código final não inclua a geração de gráficos, o processo para tal seria:

1.  Armazenar o valor de `total_error` de cada Perceptron ao final de cada época em uma lista.
2.  Ao final do treinamento, utilizar uma biblioteca como a `matplotlib` para plotar o erro em função do número de épocas.

Um gráfico típico de treinamento bem-sucedido mostraria uma **curva de erro descendente**. O erro seria alto nas épocas iniciais, quando os pesos são aleatórios, e diminuiria progressivamente à medida que os pesos são ajustados, convergindo para zero. Isso demonstra visualmente que o algoritmo está aprendendo e melhorando sua precisão a cada iteração, validando a eficácia do processo de treinamento.

*Figura 1: Exemplo ilustrativo de um gráfico de erro por época. O erro diminui à medida que o número de épocas aumenta, indicando aprendizado.*

## **4. Resultados e Testes**

Após a execução do script, o terminal exibe o log do treinamento e os resultados dos testes. A saída comprova a eficácia do modelo, que aprendeu rapidamente a classificar os dados de treinamento e, em seguida, previu corretamente os resultados esperados.

### **Saída do Terminal**

```
TREINAMENTO DO PERCEPTRON DE POTÊNCIA
Perceptron de potência aprendeu! Concluímos em 2 épocas.
Pesos finais (Potência): [-0.25091976  0.90142861  0.56398788  0.28731697 -0.67796272]
Viés final (Potência): -0.5880109593275947

TREINAMENTO DO PERCEPTRON DE VELOCIDADE
Perceptron de velocidade aprendeu! Concluímos em 10 épocas.
Pesos finais (Velocidade): [-0.18383278  0.73235229 -0.59776998 -0.23385484 -0.33883101]
Viés final (Velocidade): 0.8398197043239888

TESTE FINAL
Entrada: [1.  0.  0.  0.1 1. ]
Previsão de Potência: 1.00 (Esperado: 1.00)
Previsão de Velocidade: 5.00 (Esperado: 5.00)

Entrada: [0.  0.  1.  0.9 0.1]
Previsão de Potência: 3.00 (Esperado: 3.00)
Previsão de Velocidade: 1.00 (Esperado: 1.00)
```

A saída mostra que o Perceptron de potência convergiu em apenas 2 épocas e o de velocidade em 10. Na fase de teste, as previsões para os dois cenários de exemplo foram 100% precisas.

## **5. Conclusão**

O projeto demonstrou com sucesso a aplicação do modelo Perceptron para resolver um problema prático de controle. Através de um pré-processamento cuidadoso dos dados, da escolha de uma arquitetura com dois Perceptrons independentes e da utilização da função de ativação Degrau, foi possível treinar um sistema capaz de controlar a potência e a velocidade de um aspirador de pó com base nas condições do ambiente.

O modelo aprendeu as relações lógicas básicas, como aumentar a potência para carpetes com muita sujeira e reduzir a velocidade perto de obstáculos. Como próximos passos, o sistema poderia ser aprimorado utilizando um conjunto de treinamento maior e mais variado, ou até mesmo evoluindo a arquitetura para uma Rede Neural Multi-Layer (MLP) para capturar relações mais complexas e não-lineares entre as variáveis.
