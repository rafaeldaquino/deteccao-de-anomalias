

# DETECÇÃO DE ANOMALIAS EM TRANSAÇÕES COM PYTHON - MACHINE LEARNING 

## Resumo Executivo

Este projeto desenvolveu uma esteira modularizada e profissional de Data Science para detecção de fraudes em cartões de crédito sob um cenário de desbalanceamento extremo (~0.17% de fraudes). Após aplicar a Feature Engineering (transformação logarítmica e escalonamento) com blindagem rigorosa contra vazamento de dados (Data Leakage), foram testados e comparados os algoritmos Random Forest e XGBoost. A solução campeã foi o XGBoost com Threshold ajustado em 0.3, que alcançou o melhor equilíbrio para o negócio ao capturar 83% das fraudes reais (Recall) com uma assertividade de 87% nos alarmes disparados (Precision), reduzindo custos com fraudes e evitando o bloqueio indevido de clientes legítimos.

## Objetivo

O principal objetivo deste projeto é implementar e avaliar algoritmos de Aprendizado de Máquina (Machine Learning) focados na identificação automatizada de anomalias. 

O foco técnico está em modelar o comportamento "padrão" do sistema para que qualquer variação estatisticamente irrelevante ou maliciosa dispare alertas precisos sem a necessidade de uma rotulagem manual exaustiva.

O dataset utilizado contém transações de cartão de crédito do Kaggle.

URL: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

##  Dificuldades Encontradas

Durante o ciclo de desenvolvimento do projeto, o principal desafio técnico enfrentado foi o desbalanceamento massivo dos dados, em que, por definição, as anomalias são eventos extremamente raros (geralmente representando menos de 1% do dataset). Encontrar o equilíbrio de sensibilidade do modelo para que ele capture esses eventos sem gerar uma avalanche de alarmes falsos (falsos positivos) foi uma das tarefas mais complexas.

## Conclusão

Conseguimos construir um sistema antifraude robusto, de alta precisão e com baixo índice de alarmes falsos, superando os desafios do desbalanceamento extremo de dados.

Ao analisar a esteira de experimentos, o projeto passou por três fases de maturidade técnica que nos levaram à decisão final.

1. O Alinhamento Técnico-Operacional
   
O maior desafio inicial era o desbalanceamento severo da base, onde apenas 0,17% das transações eram fraudes.

Se usássemos os dados sem tratamento, qualquer modelo simplesmente apostaria que "toda transação é normal" para obter 99,83% de acurácia, tornando-se completamente inútil para o banco.

A engenharia de recursos (aplicação de log em Amount e o isolamento profissional do StandardScaler em Time) garantiu que os modelos não sofressem com distorções de escala ou vazamento de dados (Data Leakage).

2. A Escolha do Modelo Campeão
   
Analisando o comportamento dos três cenários testados, tomamos a decisão baseada em dados:

Random Forest (Baseline): Mostrou-se um modelo forte logo de início devido ao parâmetro class_weight="balanced". Ele capturou 80 das 98 fraudes (82% de Recall), estabelecendo uma linha de base respeitável.

XGBoost Padrão (Threshold = 0.5): Mostrou a superioridade algorítmica do Gradient Boosting em termos de assertividade. Ele elevou a precisão para 90% (reduzindo drasticamente os bloqueios indevidos de clientes legítimos), mas acabou deixando passar uma fraude a mais do que o Random Forest.

XGBoost Ajustado (Threshold = 0.3) : Aproveitando a alta precisão que o XGBoost oferecia, reduzimos o limiar de decisão para 30%. Isso gerou o melhor cenário possível para o negócio: alcançamos o maior nível de detecção (83% das fraudes capturadas) mantendo uma precisão brilhante de 87%. **CAMPEÃO** 

3. O Diagnóstico das Variáveis (O que causa a fraude?)
   
Ao exportarmos a função de importância das variáveis, o modelo revelou que a tomada de decisão não acontece ao acaso. O gráfico de barras indicou que variáveis específicas (como o forte pico observado na feature V14 e em outros componentes latentes) são os principais sinalizadores de anomalias.

Isso significa que o XGBoost conseguiu correlacionar padrões ocultos nessas variáveis que um analista humano dificilmente detectaria visualmente em uma planilha.

