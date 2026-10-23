# Anotações para o artigo

Decisões tomadas, o que cada rodada mostrou e as limitações do trabalho. Cada item vira um
parágrafo do artigo.

## 1. Decisões e justificativas

**Divisão 70/30, estratificada, seed 42.**
537 linhas para treino e 231 para teste, ambos com 35% de diabéticos. O teste só é usado uma
vez, no fim, para medir a generalização do modelo escolhido.

**Validação cruzada 5×5 estratificada.**
k = 5 ou 10 são os valores usuais. Com esta base, 5 é um bom equilíbrio: cada fold de validação
tem ~107 linhas, o suficiente para uma estimativa razoável. Com k = 10, os folds ficariam com
~54 linhas e a métrica de cada um, mais ruidosa. As 5 repetições reduzem a variação que vem da
sorte de uma única divisão. Os mesmos 25 folds valem para todos os experimentos, o que permite
compará-los fold a fold.

**Tratamentos dentro da validação cruzada.**
Imputação, seleção, balanceamento e limiar são aprendidos só nos folds de treino. Isso evita o
vazamento de dados (*data leakage*).

**Zeros impossíveis viram valores ausentes.**
Zero em Glucose, BloodPressure, SkinThickness, Insulin ou BMI não existe fisiologicamente:
significa "não medido". No treino, Insulin tem 48% de zeros e SkinThickness, 29%.

**Correção da `DiabetesPedigreeFunction`.**
No CSV original, valores menores que 1 perderam o "0." (0.627 aparece como 627). Todo valor
≥ 3 é dividido por 1000; não há ambiguidade, porque o máximo real é 2,42.

**Métrica principal: ROC-AUC.**
Mede se o modelo ordena bem doentes acima de saudáveis, independentemente do limiar. A acurácia
não serve: prever "não diabético" para todos já acerta 65,2%. No desbalanceamento (R3), o
critério é F1 ou acurácia balanceada, com recall ≥ 0,70, porque numa triagem deixar passar um
diabético custa mais que um alarme falso.

**Regra de escolha: a configuração mais simples.**
Entre as configurações sem diferença significativa para a melhor, fica a mais simples (árvore
menor, menos etapas). A significância vem do teste t corrigido de Nadeau e Bengio (2003): o
teste t comum não serve, porque os folds compartilham dados de treino.

**Sem SMOTE nem outras amostras sintéticas.**
- O SMOTE só infla métricas quando é aplicado antes da divisão dos dados; aqui ele rodaria só no
  treino de cada fold. O motivo para não usá-lo é outro.
- O peso de classe (`class_weight`) e o ajuste do limiar trocam precisão por recall da mesma
  forma, sem criar pacientes artificiais.
- Os pacientes interpolados geram cortes como `Pregnancies <= 6.18`, que atrapalham a
  interpretação, que é a principal vantagem da árvore.
- Correções de desbalanceamento distorcem as probabilidades previstas (van den Goorbergh et al.,
  2022).

**Sem tratamento de outliers e sem padronização.**
A árvore corta por limiares (`Insulin <= 123`), então só a ordem dos valores importa. Um valor
extremo cai do mesmo lado do corte que um valor apenas alto.

**Ordem das rodadas.**
O tamanho da árvore é ajustado primeiro (R1), com um pré-processamento de referência, por ser o
fator de maior impacto. Depois vem um tratamento de dados por rodada (R2 a R4). No fim (R5), o
tamanho é reajustado para verificar a interação com os tratamentos.

**KNN padronizado (`knn_scaled`).**
O KNN padrão mede distância nas unidades originais, e Insulin e Glucose dominam a escolha dos
vizinhos. Foi criada uma versão que padroniza as colunas só para escolher os vizinhos. As duas
são comparadas no R2, com e sem peso de classe.

## 2. Achados por rodada

Preencher ao fim de cada rodada: pergunta, o que venceu, números (média ± desvio) e o motivo.

- **R1 (tamanho da árvore):**
- **R2 (valores ausentes):**
- **R3 (desbalanceamento):**
- **R4 (variáveis novas):**
- **R5 (combinação):**
- **Modelo escolhido e teste final:**

## 3. Robustez (R8, opcional)

Repetir o modelo escolhido com outras sementes (além da 42) e outras divisões (80/20 e 65/35),
numa cópia do projeto. Reportar a média e a variação das métricas.

## 4. Limitações

- Base pequena (768 pacientes) e de uma só população (mulheres Pima, com 21 anos ou mais).
- Insulin tem quase metade dos valores ausentes, então qualquer variável derivada dela (como o
  HOMA-IR) depende muito da imputação.
- Árvores são instáveis: pequenas mudanças nos dados trocam cortes inteiros, o que explica o
  desvio-padrão alto entre folds.
- Em árvores rasas, a probabilidade prevista só assume poucos valores (um por folha), então a
  curva ROC fica em degraus.
- Com dezenas de configurações testadas, a nota da melhor no CV é otimista; o teste isolado
  corrige isso.
