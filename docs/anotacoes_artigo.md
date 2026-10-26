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

- **R1 (tamanho da árvore):** `max_depth=3, min_samples_leaf=10` (profundidade 3, 8 folhas),
  com zeros → NaN e mediana. ROC-AUC 0,776 ± 0,049.
- **R2 (valores ausentes):** imputação pela mediana (mantém o vencedor do R1). ROC-AUC
  0,776 ± 0,049.
- **R3 (desbalanceamento):** mesma árvore e mesma imputação, com `class_weight=balanced`.
  Recall 0,786 ± 0,120; F1 0,636 ± 0,056; acurácia balanceada 0,713 ± 0,046; ROC-AUC
  0,785 ± 0,049.
- **R4 (variáveis novas):** nenhuma variável nova nem seleção; mantém o vencedor do R3 (árvore
  3/10 com `class_weight=balanced`, mediana). ROC-AUC 0,785 ± 0,049; recall 0,786 ± 0,120.
- **R5 (combinação):** nenhum reajuste; mantém o vencedor do R3 (árvore 3/10 com
  `class_weight=balanced`, mediana). F1 0,636 ± 0,056; recall 0,786 ± 0,120.
- **Modelo escolhido e teste final:** zeros → NaN, imputação pela mediana e árvore com
  `max_depth=3`, `min_samples_leaf=10` e `class_weight=balanced` (profundidade 3, 8 folhas;
  usa Glucose, BMI e Age). Pasta `20261024_145908_a1b1c4_R3_peso_balanceado`. No CV:
  ROC-AUC 0,785 ± 0,049; recall 0,786 ± 0,120; precisão 0,543 ± 0,049; F1 0,636 ± 0,056;
  acurácia balanceada 0,713 ± 0,046; acurácia 0,690 ± 0,046. Teste final (231 pacientes, 81
  diabéticos, executado uma única vez): ROC-AUC 0,790; recall 0,802 (65 de 81 diabéticos);
  precisão 0,591 (65 de 110 alarmes); F1 0,681; acurácia balanceada 0,751; acurácia 0,736;
  especificidade 0,700 (105 de 150). Matriz de confusão [[105, 45], [16, 65]]. Todas as
  métricas ficam dentro de um desvio-padrão da média do CV, sem sinal de sobreajuste à seleção.

## 3. Observações para a discussão

O que cada rodada mostrou além do vencedor. Em todas as comparações: teste t corrigido de
Nadeau e Bengio, α = 0,05, bilateral, referência (a de maior média) contra cada uma das outras.

### R1 (tamanho da árvore, 56 configurações, ROC-AUC)

- **Os dois extremos perdem com significância.** Sem limite de crescimento, a árvore decora o
  treino: 98 folhas e AUC 0,663 (p < 0,001). Rasa demais também perde: todas as de profundidade
  2 (4 folhas, AUC 0,746 a 0,754) e as podas fortes (`ccp_alpha` ≥ 0,02, 2 folhas, AUC 0,70 a
  0,72).
- **No meio há uma faixa larga de empate.** 34 configurações, de 6 a 33 folhas e AUC de 0,767 a
  0,790, não diferem da melhor (`max_depth=6, min_samples_leaf=20`: AUC 0,790, 19 folhas). A
  maior diferença nessa faixa (0,023) é menor que o desvio entre folds (0,04 a 0,05).
- **O `min_samples_leaf` controla o tamanho; o `max_depth` muitas vezes nem atua.** Com
  `min_samples_leaf` de 20, 30 ou 50, as árvores com `max_depth` 6, 8 e sem limite são
  idênticas. O baseline provisório (`max_depth=4`, `min_samples_leaf=1`) também é igual a uma
  configuração da grade.
- **A fronteira da significância é instável.** Com profundidade 3, `min_samples_leaf=30`
  (7 folhas) foi pior que a melhor (p = 0,034), mas `min_samples_leaf=50` (6 folhas) não
  (p = 0,155). Por isso a escolha foi `min_samples_leaf=10` (8 folhas, AUC 0,776, p = 0,49):
  a mais simples longe da zona instável, não a de menos folhas.
- **A significância depende da consistência entre folds, não só da média.** O baseline e
  `max_depth=2, min_samples_leaf=50` têm a mesma média (0,754) e a mesma diferença para a
  melhor (−0,036), mas só a segunda é significativa (p = 0,042 contra 0,099): o baseline
  oscila mais de um fold para outro.
- A AUC da melhor (0,790) é otimista: é a maior entre 56 tentativas.

### R2 (valores ausentes, 14 configurações + baseline, ROC-AUC)

- **Nenhuma estratégia de imputação difere da melhor.** AUC de 0,765 (zeros mantidos como
  valor) a 0,789 (KNN com k = 10); a mediana fica com 0,776 (p = 0,62). Com uma árvore de
  profundidade 3, o modo de imputar quase não importa.
- **O tamanho da árvore não desempata:** todas têm profundidade 3 e 8 folhas. A escolha foi
  pela simplicidade do pré-processamento.
- **Opções descartadas apesar do empate:** manter os zeros (valores fisiologicamente
  impossíveis, e a pior AUC) e deixar NaN na árvore (incompatível com a subamostragem do R3 e a
  seleção `kbest` do R4).
- **A imputação muda as variáveis que a árvore usa.** Com a mediana, a árvore usa só Glucose
  (importância 0,65), BMI (0,24) e Age (0,11), variáveis quase sem ausentes. Com KNN, ela passa
  a usar Insulin (importância 0,19 com k = 10; 0,16 no KNN padronizado), que foi imputada em
  cerca de metade dos pacientes. A regra "Insulin ≤ 110" mistura valores medidos e estimados,
  o que dificulta a interpretação clínica.
- **A parte mais forte da árvore não muda:** o ramo `Glucose > 132,5` e `BMI > 29,95`, que
  concentra a maioria dos diabéticos, é idêntico com mediana, KNN e KNN padronizado.
- **KNN padrão × padronizado: sem diferença** (0,788 × 0,780 com k = 5). Só mudar o k do KNN
  (3, 5, 10) já move a AUC de 0,776 a 0,789: variações desse tamanho estão no nível do ruído.
- **Reprodutibilidade:** quatro configurações foram executadas de novo e deram o mesmo hash e
  exatamente as mesmas notas.
- **Consequência para o R4:** HOMA-IR e Insulin/Glucose serão calculados com a Insulin imputada
  pela mediana em cerca de metade dos casos. Se essas variáveis ganharem, parte do efeito pode
  ser a glicose disfarçada.

### R3 (desbalanceamento, 21 configurações, F1 e acurácia balanceada, recall ≥ 0,70)

- **Todos os tratamentos passam no filtro de recall** (0,70 a 0,80). Sem tratamento, a árvore
  encontra só metade dos diabéticos (recall 0,50).
- **Nenhum tratamento difere do melhor**, nem em F1 (melhor: grade 4/20 com peso, 0,657) nem em
  acurácia balanceada (melhor: a mesma, 0,732). Peso de classe, ajuste de limiar e subamostragem
  produzem resultados equivalentes.
- **O peso troca precisão por recall.** Comparação direta, mesma árvore, sem e com
  `class_weight=balanced`:

  | métrica | sem peso | com peso | p |
  |---|---|---|---|
  | recall | 0,501 | 0,786 | < 0,001 |
  | precisão | 0,697 | 0,543 | 0,006 |
  | acurácia | 0,745 | 0,690 | 0,047 |
  | F1 | 0,576 | 0,636 | 0,149 |
  | acurácia balanceada | 0,688 | 0,713 | 0,360 |
  | ROC-AUC | 0,776 | 0,785 | 0,682 |

  O modelo passa a encontrar 79% dos diabéticos em vez de 50%, com mais alarmes falsos. As
  métricas que equilibram os dois lados e a AUC não mudam com significância. Numa triagem, em
  que deixar passar um doente custa mais que um alarme falso, a troca se justifica.
- **O peso não muda as regras da árvore, só a decisão de duas folhas.** A árvore final com peso
  tem exatamente os mesmos cortes da árvore sem peso; duas folhas (Age ≤ 29,5 com BMI > 42,8, e
  Age > 29,5 com BMI > 26,9) passam a ser classificadas como diabéticas. Na prática, o efeito é
  o mesmo de baixar o limiar de decisão, o que explica por que peso e ajuste de limiar empatam e
  por que a AUC quase não muda.
- **O peso não exige uma árvore maior.** Na grade com peso (profundidade 3 a 6,
  `min_samples_leaf` 5 a 30), as árvores de 10 a 32 folhas não são melhores que a de 8.
- **A escolha da métrica pesa no limite.** Contra a melhor configuração, a árvore sem tratamento
  é significativamente pior em F1 (p = 0,020), mas não em acurácia balanceada (p = 0,056). Ela
  fica de fora de qualquer forma, pelo recall.
- **Limiares escolhidos:** 0,348 com o limiar ajustado sozinho (F1 ou acurácia balanceada) e
  0,469 quando combinado com o peso. Como o peso já desloca a decisão, sobra pouco para o
  limiar ajustar.

### R4 (variáveis derivadas e seleção, 7 configurações, ROC-AUC)

- **Nada difere da melhor.** AUC de 0,779 a 0,791 (p ≥ 0,45); o baseline fica com 0,785
  (p = 0,77). Variáveis derivadas e seleção externa não melhoram a árvore.
- **A árvore já faz a própria seleção.** Com as faixas de IMC e idade, o resultado é idêntico
  ao baseline: a árvore prefere os valores originais e nunca usa as faixas. Com kbest (6
  colunas) e RFE, a árvore final é a mesma do baseline (Glucose, BMI, Age). Com HOMA, as flags
  de ausente também nunca são usadas (`R4_homa` e `R4_homa_flags` dão o mesmo resultado).
- **A melhor média (kbest com 4 colunas, AUC 0,791) não é mais simples.** A árvore troca Age
  por Pregnancies e Insulin: usa 4 variáveis em vez de 3, passa a exigir a Insulin e tem recall
  menor (0,747).
- **O HOMA-IR vira a raiz da árvore (importância 0,56), mas a AUC não muda** (0,785) e a
  interpretação piora: para cerca de metade dos pacientes a Insulin é a mediana, e o corte
  "HOMA ≤ 35,6" equivale a um corte de glicose.
- **As razões empatam na AUC (0,779), mas o recall cai para 0,661**, abaixo do mínimo de 0,70.
  Glucose × BMI domina a árvore (importância 0,83). Mostra por que olhar só a métrica da rodada
  não basta.

### R5 (combinação e reajuste do tamanho, 9 configurações, F1 com recall ≥ 0,70)

- **Só 3 das 9 configurações eram novas.** Como R2 e R4 mantiveram o baseline, a combinação dos
  vencedores é o próprio vencedor do R3, e 6 pontos da grade (profundidade 3 e 4) já tinham sido
  executados no R3; o runner os reconheceu pelo hash.
- **Nenhuma difere da melhor em F1** (`max_depth=4, min_samples_leaf=20`, F1 0,657).
- **A árvore de profundidade 2 (4 folhas) empata no F1, mas fica fora pelo recall.** Ela seria a
  mais simples (p = 0,066, perto do limite), mas tem recall 0,597, abaixo do mínimo de 0,70, e
  AUC 0,756; no R1, todas as árvores de profundidade 2 foram significativamente piores na AUC.
- **O terceiro nível é o que recupera os diabéticos.** Na árvore de profundidade 2, o grupo
  "Glucose ≤ 132,5 e Age > 29,5" é uma folha só, quase empatada (80,6 × 76,1 em pesos de
  classe), classificada como não diabética. Com profundidade 3, esse grupo é dividido por
  BMI, e quem tem BMI > 26,9 passa a ser classificado como diabético.
- **Entre as que passam no recall**, as três de profundidade 3 (`min_samples_leaf` 5, 10 e 20)
  têm 8 folhas e F1 entre 0,634 e 0,636; não há motivo para sair do 3/10.

### Observações gerais

- Em todas as rodadas, p ≥ 0,05 quer dizer "sem diferença significativa", não "equivalentes":
  com 25 folds correlacionados, o teste tem pouco poder para diferenças pequenas.
- Cada rodada faz várias comparações (55, 14, 21, 7 e 8) sem correção de Holm ou Bonferroni. Como a
  regra procura as configurações que não diferem da melhor, a falta de correção torna a seleção
  mais rigorosa, não mais permissiva.

## 4. Robustez (R8, opcional)

Repetir o modelo escolhido com outras sementes (além da 42) e outras divisões (80/20 e 65/35),
numa cópia do projeto. Reportar a média e a variação das métricas.

## 5. Limitações

- Base pequena (768 pacientes) e de uma só população (mulheres Pima, com 21 anos ou mais).
- Insulin tem quase metade dos valores ausentes, então qualquer variável derivada dela (como o
  HOMA-IR) depende muito da imputação.
- Árvores são instáveis: pequenas mudanças nos dados trocam cortes inteiros, o que explica o
  desvio-padrão alto entre folds.
- Em árvores rasas, a probabilidade prevista só assume poucos valores (um por folha), então a
  curva ROC fica em degraus.
- Com dezenas de configurações testadas, a nota da melhor no CV é otimista; o teste isolado
  corrige isso.
