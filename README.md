# Árvore de decisão para triagem de diabetes (Pima Indians Diabetes)

Trabalho da disciplina Aprendizado de Máquina, do PPGCC - UFPI, 2026.2.

**Autores:** Filipe Mendes, Bétanny Alexandra, Juciane Maciel· **Orientação/professor:** Prof. Dr. Vinicius Ponte Machado.

## Objetivo

Construir e avaliar uma árvore de decisão que identifique pacientes com diabetes a partir de
exames clínicos simples, mantendo o modelo interpretável. O trabalho compara, de forma
controlada, o efeito do tamanho da árvore, do tratamento de valores ausentes, da correção do
desbalanceamento (sem amostras sintéticas) e de variáveis derivadas, sempre com validação
cruzada e um conjunto de teste isolado até o fim.

## Base de dados

**Pima Indians Diabetes Database:** 768 mulheres de ascendência Pima, com 21 anos ou mais,
descritas por 8 variáveis clínicas (gestações, glicose, pressão arterial, espessura da prega
cutânea, insulina, IMC, histórico familiar e idade). O alvo (`Outcome`) indica diabetes;
cerca de 35% das pacientes são positivas.

- Artigo original: Smith, J. W.; Everhart, J. E.; Dickson, W. C.; Knowler, W. C.; Johannes,
  R. S. *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*.
  Proceedings of the Symposium on Computer Applications and Medical Care, 1988, p. 261–265.
- Disponível em: [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv/)
  (originalmente no UCI Machine Learning Repository).
- O arquivo usado está em `data/raw/`, sem alterações. A correção de um erro de formatação
  na `DiabetesPedigreeFunction` é feita na geração do split (seção 3).

## Licença

MIT. A base de dados segue os termos da fonte original.

---

# Como usar o projeto

O projeto é um sistema de experimentos em que cada método de tratamento é uma peça
intercambiável e cada experimento é só uma configuração no YAML. Nenhum resultado é
sobrescrito: cada execução ganha uma pasta própria.

## 1. Instalação

Requer **Python 3.14**. O `requirements.txt` fixa as versões das bibliotecas: o
`pipeline.joblib` de cada execução só abre com as mesmas versões do scikit-learn e do
imbalanced-learn.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

A validação cruzada roda em paralelo com threads (`parallel_config(backend="threading")`).
O backend padrão do joblib quebra no Python 3.14 quando o caminho do projeto tem acento
("Máquina"); com ~500 linhas, threads custam o mesmo.

No Colab, monte o Drive, entre na pasta do projeto e rode os mesmos comandos com `!`:

```python
%cd "/content/drive/MyDrive/Mestrado/Aprendizado de Máquina/DecisionTree"
!pip install -r requirements.txt
!python run_experiments.py experiments/r1_tamanho_arvore.yaml
```

Todos os comandos devem ser executados **a partir da raiz do projeto**.

## 2. Rotina de uso

```bash
# 1. gerar o split treino/teste (uma única vez; 70/30, estratificado, seed 42)
python -m src.make_split

# 2. atualizar o baseline da rodada (linhas "# ATUALIZAR") e executar
python run_experiments.py experiments/r1_tamanho_arvore.yaml

# 3. ver a comparação entre todas as execuções
python summarize.py                    # ordena por roc_auc
python summarize.py --metric recall    # ou por outra métrica
python compare.py <run_A> <run_B>      # a diferença entre duas execuções é real?

# 4. no fim, avaliar a(s) melhor(es) no teste
python final_test.py --run <pasta da execução escolhida em results/runs/>
```

Opções do `run_experiments.py`:

| Opção | Efeito |
|---|---|
| `--only R1_grade R1_poda` | roda só esses experimentos (o nome de um sweep roda todos os seus valores) |
| `--force` | roda de novo mesmo se já executado (cria pasta nova, não substitui) |
| `--n-jobs 1` | número de processos da validação cruzada (padrão: todos os núcleos) |

## 3. Estrutura de pastas

```
DecisionTree/
├── data/
│   ├── raw/pima-indians-diabetes.csv   nunca alterar
│   └── split/train.csv, test.csv       gerados por src.make_split (com a correção abaixo)
├── src/
│   ├── config.py      caminhos, SEED, CV (5 folds x 5 repetições), métricas
│   ├── data.py        load_train(), load_test()
│   ├── make_split.py  gera o split
│   ├── steps/         um arquivo por etapa de tratamento
│   ├── registry.py    catálogo: nome no YAML -> função que constrói o passo
│   ├── builder.py     config -> Pipeline
│   └── runner.py      validação cruzada + gravação dos resultados
├── experiments/
│   ├── r1_tamanho_arvore.yaml ... r5_combinacao.yaml   uma rodada por arquivo (seção 6)
│   └── experiments.yaml          histórico da exploração inicial
├── docs/anotacoes_artigo.md      decisões, achados e limitações (para o artigo)
├── results/
│   ├── runs/          uma pasta por execução (só se acrescenta)
│   ├── final/         avaliações no teste
│   └── summary.csv    tabela gerada pelo summarize.py
├── run_experiments.py
├── summarize.py
├── compare.py
├── final_test.py
├── requirements.txt
├── COMO_USAR.md       anotações pessoais
└── CLAUDE.md          instruções para o Claude Code
```

**Correção nos dados:** no CSV original, os valores de `DiabetesPedigreeFunction` menores
que 1 perderam o "0." (0.627 aparece como `627`). O `make_split` divide por 1000 todo valor
≥ 3. Não há ambiguidade, porque o máximo real é 2.42. O arquivo `raw` continua intacto.

No dia a dia só se mexe em dois lugares: os YAMLs das rodadas em `experiments/` (para usar
métodos) e `src/steps/` + `src/registry.py` (para criar métodos novos).

## 4. O pipeline

Toda execução passa pelas mesmas etapas, sempre nesta ordem. Cada etapa recebe uma opção,
e `none` pula a etapa.

| # | Etapa | Opções | Observação |
|---|---|---|---|
| 1 | `zeros` | `none`, `to_nan` | 0 em Glucose, BloodPressure, SkinThickness, Insulin e BMI vira NaN |
| 2 | `missing_flags` | `none`, `flags` | cria colunas `<col>_missing` (0/1) antes da imputação |
| 3 | `imputer` | `none`, `median`, `mean`, `knn`, `knn5`, `knn_scaled`, `iterative` | preenche os NaN; com `none`, a própria árvore trata os NaN |
| 4 | `features` | `none`, `bmi_age_bins`, `ratios`, `homa` | cria colunas novas |
| 5 | `selection` | `none`, `kbest`, `kbest8`, `rfe`, `rfe6` | escolhe colunas |
| 6 | `balancing` | `none`, `smote`, `borderline_smote`, `adasyn`, `smote_tomek`, `smote_enn`, `random_over`, `random_under` | atua só no treino; exige imputer |
| 7 | `model` | hiperparâmetros do `DecisionTreeClassifier` | `random_state` já é fixado |
| — | `threshold` | `none`, `tuned` | **opcional**: escolhe o limiar de decisão (padrão 0.5) |

Detalhes de algumas opções:

- `knn` x `knn_scaled`: o `knn` mede a distância entre linhas nas unidades originais, então
  colunas de escala grande (Insulin, Glucose) dominam a escolha dos vizinhos. O `knn_scaled`
  padroniza as colunas (média e desvio aprendidos no treino) só para escolher os vizinhos,
  devolve os valores nas unidades originais e altera apenas as células ausentes.
- `homa`: cria `HOMA_IR = Glucose × Insulin / 405`, um índice de resistência à insulina.
  A árvore corta uma coluna por vez e não consegue achar esse produto sozinha.
- `smote_tomek` / `smote_enn`: aplicam o SMOTE e depois removem amostras ambíguas na
  fronteira entre as classes.
- `threshold: tuned`: em vez de prever 1 quando a probabilidade passa de 0.5, escolhe o
  limiar que maximiza `scoring`, com um CV interno feito só no treino. Não muda o `roc_auc`,
  mas muda accuracy, precision, recall, F1 e balanced_accuracy. Não use `scoring: recall`: o limiar ótimo
  seria 0, ou seja, prever tudo positivo.
- Hiperparâmetros úteis do `model`: `class_weight: balanced` (corrige o desbalanceamento sem amostras sintéticas),
  `ccp_alpha` (poda), `min_samples_leaf` e `criterion: entropy`.

Não há etapa de tratamento de outliers. A árvore divide os dados por limiares
(`Insulin <= 123`), então só a ordem dos valores importa, e um valor extremo cai do mesmo
lado do corte que um valor apenas alto. Winsorizar ou cortar outliers não muda essa ordem
e, portanto, quase não muda a árvore.

Parâmetros de cada opção (os valores mostrados são os padrões):

- `knn` / `knn_scaled`: `n_neighbors: 5`
- `iterative`: `max_iter: 10`
- `kbest`: `k: 8`, `score: f_classif` ou `mutual_info`
- `rfe`: `n_features: 6`
- `smote` / `borderline_smote`: `k_neighbors: 5`
- `adasyn`: `n_neighbors: 5`
- `tuned`: `scoring: f1` (ou `balanced_accuracy`, `accuracy`...), `n_splits: 5`

> Com as 8 colunas originais, `kbest8` mantém todas as colunas e não muda nada. Ele só tem
> efeito junto com `missing_flags` ou `features`.

## 5. Decisões do projeto

As justificativas completas estão em [`docs/anotacoes_artigo.md`](docs/anotacoes_artigo.md).

- **Divisão 70/30 estratificada** (seed 42), com o teste isolado até o fim.
- **Validação cruzada 5×5 estratificada**, igual para todos os experimentos.
- **Zeros impossíveis viram ausentes** em Glucose, BloodPressure, SkinThickness, Insulin e BMI.
- **Métrica principal: `roc_auc`.** Acurácia não serve: prever "não diabético" para todos já
  acerta 65,2%. No desbalanceamento (R3), F1 ou `balanced_accuracy`, com recall ≥ 0,70.
- **Regra de escolha:** entre as configs sem diferença significativa para a melhor
  (`compare.py`), fica a mais simples.
- **Sem SMOTE nem outras amostras sintéticas.** O desbalanceamento é tratado com `class_weight`,
  ajuste do limiar ou `random_under`. As opções de SMOTE seguem no código só para reproduzir
  execuções antigas.
- **Sem tratamento de outliers e sem padronização:** a árvore só usa a ordem dos valores.
- **Ordem das rodadas:** primeiro o tamanho da árvore, depois um tratamento de dados por rodada,
  e no fim um reajuste do tamanho.

## 6. Escrevendo experimentos

```yaml
baseline:
  zeros: to_nan
  missing_flags: none
  imputer: median
  features: none
  selection: none
  balancing: none
  model: {max_depth: 4}

experiments:
  - name: E02_knn
    override: {imputer: {name: knn, n_neighbors: 5}}

  - name: E06_profundidade6
    override: {model: {max_depth: 6}}

  - name: E07_smote_raso
    override:
      balancing: {name: smote, k_neighbors: 3}
      model: {max_depth: 3, min_samples_leaf: 10}

  - name: E10_poda
    override: {model: {max_depth: 8}}
    sweep: {model.ccp_alpha: [0.0, 0.005, 0.01]}

  - name: E12_limiar_f1
    override: {threshold: {name: tuned, scoring: f1}}
```

O `sweep` gera um experimento por valor: `E10_poda__ccp_alpha=0.0`, `E10_poda__ccp_alpha=0.005` etc.
A chave é `etapa.parametro`, e o sweep é aplicado depois do override. Com mais de uma chave,
ele roda todas as combinações. Se um valor do sweep produzir uma config que já foi
executada (por exemplo, `max_depth: 4` igual ao baseline), ela é pulada.

O sweep também pode trocar a **opção** de uma etapa, usando `name` como parâmetro. Os
parâmetros escritos no override valem para todas as opções do sweep:

```yaml
  - name: R2_knn_vs_scaled_peso
    override: {imputer: {name: knn, n_neighbors: 5}}
    sweep:
      imputer.name: [knn, knn_scaled]        # 2 opções x 2 pesos = 4 execuções
      model.class_weight: [null, balanced]
```

Regras:

- O baseline roda automaticamente como `E01_baseline`. Para usar outro nome, acrescente
  `baseline_name: R2_baseline` no topo do YAML. O nome é só um rótulo e não entra no hash.
- Um `class_weight` em dicionário (`{0: 1, 1: 1.5}`) vai só em `override`, nunca em `sweep`:
  o valor entraria no nome da pasta, e o Windows não aceita `:`.
- Cada etapa pode ser escrita como `nome` ou `{name: nome, parametro: valor}`.
- O `override` **substitui a etapa inteira**. Isso vale também para o `model`: para mudar só
  `min_samples_leaf` e manter a profundidade, repita `max_depth: 4`.
- Nomes de experimento não podem se repetir dentro do YAML.
- Antes de rodar qualquer coisa, o script valida todos os experimentos. Um erro de digitação
  para tudo com uma mensagem clara, por exemplo:
  `[R2_knn] imputer 'knnn' não existe, opções: none, median, mean, knn, knn5, knn_scaled, iterative`.

Para tirar um experimento da lista, basta apagá-lo do YAML. As execuções antigas continuam
em `results/runs/`.

### Rodadas

Os experimentos estão organizados em rodadas, um YAML por rodada. Cada rodada responde a
uma pergunta, e o vencedor de uma vira o baseline da seguinte. Do R2 em diante, os valores
marcados com `# ATUALIZAR: vencedor de Rx` são provisórios: **não rode uma rodada antes de
atualizar o baseline dela**. O `experiments/experiments.yaml` fica como histórico da
exploração inicial.

| Arquivo | Pergunta | Critério |
|---|---|---|
| `r1_tamanho_arvore.yaml` | Qual tamanho de árvore aprende sem decorar? | `roc_auc`; a config mais simples entre as sem diferença significativa para a melhor |
| `r2_ausentes.yaml` | Qual a melhor forma de lidar com exames não medidos? | `roc_auc`; a mais simples entre as sem diferença significativa |
| `r3_desbalanceamento.yaml` | Como encontrar mais diabéticos sem SMOTE? | F1 ou `balanced_accuracy`, com recall ≥ 0.70 |
| `r4_variaveis.yaml` | HOMA-IR, razões e faixas ajudam? | `roc_auc`; a mais simples entre as sem diferença significativa |
| `r5_combinacao.yaml` | Os vencedores funcionam juntos? | O critério da rodada que decidiu cada ponto |

> **O baseline de cada rodada não ganha pasta própria.** Do R2 em diante, o baseline é
> exatamente o vencedor da rodada anterior: mesma config, mesmo hash. O runner reconhece que
> ela já foi executada e pula, então o `R2_baseline` não aparece no `summary.csv`. A execução
> continua com o nome que recebeu na rodada anterior. Para comparar os experimentos de uma
> rodada com o baseline dela no `compare.py`, use como referência a pasta do vencedor da
> rodada anterior.

Depois do R5, o vencedor é escolhido (R6) e avaliado uma vez com `final_test.py` (R7). A
robustez (R8) é verificada numa cópia do projeto. Essas etapas não têm YAML.

```bash
python run_experiments.py experiments/r1_tamanho_arvore.yaml
python summarize.py
```

## 7. Onde ficam os resultados (e as árvores)

Cada execução cria uma pasta nova em `results/runs/`:

```
results/runs/20261021_142558_0e1f8f_E05_knn_smote/
├── config.yaml              config completa usada (baseline + override já resolvidos)
├── metrics.json             média e desvio de cada métrica na validação cruzada
├── fold_scores.csv          as 25 notas individuais (5 folds x 5 repetições)
├── meta.json                data, hash, versões das bibliotecas, hash do train.csv,
│                            limiar escolhido (se usar threshold)
├── pipeline.joblib          o pipeline treinado (tratamentos + árvore)
├── tree.txt                 a árvore em texto (regras)
├── tree.png                 a árvore desenhada
└── feature_importances.csv  importância de cada coluna na árvore
```

O nome da pasta é `data_hora` + `hash` + `nome`. O hash identifica o conteúdo (config +
seed + CV + métricas + arquivo de treino); o nome é só um rótulo.

### Qual árvore é salva

A validação cruzada treina **25 árvores**, uma por fold, só para medir o desempenho. Elas são
descartadas: o que fica delas são as notas em `fold_scores.csv` e `metrics.json`.

Depois disso, o runner treina **uma árvore final**, com o mesmo pipeline, usando o
`train.csv` inteiro. É essa árvore que fica salva na pasta:

- **`tree.txt`**: as regras da árvore, uma condição por linha. Cada folha mostra
  `weights: [negativos, positivos]` (quantas amostras de treino chegaram nela) e a classe prevista:
  ```
  |--- Insulin <= 123.11
  |   |--- Age <= 26.50
  |   |   |--- BMI <= 45.35
  |   |   |   |--- weights: [109.00, 2.00] class: 0
  ```
- **`tree.png`**: o desenho da árvore, com gini, número de amostras e classe em cada nó.
  Laranja indica a classe 0, azul a classe 1, e a cor mais forte indica um nó mais puro.
- **`feature_importances.csv`**: a importância de cada coluna, em ordem decrescente.
- **`pipeline.joblib`**: o objeto completo e treinado (imputador, SMOTE etc. + árvore),
  pronto para prever. Para usar num notebook:

  ```python
  import joblib
  from src.data import load_train

  pipe = joblib.load("results/runs/<pasta>/pipeline.joblib")
  arvore = pipe.named_steps["model"]        # o DecisionTreeClassifier
  print(arvore.get_depth(), arvore.get_n_leaves())

  X, y = load_train()
  pipe.predict(X.head())                    # aplica os tratamentos e a árvore
  ```

Observações:

- Os valores dos limiares estão na escala **depois** dos tratamentos. Com SMOTE, por exemplo,
  podem aparecer limiares quebrados como `Pregnancies <= 6.18`, porque o SMOTE cria amostras
  sintéticas interpoladas.
- Com `threshold`, o `pipeline.joblib` é o envoltório do ajuste de limiar. O pipeline fica em
  `pipe.estimator_`, e a árvore em `pipe.estimator_.named_steps["model"]`. O `tree.png` mostra
  a árvore normalmente, mas a classe prevista passa a depender do limiar gravado em
  `meta.json`, e não da maioria na folha.
- O `.joblib` só abre com versões compatíveis do scikit-learn e do imbalanced-learn. As
  versões usadas ficam em `meta.json`.

### Tabela comparativa

`python summarize.py` lê todas as pastas de `results/runs/` e gera `results/summary.csv`,
ordenado pela métrica escolhida. A coluna `split` identifica o `train.csv` usado (só compare
execuções do mesmo split), e `threshold` mostra o limiar de decisão. A coluna `pipeline`
resume o que cada execução usou, por exemplo
`to_nan > knn(n_neighbors=5) > smote | max_depth=4`.

Essa tabela é sempre derivada das pastas: se algo der errado, apague-a e gere de novo.

### Comparando duas execuções

A tabela mostra médias, mas não diz se a diferença entre duas execuções é real ou ruído.
Para isso, use o `compare.py`, que compara as 25 notas fold a fold:

```bash
python compare.py 20261021_173038_7fc7e0_E01_baseline 20261021_173040_1baf10_E02_knn
python compare.py <run_A> <run_B> --metric roc_auc    # só uma métrica
```

```
  métrica  média A  média B  dif (B-A)      t     p significativo (p < α)
  roc_auc    0.754    0.774      0.020  0.779 0.444                   não
```

- Ele calcula as diferenças `d = B - A` em cada um dos 25 folds e aplica o **teste t corrigido
  de Nadeau e Bengio (2003)**: `t = média(d) / sqrt((1/(k·r) + 1/(k-1)) · var(d))`, com
  `k = 5` folds, `r = 5` repetições e `k·r - 1 = 24` graus de liberdade. O p-valor é bilateral.
- O teste t comum não serve: os folds compartilham dados de treino, então as notas não são
  independentes, e ele apontaria diferenças significativas demais.
- Sem `--metric`, mostra todas as métricas. As duas execuções precisam ter o mesmo CV
  (mesmas 25 linhas `repeat, fold`) e o mesmo `train.csv`. O script confere o split sozinho,
  pelo `train_sha256` do `meta.json`, e para com uma mensagem se forem diferentes. O split
  usado aparece na saída, no mesmo formato da coluna `split` do `summary.csv`.

#### Fechando uma rodada

Para aplicar a regra de escolha de uma rodada de uma vez, use o modo `--rodada`:

```bash
python compare.py --rodada R1_                       # todas as execuções cujo nome começa com R1_
python compare.py --rodada R2_ --incluir <pasta do vencedor do R1>
python compare.py --rodada R3_ --metric f1 --incluir <pasta do vencedor do R2>
python compare.py --rodada R1_ --ref <pasta>          # referência escolhida por você
python compare.py --rodada R1_ --salvar               # grava results/comparacoes/R1_roc_auc.csv
```

- `--rodada` seleciona as execuções de `results/runs/` cujo campo `name` do `meta.json` começa
  com o prefixo (ignora as pastas `.tmp_*` e as sem `meta.json`). Use o `_` no fim (`R1_`)
  para não pegar um futuro `R10_`.
- `--incluir` acrescenta pastas à comparação. É necessário do R2 em diante: o baseline da
  rodada é pulado pelo hash e continua com o nome do vencedor da rodada anterior, então o
  prefixo não o encontra.
- A **referência** é a execução de maior média na métrica (padrão: `roc_auc`), ou a pasta
  passada em `--ref`. Ela é comparada com **cada uma** das outras (um contra todos, não todos
  os pares), com o mesmo teste t corrigido.
- Uma execução com outro `train_sha256` ou sem as mesmas 25 linhas `(repeat, fold)` da
  referência não para o script: ele avisa e pula só essa execução.
- A tabela sai ordenada pela média, com a referência na primeira linha, e traz a diferença
  para a referência, `t`, `p`, a coluna `significativo (p < α)` e a **profundidade** e o
  **número de folhas** da árvore final (lidos do `pipeline.joblib`; NaN, com aviso, se ele não
  abrir). No fim, lista as execuções sem diferença significativa para a referência: as
  candidatas à mais simples. O script **não escolhe** o vencedor; essa decisão é sua.
- `--salvar` grava a tabela em `results/comparacoes/<prefixo>_<métrica>.csv` (o `_` final do
  prefixo é removido do nome do arquivo). Nada é escrito em `results/runs/`.

#### Como ler o p-valor

- O nível de significância é `ALPHA = 0.05`, em `src/config.py` (bilateral). Ele não entra no
  hash das execuções: mudá-lo só muda a leitura das comparações.
- p < α: há evidência de diferença entre as duas execuções na métrica.
- **p ≥ α quer dizer "sem diferença significativa", não "equivalentes".** O teste não provou
  que as duas são iguais; só não encontrou evidência de que sejam diferentes, e com 25 folds
  correlacionados ele tem pouco poder para diferenças pequenas. É isso que permite escolher a
  config mais simples entre as que empatam com a melhor (critério das rodadas), mas o empate
  é "não deu para distinguir", não "são iguais".
- Cada rodada faz **várias comparações** (uma por execução contra a referência) **sem
  correção para comparações múltiplas** (Holm ou Bonferroni). Com muitas comparações, algumas
  podem dar p < α por acaso. Como a regra procura as que **não** diferem da melhor, a falta de
  correção deixa a seleção mais rigorosa (menos candidatas), não mais permissiva; ainda assim,
  trate p-valores perto de α com cautela e registre isso no relatório.

### Garantias contra sobrescrita

- O runner só **cria** pastas. Ele nunca apaga nem edita nada em `results/runs/`.
- Se a mesma config, com os mesmos dados, já foi executada, ele avisa e pula. Com
  `--force`, cria uma pasta nova ao lado da antiga.
- Enquanto roda, a execução fica numa pasta `.tmp_...`. Ela só ganha o nome definitivo
  quando tudo foi gravado, então uma execução interrompida nunca conta como feita.
- Se você mudar o **código** de um método já existente (não a config), o hash não muda e o
  runner vai pular. Nesse caso, use `--force`.

## 8. Teste final

Só o `final_test.py` lê o `test.csv`. Ele:

1. carrega o `pipeline.joblib` da execução indicada, ou seja, **a mesma árvore** de `tree.txt`/`tree.png`;
2. avalia uma vez no teste;
3. grava `metrics.json`, `confusion_matrix.csv` e `meta.json` em `results/final/<run>/`.

Se a execução já foi avaliada, ele apenas mostra o resultado guardado, sem avaliar de novo.

## 9. Criando um método novo

Exemplo: um imputador que preenche com a moda.

1. Em `src/steps/imputers.py`:
   ```python
   def most_frequent():
       return SimpleImputer(strategy="most_frequent")
   ```
2. Em `src/registry.py`, na entrada `"imputer"`:
   ```python
   "most_frequent": imputers.most_frequent,
   ```
3. No YAML:
   ```yaml
   - name: E08_moda
     override: {imputer: most_frequent}
   ```

Transformadores que aprendem algo com os dados (quantis, médias...) devem calcular esses
valores no `fit` e só aplicá-los no `transform`, como o `SimpleImputer` faz com a mediana.
É isso que impede a contaminação com os dados de validação.