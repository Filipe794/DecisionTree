# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Projeto de mestrado: árvore de decisão (scikit-learn) na base Pima Indians Diabetes, com um sistema
declarativo de experimentos. A documentação de uso para humanos está em `README.md`, em português;
mantenha o código, os comentários e as mensagens em português. `COMO_USAR.md` guarda anotações do usuário.

## Comandos

Todos rodam a partir da raiz, com o `.venv` (Python 3.14):

```bash
pip install -r requirements.txt
python -m src.make_split                       # gera data/split/{train,test}.csv (uma vez; --force regera)
python run_experiments.py experiments/r1_tamanho_arvore.yaml                 # uma rodada
python run_experiments.py experiments/r1_tamanho_arvore.yaml --only R1_grade # um experimento (ou um sweep inteiro)
python run_experiments.py experiments/r1_tamanho_arvore.yaml --force         # repete, cria pasta nova
python summarize.py [--metric recall]          # regenera results/summary.csv
python compare.py <run_A> <run_B> [--metric roc_auc]   # teste t corrigido (Nadeau-Bengio) fold a fold
python final_test.py --run <pasta em results/runs>
```

Não há testes, linter nem build. Para validar uma mudança, rode os experimentos afetados com
`--only`. Evite gravar resultados de teste em `results/` do usuário; use uma cópia do projeto.
Para validar um YAML sem rodar nada, use `load_experiments()` + `build_pipeline()` de
`run_experiments.py` num script temporário.

## Arquitetura

O fluxo é YAML → `resolve` → `build_pipeline` → `cross_validate` → pasta imutável em `results/runs/`.

- **Etapas fixas e ordenadas:** `src/registry.py` define `STEP_ORDER` (zeros → missing_flags →
  imputer → features → selection → balancing), seguida do `model` (`DecisionTreeClassifier`), e
  `OPTIONAL_STEPS` (`threshold`, que envolve o pipeline inteiro em `TunedThresholdClassifierCV`).
  Cada etapa recebe uma opção pelo nome em `REGISTRY`; `none` pula a etapa.
- **Método novo:** escreva a função-fábrica em `src/steps/<etapa>.py` e acrescente uma linha no
  `REGISTRY`. O builder, o runner e o YAML não mudam. Transformadores que aprendem algo dos dados
  calculam no `fit` (anti-vazamento). Os customizados devolvem DataFrame e implementam
  `get_feature_names_out` quando necessário (ex.: `KNNScaledImputer` em `src/steps/imputers.py`).
- **Não altere o comportamento de uma opção existente:** o hash não inclui o código, então
  execuções antigas ficariam com o mesmo hash e outro resultado. Crie uma opção nova (foi o caso
  de `knn_scaled`, ao lado de `knn`, que fica para comparação).
- **Config:** `src/builder.py:resolve` junta baseline + override, e o override **substitui a etapa
  inteira**, inclusive `model`. Etapas são normalizadas para `{name: ..., **params}`. As opcionais
  só entram na config quando escritas, para não alterar o hash de configs antigas (por isso o
  `threshold` fica comentado, e não `none`, nos YAMLs que não o usam).
- **Sweep:** `run_experiments.py:expand_sweep` expande `sweep: {etapa.param: [...]}` em
  experimentos `nome__param=valor` (várias chaves = produto cartesiano). `param` pode ser `name`
  (ex.: `imputer.name: [knn, knn_scaled]`). Valores com `:` (ex.: `class_weight: {0: 1, 1: 1.5}`)
  só em `override`: no sweep entrariam no nome da pasta, e o Windows não aceita `:`.
- **`baseline_name`** (chave opcional do YAML, padrão `E01_baseline`) é só o rótulo do baseline e
  não entra no hash.
- **Pipeline:** usa `imblearn.pipeline.Pipeline` (necessário para os balanceadores). O
  `src/__init__.py` aplica `set_config(transform_output="pandas")` globalmente, e as colunas mantêm
  o nome entre as etapas. Isso é necessário também ao carregar um `pipeline.joblib`.
- **Runner (`src/runner.py`):**
  - o hash da config completa + seed + CV + métricas + sha256 do `train.csv` define a identidade
    da execução; um hash já existente é pulado;
  - grava em `.tmp_*` e renomeia no fim; nunca apaga nem edita execuções existentes;
  - depois do CV, treina a árvore final no treino inteiro e salva `pipeline.joblib`, `tree.txt`,
    `tree.png` e `feature_importances.csv`;
  - use `builder.unwrap()` para chegar ao pipeline sob o wrapper de threshold.
- **Métricas:** `SCORING` em `src/config.py` (accuracy, precision, recall, f1, balanced_accuracy,
  roc_auc; `PRIMARY_METRIC = roc_auc`) é a única lista: o runner, o `summarize.py`, o `compare.py`
  e o `final_test.py` (via `sklearn.metrics.get_scorer`) leem dela. Mudar o `SCORING` muda o hash
  de todas as configs. O `summarize.py` preenche com NaN uma métrica ausente numa execução antiga.
- **`summarize.py`** só lê as pastas; a tabela é sempre derivada.
- **`compare.py`** exige as mesmas 25 linhas (repeat, fold) e o mesmo `train_sha256` no
  `meta.json` (confere o split sozinho e para com `sys.exit` se divergir ou se faltar o `meta.json`).

## Rodadas de experimentos

Um YAML por rodada em `experiments/`; o vencedor de uma rodada vira o baseline da seguinte.
`experiments/experiments.yaml` é só o histórico da exploração inicial.

| Arquivo | Pergunta | Critério |
|---|---|---|
| `r1_tamanho_arvore.yaml` | tamanho da árvore (`max_depth`, `min_samples_leaf`, `ccp_alpha`, `max_leaf_nodes`) | `roc_auc` |
| `r2_ausentes.yaml` | zeros e imputação (inclui `knn` x `knn_scaled`, com e sem `class_weight`) | `roc_auc` |
| `r3_desbalanceamento.yaml` | recall sem SMOTE (`class_weight`, `threshold`, `random_under`) | F1 ou `balanced_accuracy`, recall ≥ 0.70 |
| `r4_variaveis.yaml` | variáveis derivadas e seleção | `roc_auc` |
| `r5_combinacao.yaml` | combinação dos vencedores + grade pequena de complexidade | o da rodada que decidiu cada ponto |

Em todas: entre as configs sem diferença significativa para a melhor (`compare.py`), fica a mais
simples. R6 (escolha), R7 (`final_test.py`, uma vez) e R8 (robustez, numa cópia do projeto) não
têm YAML.

- Do R2 em diante, linhas com `# ATUALIZAR: vencedor de Rx` são provisórias e devem ser trocadas
  pelo usuário antes de rodar. Não preencha vencedores por conta própria.
- O baseline de cada rodada é a mesma config do vencedor anterior, então o runner o pula pelo
  hash e ele não aparece com o nome `Rx_baseline` no `summary.csv`: a referência é a pasta do
  vencedor da rodada anterior.

## Restrições importantes

- **O conjunto de teste é isolado:** só `final_test.py` chama `load_test()`. Nunca use
  `test.csv` durante os experimentos e não rode `final_test.py` sem o usuário pedir.
- **Sem SMOTE, por decisão do usuário:** as opções `smote`, `borderline_smote`, `adasyn`,
  `smote_tomek` e `smote_enn` continuam no `REGISTRY` (há execuções antigas que as usam), mas não
  devem entrar nos YAMLs das rodadas. O desbalanceamento é tratado com `class_weight`, ajuste de
  limiar ou `random_under`.
- **Sem etapa de outliers, por decisão do usuário:** árvores não são sensíveis a valores extremos.
  Uma chave desconhecida no YAML (ex.: `outliers`) gera erro.
- **CV fixo** em `src/config.py` (`RepeatedStratifiedKFold` 5x5, `SEED=42`), igual para todos os
  experimentos. Componentes aleatórios devem receber `random_state=SEED`.
- **Paralelismo:** a validação cruzada roda com `parallel_config(backend="threading")`. O backend
  padrão do joblib (loky) quebra no Python 3.14 porque o caminho do projeto tem acento ("Máquina").
- **Dado corrompido:** em `data/raw/` (nunca alterar), a `DiabetesPedigreeFunction` < 1 perdeu o
  "0." (0.627 → 627). `make_split.fix_dpf` divide por 1000 os valores ≥ 3.
- **NaN no pipeline (`imputer: none`):** os balanceadores do imblearn não aceitam NaN, e o
  `build_pipeline` rejeita a combinação com `balancing` (isso derruba a validação do YAML inteiro,
  ex.: `R3_subamostragem`). O `kbest` (`f_classif`) também não aceita NaN, mas só falha durante o
  CV. Se o R2 escolher `imputer: none`, esses experimentos precisam sair do R3 e do R4.
