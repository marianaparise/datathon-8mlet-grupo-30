# Plano de implementação — TC5 / Datathon

Formulação escolhida: braços de ação real + **avaliação A (ambiente calibrado) e C (replay)**.
Decisões e restrições em `CLAUDE.md`.

---

## Mapa: fase → etapa do enunciado

| Fase | Entrega | Etapas cobertas |
|---|---|---|
| 0 ✅ | Fundação do repositório | Etapa 0 |
| 1 ✅ | Dados e EDA | Etapas 1 e 2 |
| 2 ✅ | Braços e ambiente calibrado | base da Etapa 3 |
| 3 ✅ | Políticas e experimentos | Etapas 3 e 7 |
| 4 ✅ | Replay sobre o log real | validação da Etapa 3 |
| 5 ✅ | Métricas e Golden Set | Etapa 4 |
| 6 | API e Docker | Etapa 5 |
| 7 | README consolidado e arquitetura em nuvem | Etapas 0, 1, 6 |
| 8 | Vídeo pitch | Etapa 8 |

---

## Estrutura do repositório

```
tc5/
├── README.md                 # documentação consolidada — o entregável de texto
├── CLAUDE.md                 # regras do projeto
├── CHANGELOG.md              # histórico de modificações
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        # api + mlflow ui
├── Makefile                  # atalhos de execução
├── train.py                  # entrypoint: prepara, calibra, roda experimentos, serializa
├── docs/PLANO.md             # este arquivo
├── data/raw/                 # bank-additional-full.csv        (gitignored)
├── data/processed/           # dataset tratado                 (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb              # Etapas 1 e 2
│   └── 02_experimentos.ipynb     # Etapas 3 e 4
├── src/
│   ├── config.py             # paths, SEED, definição dos braços, listas de features
│   ├── data.py               # carga, limpeza, split contexto/ação, split treino/teste
│   ├── arms.py               # construção e encoding do espaço de braços
│   ├── environment.py        # P(y | contexto, braço) calibrado — track A
│   ├── policies.py           # baseline, ε-greedy, UCB1, Thompson, LinTS
│   ├── replay.py             # rejection sampling sobre o log real — track C
│   ├── evaluation.py         # métricas, runner multi-seed, integração MLflow
│   └── golden_set.py         # seleção e explicação dos 5 casos
├── api/
│   ├── app.py                # FastAPI
│   └── schemas.py            # contratos pydantic
├── models/                   # artefatos serializados            (gitignored)
├── reports/figures/          # gráficos exportados para o README
├── tests/
└── mlruns/                   # MLflow local                      (gitignored)
```

---

## Fase 0 — Fundação

**Cobre a Etapa 0.**

- [ ] `git init`, primeiro commit, repositório público `datathon-7mlet-grupo-XX`
- [ ] `.gitignore` — `data/raw/`, `data/processed/`, `mlruns/`, `models/*.joblib`, `.venv/`, `__pycache__/`
- [ ] `requirements.txt` pinado e validado com `pip install` real
- [ ] `Makefile` — `setup`, `data`, `train`, `api`, `mlflow`, `test`
- [ ] esqueleto de `README.md` com as seções que serão preenchidas ao longo do caminho

**Saída:** repositório instalável do zero em máquina limpa.

---

## Fase 1 — Dados e EDA

**Cobre as Etapas 1 e 2.**

- [x] Baixar a base e documentar o procedimento no README (link Kaggle, versão, licença)
- [x] `notebooks/01_eda.ipynb`:
  - [x] perfil geral, tipos, cardinalidade, tratamento de `unknown`
  - [x] taxa de conversão global e por segmento
  - [x] **taxa de conversão por candidato a braço, com contagem por célula**
  - [x] evidência do vazamento de `duration` — justificar o descarte com número, não com citação
  - [x] **checagem de confounding temporal**: distribuição de `contact` ao longo de `month`
  - [x] comportamento de `euribor3m` e `nr.employed` no tempo
- [x] `src/data.py`: pipeline de limpeza determinístico
  - [x] descarte de `duration`
  - [x] separação explícita entre colunas de **contexto** e colunas de **ação**
  - [x] split treino/teste estratificado com seed fixa

**Decisão que sai desta fase:** o espaço de braços definitivo, escolhido pelo suporte amostral
observado — não antes dele.

**Risco:** granularidade fina demais esvazia células. Mitigação: agregar `day_of_week` em janelas
(início/meio/fim de semana) e exigir um piso de eventos e de conversões por braço.

**Concluída.** Espaço fixado em `contact × week_window` — 6 braços, o pior com 2.979 eventos e 139
conversões contra um piso de 1.000 e 100. A agregação de `day_of_week` foi de fato necessária, mas
não pelo motivo previsto: `contact × day_of_week` também passava no piso, e o que a descartou foi a
sobreposição dos intervalos de Wilson e a queda do aproveitamento do replay.

Desvio: a análise foi para `src/eda.py`, não para `src/data.py`. `data.py` está no caminho de import
da API da Fase 6 e não pode arrastar matplotlib para o container.

---

## Fase 2 — Braços e ambiente calibrado

**Base da Etapa 3. É o track A.**

- [x] `src/arms.py`: espaço de braços, encoding, mapeamento log → braço
- [x] `src/environment.py`: estimar `P(y | contexto, braço)`
  - [x] `HistGradientBoostingClassifier` com `CalibratedClassifierCV`
  - [x] **validar a calibração**: Brier score e curva de confiabilidade no conjunto de teste.
        Ambiente mal calibrado invalida todo o resto
  - [x] comparar com um baseline de propensão (regressão logística) para sanidade
- [x] Testar **heterogeneidade braço × contexto**: o melhor braço muda conforme o perfil?
  - [x] se não muda, o bandit contextual não tem o que ganhar → revisar o espaço de braços

**Saída:** ambiente que, dado um cliente e um braço, devolve recompensa Bernoulli calibrada,
mais o oráculo `argmax_a p̂(y|x,a)` que permite calcular regret verdadeiro.

**Risco declarado:** o ambiente herda o viés do log histórico. Isso é limitação a documentar no
README, e é justamente o que a Fase 4 existe para contrabalançar.

---

## Fase 3 — Políticas e experimentos

**Cobre as Etapas 3 e 7.**

- [x] `src/policies.py`, todas com a mesma interface `select` / `update`:
  - [x] `FixedPolicy` — **baseline determinístico**: a regra que a operação de fato seguia
  - [x] `BestHistoricalArm` — segundo comparador, mais difícil de bater
  - [x] `EpsilonGreedy`
  - [x] `UCB1`
  - [x] `ThompsonSampling` — Beta-Bernoulli, **priors documentados** (exigência explícita do PDF)
  - [x] `LinTS` (ou `LinUCB`) — a política contextual, onde mora o diferencial
- [x] `src/evaluation.py`:
  - [x] runner de N rodadas × M seeds
  - [x] métricas: CVR acumulada, regret acumulado vs oráculo, uplift % vs baseline, IC entre seeds
  - [x] curvas exportadas para `reports/figures/`
- [x] MLflow: um run por (política, seed)
  - [x] params: `policy`, `epsilon`, `alpha`, priors Beta, `n_rounds`, `seed`, `n_arms`
  - [x] métricas: `cvr_final`, `regret_final`, `uplift_vs_baseline`, `exploration_rate`
- [x] `train.py` amarra o pipeline ponta a ponta e serializa os artefatos

**Ponto de atenção de design:** o enunciado permite escolher o baseline entre regra fixa, melhor braço
histórico ou segmentação inicial. Se o baseline for "melhor braço histórico" e existir um braço
globalmente dominante, um bandit não-contextual converge para ele e o ganho fica em zero. Por isso
o baseline principal é a **regra fixa**, com o melhor braço histórico como comparador secundário —
é contra ele que o **contextual** precisa mostrar vantagem.

---

## Fase 4 — Replay sobre o log real

**Validação da Etapa 3. É o track C.**

- [x] `src/replay.py`: rejection sampling (Li et al., WSDM 2011) sobre o conjunto de teste
  - [x] percorre o log; aceita o evento quando o braço escolhido coincide com o braço registrado
  - [x] recompensa = `y` observado, nunca estimado
  - [x] reporta taxa de aceitação e tamanho efetivo da amostra
- [x] Ponderação por propensity (IPS) para atenuar o viés da política de log
- [x] Comparar o ranking de políticas obtido no replay com o obtido no ambiente calibrado
- [x] Documentar as ressalvas: a política de log não era aleatória uniforme, então a garantia de
      não-viés do replay não se aplica integralmente

**Critério de sucesso:** o ranking das políticas se preserva entre os dois tracks. Se divergir, isso
é achado relevante e vai para o README — não é problema a esconder.

---

## Fase 5 — Métricas e Golden Set

**Cobre a Etapa 4.**

- [x] Consolidar a tabela comparativa: baseline vs cada política, nos dois tracks
- [x] `src/golden_set.py`: **5 clientes** escolhidos por critério, não por sorteio — perfis que
      exercitam decisões diferentes (ex.: sem histórico, `poutcome=success`, muitos contatos prévios,
      perfil médio, perfil de canal atípico)
- [x] Para cada caso: features de entrada, braço recomendado, `p̂` de todos os braços e a justificativa
      em linguagem de negócio — "a decisão fez sentido?" é a pergunta literal do enunciado
- [x] Tabela replicada no README

---

## Fase 6 — API e Docker

**Cobre a Etapa 5.**

- [ ] `api/schemas.py`: contrato de entrada com validação pydantic
- [ ] `api/app.py`:
  - [ ] `POST /recommend` → braço recomendado, `p̂` estimada, ranking completo e versão da política
  - [ ] `GET /health`
  - [ ] `/docs` Swagger — é o que aparece no vídeo
- [ ] Carga dos artefatos de `models/` no startup, sem retreinar
- [ ] `Dockerfile` multi-stage
- [ ] `docker-compose.yml` subindo API + MLflow UI juntos
- [ ] `tests/test_api.py` com `TestClient`, incluindo payload inválido

---

## Fase 7 — README consolidado e arquitetura em nuvem

**Cobre as Etapas 0, 1 e 6. É onde os 30% de negócio são ganhos ou perdidos.**

- [ ] Problema de negócio e por que bandit em vez de A/B
- [ ] Link da base no Kaggle, versão, licença, colunas, target
- [ ] Formulação: o que é um braço aqui e por que essa escolha
- [ ] Resultados: tabela comparativa e curvas de regret
- [ ] Golden Set
- [ ] Instruções de execução local — precisam funcionar em máquina limpa
- [ ] **Arquitetura-alvo em nuvem**, 1–2 parágrafos (diagrama opcional): S3 para dados, ECR + ECS
      Fargate para a API, Kinesis Firehose para o log de recompensas, DynamoDB para o estado dos
      braços, MLflow gerenciado, CloudWatch para observabilidade
- [ ] **Governança**: base legal, finalidade, minimização, retenção, humano no loop
- [ ] **Limitações**, sem maquiagem: viés do log histórico, confounding temporal de `contact`,
      ambiente calibrado não é tráfego real, replay com política de log não aleatória
- [ ] Parágrafo de roadmap: o motor é agnóstico ao espaço de ações; migrar para catálogo de produtos
      exige log de recompensa por produto, que esta base não tem

---

## Fase 8 — Vídeo pitch

**Cobre a Etapa 8. Até 5 minutos.**

Roteiro sugerido:

| Tempo | Conteúdo |
|---|---|
| 0:00–1:00 | Problema de negócio: regra fixa e A/B longo desperdiçam tráfego |
| 1:00–2:00 | Base, formulação dos braços e por que não sintetizamos recompensa |
| 2:00–3:30 | Resultados: adaptativo vs baseline, curvas de regret, MLflow na tela |
| 3:30–5:00 | Demo ao vivo: `POST /recommend` no Swagger devolvendo recomendação |

- [ ] Gravar com a API rodando de verdade, não com slide de print

---

## Riscos principais

| Risco | Impacto | Mitigação |
|---|---|---|
| Um braço domina globalmente | Bandit empata com o baseline — falha a Etapa 3 | Baseline ancorado em regra fixa; contextual explora heterogeneidade; verificar a heterogeneidade já na Fase 2 |
| Células de braço com poucos eventos | Ambiente mal calibrado | Agregar `day_of_week` em janelas; piso de suporte por braço definido na Fase 1 |
| `euribor3m` domina o modelo | "Acerto" por calendário, não por cliente | Avaliar importância de features; considerar remover ou destemporalizar |
| Ambiente calibrado soa fabricado para a banca | Perda nos 70% técnicos | É exatamente o que a Fase 4 (replay) responde, com evidência |
| Docker pesado atrasa a demo | Vídeo travado | Sem framework pesado; imagem slim, artefatos pré-serializados |

---

## Stack

`pandas`, `numpy`, `scikit-learn`, `scipy` para dados e modelagem.
`matplotlib`, `seaborn` para as figuras. `mlflow` para rastreio.
`fastapi`, `uvicorn`, `pydantic` para o serviço. `pytest`, `httpx` para os testes.

Sem LightGBM, XGBoost ou torch — `HistGradientBoostingClassifier` cobre a necessidade sem peso extra
na imagem.
