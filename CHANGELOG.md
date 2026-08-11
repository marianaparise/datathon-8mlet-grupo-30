# Histórico de modificações

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
Datas em `AAAA-MM-DD`.

Este arquivo é ferramenta de desenvolvimento, não entregável do Datathon — a documentação
avaliada pela banca mora no `README.md` (ver `CLAUDE.md`, seção 3).

---

## [Não lançado]

Próxima: **Fase 2** — `src/arms.py` e `src/environment.py`, com `P(y | contexto, braço)` calibrado
e validado por Brier score e curva de confiabilidade. O teste de heterogeneidade braço × contexto
sobe de prioridade: a Fase 1 mostrou que ela é fraca no espaço de braços escolhido.

---

## [0.4.0] — 2026-08-10 — Fase 1: dados, EDA e o espaço de braços

Cobre as Etapas 1 e 2 do enunciado. As quatro decisões que a fase existia para tomar foram
fechadas com número, e o critério de cada uma foi registrado antes de rodar a análise.

### Adicionado
- `notebooks/01_eda.ipynb` — análise exploratória e tratamento de dados. Versionado **sem saídas**;
  só chama funções de `src/`, nenhuma célula passa de 4 linhas e nenhuma define lógica de negócio.
- `src/eda.py` — agregações e figuras: `wilson_interval()`, `conversion_by()`, `arm_support()`,
  `screen_arm_spaces()`, `duration_leakage()`, `contact_by_month()`, `macro_calendar_report()`,
  `period_index()`, `month_run_count()` e as funções de plot.
- `src/data.py` — `add_week_window()`, `add_first_contact()`, `build_arm()`, `prepare()` e
  `split_train_test()`.
- `src/config.py` — `MACRO_COLUMNS`, `CLIENT_COLUMNS`, `WEEK_WINDOWS`, `ARM_COLUMNS`,
  `ARM_SPACE_CANDIDATES`, `MIN_EVENTS_PER_ARM`, `MIN_CONVERSIONS_PER_ARM`, `PDAYS_SENTINEL`,
  `UNKNOWN_TOKEN`, `TEST_SIZE` e os nomes das colunas derivadas.
- `conftest.py` na raiz, com as fixtures `raw`, `prepared` e `toy`.
- `tests/test_eda.py` e ampliação de `tests/test_data.py` — 41 testes no total, cobrindo as 8
  funções públicas de `data.py` e as 17 de `eda.py`.
- `reports/figures/` — quatro figuras geradas pelo notebook.
- `.ruff.toml` e alvo `make lint`. O projeto não tinha verificação estática nenhuma, então
  "está limpo" era opinião, não checagem. Regras: pycodestyle, pyflakes, isort, pyupgrade,
  bugbear, simplify, type hints e docstrings Google, em 88 colunas — a largura que `config.py` e
  `data.py` já praticavam. Testes ficam isentos de docstring e de anotação em fixture.

### Dependências
- `ruff==0.16.2`. Ferramenta única para lint e ordenação de import, binário sem dependência
  transitiva — não conflita com a regra de não adicionar framework pesado. Fica em
  `requirements.txt` junto do `pytest`, seguindo a convenção que o arquivo já usava para
  dependência de desenvolvimento.

### Corrigido
- **`make test` não coletava nenhum teste.** `pytest tests/` insere no `sys.path` o diretório do
  arquivo de teste, não a raiz do repositório, então `from src import config` levantava
  `ModuleNotFoundError` e a coleta abortava — só `python -m pytest` funcionava. O `conftest.py` na
  raiz resolve, porque o pytest insere o diretório dele no path. O bug passou despercebido na
  0.3.0 porque a base ainda não estava baixada e a suíte parecia apenas "skipada".
- `tests/test_data.py` usava `pytestmark` de módulo para pular tudo quando a base falta. Trocado
  por skip na fixture: em clone limpo a suíte agora roda os testes que não dependem do CSV, em vez
  de ficar verde e vazia.

### Decidido
- **Espaço de braços: `contact` × `week_window`, 6 braços.** Critério pré-registrado — o pior braço
  precisa de ≥ 1.000 eventos e ≥ 100 conversões; ele tem 2.979 e 139. Os três candidatos passaram
  no piso, então o desempate foi aplicado depois de ver o dado, e isso está declarado no notebook:
  `contact × day_of_week` cria braços com intervalos de Wilson sobrepostos e derruba o
  aproveitamento do replay de 16,7% para 10%.
- **`month` fica fora do braço** — 100 células, e é o confundidor temporal principal.
- **Baseline: a política de log (11,27%)**, não o melhor braço histórico. O braço modal
  (`cellular|mid`, 38,8% do volume) é também o de maior conversão (15,47%), então "melhor braço
  histórico" como baseline garantiria ganho zero para qualquer bandit não-contextual — a armadilha
  que a decisão em aberto #4 do briefing antecipava. Fecha #4.
- **Indicadores macro separados do contexto de cliente.** Critério pré-registrado: R² contra o
  índice de período ≥ 0,95. Todos deram 1,0000, exceto `euribor3m` com 0,9996 — são constantes
  dentro do período e não personalizam nada. Ficam na base como ablação, fora do contexto padrão.
  Destemporalizar foi descartado: sem coluna de data, a variação seria tão temporal quanto o nível.
  Fecha #5.
- **Split estratificado por alvo × braço, seed 42.** O temporal produz 24,5 p.p. de deriva de
  conversão entre treino e teste (6,37% contra 30,83%), contra 0,014 p.p. do estratificado. Fecha
  #6. A estratificação inclui o braço porque o ambiente da Fase 2 estima `P(y | contexto, braço)`
  e precisa dos seis braços povoados nos dois folds.
- **`unknown` preservado como nível**, nunca imputado — é resposta registrada, não ausência.
- **`pdays == 999` tratado como sentinela**, virando a flag `first_contact`.

### Notas
- O vazamento de `duration` agora tem número: sozinha ela rende AUC 0,819, acima das 19 features
  legítimas juntas (0,816); no conjunto completo a AUC vai a 0,955.
- A ordem do arquivo é cronológica — 26 blocos de meses consecutivos em 41.188 linhas. É o que
  torna o índice de período utilizável sem coluna de data.
- **Alerta para a Fase 2:** a heterogeneidade braço × contexto é fraca. Onde o melhor braço muda
  entre estratos, os intervalos de Wilson dos concorrentes se sobrepõem — nenhuma troca é
  estatisticamente distinguível.
- Desvio do `docs/PLANO.md`, que previa tudo em `src/data.py`: a EDA foi para `src/eda.py` porque
  `data.py` está no caminho de import da API da Fase 6 e não pode arrastar matplotlib para o
  container.

### Verificado
- `make test` — 41 testes, todos passando. `make lint` sem achados.
- Notebook executado ponta a ponta com `nbconvert --execute` sobre cópia descartável; o arquivo
  versionado permanece sem saídas.
- `duration` ausente de `prepare()` e de tudo que a Fase 2 consome.

---

## [0.3.0] — 2026-08-10 — Briefing do time e automação do download

### Adicionado
- `src/config.py` — schema e constantes de reprodutibilidade, sem efeito colateral no import:
  caminhos, `SEED`, `RAW_SEPARATOR`, e a partição explícita das colunas em
  `CONTEXT_COLUMNS` (atributos do cliente), `ACTION_COLUMNS` (decisões da campanha, de onde saem os
  braços), `FORBIDDEN_COLUMNS` (`duration`) e `TARGET`.
- `src/data.py` — `load_raw()`, `drop_forbidden()` e `binarize_target()`.
- `tests/test_data.py` — 8 testes cobrindo shape, schema, separador errado, arquivo ausente,
  imutabilidade do DataFrame original e a taxa de conversão de 11,27%.
- `scripts/download_data.sh` e reescrita do alvo `make data`. Tenta o Kaggle (fonte citada no
  enunciado) e cai no **UCI** quando não há credencial, tornando o repositório clonável e executável
  por qualquer pessoa — inclusive pela banca. Ambas as vias passam por verificação **SHA-256**
  (`74adfc57…afb4d8`), porque duas fontes só servem se entregarem exatamente o mesmo arquivo.
  O alvo é idempotente e redetecta arquivo corrompido.
- Proveniência da base documentada no README com dados verificados: 41.188 × 21, separador `;`,
  conversão global de 11,27% (4.640 `yes` / 36.548 `no`), licença CC BY 4.0.
- `docs/BRIEFING.md` — documento de contexto para membros do time que não acompanharam as decisões:
  resumo do desafio, explicação do problema de *bandit feedback*, as três formulações avaliadas e o
  racional da escolha A+C, decisões fechadas, **decisões em aberto** e roteiro de estudo priorizado.
- Ponteiro para o briefing no topo do `README.md`.

### Corrigido
- Citação do paper de replay em `README.md` e `docs/PLANO.md`: era "Li et al., 2010", mas
  `arXiv:1003.5956` é **WSDM 2011** (o preprint é de março de 2010, daí a confusão). O paper de 2010 é
  o outro, `arXiv:1003.0146`, que introduz o LinUCB. Ambos verificados na fonte.

### Notas
- O zip do UCI contém **zips aninhados** (`bank.zip` e `bank-additional.zip`); o arquivo de interesse
  está no segundo, junto de `bank-additional-names.txt` com a descrição oficial das colunas.
- O CSV usa **`;` como separador**, não vírgula — e ler com o separador padrão do pandas devolve
  `(41188, 1)` **sem levantar exceção**. Por isso o separador virou constante em `config.py` e
  `load_raw()` valida o schema depois de ler, em vez de confiar no parse. O arquivo bruto **não** foi
  convertido: alterar dado cru invalidaria o checksum que acabamos de estabelecer.
- Registrado no briefing o enquadramento teórico da escolha A+C: as opções correspondem aos dois
  estimadores canônicos de **Off-Policy Evaluation** — Direct Method (A) e Importance Sampling (C) —
  cuja combinação formal é o **Doubly Robust** (Dudík, Langford & Li, ICML 2011).

---

## [0.2.0] — 2026-08-03 — Fase 0: fundação do repositório

Cobre a **Etapa 0** do enunciado.

### Adicionado
- `CLAUDE.md` com contexto, decisões fechadas, restrições do enunciado e convenções de código.
- `docs/PLANO.md` com o plano de implementação em 8 fases, mapeado para as Etapas 0–8.
- `README.md` inicial: problema de negócio, base, formulação dos braços, estratégia de avaliação
  em duas camadas, políticas comparadas, instruções de execução e limitações conhecidas.
  Seções de resultados, Golden Set, arquitetura em nuvem e governança marcadas como pendentes.
- `requirements.txt` com versões **resolvidas a partir de uma instalação real** em Python 3.12.3,
  não estimadas.
- `Makefile` com `setup`, `data`, `train`, `api`, `mlflow`, `test`, `docker-*` e `clean`.
- `.gitignore` cobrindo `data/`, `models/`, `mlruns/`, `.venv/` e `kaggle.json`.
- Esqueleto de diretórios: `src/`, `api/`, `notebooks/`, `tests/`, `data/{raw,processed}/`,
  `models/`, `reports/figures/`, `docs/`.

### Dependências
Justificativa das escolhas, conforme exigido pelo `CLAUDE.md`:

- `scikit-learn` cobre modelagem e calibração via `HistGradientBoostingClassifier` +
  `CalibratedClassifierCV`. **Sem LightGBM, XGBoost ou torch** — mesma capacidade aqui, sem peso
  extra na imagem Docker.
- `httpx2` entrou porque o `TestClient` do Starlette nesta versão o exige; sem ele o import emite
  `StarletteDeprecationWarning`.
- `kaggle` para tornar o download da base reprodutível via `make data`.

### Verificado
- `pip install` completo em venv limpa, com `EXIT=0`.
- Smoke test de imports e do pipeline `CalibratedClassifierCV(HistGradientBoostingClassifier)`,
  com Brier score computado.
- `from fastapi.testclient import TestClient` sem warnings sob `-W error::DeprecationWarning`.

---

## [0.1.0] — 2026-08-03 — Planejamento

Sessão de leitura do enunciado e fechamento das decisões estruturais. Sem código ainda.

### Decidido
- **Base de dados:** Kaggle `bank-marketing` (henriqueyamahata), equivalente ao
  `bank-additional-full.csv` do UCI (~41k linhas, 20 features, target binário `y`).
  Escolhida por ser a única entre as sugeridas que tem `day_of_week` **e** contexto socioeconômico —
  sustenta ao mesmo tempo braços reais bem definidos e um bandit contextual.
- **Formulação dos braços:** derivados de colunas reais de ação (`contact` × janela de contato),
  e não de ofertas de produto fictícias. Motivo: todos os braços existem no log histórico, então
  `P(y | contexto, braço)` é estimável de dado observado, sem sintetizar recompensa.
- **Estratégia de avaliação A + C:** ambiente calibrado nos dados como track principal (curvas de
  regret contra oráculo) e replay/rejection sampling sobre o log real como validação metodológica.
- **Serviço:** FastAPI + Docker, com endpoint de recomendação e Swagger para a demo em vídeo.
- **Rastreio de experimentos:** MLflow local.

### Descartado
- **Braços como ofertas de produto sintéticas (opção B).** Exigiria escrever à mão a função de
  resposta de 4 produtos sem nenhuma observação real, contrariando a Etapa 2 do enunciado
  ("sem precisar gerar dados sintéticos complexos"). Além disso duplicaria Etapas 2 a 7 sob uma
  segunda formulação de negócio, diluindo os 30% de clareza. Sobrevive apenas como parágrafo de
  roadmap no README, sem métrica associada.
- **Replay puro (opção C isolada).** Descarta ~85% dos eventos com 6 braços, gera curvas ruidosas e
  é a alternativa com maior risco de o bandit não superar o baseline com clareza — requisito
  explícito da Etapa 3. Mantido como camada de validação, não como track principal.

### Notas
- `duration` marcada como coluna proibida (vazamento temporal, citada nominalmente no PDF).
- Confounding conhecido a documentar: `contact` é correlacionado com o tempo — telefone fixo domina
  o início da série e celular o fim. Parte da diferença entre braços é época, não canal.
- `euribor3m` e `nr.employed` são proxies temporais fortes; exigem tratamento consciente para o
  modelo não "acertar" por calendário.
