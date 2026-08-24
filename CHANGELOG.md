# Histórico de modificações

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
Datas em `AAAA-MM-DD`.

Este arquivo é ferramenta de desenvolvimento, não entregável do Datathon — a documentação
avaliada pela banca mora no `README.md` (ver `CLAUDE.md`, seção 3).

---

## [Não lançado]

Próxima: **Fase 6** — API FastAPI servindo o ranking por braço, mais Docker.

### Adicionado
- `src/golden_set.py` e `tests/test_golden_set.py` (18 testes) — os cinco casos da Etapa 4,
  escolhidos por critério e não sorteados, com `p̂` de todos os seis braços por cliente e
  justificativa gerada a partir dos números.
- Seção Golden Set no README e `models/golden_set.csv` gerado por `make train`.
- `src/scenarios.py` e `tests/test_scenarios.py` (13 testes) — análise de sensibilidade temporal.
  `channel_confounding_report()` mede quanto do efeito de canal é, na verdade, calendário.
- Seção dedicada no README quantificando a limitação principal do projeto.

### Corrigido
- **O portão de calibração do ambiente estava errado desde a Fase 2.** Usava piso **absoluto** de
  Brier em 0,10 — que por coincidência é exatamente `p(1-p)` da base completa, o que fazia o número
  parecer princípio. Num recorte com taxa-base de 14,5% a referência é 0,1240, e um Brier de 0,1061
  seria rejeitado apesar de bom. Trocado por **Brier Skill Score** (`1 - brier/p(1-p)`), que é
  relativo e transfere entre recortes. O ambiente principal tem skill de 0,141.

### Notas — Golden Set
- **Três dos cinco casos dão empate entre os braços do topo**, e isso é o teto contextual de +4,44%
  visto cliente a cliente: na maioria dos perfis os líderes são indistinguíveis, numa minoria a
  escolha vale muito. O caso do estudante de 18 anos tem 38 p.p. entre o melhor e o pior braço.
- **Empate não é vendido como troca de braço.** `TIE_THRESHOLD` de 0,5 p.p. — menor que o desvio de
  calibração do pior braço — separa preferência real de desempate numérico. Há teste para isso.
- **A calibração isotônica é função em degraus**, então clientes de alta propensão caem no mesmo
  patamar: um dos casos tem cinco braços empatados em 71,11%. Documentado como artefato de
  modelagem em vez de apresentado como resultado.
- Um caso rende recomendação **por exclusão**: os dois líderes empatam entre si mas superam o braço
  médio em 1,07 p.p., então a decisão útil é sair do braço global, não escolher entre os líderes.

### Notas — a limitação principal, com número
- **A vantagem do celular sobre o telefone fixo é inflada 9,4x pelo confounding temporal.** Medida
  na base completa dá **+181,7%**; medida dentro da janela onde os dois canais rodaram lado a lado,
  **+19,3%**. Os braços de fixo saltam de ~5% para ~12,5% de conversão quando o calendário sai da
  comparação; os de celular não se movem, porque ele só existiu na janela tardia.
- **Consequência sobre o resultado principal:** o uplift de +18,2% pressupõe que o efeito de canal
  seja causal. Sob a leitura conservadora fica em torno de **+8% a +9%**. A decisão da política
  segue correta — `cellular|mid` é o melhor braço dado o log — mas a magnitude precisa da ressalva.
- **Restringir a base à coexistência foi testado e descartado:** lá o baseline já usa celular em 90%
  das ligações e nenhuma política adaptativa o supera (14,54% contra 14,50%). Ficaria causalmente
  limpo e inteiramente nulo.
- **O cenário de "braço novo" foi testado e deu negativo:** uma regra congelada antes de o celular
  existir perde só +2,8% depois que ele aparece. Este dataset não oferece o contraexemplo de "regra
  fixa quebra quando o mundo muda", e o README diz isso em vez de sugerir o contrário.
- **Trocar de dataset foi avaliado e descartado.** As quatro bases sugeridas pelo enunciado são
  variações da mesma campanha bancária portuguesa — mesmo confounding. Sair da lista exigiria
  refazer as Fases 1 a 4 e perder o domínio financeiro, com risco de cronograma desproporcional ao
  ganho. O enunciado pede documentar limitações, e é o que fazemos.

---

## [0.6.0] — 2026-08-24 — Fase 4: replay sobre o log real

Fecha o track C e, com ele, a estratégia de avaliação A+C completa. O ambiente calibrado deixa de
ser palavra contra palavra: existe agora uma segunda medida, por um caminho que não passa por
modelo nenhum.

### Adicionado
- `src/replay.py` — rejection sampling (Li, Chu, Langford & Wang, WSDM 2011). Percorre o log
  embaralhado e só conta o evento quando a política escolhe o braço que foi de fato jogado; a
  recompensa é o `y` observado. Eventos rejeitados são descartados inteiros, sem update, que é o
  que mantém o fluxo aceito consistente com um mundo onde a política teria decidido.
- Estimador **IPS auto-normalizado** (Hájek) com piso de propensão, mais taxa de aceitação e
  tamanho efetivo de amostra (Kish) — as três coisas que dizem quanto confiar em cada linha.
- `compare_tracks()` e `rank_agreement()` — os dois tracks lado a lado, com Spearman.
- `fit_arm_propensity()` em `src/environment.py`, agora compartilhado entre o diagnóstico de
  positividade da Fase 2 e a ponderação da Fase 4.
- `tests/test_replay.py` — 17 testes. **130 no total.**

### Resultado
| Política | Ambiente | Replay (IPS) | Rank A | Rank C |
|---|---:|---:|:-:|:-:|
| `FixedArm[cellular\|mid]` | 13,31% | 14,00% | 1 | 1 |
| `ThompsonSampling[1.13, 8.87]` | 13,01% | 12,37% | 2 | 4 |
| `EpsilonGreedy` | 12,88% | 13,50% | 3 | 2 |
| `ThompsonSampling[1, 1]` | 12,81% | 12,55% | 4 | 3 |
| `UCB1` | 12,70% | 12,07% | 5 | 5 |
| `LinTS` | 12,41% | 11,27% | 6 | 7 |
| `LoggingPolicy` | 11,01% | 11,48% | 7 | 6 |

**Spearman = 0,857.** Os dois métodos ordenam quase igual, apesar de um simular 20.000 decisões
contra um modelo e o outro peneirar 8.238 linhas de log real.

### Notas
- **A triangulação da `FixedArm` é o achado mais forte do projeto.** Replay cru 15,47% → replay com
  IPS 14,00% → ambiente calibrado 13,31%. O replay cru é otimista por seleção: quem recebeu
  `cellular|mid` não era recorte aleatório. A ponderação move a estimativa três quartos do caminho
  até o número do ambiente. Dois métodos sem premissa em comum chegando ao mesmo lugar.
- **A vantagem do prior informado não se replica.** 2º no ambiente, 4º no replay. O que reproduz nos
  dois tracks é a diferença de exploração (23,0% contra 38,0%) — o prior acelera a convergência, mas
  o efeito na conversão final está dentro do ruído.
- **A `LinTS` cai para último no replay, abaixo do baseline.** Segunda evidência independente, sem
  modelo no caminho, de que a política contextual não entrega nestes dados. Os intervalos se
  sobrepõem, então "pior que o baseline" não é afirmável — mas "melhor" também deixa de ser.
- Custo do método: o replay descarta de 61% a 83% do log conforme a política. A `LinTS` fica com 787
  eventos efetivos de 8.238, e por isso tem os intervalos mais largos da tabela.

### Verificado
- `make test` — 130 testes, todos passando. `make lint` sem achados.
- `make train` roda os dois tracks ponta a ponta em ~3 min.

---

## [0.5.1] — 2026-08-24 — Nome do repositório

### Decidido
- **`datathon-8mlet-grupo-30`** — turma 8MLET, grupo 30. O `7mlet` do enunciado era exemplo, não
  exigência. Fecha a decisão em aberto #3 do briefing.

### Alterado
- `git remote` local atualizado para a URL nova. O GitHub redirecionava o nome antigo, então tudo
  funcionava, mas todo push imprimia `This repository moved`.

---

## [0.5.0] — 2026-08-24 — Fases 2 e 3: ambiente calibrado, políticas e o experimento

Cobre as **Etapas 3 e 7** do enunciado. O requisito central — algoritmo adaptativo superando o
baseline — está cumprido e medido: **+18,2%** da melhor política sobre a política de log.

### Adicionado
- `src/arms.py` — `ArmSpace` com ordenação estável e mapeamento rótulo ↔ índice, mais
  `arm_distribution()` (a mistura histórica que alimenta o baseline) e `best_historical_arm()`.
  A ordem é fixada na construção e nunca re-derivada: índices entram nos artefatos serializados,
  e uma reordenação entre execuções rotularia todo resultado errado em silêncio.
- `src/environment.py` — ambiente calibrado do track A. `P(y | contexto, braço)` por
  `HistGradientBoostingClassifier` + `CalibratedClassifierCV` isotônica, com **três portões que
  levantam exceção em vez de seguir**: calibração global e por braço, sanity check contra
  regressão logística, e diagnóstico de sobreposição por propensão. Mais `contextual_ceiling()`,
  que mede o teto do ganho contextual antes de qualquer política existir.
- `src/policies.py` — `Policy` como ABC com a interface do `CLAUDE.md`, e seis implementações:
  `LoggingPolicy`, `FixedArm`, `EpsilonGreedy`, `UCB1`, `ThompsonSampling` e `LinTS`.
  RNG injetado em todas; nenhuma toca o estado global do numpy, e há teste que verifica isso.
- `src/evaluation.py` — `Environment` como `typing.Protocol` estrutural, então o ambiente calibrado
  e os testbeds sintéticos são intercambiáveis sem que nenhum módulo importe o outro. Runner
  multi-seed, métricas com intervalo-t entre seeds e integração MLflow com runs aninhados.
- `train.py` — entrypoint do experimento. Aceita `--rounds`, `--seeds` e `--no-mlflow`.
- `tests/bandit_testbed.py`, `tests/test_arms.py`, `tests/test_environment.py`,
  `tests/test_policies.py`, `tests/test_evaluation.py` — 72 testes novos, 113 no total.

### Resultado
20.000 rodadas × 10 seeds, baseline = política de log (11,01% no ambiente):

| Política | CVR | Uplift | Exploração |
|---|---:|---:|---:|
| `FixedArm[cellular\|mid]` | 13,31% | +20,85% | 0,0% |
| `ThompsonSampling[1.13, 8.87]` | 13,01% | +18,16% | 23,0% |
| `EpsilonGreedy[0.05]` | 12,88% | +16,95% | 16,1% |
| `ThompsonSampling[1, 1]` | 12,81% | +16,36% | 38,0% |
| `UCB1[0.25]` | 12,70% | +15,38% | 41,7% |
| `LinTS[0.05]` | 12,41% | +12,69% | 59,5% |

Todas as adaptativas superam o baseline sem sobreposição de intervalos.

### Decidido
- **Hiperparâmetros por sweep, não por default.** `ε=0.05`, `c=0.25`, `v=0.05`. O `c=1.0` de
  livro-texto do UCB1 é o **pior** valor testado (11,58%): o bônus `√(2·ln t / n)` pressupõe
  recompensa em toda a faixa [0,1], mas aqui as médias vivem entre 5% e 15%, então o bônus domina
  o sinal e a política explora sem parar. O sweep completo está no README, e a otimização no mesmo
  ambiente em que se reporta virou limitação declarada.
- **Prior informado `Beta(1.13, 8.87)` supera o uniforme** — 13,01% contra 12,81%, com 23,0% de
  exploração contra 38,0%. Ambos permanecem no experimento: o PDF pede análise da escolha de prior.
- **A `LinTS` não se paga, e isso é resultado, não falha.** O teto contextual medido é de apenas
  +4,44%, e capturá-lo exigiria estimar 41 features × 6 braços = 246 parâmetros sobre recompensa
  binária de ~11%. O custo de exploração excede o prêmio. A implementação está validada por um
  teste que a coloca contra a Thompson comum num ambiente onde o braço ótimo depende do cliente —
  lá ela vence. **Nestes dados o efeito do braço é quase todo não-contextual.**
- **MLflow em SQLite, não em file store.** A partir do MLflow 3 o backend de arquivos está em modo
  de manutenção e levanta exceção. `MLFLOW_TRACKING_URI` aponta para `sqlite:///mlflow.db` e o
  alvo `make mlflow` passou a informar `--backend-store-uri`.

### Notas
- O ambiente carrega encoder e modelo além da matriz pré-calculada, e expõe `predict()` para linhas
  novas. Sem isso ele serviria só para simulação, e a API da Fase 6 não teria como pontuar um
  cliente que chega na requisição.
- `LinTS` usa Sherman-Morrison para a inversa e cacheia o fator de Cholesky por braço, recomputando
  só o braço que mudou. Sem isso o experimento não fecharia em tempo aceitável; com isso são 33 s
  para 20.000 rodadas × 10 seeds.
- A sobreposição revelou que `telephone|early` e `telephone|late` ficam abaixo de 1% de propensão
  para 3,76% e 4,37% dos clientes — consequência direta do confounding temporal achado na Fase 1.
  Declarado como extrapolação no README.

### Verificado
- `make test` — 113 testes, todos passando. `make lint` sem achados.
- `make train` ponta a ponta em ~3 min, gerando artefatos, figuras e 77 runs no MLflow.
- `duration` ausente de todo o código das Fases 2 e 3 — a única ocorrência é o teste que verifica
  a ausência dela. As 12 colunas de contexto do ambiente foram conferidas explicitamente.

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
