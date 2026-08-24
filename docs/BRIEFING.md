# Briefing — TC5 / Datathon POSTECH MLET

> Documento de contexto para quem está entrando no projeto agora. Explica o desafio, o que já foi
> decidido e **por quê**, o que ainda precisa ser decidido, e o que vale estudar.
>
> Não é entregável — a documentação que a banca lê fica no [`README.md`](../README.md).

**Índice**
1. [O desafio em um minuto](#1-o-desafio-em-um-minuto)
2. [Onde estamos](#2-onde-estamos)
3. [O problema central do projeto](#3-o-problema-central-do-projeto)
4. [Decisões fechadas](#4-decisões-fechadas)
5. [Decisões em aberto](#5-decisões-em-aberto--precisamos-bater-martelo)
6. [Riscos conhecidos](#6-riscos-conhecidos)
7. [Base teórica](#7-base-teórica--o-que-vale-estudar)
8. [Como rodar o que existe](#8-como-rodar-o-que-existe)
9. [Mapa dos arquivos](#9-mapa-dos-arquivos)

---

## 1. O desafio em um minuto

Enunciado oficial: `POSTECH - MLET - DATATHON (1).pdf`, na raiz do repo.

Uma instituição financeira digital precisa decidir, em cada canal, **qual oferta, mensagem ou próximo
passo apresentar a cada cliente elegível**. Regra fixa congela a decisão; teste A/B longo desperdiça
metade do tráfego por semanas. A proposta é uma abordagem **adaptativa (multi-armed bandit)**, que
realoca tráfego em tempo real conforme aprende.

**Nove etapas obrigatórias (0 a 8):**

| Etapa | Entrega |
|---|---|
| 0 | Repo público, README, requirements |
| 1 | Link da base Kaggle + notebook de EDA |
| 2 | Preparação da base |
| 3 | Baseline + algoritmo adaptativo **superando o baseline** |
| 4 | Métricas + **Golden Set de 5 clientes** |
| 5 | Serviço demonstrável (script, notebook ou API) |
| 6 | 1–2 parágrafos sobre arquitetura em nuvem |
| 7 | **MLflow** registrando params e métricas |
| 8 | **Vídeo pitch de até 5 minutos** |

**Nota:** 30% clareza do problema e impacto de negócio · 70% validação técnica (código organizado,
modelo superando baseline, MLflow, demo funcionando).

**Restrições que o PDF impõe:**
- `duration` é proibida — vazamento temporal, citada nominalmente
- Sem dados reais de cliente, identificadores, patrimônio, renda, gênero ou raça
- Toda documentação **consolidada no README**, sem arquivos soltos de governança
- README precisa cobrir base legal, finalidade, minimização, retenção e humano no loop

---

## 2. Onde estamos

**Fases 0 a 3 de 8 concluídas.** Plano completo em [`PLANO.md`](PLANO.md).
Etapas 0, 1, 2, 3 e 7 do enunciado entregues.

✅ **Fase 0** — estrutura, `requirements.txt` validado com instalação real, `Makefile`, `.gitignore`,
regras em [`CLAUDE.md`](../CLAUDE.md). `make data` roda sem credencial, com checksum.

✅ **Fase 1** — `src/data.py`, `src/eda.py`, `notebooks/01_eda.ipynb`. Espaço de braços fixado em
`contact × week_window` (6 braços).

✅ **Fase 2** — `src/arms.py` e `src/environment.py`. Ambiente calibrado com três portões:
Brier **0,0860**, desvio máximo por braço de **0,96 p.p.**, AUC 0,7413 contra 0,7274 da logística,
e diagnóstico de sobreposição.

✅ **Fase 3** — `src/policies.py`, `src/evaluation.py`, `train.py`. Seis políticas, 20.000 rodadas
× 10 seeds, 77 runs no MLflow. **113 testes no total.**

❌ `api/` continua vazio. `src/replay.py` (Fase 4) e o Golden Set (Fase 5) ainda não existem.

### O requisito da Etapa 3 está cumprido

Todas as políticas adaptativas superam o baseline, sem sobreposição de intervalos. A melhor
(`ThompsonSampling` com prior informado) entrega **+18,2%** sobre a política de log.

### Dois achados que mudam a conversa

**A heterogeneidade fraca virou número.** O ambiente calibrado mede o teto do ganho contextual em
**+4,44%** sobre o melhor braço fixo, com 47% dos clientes tendo outro braço ótimo — as trocas são
quase todas entre braços de probabilidade praticamente igual. A `LinTS` não captura esse teto: fica
em 12,41% contra 13,01% da Thompson comum, ainda com 59,5% de exploração no fim do horizonte.
Estimar 246 parâmetros sobre recompensa binária de 11% custa mais do que rende.

Isso **não é bug**: existe um teste que roda a mesma `LinTS` num ambiente onde o braço ótimo depende
do cliente, e lá ela vence a Thompson. A máquina funciona; **estes dados é que são quase todos
não-contextuais**.

**O `c=1.0` de livro-texto do UCB1 é o pior valor possível aqui.** O bônus `√(2·ln t / n)` pressupõe
recompensa em toda a faixa [0,1]; com médias entre 5% e 15% ele domina o sinal e a política nunca
se decide. Com `c=0.25` a conversão sobe de 11,58% para 12,74%.

---

## 3. O problema central do projeto

Esta seção é a mais importante do documento. É o raciocínio que sustenta o projeto inteiro.

### 3.1. A base não tem braços

Escolhemos a base `bank-marketing`. Ela registra campanhas de telemarketing de um banco português:
cada linha é um cliente, e o target `y` diz se ele assinou um depósito a prazo.

O problema: **existe uma única ação registrada** — "ligamos e oferecemos depósito a prazo". Cada linha
é `(cliente, converteu?)`.

Um bandit precisa escolher entre **múltiplas ações**. Sem múltiplas ações no log, não existe bandit.

### 3.2. Por que isso é difícil: bandit feedback

A raiz do problema cabe numa frase: **você só observa a recompensa da ação que tomou.**

O log diz "ligamos no celular da Maria numa terça e ela não converteu". Ele **nunca** diz o que teria
acontecido se tivéssemos ligado no fixo numa sexta. Esse resultado não observado é o **contrafactual**,
e não está no dataset — não por falha de coleta, mas por impossibilidade lógica.

Isso se chama *partial-label problem* ou *bandit feedback*, e é o que separa este projeto de uma
classificação supervisionada comum. Em classificação você tem o rótulo certo de toda linha. Aqui você
tem o rótulo de **uma** ação por linha.

### 3.3. As três saídas possíveis

Avaliamos três formas de construir um problema multi-ação legítimo a partir desse log:

**Opção A — braços vindos de colunas reais de ação.**
Nem toda coluna descreve o cliente. Algumas registram **decisões que a campanha tomou**:

| Contexto (quem é o cliente) | Ação (o que o banco escolheu) |
|---|---|
| `age`, `job`, `marital`, `education`, `housing`, `loan`, `campaign`, `pdays`, `previous`, `poutcome`, indicadores socioeconômicos | `contact` (celular/fixo), `month`, `day_of_week` |

Os braços saem das colunas de ação. Consequência decisiva: **todo braço aparece no log**, então
`P(conversão | contexto, braço)` é estimável de dado observado para qualquer braço. Isso vira um
ambiente de simulação **calibrado nos dados**, sem inventar taxa de conversão.

**Opção B — braços como ofertas de produto.**
Definir 4–5 produtos (cartão, crédito, seguro...) e fazer o bandit escolher entre eles. Problema: só
existe dado real de **um** produto. Os outros exigiriam escrever à mão a função de resposta. Isso é
recompensa sintética, e a Etapa 2 do PDF diz explicitamente para usar a base direto *"sem precisar
gerar dados sintéticos complexos"*.

**Opção C — replay sobre o log real.**
Não simular nada. Percorrer o log; quando o braço escolhido pelo bandit coincide com o que foi de fato
registrado, aceita o evento e usa o `y` real. Senão, descarta a linha.

### 3.4. O que escolhemos: A + C

**A como track principal, C como validação.**

```
log real → braços de ação (contact × janela de contato)
              ├── A: P(y|x,a) calibrado → ambiente → regret contra oráculo   [principal]
              └── C: rejection sampling sobre o log → CVR observada          [validação]
```

### 3.5. Por que essa combinação, e não uma só

Isso não é preferência de projeto — **tem nome na literatura**. O campo se chama
**Off-Policy Evaluation (OPE)**: estimar como uma política nova teria performado usando apenas o log
de uma política antiga, sem nunca colocá-la no ar. Existem duas famílias de estimadores, e as nossas
opções eram exatamente elas:

| Nossa opção | Nome na literatura | Fraqueza |
|---|---|---|
| **A** — ambiente calibrado | **Direct Method (DM)** — modela a recompensa | Baixa variância, **mas herda o viés do modelo** |
| **C** — replay | **Importance Sampling / IPS** | Pouco viés, **mas variância alta e descarta dados** |
| **A + C** | as duas famílias juntas — combiná-las formalmente é o **Doubly Robust** | — |

As fraquezas são complementares. Usar as duas e comparar é prática padrão. Se os rankings de políticas
concordarem entre os dois tracks, o resultado é sólido; se divergirem, isso é achado relevante e vai
para o README.

**Por que B foi descartada:** exigiria inventar a resposta de 4 produtos sem nenhuma observação,
contrariando a Etapa 2. E duplicaria as Etapas 2 a 7 sob uma **segunda formulação de negócio**,
diluindo os 30% de clareza. Sobrevive só como parágrafo de roadmap no README.

**Por que C sozinha foi descartada:** com 6 braços descarta ~85% dos eventos, gera curvas ruidosas e é
a alternativa com maior risco de o bandit **não** superar o baseline com clareza — requisito explícito
da Etapa 3.

---

## 4. Decisões fechadas

| Tema | Decisão | Motivo curto |
|---|---|---|
| Base | Kaggle `bank-marketing` (henriqueyamahata) = `bank-additional-full.csv` do UCI | Única das sugeridas com `day_of_week` **e** contexto socioeconômico |
| Formulação | Braços de colunas reais de ação (canal × janela) | Todo braço tem suporte no log → nada sintético |
| Avaliação | **A + C** | Direct Method + IPS têm fraquezas complementares |
| Serviço (Etapa 5) | FastAPI + Docker | Mais evidência de ML Engineering; boa demo em vídeo |
| Rastreio (Etapa 7) | MLflow local | Exigência do PDF |
| Modelagem | `HistGradientBoostingClassifier` + `CalibratedClassifierCV` | Cobre o que LightGBM cobriria, sem dependência extra no Docker |
| Ofertas sintéticas | Descartado | Ver 3.5 |

---

## 5. Decisões em aberto — precisamos bater martelo

**6 das 8 fechadas.** Restam a **#7** (divisão de trabalho das Fases 4–7 e o vídeo) e a **#8**
(o que fazer com a política contextual). As resolvidas ficam registradas com o racional — quem
chegar depois precisa saber por que, não só o quê.

### ~~1. Como obter a base~~ — ✅ **RESOLVIDA**
`make data` funciona sem credencial nenhuma. O script
[`scripts/download_data.sh`](../scripts/download_data.sh) tenta o Kaggle e cai no UCI (origem real do
dataset, CC BY 4.0) quando não há token. As duas vias são conferidas por SHA-256, então nunca se treina
em cima de um arquivo diferente sem perceber.

Não conflita com o enunciado: o PDF manda *preservar a referência ao Kaggle*, que é bibliográfica — o
README cita fonte, link e licença. De onde os bytes vêm é detalhe de infraestrutura.

**Se você quiser usar o Kaggle mesmo assim:** gere o token em *Kaggle → Settings → Create New Token*,
salve em `~/.kaggle/kaggle.json` e rode `chmod 600 ~/.kaggle/kaggle.json`. O script passa a preferi-lo
automaticamente.

### ~~2. Por onde começar~~ — ✅ **RESOLVIDA**
Seguimos a ordem do plano. A Fase 1 fez EDA e pipeline de preparação; `src/policies.py` e
`src/evaluation.py` vêm na Fase 3, depois de a Fase 2 fixar o ambiente calibrado.

### ~~3. Nome do repositório~~ — ✅ **RESOLVIDA**
**`datathon-8mlet-grupo-30`** — turma 8MLET, grupo 30. O `7mlet` do PDF era só exemplo (aparece
como "ex:"), e a turma confirmada é a 8.

```
https://github.com/marianaparise/datathon-8mlet-grupo-30
```

⚠️ **Doglas: rode isto no seu clone.** O GitHub redireciona o nome antigo, então tudo funciona, mas
o aviso `This repository moved` aparece a cada push até você atualizar:

```bash
git remote set-url origin https://github.com/marianaparise/datathon-8mlet-grupo-30.git
```

### ~~4. Qual baseline usar~~ — ✅ **RESOLVIDA**
**A armadilha era real e disparou.** O braço modal do log (`cellular|mid`, 38,8% do volume) é
*também* o de maior conversão (15,47%): modal e melhor histórico são o mesmo braço. Com "melhor
braço histórico" como baseline, qualquer bandit não-contextual convergiria para ele e o ganho seria
exatamente zero.

**Decisão: regra fixa = a política de log**, cuja conversão realizada é 11,27%. Concentrar em
`cellular|mid` rende 15,47% — uplift de +37%, e legítimo, porque a operação jogava uma mistura, não
o melhor braço. `BestHistoricalArm` fica como comparador secundário, e é contra ele que só a
política **contextual** pode ganhar.

✅ **A Fase 3 confirmou a decisão com número.** No ambiente calibrado, a política de log rende
11,01% e a melhor adaptativa 13,01% — **+18,2%**, sem sobreposição de intervalos. A `FixedArm` em
`cellular|mid` chega a 13,31%, acima de todas as adaptativas, o que era esperado num ambiente
estacionário com braço dominante: ela já começa sabendo o que as outras gastam exploração para
descobrir. Ela é o teto de um oráculo estacionário, não uma alternativa disponível no dia zero.

⚠️ **A ressalva sobre heterogeneidade se confirmou.** Ver decisão #8.

### ~~5. O que fazer com `euribor3m` e `nr.employed`~~ — ✅ **RESOLVIDA**
Pior do que se supunha: **todos os cinco indicadores macro são constantes dentro do período**. O R²
contra o índice de período é 1,0000 para `emp.var.rate`, `cons.price.idx`, `cons.conf.idx` e
`nr.employed`, e 0,9996 para `euribor3m`. Para `cons.price.idx`, cada valor identifica um único
período — saber o índice é saber a data.

**Decisão: manter na base, separados em `MACRO_COLUMNS`, fora do contexto padrão do ambiente.** Um
indicador constante no período não personaliza nada: move a taxa-base, não distingue cliente. A
Fase 2 usa `CLIENT_COLUMNS` por padrão e roda a versão com macro como ablação, então "quanto do
acerto vem do calendário" vira medida em vez de retórica.

Destemporalizar foi descartado: sem coluna de data, a variação só seria computável sobre a ordem de
linha — tão temporal quanto o nível, e menos interpretável.

### ~~6. Split treino/teste~~ — ✅ **RESOLVIDA**
O temporal é implementável (a ordem do arquivo é cronológica: 26 blocos de meses consecutivos em
41.188 linhas), mas produz **24,5 p.p. de deriva** — conversão de 6,37% no treino contra 30,83% no
teste. O estratificado fica em 0,014 p.p.

**Decisão: estratificado como principal, por alvo × braço**, seed 42. A estratificação inclui o
braço porque o ambiente da Fase 2 estima `P(y | contexto, braço)` e precisa dos seis braços
povoados nos dois folds — o mais magro fica com 596 eventos e 28 conversões no teste. O temporal
vira análise de sensibilidade na Fase 3.

### 7. Divisão de trabalho e vídeo
Quem toca o quê nas Fases 4–7, quem grava o vídeo da Etapa 8, e qual a data-limite interna.

Até aqui: Doglas fez a Fase 1, Mariana as Fases 2 e 3. **A Fase 2 estava declarada como próximo
passo do Doglas no CHANGELOG e acabou sendo feita pela Mariana** — vale alinhar para não repetir.

### 8. O que fazer com a política contextual — **NOVA, precisa de decisão**

A `LinTS` ficou em 12,41%, **abaixo** da Thompson não-contextual (13,01%), com 59,5% de exploração
ainda no fim das 20.000 rodadas. A causa está medida: o teto do ganho contextual é de apenas
**+4,44%** sobre o melhor braço fixo, e capturá-lo exige estimar 41 features × 6 braços = 246
parâmetros a partir de recompensa binária que sai 1 em cada 9 vezes.

A implementação está validada — há teste que a coloca contra a Thompson num ambiente onde o braço
ótimo depende do cliente, e lá ela vence. O problema é o dado, não o código.

**Opções:**
1. **Reportar como está** (recomendado). "Testamos, medimos o teto, a contextual não se paga nestes
   dados" é resultado maduro, e a evidência está toda no README. O enunciado não exige que o
   contextual vença — exige que o adaptativo supere o baseline, o que já acontece.
2. **Reduzir a dimensão do contexto.** Menos features = menos parâmetros = convergência mais rápida.
   Custa tempo e pode não mudar nada, já que o teto continua sendo 4,44%.
3. **Revisar o espaço de braços** para um em que a heterogeneidade seja maior. Reabre a Fase 1
   inteira; não recomendo a esta altura.

---

## 6. Riscos conhecidos

| Risco | Impacto | Mitigação |
|---|---|---|
| Um braço domina globalmente | Bandit empata com baseline — **falha a Etapa 3** | Baseline em regra fixa; verificar heterogeneidade braço × contexto já na Fase 2 |
| Células de braço com poucos eventos | Ambiente mal calibrado | Agregar `day_of_week` em janelas; piso de suporte por braço |
| `euribor3m` domina o modelo | Acerto por calendário, não por cliente | Decisão em aberto #5 |
| Ambiente calibrado soa fabricado | Perda nos 70% técnicos | É exatamente o que o track C responde, com evidência |
| Confounding temporal em `contact` | Parte da diferença entre braços é época, não canal | Documentar como limitação no README |

---

## 7. Base teórica — o que vale estudar

Ordenado por prioridade. **As três primeiras em ~10h já permitem defender a decisão A+C na banca.**

**Essencial**

1. **Sutton & Barto, *Reinforcement Learning: An Introduction* (2ª ed.), capítulo 2** —
   [PDF gratuito](http://incompleteideas.net/book/the-book-2nd.html). Exploração × explotação,
   ε-greedy, UCB. O capítulo mais didático que existe sobre bandits. *(~4h)*

2. **Russo et al., [*A Tutorial on Thompson Sampling*](https://arxiv.org/abs/1707.02038), seções 1–4** —
   melhor porta de entrada para TS, com intuição bayesiana e pseudocódigo. **É aqui que se entende os
   "priors documentados" que o enunciado exige.** *(~3h)*

3. **Li, Chu, Langford & Wang, [*Unbiased Offline Evaluation of Contextual-bandit-based News Article
   Recommendation Algorithms*](https://arxiv.org/abs/1003.5956) (WSDM 2011), seções 1–3** —
   **é literalmente a nossa opção C.** O abstract contrasta o método com *"simulator-based approaches"*,
   que é a nossa opção A. O paper apresenta os dois lados da nossa decisão. *(~3h)*

**Importante**

4. **Li, Chu, Langford & Schapire, [*A Contextual-Bandit Approach to Personalized News Article
   Recommendation*](https://arxiv.org/abs/1003.0146) (WWW 2010)** — o paper do **LinUCB**, nossa
   política contextual. Cenário quase idêntico ao nosso.

5. **Calibração de probabilidade** — Niculescu-Mizil & Caruana (2005) e a
   [doc do scikit-learn](https://scikit-learn.org/stable/modules/calibration.html). Crítico: o ambiente
   do track A **só é válido se as probabilidades forem calibradas**. AUC boa com probabilidade
   descalibrada gera um ambiente mentiroso.

6. **Dudík, Langford & Li, [*Doubly Robust Policy Evaluation and
   Learning*](https://icml.cc/2011/papers/554_icmlpaper.pdf) (ICML 2011)** — mostra formalmente por que
   combinar DM e IPS é melhor que qualquer um sozinho. **É a justificativa teórica do nosso A+C.**

**Contexto**

7. **Auer, Cesa-Bianchi & Fischer (2002)** — origem do UCB1.
8. **Chapelle & Li (2011), *An Empirical Evaluation of Thompson Sampling*** — o paper que fez TS virar
   prática de indústria.
9. **Propensity score / confounding** — Rosenbaum & Rubin (1983). Fundamenta a ressalva de que a
   política de log do banco não era aleatória, e portanto a garantia de não-viés do replay não vale
   integralmente.
10. **Lattimore & Szepesvári, [*Bandit Algorithms*](https://tor-lattimore.com/downloads/book/book.pdf)
    (Cambridge, 2020)** — gratuito, rigoroso e pesado. Use como dicionário, não leia linear.

---

## 8. Como rodar o que existe

```bash
make setup   # cria .venv e instala dependências
make data    # baixa a base e confere o SHA-256 (não precisa de credencial)
make help    # lista todos os alvos
```

`make train` e `make api` **ainda não funcionam** — dependem de código que não existe.
Ver [decisão em aberto #2](#2-por-onde-começar).

Stack: Python 3.12, scikit-learn 1.9, pandas 2.3, MLflow 3.15, FastAPI 0.141. Versões pinadas em
[`requirements.txt`](../requirements.txt), resolvidas a partir de uma instalação real.

---

## 9. Mapa dos arquivos

| Arquivo | O que é |
|---|---|
| [`README.md`](../README.md) | **O entregável de texto.** É o que a banca lê |
| [`CLAUDE.md`](../CLAUDE.md) | Regras do projeto: decisões fechadas, restrições, convenções de código |
| [`docs/PLANO.md`](PLANO.md) | Plano de implementação em 8 fases, com checklists e riscos |
| [`docs/BRIEFING.md`](BRIEFING.md) | Este documento |
| [`CHANGELOG.md`](../CHANGELOG.md) | Histórico de modificações, com o racional de cada decisão |
| `POSTECH - MLET - DATATHON (1).pdf` | Enunciado oficial |

`CLAUDE.md`, `PLANO.md`, `BRIEFING.md` e `CHANGELOG.md` são ferramentas de desenvolvimento, **não
entregáveis** — o PDF pede documentação consolidada no README.
