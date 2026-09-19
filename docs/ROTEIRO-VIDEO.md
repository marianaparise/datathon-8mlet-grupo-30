# Roteiro do vídeo pitch — Etapa 8

Texto falado, palavra por palavra, cronometrado para **4:50** dentro do limite de 5 minutos.
As requisições e as saídas esperadas estão em [`DEMO.md`](DEMO.md) — este arquivo é a narração.

**Regra de ouro:** os 90 segundos da demo valem mais que todo o resto. Se algo estourar o tempo,
corte narração, nunca a API rodando.

---

## Divisão entre os quatro

O grupo tem quatro integrantes e o vídeo tem cinco blocos. Sugestão:

| Bloco | Tempo | Quem |
|---|---|---|
| 1 — Problema e base | 0:00–1:10 | **Doglas** |
| 2 — Modelo e políticas | 1:10–1:55 | **Silvio** |
| 3 — Resultados e MLflow | 1:55–3:05 | **Ricardo** |
| 4 — **A API rodando** | 3:05–4:35 | **Mariana** |
| 5 — Fechamento | 4:35–4:50 | **Silvio** |

Gravando sozinho, ignore a coluna "quem" e leia direto. Gravando em quatro, **grave cada bloco
separado e edite depois** — tentar revezar ao vivo custa tempo em transição, e o limite é rígido.

---

## Bloco 1 — Problema e base · 0:00–1:10

**Tela:** capa simples com o título e os quatro nomes (0:00–0:15), depois o README no navegador.

> Somos o grupo 30 da turma 8MLET. O problema é o seguinte: um banco precisa decidir, para cada
> cliente elegível, **por qual canal e em que dia da semana abordar**.
>
> Hoje isso é regra fixa ou teste A/B longo. A regra fixa congela a decisão e não reage a mudança
> de contexto. O A/B manda metade do tráfego para o braço pior durante semanas inteiras, e só
> decide no fim. Os dois desperdiçam. A alternativa é uma política adaptativa — um **multi-armed
> bandit** — que realoca tráfego conforme a evidência chega.
>
> A base é a **bank-marketing do Kaggle**: 41 mil campanhas de telemarketing de um banco português,
> com conversão de 11,3%. E aqui está a decisão que define o projeto: nós **não inventamos** braço
> nenhum. Três colunas do dataset não descrevem o cliente, descrevem o que a campanha escolheu
> fazer — canal e dia do contato. É delas que saem os seis braços: celular ou telefone fixo, vezes
> três janelas de semana.
>
> Como todo braço aparece no log histórico, a probabilidade de conversão de cada um é estimável a
> partir de linha observada. **Nenhuma recompensa é sintética.** E a coluna `duration`, que o
> enunciado proíbe, está fora de tudo — ela sozinha dá AUC de 0,82, porque ligação que termina em
> venda dura mais. No momento de decidir para quem ligar, esse número ainda não existe.

**Mostrar na tela, nesta ordem:** o link do Kaggle no README → a tabela de conversão por braço
(15,47% em `cellular|mid` contra 4,67% em `telephone|early`) → o gráfico `duration_vazamento.png`.

---

## Bloco 2 — Modelo e políticas · 1:10–1:55

**Tela:** o diagrama das duas camadas de avaliação no README, depois a tabela de políticas.

> Avaliamos em duas camadas independentes. A primeira é um **ambiente calibrado**: um
> `HistGradientBoosting` com calibração isotônica estima a probabilidade de conversão de cada braço
> para cada cliente, e isso permite rodar 20 mil decisões e medir arrependimento contra o oráculo.
> Ele só entra em uso depois de passar por três portões — calibração global **e por braço**, sanity
> check contra regressão logística, e teste de sobreposição.
>
> A segunda camada é **replay** por rejection sampling sobre o log real: o evento só conta quando a
> política escolhe exatamente o braço que foi de fato jogado, e a recompensa é o `y` observado,
> nunca estimado. É a contraprova sem modelo no meio.
>
> Implementamos sete políticas com a mesma interface: epsilon-greedy, UCB1, Thompson Sampling com
> prior informado e uniforme, gradient bandit e um bandit contextual. O baseline é a **política de
> log** — a mistura de braços que a operação de fato executou.

---

## Bloco 3 — Resultados e MLflow · 1:55–3:05

**Tela:** tabela de resultados → `regret_acumulado.png` → MLflow em `localhost:5000`.

> Vinte mil rodadas, dez sementes, intervalo de confiança de 95%. **Todas as políticas adaptativas
> superam o baseline**, sem sobreposição de intervalos. A melhor entrega **13% de conversão contra
> 11% do baseline — um ganho relativo de 18%**.
>
> Três leituras honestas, porque elas valem mais que o número bonito.
>
> Primeira: o **prior informado vence o uniforme**. Codificar a taxa-base de 11% no Beta poupa
> exatamente as rodadas que o prior uniforme gasta descobrindo que nenhum braço converte a 50% — a
> exploração cai de 38% para 23%.
>
> Segunda: a **política contextual não se paga**. E nós sabíamos disso antes de implementá-la,
> porque medimos o teto: comparando o melhor braço fixo com um oráculo que escolhe cliente a
> cliente, o ganho máximo da personalização nesta base é de **4,4%**. Capturar isso exige estimar
> 246 parâmetros a partir de recompensa binária que sai uma vez a cada nove. O custo de exploração
> é maior que o prêmio.
>
> Terceira: os **dois tracks concordam** — Spearman de 0,857 entre o ranking do ambiente e o do
> replay, que não compartilham premissa nenhuma.
>
> E tudo isso está no **MLflow**: 88 runs, um pai por política com a média entre sementes, e um
> filho por semente. Params, métricas e intervalos — a média fica citável e cada semente continua
> auditável.

**No MLflow, mostrar:** a lista de runs → abrir um run pai → a aba de métricas com `cvr_final` e
`uplift_vs_baseline`.

---

## Bloco 4 — A API rodando · 3:05–4:35

**Tela:** terminal, fonte grande. Comandos já no histórico — só seta pra cima e Enter.
**Esta é a parte que a banca precisa ver funcionando.** Não narre por cima do comando rodando:
dispare, espere a resposta aparecer, e aí fale.

### 4.1 — O serviço está de pé (3:05–3:20)

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

> Etapa 5: FastAPI, em container. Seis braços carregados e a versão do artefato — é isso que amarra
> a resposta a um modelo específico.

### 4.2 — Cliente A: estudante de 18 anos (3:20–3:50)

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":18,"job":"student","marital":"single","education":"high.school",
       "default":"no","housing":"no","loan":"no","campaign":1,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}'
```

> A API devolve o braço recomendado e **o ranking inteiro dos seis**. Estudante converte muito acima
> da taxa-base de 11%, celular domina, e o pior braço vale um quarto do melhor.

### 4.3 — Cliente B: o ranking vira (3:50–4:20)

> Agora o mesmo endpoint, cliente diferente: técnico de 49 anos que **já levou seis ligações**.

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":49,"job":"technician","marital":"married","education":"professional.course",
       "default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}'
```

> O ranking virou. **`telephone|early` assume a liderança**, e `cellular|mid` — que é o melhor braço
> na média da base inteira — cai para quarto lugar. Para quem já cansou de seis ligações no celular,
> o canal alternativo deixa de ser inferior. **Uma regra fixa mandaria celular para os dois.**

### 4.4 — O contrato recusa lixo (4:20–4:35)

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"age":49,"job":"astronauta","marital":"married","education":"professional.course",
       "default":"no","housing":"yes","loan":"no","campaign":6,"pdays":999,
       "previous":0,"poutcome":"nonexistent"}' -w '\nHTTP %{http_code}\n'
```

> Categoria desconhecida é recusada na porta, com 422, nomeando os valores válidos. Sem isso o
> encoder receberia o valor, devolveria um vetor de zeros, e a API responderia algo plausível a
> partir de lixo — que é a pior falha possível, porque é silenciosa.

---

## Bloco 5 — Fechamento · 4:35–4:50

**Tela:** volta ao README, na seção de limitações.

> Fechando com a ressalva, não com o número: os 18% de ganho pressupõem que o efeito de canal seja
> causal, e ele não é inteiramente — a campanha trocou de canal no meio, então parte da vantagem do
> celular é calendário. Medimos: na janela em que os dois canais coexistem, o ganho fica mais perto
> de **8 a 9%**. Está tudo no README, com os números.
>
> Um sistema que sabe onde erra vale mais que um que só mostra o resultado bom. Obrigado.

---

## Antes de gravar

```bash
make train && make docker-up
```

- [ ] `curl /health` responde `"model_loaded": true` — se vier `degraded`, falta `make train`
- [ ] Os quatro comandos do Bloco 4 já rodados uma vez, para estarem no histórico do shell
- [ ] MLflow aberto em `localhost:5000` numa aba, README em outra
- [ ] Fonte do terminal em pelo menos 18pt — vídeo comprimido come texto pequeno
- [ ] Notificações do sistema desligadas
- [ ] Um ensaio cronometrado completo. **O limite de 5 minutos é rígido.**

## Se travar na gravação

| Sintoma | O que fazer ao vivo |
|---|---|
| A API não responde | Não debugue na câmera. Corte, `make docker-up`, regrave o bloco |
| Números diferentes do roteiro | Normal se houve retreino — leia o que está na tela, não o que está aqui |
| Passou de 5 minutos | Corte o Bloco 4.4 (o 422) e encurte o Bloco 3 à primeira leitura |
