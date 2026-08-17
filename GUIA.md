# Guia completo — do jogo ao robô treinado

Este arquivo é o manual de uso do projeto: como jogar, como calibrar, como
desenhar pistas, como treinar a sua rede neural, como competir contra ela, e
onde mexer quando quiser mudar alguma coisa. O [README.md](README.md) explica
as decisões de design; este arquivo é o passo a passo prático.

## Índice

1. [Instalação](#1-instalação)
1.1. [Usando o VSCode](#11-usando-o-vscode)
2. [Jogar](#2-jogar)
3. [Calibrar o robô](#3-calibrar-o-robô)
4. [Desenhar uma pista](#4-desenhar-uma-pista)
5. [Como o robô funciona](#5-como-o-robô-funciona)
6. [Treinar a rede neural](#6-treinar-a-rede-neural)
7. [Salvar e carregar a rede](#7-salvar-e-carregar-a-rede)
8. [Competir contra a IA](#8-competir-contra-a-ia)
9. [O que mudar e onde](#9-o-que-mudar-e-onde)
10. [Testes](#10-testes)
11. [Levando para o Arduino](#11-levando-para-o-arduino)
12. [Perguntas frequentes](#12-perguntas-frequentes)

---

## 1. Instalação

O projeto já vem com um ambiente virtual pronto em `.venv`. Todo comando deste
guia roda a partir da pasta do projeto:

```bash
cd D:\Claude\robo-desviador
```

E usa o Python de dentro do `.venv`, nunca o `python` do sistema:

```bash
.venv\Scripts\python.exe -m robo.game
```

Se preferir não repetir o caminho toda hora, ative o ambiente uma vez por
terminal:

```bash
.venv\Scripts\Activate.ps1
```

Depois disso, `python -m robo.game` funciona direto nesse terminal.

Se o `.venv` não existir (pasta nova, clone novo), recrie com:

```bash
py -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1.1. Usando o VSCode

O projeto já vem com três arquivos em `.vscode/` prontos: `launch.json` (o quê
rodar com `F5`), `settings.json` (interpretador fixo) e `tasks.json` (tarefas
de terminal, como rodar todos os testes de uma vez).

**Abra a pasta certa.** No VSCode, `Arquivo → Abrir Pasta` e escolha
`D:\Claude\robo-desviador` — a pasta do projeto em si, não `D:\Claude`. Se
abrir a pasta errada, `${workspaceFolder}` aponta para o lugar errado e nada
dos passos abaixo funciona.

**Selecione o interpretador**, uma vez. `Ctrl+Shift+P` → digite *Python: Select
Interpreter* → escolha o que aparece como `.venv\Scripts\python.exe` (às
vezes rotulado como `('venv': venv)`). Isso já está fixado em
`.vscode/settings.json`, então normalmente o VSCode acha sozinho — faça esse
passo manual só se ele pedir ou se o autocomplete não estiver funcionando.

#### Rodar e depurar com F5

Aperte `F5` (ou vá no ícone de inseto na barra lateral, *Run and Debug*). No
topo aparece um menu suspenso com todas as opções já configuradas — as
principais:

| escolha no menu | o que faz |
|---|---|
| **Jogar** | abre o jogo na pista `zigue-zague` |
| **Jogar (escolher pista)** | pergunta qual pista antes de abrir |
| **Editor de pista** | abre o editor em branco |
| **Editor (abrir pista salva em tracks/)** | pede o caminho de um `.json` seu |
| **Rede de exemplo (com janela)** | roda `brains/exemplo.py --ver`, para ver o contrato funcionando |
| **Corrida (contra regra fixa)** | corrida sem `--rede`, contra o adversário embutido |
| **Corrida (escolher rede treinada)** | pede o caminho do `.npz` |
| **Treinar (escolher pista no menu)** | pergunta a pista e treina, sem tela |
| **Treinar (assistir a população inteira)** | vê todos os robôs treinando ao vivo |
| **Treinar na GPU (com os seus valores)** | liga a placa e mais nada |
| **Treinar na GPU, assistindo a população inteira** | os dois juntos |
| **Treinar (continuar de um checkpoint)** | retoma de onde parou |
| **Testes: ...** | um por arquivo de teste (nove ao todo) |
| **Arquivo atual** | roda com `F5` o próprio arquivo `.py` que está aberto no editor — útil quando você já tem `brains/minha_rede.py` e quer rodar exatamente ele |

**Nenhuma entrada de treino fixa população, camadas ou gerações** — todas leem a
classe `Hiperparametros` de [brains/treino_ga.py](brains/treino_ga.py), então
editar lá muda o que roda no `F5`. As únicas coisas que essas entradas passam
são comportamento (assistir, GPU, pasta, continuar), nunca hiperparâmetro.

Troque de opção no menu suspenso ao lado do botão verde de play (ou
`Ctrl+Shift+D` para abrir o painel, depois a seta para baixo do play).

**Por que isso importa mais do que rodar pelo terminal**: com `F5`, um
breakpoint (clique à esquerda do número da linha, vira uma bolinha vermelha)
pausa a execução ali, e você inspeciona variáveis no painel *Variables* à
esquerda, ou digita expressões no *Debug Console* embaixo. Alguns lugares onde
isso ajuda de verdade neste projeto:

- Em `robo/sensors.py`, dentro de `_disparar`, para ver `t` e `cos_inc` e
  entender por que um sensor não está vendo uma parede inclinada.
- Em `brains/exemplo.py`, dentro de `act_batch`, para conferir a forma real de
  `obs` e da saída antes de escrever a sua própria rede por cima.
- No seu módulo de rede, dentro do laço de treino, para parar numa geração
  específica e olhar a telemetria (`dados["progress"]`, `dados["collided"]`
  etc.) antes de confiar nela.

#### Rodar sem depurar

Se não precisa de breakpoint, o botão de play (▷) ao lado do menu suspenso no
painel *Run and Debug* roda a mesma configuração sem parar em nada — mais
rápido para só ver o resultado.

#### Tarefas de terminal (`tasks.json`)

Para coisas que você quer rodar rápido sem abrir o painel de depuração:
`Ctrl+Shift+P` → *Tasks: Run Task* → escolha uma:

- **Rodar todos os testes** — roda os nove arquivos de teste em sequência e
  resume no fim quais falharam, se algum falhar. Também roda com
  `Ctrl+Shift+B` (é a tarefa de build padrão do projeto).
- **Testes: núcleo** — só o arquivo de física/sensores, o que muda com mais
  frequência.
- **Jogar** / **Editor de pista** — abrem num painel de terminal dedicado, sem
  o overhead do depurador.

#### O terminal integrado

`` Ctrl+` `` abre um terminal dentro do VSCode já posicionado na pasta do
projeto. Com o interpretador selecionado, `python` dentro desse terminal já
aponta para o do `.venv` — não precisa escrever `.venv\Scripts\python.exe` toda
vez. Se não estiver apontando certo, feche o terminal (lixeira no canto) e
abra outro: ele é recriado com o ambiente atualizado.

#### Se algo não funcionar

- **"module not found" ou erro de import**: quase sempre é a pasta errada
  aberta no VSCode, ou o interpretador errado selecionado. Confira os dois.
- **`F5` não mostra as opções do projeto**: o VSCode não achou
  `.vscode/launch.json` — sinal de que a pasta aberta não é
  `robo-desviador`.
- **A extensão Python não tem "debugpy" como tipo**: extensão desatualizada.
  Abra `.vscode/launch.json`, troque `"type": "debugpy"` por `"type":
  "python"` em todas as entradas (ou atualize a extensão Python pela aba de
  extensões).

---

## 2. Jogar

```bash
.venv\Scripts\python.exe -m robo.game
```

```bash
.venv\Scripts\python.exe -m robo.game --pista diagonais
```

| tecla | ação |
|---|---|
| setas ou WASD | dirigir (acelerar, ré, esquerda, direita) |
| `R` | reiniciar |
| `C` | alternar câmera (seguir o robô / ver a pista inteira) |
| `P` | abrir o painel de calibragem (seção 3) |
| `Esc` | sair |

Pistas embutidas: `curva-u`, `zigue-zague`, `caracol`, `diagonais`. Também
aceita o caminho de uma pista sua: `--pista tracks/minha.json`.

Por padrão, bater não encerra a tentativa — você dá ré e continua, o que ajuda
a sentir a física. Para bater derrubar a tentativa (como no treino):

```bash
.venv\Scripts\python.exe -m robo.game --fim-ao-bater
```

O HUD mostra, por sensor: distância, idade da leitura (o tempo desde a última
varredura daquele sensor — veja a seção 5) e uma barra. Embaixo, velocidade das
rodas, velocidade linear/angular e o percurso andado.

---

## 3. Calibrar o robô

Aperte `P` dentro do jogo. Abre um painel de 15 barras, divididas em três
grupos:

| grupo | o que ajusta |
|---|---|
| **motores** | velocidade máxima, zona morta do PWM, tempo para acelerar, assimetria entre motores, força da curva |
| **chassi** | entre-eixos, raio do corpo |
| **sensores** | quantidade, leque, cone, sub-raios, alcance, tempo de leitura, limite de eco, ruído |

Você continua dirigindo com as setas enquanto o painel está aberto — arraste
uma barra, sinta o efeito, ajuste de novo. Atalhos com o painel aberto:

- **arrastar** uma barra muda o valor
- `S` grava tudo em `config.json`
- `Z` restaura os valores de fábrica (não grava sozinho — aperte `S` depois se quiser manter)
- `P` fecha o painel

**Por que gravar importa**: o jogo e o treino rodam em processos separados.
Ajustar as barras sem apertar `S` só vale para a sessão atual do jogo — o
treino nunca vai ver a mudança. Todo script deste projeto que precisa da sua
calibragem carrega assim:

```python
from robo.config import SimConfig
cfg = SimConfig.carregar_ou_padrao()   # lê config.json; sem ele, usa o de fábrica
```

### O que medir primeiro

Os dois números que mais separam a simulação do robô real:

1. **Velocidade máxima** — cronometre o seu robô andando 1 metro em linha reta
   com o PWM no máximo, e calcule m/s.
2. **Zona morta do PWM** — vá subindo o PWM aos poucos a partir de zero até o
   robô sair do lugar. O valor onde ele começa a mexer é a zona morta (em
   fração de 0 a 1, não em unidades de 0-255 do Arduino — divida por 255).

Ajuste essas duas barras primeiro, grave, e só depois mexa no resto.

---

## 4. Desenhar uma pista

```bash
.venv\Scripts\python.exe -m robo.editor
```

```bash
.venv\Scripts\python.exe -m robo.editor --abrir tracks/pista1.json
```

Você desenha a **linha central** do corredor; as paredes, o chão e os
checkpoints saem dela automaticamente. Em todo modo, **clicar e arrastar**
significa "colocar e dimensionar":

| tecla | modo | clique | arrastar |
|---|---|---|---|
| `1` | linha central | acrescenta ponto no fim | move um ponto existente |
| `2` | caixas (obstáculos) | — | de um canto ao outro, com a medida em cm |
| `3` | largada | posiciona o robô | define para onde ele olha |
| `4` | objetivo | posiciona o centro | define o raio |

Outros controles:

- **botão direito**: apaga o que estiver sob o cursor no modo atual
- **roda do mouse**: zoom, em qualquer modo
- **botão do meio, arrastar**: move a câmera
- `[` / `]`: ajusta um número, que muda de sentido por modo — largura do
  corredor no modo 1, giro da caixa sob o cursor no modo 2, raio no modo 4
- `A`: preenche largada e objetivo automaticamente a partir do traçado (atalho;
  você pode continuar ajustando à mão depois)
- `T`: testa a pista dirigindo de verdade — `Esc` volta para a edição
- `S`: salva em `tracks/` (nome automático, `pista1.json`, `pista2.json`...)
- `L`: abre a última pista salva
- `G`: liga/desliga a grade de 5 cm
- `Z`: desfaz
- `N`: começa uma pista nova

**Salvar e testar ficam bloqueados até existirem largada e objetivo** — o
painel esquerdo mostra o que falta.

### Os avisos em laranja

Aparecem em vértices da linha central quando o corredor gerado ali sairia
errado:

- **curva fechada demais** para a largura atual — as duas paredes se cruzariam
- **trecho mais curto que a largura** — a ponta do corredor sai com uma aba
  torta

Nos dois casos, estreitar o corredor com `[` costuma resolver.

---

## 5. Como o robô funciona

Resumo rápido — detalhes e os porquês estão no [README](README.md).

**Entrada (sensores)**: um vetor de N números em `[0, 1]`, um por sensor
ultrassônico. `0` = obstáculo colado, `1` = livre (ou sem eco — o sensor real
não distingue as duas coisas). Nada de posição, velocidade ou ângulo: só o que
um HC-SR04 de verdade entregaria.

**Saída (motores)**: 4 números em `[0, 1]` — `[acelerar, ré, esquerda,
direita]`. Acima de 0,5 conta como "botão apertado". São os mesmos 4 botões que
você aperta jogando — não existe um canal separado "só para a IA".

**Detalhes que a simulação leva a sério, porque afetam se a rede treinada
funciona no robô real**:

- O sensor mede um **cone**, não um raio fino (por padrão 30°).
- Parede muito inclinada (mais de 65° por padrão) **não devolve eco** — lê
  igual a "livre". É a maior causa de robô batendo de frente numa parede em
  ângulo.
- Os sensores disparam **um de cada vez**, então a leitura de cada um fica
  desatualizada por um tempo (60 ms × quantidade de sensores, por padrão). A
  rede sempre decide com dado um pouco velho.
- Motor tem **zona morta** (PWM baixo não move nada) e **inércia** (leva um
  tempo até chegar na velocidade pedida).

---

## 6. Treinar a rede neural

Esta é a parte que é sua — o simulador te entrega dois jeitos de rodar
episódios, dependendo do tipo de aprendizado.

### O contrato: o que toda rede precisa ter

```python
class MinhaRede:
    def act_batch(self, obs):
        """obs: (P, n_sensores) em [0,1] -> ação: (P, 4) em [0,1]"""
        ...
```

Onde `P` é o número de robôs rodando ao mesmo tempo (a população, no caso de
algoritmo evolutivo; ou 1, se você só quer testar um robô). Isso é tudo que o
simulador exige — nenhuma suposição de framework.

Opcionalmente, para o painel de visualização (`Esc`/`R`/`N` na corrida, ou o
`Viewer`) desenhar a sua rede:

```python
def inspect(self, i=0):
    """Estado interno do robô i, só para desenhar. Pode devolver None."""
    return {
        "ativacoes": [array_por_camada, ...],   # da entrada até a saída
        "pesos": [matriz_saida_x_entrada, ...], # uma por conexão entre camadas
    }
```

Antes de treinar de verdade, confira o contrato:

```python
from robo.brain import check_brain
check_brain(minha_rede, n_sensores=4, n_robos=10)
```

Isso roda uma vez dentro do `EpisodeRunner` automaticamente e já avisa, com o
motivo escrito, se a forma da saída estiver errada, se vier NaN, ou se o
`inspect` estiver inconsistente — evita descobrir isso 200 gerações depois.

### Caminho A — episódio inteiro (neuroevolução, GA, CMA-ES)

Use quando o seu algoritmo avalia uma população inteira e não precisa de
recompensa a cada passo — só do resultado final do episódio.

```python
from robo.config import SimConfig
from robo.pistas import carregar
from robo.training import EpisodeRunner, resumo

cfg = SimConfig.carregar_ou_padrao()
track = carregar("zigue-zague")
runner = EpisodeRunner(track, cfg, n_robots=100)

for geracao in range(500):
    populacao = minha_populacao_atual()          # (P, 4) por robô -> uma rede
    dados = runner.run(populacao, seed=geracao)  # mesma pista para todos, sorteio fixo
    print(resumo(dados))

    fitness = meu_fitness(dados)                 # ver a fórmula abaixo
    minha_populacao_atual = evoluir(populacao, fitness)
```

`dados` (a telemetria) traz, por robô:

| campo | o quê |
|---|---|
| `progress` / `remaining` | quanto avançou **ao longo do traçado** (não em linha reta), em metros |
| `collided` | bateu (bool) |
| `bumps` | quantos passos ficou encostado numa parede |
| `finished` | chegou ao objetivo (bool) |
| `time` / `steps_alive` | quanto tempo rodou |
| `checkpoints` | quantos checkpoints passou, em ordem |
| `distance` | metros efetivamente rodados |
| `readings` | últimas leituras dos sensores |
| `position` | posição final |

**Um exemplo de fitness** (o que você descreveu: bateu perde, tempo perde,
chegar perto ganha) está pronto em [brains/exemplo.py](brains/exemplo.py),
função `fitness_exemplo`:

```python
def fitness_exemplo(dados, peso_tempo=0.5, punicao_batida=50.0, bonus_chegada=100.0):
    pontos = dados["progress"].copy()
    pontos -= peso_tempo * dados["time"]
    pontos -= punicao_batida * dados["collided"]
    pontos += bonus_chegada * dados["finished"]
    return pontos
```

Rode o exemplo (rede aleatória, só para ver o contrato funcionando):

```bash
.venv\Scripts\python.exe -m brains.exemplo --ver
```

Para assistir ao treino em uma janela, com o painel da rede:

```python
from robo.viewer import Viewer

viewer = Viewer(track, cfg, brain=melhor_da_geracao)
dados = runner.run(melhor_da_geracao, on_step=viewer.on_step)
viewer.close()
```

Dentro da janela: `espaço` liga o modo turbo (desenha só de vez em quando, para
não travar o treino), `R` liga/desliga o painel da rede, `Esc` fecha.

### Um exemplo completo e funcional: `brains/treino_ga.py`

O trecho acima é o conceito. [brains/treino_ga.py](brains/treino_ga.py) é a
implementação de verdade — um script que você roda hoje e que já treina,
salva, retoma depois de interromper, e compete. Ele existe para responder
exatamente três perguntas: onde você define os valores, onde mora a função de
treino, e como manter várias redes evoluindo por várias épocas sem perder o
trabalho se o processo cair no meio.

```bash
.venv\Scripts\python.exe -m brains.treino_ga
```

Isso abre um **menu para escolher a pista** e depois treina 120 redes por 300
gerações, salvando tudo em `checkpoints/`. Sem tela nenhuma — pygame só entra
se você pedir `--ver`. Dá para deixar rodando num servidor sem monitor, ou numa
aba de terminal enquanto você faz outra coisa.

#### 1. Escolher a pista de treino

Sem `--pista`, o treino pergunta antes de começar:

```
Escolha a pista de treino:

  embutidas
    1) curva-u                    8.00 m  sem caixas
    2) zigue-zague               15.00 m  sem caixas  <- padrão
    3) caracol                   24.40 m  sem caixas
    4) diagonais                 12.17 m  sem caixas
  suas (tracks/)
    5) tracks/pista1.json        19.95 m   5 caixas
    6) tracks/pista2.json        32.23 m  21 caixas
    7) tracks/pista3.json        19.95 m   5 caixas

número, nome/caminho, ou Enter para zigue-zague:
```

Aceita o número, o nome embutido (`caracol`) ou um caminho (`tracks/pista2.json`).
As suas pistas aparecem sozinhas — é a pasta `tracks/`, onde o editor salva.

Para pular o menu, passe a pista direto:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --pista tracks/pista2.json
```

```bash
.venv\Scripts\python.exe -m brains.treino_ga --listar-pistas
```

O menu só aparece em terminal interativo. Rodando por script ou tarefa
agendada, ele usa a padrão e avisa, em vez de travar esperando uma digitação
que nunca vem. E ao usar `--continuar`, ele não pergunta: a pista vem do
próprio checkpoint (veja o item 7).

**Qual escolher para começar:** `curva-u` é a mais curta e a rede aprende algo
nela bem antes; `caracol` tem curvas fechando progressivamente e é a mais
difícil; `diagonais` tem paredes em ângulo, onde o eco perdido do ultrassom
aparece — boa para testar se a rede lida com sensor mentindo.

#### 2. Onde você define os valores

Tudo que controla *como* a evolução se comporta está numa classe só, no topo
do arquivo:

```python
@dataclass
class Hiperparametros:
    populacao: int = 120        # quantas redes competem ao mesmo tempo
    geracoes: int = 300         # quantas vezes a população evolui
    camadas_ocultas: tuple = (12, 12)
    elite_frac: float = 0.12    # fração que passa direto, sem mudar
    torneio: int = 4            # quantos disputam por vaga de pai
    taxa_mutacao: float = 0.15  # fração dos pesos que sofre mutação
    forca_mutacao: float = 0.2  # tamanho do passo da mutação
    checkpoint_a_cada: int = 20
    seed: int = 0
    ver_a_cada: int = 10        # de quantas em quantas gerações a janela aparece
```

Isto é **diferente** da `SimConfig` do capítulo anterior. `SimConfig` é a
física do robô (sensores, motores) — a mesma para qualquer algoritmo de
treino. `Hiperparametros` é a receita do algoritmo genético em si: nada aqui
fala de metros ou m/s.

Você muda esses valores de duas formas. Editando o `default=` na classe (se
quiser um novo padrão permanente), ou pela linha de comando, sem editar nada:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --populacao 200 --geracoes 500 --camadas 16,16 --mutacao 0.2
```

**As duas formas conversam**: o `argparse` puxa os padrões desta classe em vez
de repetir números próprios. Então editar aqui muda também o que roda no
terminal e no `F5` do VSCode. Para conferir o que está valendo sem abrir o
arquivo:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --help
```

Cada campo tem um comentário no próprio arquivo explicando o efeito de
aumentar ou diminuir — vale ler antes de sair mudando número.

#### 3. Onde moram "várias redes ao mesmo tempo"

A classe `PopulacaoGA` é a população inteira — não uma rede, **P redes**, uma
por "robô" da simulação. O truque que faz isto rápido: os pesos de todo mundo
ficam juntos num único array `(P, saída, entrada)` por camada, e um `einsum`
calcula a saída da população inteira de uma vez, sem laço em Python:

```python
def act_batch(self, obs):
    x = obs
    for W, b in self.pop:                       # (P, saída, entrada), (P, saída)
        x = np.einsum("poi,pi->po", W, x) + b    # as P redes de uma vez
        x = tanh(x)  # ou sigmoide na última camada
    return x
```

Isso é o que já vinha em `brains/exemplo.py` (a `RedeAleatoria`) — aqui ela
ganhou um método a mais, `evoluir()`, que é a genética em si: separa os
melhores (elite), sorteia pais por torneio, cruza os pesos dos dois pais
(cada peso do filho vem de um ou de outro, por sorteio) e aplica mutação
(ruído gaussiano numa fração dos pesos). Tudo isso também vetorizado — sem
laço por indivíduo.

#### 4. Onde está a função de fitness — troque esta

```python
# ======================================================================
# 3. FUNÇÃO DE FITNESS — troque esta função pela sua fórmula
# ======================================================================
minha_fitness = fitness_exemplo
```

É só isso. `fitness_exemplo` vem de `brains/exemplo.py` e implementa a fórmula
que você descreveu (avançar ganha, tempo perde, bater perde, chegar dá
bônus). Para usar a sua, escreva uma função que recebe a telemetria do
episódio e devolve um array `(P,)` — maior é melhor — e troque essa linha:

```python
def minha_fitness(dados):
    pontos = dados["progress"] * 2.0
    pontos -= 0.3 * dados["time"]
    pontos -= 80.0 * dados["collided"]
    pontos += 200.0 * dados["finished"]
    pontos -= 5.0 * dados["bumps"]          # cada passo encostado na parede também custa
    return pontos
```

Todos os campos disponíveis estão na tabela da seção anterior.

#### 5. A função de treino: o laço de muitas épocas

```python
def treinar(hiper, track, cfg, pasta, geracao_inicial=0, pop_inicial=None,
           melhor_fitness_global=-np.inf, rng=None, ver=False, ver_a_cada=10):
    populacao = PopulacaoGA(cfg.n_sensors, hiper, rng, pop_inicial=pop_inicial)
    runner = EpisodeRunner(track, cfg, n_robots=hiper.populacao)

    for geracao in range(geracao_inicial, hiper.geracoes):
        dados = runner.run(populacao, seed=hiper.seed * 100_000 + geracao)
        fitness = minha_fitness(dados)

        print(...)                    # uma linha de progresso no terminal
        log.write(...)                # a mesma linha, num CSV

        if fitness.max() é o melhor de todos os tempos:
            salvar(".../melhor.npz", ...)     # a rede pronta para competir/implantar

        populacao.evoluir(fitness)            # a próxima geração nasce aqui

        if (geracao + 1) % hiper.checkpoint_a_cada == 0:
            salvar_checkpoint(".../geracao_NNNN.ga.npz", populacao, ...)
```

(o trecho acima é simplificado — o arquivo de verdade tem mais alguns
detalhes, como o `--ver` periódico; a estrutura é exatamente essa)

Cada iteração do `for` é uma **época** (uma geração): avalia todo mundo num
episódio, calcula o fitness de cada um, e evolui para a próxima leva de
redes. É por isso que "treinar várias redes" e "treinar por várias épocas"
são a mesma coisa aqui — cada iteração avalia P redes de uma vez, e você roda
centenas dessas iterações.

Cada geração imprime uma linha assim:

```
geração   42 | melhor    287.3 | média    134.9 | pior    -95.2 | chegaram   6/120 | bateram  38/120 |  0.91 s
```

E a mesma linha vai para `checkpoints/historico.csv`, para você plotar a
curva de aprendizado depois (Excel, `pandas.read_csv` + `matplotlib`, o que
preferir).

#### 6. Dois arquivos diferentes saem do treino — e por um motivo

| arquivo | o quê | formato |
|---|---|---|
| `checkpoints/melhor.npz` | a **melhor rede já vista**, pronta para competir ou virar firmware | `robo.persist` — o mesmo formato da seção 7 |
| `checkpoints/geracao_0100.ga.npz` | a **população inteira**, para retomar o treino de onde parou | formato próprio do script, não é para implantar |

São propósitos diferentes. `melhor.npz` você carrega com `RedeSalva.carregar(...)`
e usa direto na corrida (seção 8). `geracao_0100.ga.npz` só serve para o
próprio `treino_ga.py` continuar treinando — ele guarda a população inteira
(todas as P redes, não só a campeã) e o estado exato do gerador de números
aleatórios, para a evolução continuar bit a bit igual a se nunca tivesse
parado.

#### 7. Interromper e continuar

Você pode fechar o terminal (ou apertar `Ctrl+C`) a qualquer momento — o
último checkpoint gravado (a cada `--checkpoint-a-cada` gerações, 20 por
padrão) é o ponto de retomada:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --continuar checkpoints/geracao_0100.ga.npz --geracoes 500
```

Isso carrega a população, os hiperparâmetros originais (a arquitetura da rede
não pode mudar no meio do treino, então eles vêm do checkpoint, não da linha
de comando), **a pista em que estava treinando**, e continua a partir da
geração 101. **`--geracoes` é o único valor que você pode esticar ao retomar** —
é o alvo até onde continuar.

Isto foi verificado de um jeito que importa: treinei 10 gerações sem parar, e
separadamente treinei 6 e retomei até 10 — os dois produziram exatamente a
mesma sequência de números em cada geração, não só "parecida". Sem isso,
interromper o treino silenciosamente reiniciaria a sorte da mutação a cada
vez, e você nunca saberia.

**A pista volta sozinha.** O checkpoint registra em qual percurso o treino
aconteceu, então retomar sem dizer nada continua no mesmo lugar. E se você
trocar de propósito, ele avisa:

```
ATENÇÃO: mudando de pista — treinado em 'curva-u' (8.00 m), continuando em
'caracol' (24.40 m). Uma queda no fitness aqui é esperada, não é falha da rede.
```

Treinar no fácil e migrar para o difícil é uma tática legítima, então isso não
bloqueia nada — só deixa de ser silencioso. Junto com a referência da pista vai
uma **impressão digital da geometria**, porque só o nome não bastaria: você pode
reabrir `tracks/pista1.json` no editor, mexer no traçado e salvar com o mesmo
nome. Nesse caso o aviso é outro:

```
ATENÇÃO: a pista 'tracks/pista1.json' foi EDITADA desde o treino
(19.95 m -> 19.69 m). A rede foi treinada noutro percurso com o mesmo nome.
```

Checkpoints gravados antes desta versão não têm a marca; eles continuam
carregando normalmente, só avisam que não dá para conferir.

#### 8. Acompanhar visualmente enquanto treina

```bash
.venv\Scripts\python.exe -m brains.treino_ga --ver --ver-a-cada 5
```

A cada 5 gerações, abre uma janela mostrando a campeã atual rodando sozinha
na pista — a mesma janela fica aberta a evolução inteira, só atualiza durante
as gerações de demonstração (fica "parada" entre uma e outra, o que é
esperado: ela não está desenhando as gerações que só treinam).

**Para assistir a população inteira**, e não só a campeã:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --ver-populacao --ver-a-cada 5
```

A diferença não é só visual. Com `--ver`, o que você vê é uma *demonstração à
parte*: a campeã roda sozinha num episódio extra, depois que a geração já foi
avaliada. Com `--ver-populacao`, você está vendo o **treino de verdade
acontecendo** — é o mesmo episódio que gera o fitness, com todos os
indivíduos ao mesmo tempo.

Isso tem um custo, e ele é grande: desenhar entra no laço quente do treino, e
a geração passa a rodar na velocidade da tela (~20 fps). Medido: com 60 robôs,
uma geração normal levou **1 s** e a mesma geração com janela levou **101 s** —
não é travamento, é o tempo real de assistir 2000 passos a 20 quadros por
segundo. As gerações com janela saem marcadas com `[na tela]` no terminal.

Por isso o `ver_a_cada` (na classe `Hiperparametros`, ou `--ver-a-cada` na linha de comando) importa aqui mais do que no `--ver`: nas gerações que
não são de exibição nada é desenhado e o treino corre na velocidade máxima.
Para treinar sério assistindo de vez em quando, use algo como `--ver-a-cada 20`
ou mais. Para só ver como está indo, `--ver-a-cada 5` e menos gerações.

Na tela, o robô de índice 0 aparece destacado (cor cheia, com os cones dos
sensores) e o resto da população em tom apagado. Dois detalhes que precisaram
de correção para isso funcionar, e que valem saber caso você mexa no desenho:

- os robôs de fundo usam uma **cor sólida escurecida**, não transparência —
  `pygame.draw.circle` ignora o canal alfa numa superfície comum (pedir alfa
  120 devolve um pixel opaco, com alfa 255)
- o destacado é desenhado **por último**. No começo do treino a população
  fica quase toda empilhada no mesmo ponto (medido: 52 de 80 robôs sobre o
  mesmo pixel), e desenhá-lo primeiro o fazia sumir debaixo da pilha
  justamente quando você mais quer olhar para ele

#### 9. Treinar na GPU

Só a **passada da rede** vai para a GPU; a física (sensores, colisão,
movimento) continua na CPU. Isso não é preguiça — é o que os números pedem:

```bash
.venv\Scripts\python.exe -m brains.treino_ga --dispositivo auto --populacao 2000 --camadas 128,128
```

`--dispositivo` aceita `cpu` (padrão), `cuda` ou `auto` (usa GPU se houver).
Pedir `cuda` numa máquina sem GPU não quebra o treino: avisa o motivo e
continua na CPU.

**Quando a GPU compensa** — medido nesta máquina (RTX 4060), tempo da passada
da rede por passo de simulação:

| população | camadas | CPU | GPU | ganho |
|---|---|---|---|---|
| 2000 | 12, 12 | 1,0 ms | 0,8 ms | 1,3× |
| 2000 | 64, 64 | 7,6 ms | 0,8 ms | **9,2×** |
| 2000 | 128, 128 | 21,0 ms | 1,7 ms | **12,6×** |
| 8000 | 128, 128 | 85,7 ms | 6,6 ms | **12,9×** |

Com a rede pequena do padrão (12,12) o ganho é quase nada, e num treino curto
a GPU chega a ficar **mais lenta** — o custo de mandar os dados para a placa e
trazer de volta não se paga quando a conta é minúscula. Para redes de 64
neurônios por camada ou mais, o quadro inverte completamente.

O outro lado: com rede pequena, o gargalo não é a rede, é a física. Com 2000
robôs e camadas (12,12), o `World.step` custa ~7,5 ms/passo contra ~0,9 ms da
rede. Mandar só a rede para a GPU aí não adianta — o tempo está no outro lugar.
Portar a física exigiria reescrever o raycast inteiro
([robo/geometry.py](robo/geometry.py), [robo/sensors.py](robo/sensors.py)) para
tensores, um projeto bem maior; se um dia isso virar o gargalo de verdade, é
por aí que se começa.

**A armadilha: aumentar só a população não é acelerado pela GPU.** A física
custa proporcionalmente ao número de robôs, e ela roda na CPU. Medido com
20.000 indivíduos e rede (16,16):

| | física (CPU) | rede | total | quanto a GPU pode ajudar |
|---|---|---|---|---|
| CPU | 93,2 ms | 14,0 ms | 107,2 ms | — |
| GPU | 93,2 ms | 1,5 ms | 94,7 ms | 12% |

A rede é só 13% do trabalho; mesmo zerando-a, o teto de ganho é ~13%. Compare
com 2.000 indivíduos e rede (128,128), onde o ganho de ponta a ponta medido foi
**3,5×** (60 s → 17 s por geração): ali a rede era a maior parte do tempo.

**Resumo prático:**

- **rede grande** (64+ por camada) → GPU compensa muito, ligue `--dispositivo auto`
- **população grande com rede pequena** → o gargalo é a física, na CPU; a GPU
  quase não muda nada
- **quer os dois** → aumente as camadas junto com a população, senão você está
  pagando o custo da população sem usar a placa

#### 10. Rodando do zero, na prática

```bash
.venv\Scripts\python.exe -m brains.treino_ga --populacao 150 --geracoes 400 --pista curva-u --ver --ver-a-cada 10
```

Deixe rodando. De vez em quando, teste a campeã atual contra você:

```bash
.venv\Scripts\python.exe -m robo.corrida --rede checkpoints/melhor.npz --pista curva-u
```

Como `melhor.npz` é reescrito toda vez que aparece uma rede melhor, basta
rodar a corrida de novo mais tarde para competir contra uma versão mais
treinada — não precisa parar o treino para isso.

### Caminho B — passo a passo (PPO, DQN, qualquer coisa com gradiente)

Use quando você precisa de recompensa a cada passo, não só no fim do episódio.

```python
obs = runner.reset(seed=0)
total = 0.0
while True:
    acao = minha_politica(obs)              # (P, 4)
    obs, info, fim = runner.step(acao)
    r = minha_recompensa(info)              # a fórmula é sua, ver abaixo
    total += r
    # aqui entra o passo de otimização do PPO/DQN, com (obs, acao, r, ...)
    if fim:
        break
```

`info` traz **fatos do passo**, nunca recompensa pronta:

| campo | o quê |
|---|---|
| `delta_progress` | quanto avançou no traçado **neste passo** (m; pode ser negativo, se recuar) |
| `collided_now` | bateu **neste passo** (sinalizado uma vez só) |
| `finished_now` | chegou **neste passo** (sinalizado uma vez só) |
| `alive` | ainda está rodando |
| `active` | estava rodando durante este passo (para não contar passo de robô já parado) |
| `min_reading` | leitura mais próxima entre todos os sensores, em `[0,1]` |
| `forward_speed` | velocidade para frente, m/s |
| `dt` | duração do passo, s |

Exemplo da mesma ideia de recompensa (bateu perde, demora perde, avançar
ganha), agora por passo:

```python
def minha_recompensa(info, peso_avanco=10.0, peso_tempo=0.5,
                     punicao_batida=50.0, bonus_chegada=100.0):
    r  = info["delta_progress"] * peso_avanco
    r -= peso_tempo * info["dt"] * info["active"]
    r -= punicao_batida * info["collided_now"]
    r += bonus_chegada * info["finished_now"]
    return r
```

### Esboço de rede em PyTorch

Só para ilustrar o formato — o `act_batch` precisa devolver numpy, então
converta na saída:

```python
import torch
import torch.nn as nn
import numpy as np

class RedePPO(nn.Module):
    def __init__(self, n_sensores, ocultas=(32, 32)):
        super().__init__()
        camadas = []
        tam = n_sensores
        for h in ocultas:
            camadas += [nn.Linear(tam, h), nn.Tanh()]
            tam = h
        camadas += [nn.Linear(tam, 4), nn.Sigmoid()]   # sigmoide: saída em [0,1]
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

    def act_batch(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            return self.rede(x).numpy()
```

Isso já satisfaz o contrato (`act_batch`) e pode ir direto no `EpisodeRunner`
ou no `Viewer`. O `Sigmoid()` no fim é importante — a saída precisa estar em
`[0,1]` para o limiar de 0,5 fazer sentido; se você usar `PPO` de verdade com
distribuição de ação, amostre da distribuição e passe o resultado por um
`sigmoid`/`clip` antes de devolver.

### Bater encerra o episódio?

Por padrão, sim (`cfg.collision_ends_episode = True`), que é o que algoritmo
evolutivo espera — economiza passos simulando. Se você quer gradiente mais
denso (o robô continua "vivo", parado contra a parede, acumulando penalidade),
desligue:

```python
cfg.collision_ends_episode = False
```

---

## 7. Salvar e carregar a rede

O destino final é um Arduino, que não tem PyTorch. Por isso o formato de
salvar é **neutro**: matrizes de peso, vieses, o nome da ativação de cada
camada, e a `SimConfig` usada no treino — nada de classe Python amarrada.

```python
from robo.persist import de_torch, salvar

camadas, ativacoes = de_torch(minha_rede.rede)   # varre um nn.Sequential
salvar("modelos/v1.npz", camadas, ativacoes, cfg, nota="geração 500, fitness 340")
```

`de_torch` reconhece `Linear`, `Tanh`, `ReLU`, `Sigmoid`, `LeakyReLU` dentro de
um `nn.Sequential` (inclusive aninhado). Se a sua rede tiver algo mais exótico
(atenção, recorrência, blocos residuais), monte a lista à mão — o formato é só
`[(W, b), (W, b), ...]` com `W` em `(saída, entrada)`:

```python
from robo.persist import salvar

W1, b1 = minha_rede.camada1.weight.detach().numpy(), minha_rede.camada1.bias.detach().numpy()
W2, b2 = minha_rede.camada2.weight.detach().numpy(), minha_rede.camada2.bias.detach().numpy()
salvar("modelos/v1.npz", [(W1, b1), (W2, b2)], ["tanh", "sigmoid"], cfg)
```

Para carregar de volta (não precisa de torch instalado):

```python
from robo.persist import RedeSalva

rede = RedeSalva.carregar("modelos/v1.npz")   # já é um Brain, roda em numpy puro
rede.act_batch(obs)
```

**Guarde o `state_dict` do PyTorch também**, num arquivo separado, se quiser
retomar o treino de onde parou — mas é o `.npz` que a corrida e o firmware
usam.

A `SimConfig` viaja dentro do `.npz` de propósito: uma rede treinada com 4
sensores a 90° não sabe o que fazer com 6 leituras a 150°. Se você carregar
essa rede numa sessão com config diferente, o simulador avisa e usa a config
do treino.

---

## 8. Competir contra a IA

```bash
.venv\Scripts\python.exe -m robo.corrida --rede modelos/v1.npz
```

```bash
.venv\Scripts\python.exe -m robo.corrida --rede modelos/v1.npz --pista tracks/minha.json
```

Sem `--rede`, você corre contra um adversário de **regra fixa** embutido
(acelera sempre, vira para o lado com o sensor mais livre) — serve de linha de
base: se a sua rede treinada não vence essa regra simples, ela ainda não
aprendeu nada de útil.

```bash
.venv\Scripts\python.exe -m robo.corrida
```

Os dois robôs largam lado a lado (com uma contagem de 3 segundos antes de
começar) e apertam os mesmos 4 botões — você pelo teclado, a rede pela saída
dela — com a mesma física e os mesmos sensores.

**Vence quem chega primeiro.** Se o tempo do episódio esgotar antes de alguém
chegar, vence quem avançou mais no traçado (empate se a diferença for menor
que 5 cm). Bater não elimina — você (ou a rede) dá ré e continua, perdendo só
tempo.

| tecla | ação |
|---|---|
| setas ou WASD | dirigir |
| `R` | reiniciar a corrida |
| `N` | esconder/mostrar o painel da rede |
| `Esc` | sair |

---

## 9. O que mudar e onde

| quero mudar... | vá em... |
|---|---|
| Velocidade, zona morta, sensores (temporariamente, para testar) | painel `P` dentro do jogo |
| Velocidade, zona morta, sensores (permanente, valores de fábrica) | [robo/config.py](robo/config.py) — `RobotConfig`, `SensorConfig` |
| Uma pista nova | editor (`robo.editor`), ou escreva à mão em [robo/pistas.py](robo/pistas.py) seguindo o padrão de `zigue_zague()` |
| Como os 4 botões viram movimento | [robo/physics.py](robo/physics.py), função `buttons_to_pwm` |
| O modelo do sensor ultrassônico (cone, eco perdido, rodízio) | [robo/sensors.py](robo/sensors.py) |
| Cores, HUD, desenho | [robo/render.py](robo/render.py) |
| O painel de calibragem (novas barras) | [robo/calibra.py](robo/calibra.py), lista `self.grupos` |
| A arquitetura da rede, o algoritmo de treino | [brains/treino_ga.py](brains/treino_ga.py) — copie o arquivo e mude à vontade |
| A fórmula de fitness / recompensa | `minha_fitness` em [brains/treino_ga.py](brains/treino_ga.py), ou `fitness_exemplo` em [brains/exemplo.py](brains/exemplo.py) — não mora no simulador |
| Hiperparâmetros do treino (população, gerações, mutação) | classe `Hiperparametros` em [brains/treino_ga.py](brains/treino_ga.py), ou pela linha de comando |

**Não devia precisar mudar** (mas pode, se souber o motivo): `robo/world.py`
(física central), `robo/geometry.py` (raycast), `robo/track.py` (como a pista
é derivada da linha central), `robo/training.py` (o contrato do runner),
`robo/persist.py` (formato de salvar).

### Estrutura completa

```
robo/config.py       parâmetros do robô/sensores/simulação; calibragem salva em config.json
robo/geometry.py      raycast e distâncias, vetorizados
robo/sensors.py       modelo do HC-SR04 (cone, eco perdido, rodízio, latência)
robo/physics.py       4 botões -> PWM -> movimento (tração diferencial)
robo/track.py         a receita da pista (linha central, largura, caixas); deriva as paredes
robo/pistas.py        pistas embutidas prontas (curva-u, zigue-zague, caracol, diagonais)
robo/world.py         P robôs num percurso; devolve telemetria crua, nunca fitness
robo/render.py        desenho e HUD
robo/game.py          modo de jogar
robo/editor.py         editor de pista
robo/calibra.py        painel de barras de calibragem
robo/brain.py          o contrato Brain (act_batch, inspect) + verificação
robo/training.py       EpisodeRunner: episódio inteiro OU passo a passo
robo/netviz.py         painel que desenha qualquer arquitetura de rede
robo/viewer.py         janela para assistir ao treino
robo/persist.py        salvar/carregar a rede em formato neutro (.npz)
robo/corrida.py        você contra a rede, na mesma pista
brains/exemplo.py      rede aleatória de referência — o contrato Brain, mínimo
brains/treino_ga.py    treino completo: hiperparâmetros, laço de gerações, checkpoint
tracks/                suas pistas salvas pelo editor
checkpoints/            saída de brains/treino_ga.py: melhor.npz, gerações, historico.csv
tests/                 verificações automáticas de cada parte
arduino/                firmware de referência (precisa ser atualizado — ver seção 11)
config.json             sua calibragem (criado ao apertar S no painel)
.vscode/launch.json     todo comando deste guia, pronto para rodar com F5
.vscode/tasks.json      tarefas de terminal, como "rodar todos os testes"
.vscode/settings.json   interpretador fixo no .venv do projeto
```

---

## 10. Testes

```bash
.venv\Scripts\python.exe -m tests.test_nucleo
```

Existem nove arquivos de teste, um por área:

| arquivo | cobre |
|---|---|
| `test_nucleo` | física, sensores, raycast, colisão |
| `test_editor` | editor de pista |
| `test_training` | contrato `Brain`, medida de progresso, `EpisodeRunner.run` |
| `test_netviz` | painel de visualização da rede |
| `test_calibra` | painel de calibragem |
| `test_persist` | laço passo a passo, salvar/carregar |
| `test_corrida` | modo corrida |
| `test_treino_ga` | `brains/treino_ga.py` — o ponto que mais importa: retomar um checkpoint precisa reproduzir bit a bit o que teria acontecido sem parar; e a GPU tem que dar o mesmo resultado da CPU |
| `test_render` | desenho: robô destacado não pode sumir sob a população empilhada |

Rode todos em sequência:

```bash
for %f in (test_nucleo test_editor test_training test_netviz test_calibra test_persist test_corrida test_treino_ga test_render) do .venv\Scripts\python.exe -m tests.%f
```

No VSCode, é a tarefa **Rodar todos os testes** (`Ctrl+Shift+B`, ou
`Ctrl+Shift+P` → *Tasks: Run Task*) — veja a [seção 1.1](#11-usando-o-vscode).
Cada arquivo também tem sua própria entrada em `F5`, útil para depurar um teste
que falhou com um breakpoint dentro dele.

Se você mudar algo em `robo/`, vale rodar o arquivo de teste correspondente
antes de treinar de verdade — é mais rápido descobrir um erro de física ali do
que 500 gerações depois.

---

## 11. Levando para o Arduino

O arquivo [arduino/robo_desviador.ino](arduino/robo_desviador.ino) e o
`arduino/policy.h` que está na pasta hoje **são de uma versão anterior do
projeto** (a com obstáculos circulares, antes do redesenho para o jogo com
pista). Eles não vão bater com o formato atual — trate como referência de como
ligar os pinos, não como pronto para usar.

O caminho para gerar um `policy.h` atualizado a partir do `.npz` desta versão
ainda não foi escrito. Quando você tiver uma rede treinada e quiser embarcar,
isso é um script pequeno: ler o `.npz` com `robo.persist.carregar`, e para cada
camada emitir um array C `const float W[] = {...};` e uma função
`forward()` que faz `saida[i] = ativacao(soma(W[i][j] * entrada[j]) + b[i])`
camada a camada — mesma lógica de `RedeSalva.act_batch`, só que em C ao invés
de numpy. Peça para eu escrever esse script quando chegar nessa etapa.

---

## 12. Perguntas frequentes

**"python" não é reconhecido, ou abre a Microsoft Store**
Use `.venv\Scripts\python.exe`, nunca `python` sozinho — o `python` do PATH do
Windows é um atalho da loja, não um Python de verdade.

**Salvei a calibragem mas o treino não mudou**
Confira se o seu script chama `SimConfig.carregar_ou_padrao()` (ou
`SimConfig.load()`) em vez de `SimConfig()`. `SimConfig()` sozinho sempre volta
aos valores de fábrica.

**Minha rede passa no `check_brain` mas o comportamento é estranho**
Confira o intervalo da saída. `check_brain` só reclama de forma errada, NaN ou
`inspect` inconsistente — não de "valores plausíveis". Se a última camada não
tiver uma ativação que limite a saída a `[0,1]` (sigmoide, por exemplo), o
limiar de 0,5 não vai se comportar como esperado.

**A corrida diz que a rede espera N sensores mas minha config tem outro
número**
A rede foi treinada com uma `SensorConfig` diferente da atual. Ou recalibre a
config atual para bater (`P` no jogo, ajustar "quantidade" em sensores, `S`
para gravar), ou deixe — se o `.npz` trouxe a config do treino, a corrida usa
essa automaticamente e só avisa.

**Quero comparar dois cérebros de forma justa**
Use o mesmo `seed` nos dois `runner.run(...)` — isso fixa o sorteio de
obstáculos aleatórios e o ponto de partida, então a única diferença entre as
duas rodadas é a rede.
