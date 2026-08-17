# Robô desviador — simulador jogável

Simulador de um robô diferencial com HC-SR04, num percurso de paredes. Você joga
no teclado; depois a sua rede neural joga o mesmo jogo, pelos mesmos 4 botões.

**Estado: as 5 etapas do plano estão prontas.** Jogo, editor de pista, contrato
da rede, painel de visualização e corrida contra a IA.

## Rodar

```bash
.venv\Scripts\python.exe -m robo.game
```

```bash
.venv\Scripts\python.exe -m robo.game --pista diagonais
```

Pistas embutidas: `curva-u`, `zigue-zague`, `caracol`, `diagonais`.
Setas ou WASD dirigem, `R` reinicia, `C` alterna a câmera, `Esc` sai.

```bash
.venv\Scripts\python.exe -m tests.test_nucleo
```

```bash
.venv\Scripts\python.exe -m tests.test_editor
```

São sete: `test_nucleo`, `test_editor`, `test_training`, `test_netviz`,
`test_calibra`, `test_persist` e `test_corrida`.

## Editor de pista

```bash
.venv\Scripts\python.exe -m robo.editor
```

```bash
.venv\Scripts\python.exe -m robo.editor --abrir tracks/pista1.json
```

Você desenha a linha do meio do corredor e o resto sai dela: as duas paredes, o
chão, os checkpoints. Em todo modo, **apertar e arrastar** quer dizer "coloca e
dimensiona".

| tecla | modo | clique | arrastar |
|---|---|---|---|
| `1` | linha central | acrescenta ponto | move um ponto existente |
| `2` | caixas | — | de um canto ao outro, com a medida em cm |
| `3` | largada | põe o robô | para onde ele olha |
| `4` | objetivo | põe o centro | o raio |

Botão direito apaga o que estiver sob o cursor. A roda é zoom sempre, e os
escalares ficam em `[` e `]` — largura do corredor no modo 1, giro da caixa no
modo 2, raio no modo 4. Botão do meio arrasta a tela.

`t` testa dirigindo e `Esc` volta pra edição. `s` salva em `tracks/`, `l` abre a
última, `a` preenche largada e objetivo a partir do traçado, `g` liga a grade de
5 cm, `z` desfaz, `n` começa do zero.

Salvar e testar ficam bloqueados até existirem largada e objetivo — o painel
mostra o que falta.

### Os avisos em laranja

O corredor nasce deslocando a linha central para os dois lados, e isso quebra
de duas formas que não dão erro nenhum, só saem tortas:

- **curva fechada demais** para a largura: as paredes se cruzam e o corredor
  vira um nó
- **trecho mais curto que a largura**: a esquadria invade a tampa da ponta e
  sobra uma aba enviesada

Os vértices problemáticos aparecem circulados em laranja enquanto você desenha.
Estreitar o corredor com `[` costuma resolver os dois.

## Os 4 botões

Acelerar, ré, esquerda e direita — e é só isso que existe. O teclado e a rede
entram pelo mesmo caminho, então o que você sente jogando é exatamente o que a
rede vai enfrentar. Eles viram PWM dos dois motores em
[`buttons_to_pwm`](robo/physics.py):

```
base = acelerar - ré           # -1, 0, +1
giro = direita - esquerda      # -1, 0, +1
esquerda = base + giro * turn_gain
direita  = base - giro * turn_gain     # saturado preservando a proporção
```

Acelerar sozinho anda reto, virar sozinho gira no lugar, e os dois juntos fazem
curva aberta.

## O que o robô enxerga

A observação é só um vetor `(n_sensores,)` em `[0, 1]`: `0` = obstáculo colado,
`1` = livre. Nada de posição, velocidade ou ângulo — o robô real não teria isso.

O modelo do HC-SR04 em [`sensors.py`](robo/sensors.py) tenta ser honesto:

- **Cone, não raio.** Ele devolve o obstáculo mais próximo num cone de ~30°,
  simulado com sub-raios pegando o menor. Verificado: um obstáculo lateral que um
  raio fino não vê é detectado a 0,48 m por um cone de 40°.
- **Eco perdido.** Parede inclinada além de 65° reflete o som para longe e não
  volta eco — e "sem eco" lê igual a "livre". Verificado: a mesma parede lê
  0,91 m de frente e "livre" a 75°. É assim que robô com ultrassom entra de
  cara numa parede em ângulo.
- **Rodízio e latência.** Um sensor por vez, ~60 ms cada, porque disparar juntos
  dá crosstalk. Com 4 sensores a varredura leva 240 ms e a rede sempre decide com
  dado velho. O HUD mostra a idade de cada leitura.

Ruído e falha aleatória de leitura existem em `SensorConfig` mas vêm desligados.

## Onde montar os sensores importa mais do que parece

Com número **par** de sensores nenhum aponta para frente. Medido num corredor de
0,9 m, a 1 m de uma parede frontal, o menor valor lido:

| arranjo | ângulos | lê |
|---|---|---|
| 4 em 150° | ±25, ±75 | 0,61 m — é a **parede lateral**; a frontal só aparece a ~0,6 m |
| 4 em 90° (padrão) | ±15, ±45 | 0,81 m |
| 4 em 60° | ±10, ±30 | 0,97 m |
| 5 em 150° | ±75, ±38, **0** | 1,00 m — o sensor central enxerga a parede inteira |

Leque largo enxerga os lados e perde a frente; um sensor central resolve a frente.
E o cone de 30° bate na parede lateral por volta de 1 m num corredor estreito, então
nenhum arranjo enxerga muito além disso — corredor mais largo ou cone mais estreito
é o que compra distância. Isso limita a velocidade útil: a 0,35 m/s com 240 ms de
varredura, o robô anda 8 cm entre leituras do mesmo sensor.

Mudei o padrão para 90° por isso, mas a decisão é sua — `SensorConfig` aceita
`angles_deg` explícito se você quiser um arranjo assimétrico.

## Onde a sua rede entra

O contrato inteiro cabe num método:

```python
class MinhaRede:
    def act_batch(self, obs):      # (P, n_sensores) em [0,1] -> (P, 4) em [0,1]
        ...
```

A saída é `[acelerar, ré, esquerda, direita]`, e acima de 0,5 conta como tecla
apertada — os mesmos 4 botões do teclado. Nada de framework assumido.

```python
from robo.training import EpisodeRunner, resumo

runner = EpisodeRunner(track, cfg, n_robots=100)
while True:
    cerebros = voce_monta_a_populacao()
    dados = runner.run(cerebros, seed=geracao)
    voce_evolui(dados)             # o fitness é sua conta
```

`runner.run` devolve **telemetria crua** e nenhum fitness. `check_brain` roda
antes e falha com o motivo escrito se a forma estiver errada, em vez de deixar o
numpy transformar isso em comportamento estranho lá na frente.

Rodando o exemplo, que é uma rede aleatória só para provar que o contrato fecha:

```bash
.venv\Scripts\python.exe -m brains.exemplo --ver
```

### A telemetria

| campo | o quê |
|---|---|
| `progress` / `remaining` | quanto avançou **ao longo do traçado**, em metros |
| `collided` / `bumps` | bateu; e quantos passos ficou em contato |
| `finished` | chegou ao objetivo |
| `time` / `steps_alive` | quanto tempo levou |
| `checkpoints` | quantos passou, em ordem |
| `distance` | metros rodados (≠ avanço: girar em falso não conta) |
| `readings` / `position` | leituras e pose no fim |

Para "o quanto chegou perto da chegada" use **`progress`, não `distance_to_goal`**.
A distância em linha reta ignora as paredes: num corredor sinuoso o robô pode
estar a 1 m do objetivo em linha reta e a 5 m de percurso, do outro lado de uma
parede. Premiar isso ensina o robô a encostar na parede certa, não a progredir.
Tem teste medindo exatamente essa inversão.

Um fitness como o que você descreveu — avança ganha, demora perde, bate perde —
está escrito em [brains/exemplo.py](brains/exemplo.py) como `fitness_exemplo`, do
lado de fora do simulador, que é onde ele deve morar.

### Treinando com recompensa por passo (PPO, DQN)

O `run` acima entrega o episódio inteiro, que é o que neuroevolução quer. Para
gradiente existe o laço passo a passo:

```python
obs = runner.reset(seed=0)
while True:
    acao = politica(obs)
    obs, info, fim = runner.step(acao)
    r = minha_recompensa(info)        # a fórmula é sua
    if fim:
        break
```

`info` traz os **fatos** do passo, por robô — e nenhuma recompensa:

| campo | o quê |
|---|---|
| `delta_progress` | quanto avançou no traçado **neste passo** (m, pode ser negativo) |
| `collided_now` / `finished_now` | bateu ou chegou neste passo — sinalizados uma vez só |
| `alive` / `active` | está rodando; estava rodando durante o passo |
| `min_reading` | leitura mais próxima, em [0,1] |
| `forward_speed` | velocidade para frente (m/s) |
| `dt` | duração do passo |

A recompensa que você descreveu sai daí em quatro linhas:

```python
r  = info["delta_progress"] * 10.0            # mais perto da chegada, mais ponto
r -= 0.5 * info["dt"] * info["active"]        # demorar custa
r -= 50 * info["collided_now"]                # bateu, perdeu
r += 100 * info["finished_now"]               # chegou, bônus
```

### Bater encerra o episódio?

Por padrão sim (`cfg.collision_ends_episode`), que é o que algoritmo evolutivo
espera. Desligando, o robô fica vivo parado contra a parede acumulando tempo e
batidas, o que dá gradiente mais denso se você for de gradiente.

## Salvando a rede treinada

O destino final desta rede é um Arduino, e lá não existe torch. O `state_dict`
também amarra o arquivo à classe Python que o criou — renomeie a classe e ele
vira lixo. Então salve um artefato **neutro**: matrizes, vieses, o nome de cada
ativação e a `SimConfig` do treino.

```python
from robo.persist import de_torch, salvar, RedeSalva

camadas, ativacoes = de_torch(minha_rede_torch)   # varre nn.Sequential
salvar("modelos/v1.npz", camadas, ativacoes, cfg, nota="geração 42")

brain = RedeSalva.carregar("modelos/v1.npz")      # já é um Brain, sem torch
```

Guarde o `state_dict` também se quiser retomar o treino — mas é o `.npz` que vai
virar firmware, e é ele que a corrida contra a IA carrega. Verificado: o numpy
reproduz a saída do torch com erro de 4e-8.

A `SimConfig` vai junto de propósito. Rede treinada com 4 sensores a 90° é lixo
num robô com 5 a 150°, e sem essa marca você descobre isso depurando firmware.

## Painel da rede

```python
def inspect(self, i=0):
    return {"ativacoes": [...],   # um array por camada, da entrada à saída
            "pesos": [...]}       # uma matriz (saída, entrada) por conexão
```

Opcional: devolver None, não implementar, ou até estourar, e o painel apenas não
aparece — o treino continua. Ele lê qualquer arquitetura, colore os neurônios
pela ativação e as conexões pelo peso (vermelho excita, azul inibe), e acende em
verde os botões que passaram de 0,5. Rede grande desenha só as conexões mais
fortes, senão vira borrão preto e custa caro.

```python
from robo.viewer import Viewer

viewer = Viewer(track, cfg, brain=minha_rede)
dados = runner.run(minha_rede, on_step=viewer.on_step)
viewer.close()
```

Na janela: espaço alterna turbo, `r` liga/desliga o painel, `Esc` encerra.

## Calibragem: aproximar a simulação do seu robô

No jogo, `p` abre um painel de barras. Você dirige com as setas enquanto ajusta,
e sente o efeito na hora.

| grupo | barras |
|---|---|
| motores | velocidade máx, zona morta PWM, tempo p/ acelerar, assimetria, força da curva |
| chassi | entre-eixos, raio do corpo |
| sensores | quantidade, leque, cone, sub-raios, alcance, tempo de leitura, limite de eco, ruído |

`s` grava em `config.json`, `z` volta ao de fábrica. **Gravar é o que importa** —
o treino roda noutro processo e lê o mesmo arquivo:

```python
cfg = SimConfig.carregar_ou_padrao()   # sua calibragem, ou os valores de fábrica
```

Mudar a montagem dos sensores (quantidade, leque, cone) remonta o array **sem
perder a pose do robô**, para você comparar dois arranjos no mesmo ponto da pista
em vez de voltar para a largada a cada ajuste.

Os dois números que mais afastam a simulação do robô real são a **velocidade
máxima** (cronometre o seu no chão) e a **zona morta do PWM** (o menor duty que
tira ele do lugar). Meça esses dois primeiro.

## Fidelidade dos motores

Zona morta (PWM baixo não vence o atrito), inércia (a roda leva `accel_time` para
chegar na velocidade pedida) e assimetria (`motor_bias`: com 6% o robô desvia
22 cm em 3 s andando "reto"). Meça os seus e ajuste em
[`config.py`](robo/config.py) — é isso que decide se a rede treinada aqui
funciona no robô de verdade.

## Estrutura

```
robo/config.py     todos os parâmetros, comentados com o porquê
robo/geometry.py   raycast e distâncias, vetorizados sobre todos os robôs
robo/sensors.py    o modelo do HC-SR04
robo/physics.py    4 botões -> PWM -> movimento
robo/track.py      a receita da pista; as paredes saem dela
robo/world.py      P robôs num percurso; devolve telemetria crua
robo/render.py     desenho e HUD
robo/game.py       modo de jogar
robo/editor.py     editor de pista
robo/brain.py      o contrato da sua rede
robo/training.py   EpisodeRunner: roda episódios, devolve telemetria crua
robo/corrida.py    você contra a rede, na mesma pista
robo/persist.py    salvar/carregar a rede em formato neutro
robo/calibra.py    painel de barras para calibrar
robo/netviz.py     painel da rede
robo/viewer.py     janela para assistir ao treino
brains/exemplo.py  rede aleatória de referência — você substitui
tracks/            suas pistas salvas
tests/             verificações
```

O `Track` guarda a **receita** (linha central, largura, caixas, largada,
objetivo) e deriva as paredes num `rebuild()`. Por isso uma pista salva continua
editável: se guardasse só as paredes prontas, você não conseguiria mudar a
largura depois. As caixas são retângulos que viram 4 segmentos de parede, então
a física não as distingue de qualquer outra parede — herdam o cone, o eco
perdido e a colisão de graça.

`World` aceita qualquer P: com 1 é você jogando, com 100 é a população treinando.
O simulador não sabe a diferença — e roda 100 robôs a ~2400 passos/s, folgado
para assistir ao treino em tempo real.

`World.telemetry()` devolve dado cru (distância, checkpoints, batidas, tempo,
chegou ou não) e **nenhum fitness**. O que conta como "bom" é decisão sua.

## Corrida: você contra a sua rede

```bash
.venv\Scripts\python.exe -m robo.corrida --rede modelos/v1.npz
```

```bash
.venv\Scripts\python.exe -m robo.corrida
```

Sem `--rede`, o adversário é uma **regra fixa** embutida (acelera e vira para o
lado mais livre) — serve de linha de base: se a sua rede não ganhar dela, ela
ainda não aprendeu nada.

Os dois largam lado a lado e apertam os mesmos 4 botões, com a mesma física, os
mesmos sensores e a mesma latência. Vence quem chega primeiro; se o tempo acabar,
vence quem avançou mais no traçado. `r` repete, `n` esconde o painel da rede.

Se o `.npz` trouxer a `SimConfig` do treino e ela não bater com o `config.json`
atual, **a do treino ganha**, com aviso no terminal. Uma rede treinada com 6
sensores não sabe o que fazer com 4 leituras, e correr assim mediria o robô errado.
