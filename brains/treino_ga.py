"""Treino por neuroevolução: hiperparâmetros, função de treino, e o laço de
muitas épocas (gerações) com checkpoint automático.

Isto é um exemplo **completo e funcional**, não um esqueleto para preencher.
Rode como está para ver uma população evoluindo; depois copie o arquivo e mude
o que quiser — a arquitetura, a fórmula de fitness, os hiperparâmetros. É a
continuação natural de `brains/exemplo.py`: aqui a "rede aleatória" ganha um
laço de evolução por cima.

    python -m brains.treino_ga
    python -m brains.treino_ga --populacao 200 --geracoes 500 --camadas 16,16
    python -m brains.treino_ga --continuar checkpoints/geracao_0100.ga.npz
    python -m brains.treino_ga --ver --ver-a-cada 5

Não precisa de tela para treinar — pygame só entra em cena se você pedir
`--ver`. Dá para rodar isto num servidor sem monitor.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from robo.config import SimConfig
from robo.persist import ATIVACOES, RedeSalva, salvar
from robo.pistas import PISTAS, carregar, catalogo, descrever
from robo.training import EpisodeRunner

from .exemplo import (MARCOS_PADRAO, PESOS, fitness_com_antienrolação,
                      fitness_exemplo)

PASTA_PADRAO = "checkpoints"
EXT_CHECKPOINT = ".ga.npz"      # estado completo da população, para retomar
EXT_MODELO = ".npz"             # um indivíduo só, formato de robo.persist (implantável)


# ======================================================================
# 1. HIPERPARÂMETROS — é aqui que você define os valores do treino
# ======================================================================
@dataclass
class Hiperparametros:
    """Tudo que controla *como* a evolução acontece, num lugar só.

    Isto não é a física do robô (isso é `SimConfig`, em `robo/config.py`) —
    é a receita do algoritmo de treino: quantos indivíduos, quantas gerações,
    o quanto cada geração muda em relação à anterior.
    """

    populacao: int = 200
    """Quantas redes competem ao mesmo tempo, em cada geração. Maior = explora
    mais opções por geração, mas cada geração custa mais para simular."""

    geracoes: int = 300
    """Quantas vezes a população evolui. Uma "época" aqui é uma geração
    inteira: avaliar todo mundo, escolher os melhores, gerar os próximos."""

    camadas_ocultas: tuple = (16, 8)
    """Tamanho de cada camada oculta da rede. `()` = sem camada oculta,
    entrada ligada direto na saída."""

    elite_frac: float = 0.1
    """Fração da população que passa para a próxima geração sem mudar nada.
    Garante que a geração nunca piora em relação à anterior."""

    torneio: int = 4
    """Quantos indivíduos disputam por vaga de pai. Maior = a seleção fica
    mais dura (só os bons têm chance real de reproduzir)."""

    taxa_mutacao: float = 0.2
    """Fração dos pesos de cada filho que sofre mutação. O resto vem exato de
    um dos dois pais (crossover)."""

    forca_mutacao: float = 0.15
    """Tamanho do passo aleatório aplicado a cada peso mutado (desvio padrão
    de um ruído gaussiano)."""

    checkpoint_a_cada: int = 3
    """De quantas em quantas gerações grava o estado completo em disco, para
    poder retomar depois de uma queda de luz ou um Ctrl+C."""

    seed: int = 0
    """Semente do sorteio. Mesma semente + mesmos hiperparâmetros = mesma
    evolução, sempre — útil para comparar duas mudanças de forma justa."""

    ver_a_cada: int = 1
    """De quantas em quantas gerações a janela aparece (com --ver ou
    --ver-populacao). Só tem efeito se você pedir para assistir.

    Este é o único campo aqui que não muda a evolução: é preferência de
    exibição, não parte do algoritmo. Duas consequências disso: mexer nele não
    altera o resultado do treino, e ao retomar um checkpoint vale o valor
    atual, não o que estava gravado — o resto dos hiperparâmetros vem do
    checkpoint, porque mudá-los no meio quebraria a continuidade."""


# ======================================================================
# 2. A REDE — P indivíduos avaliados juntos, cada um com os próprios pesos
# ======================================================================
class PopulacaoGA:
    """A população inteira como um único `Brain`.

    O truque que faz isto rápido: os pesos de cada indivíduo ficam num tensor
    `(P, saída, entrada)` por camada, e um `einsum` avalia a população inteira
    de uma vez — sem laço em Python por indivíduo. É o mesmo padrão de
    `RedeAleatoria` em `brains/exemplo.py`, só que agora com evolução.
    """

    def __init__(self, n_sensores, hiper: Hiperparametros, rng, pop_inicial=None,
                dispositivo="cpu"):
        self.n_sensores = n_sensores
        self.hiper = hiper
        self.rng = rng
        self.pop = pop_inicial if pop_inicial is not None else self._inicializar()
        self._ativ = None

        # A genética (elitismo, torneio, crossover, mutação) continua sempre em
        # numpy, na CPU — roda uma vez por geração, custa quase nada, e é onde
        # o determinismo do checkpoint/retomada foi verificado a fundo. Só a
        # passada da rede (`act_batch`), chamada centenas de vezes por geração,
        # é que opcionalmente vai para a GPU.
        self.dispositivo = dispositivo
        self._pop_torch = None
        if self.dispositivo != "cpu":
            self._atualizar_cache_torch()

    def _inicializar(self):
        tamanhos = [self.n_sensores, *self.hiper.camadas_ocultas, 4]
        pop = []
        for entrada, saida in zip(tamanhos, tamanhos[1:]):
            escala = np.sqrt(2.0 / (entrada + saida))       # Xavier
            W = self.rng.normal(0, escala, (self.hiper.populacao, saida, entrada))
            b = np.zeros((self.hiper.populacao, saida))
            pop.append((W, b))
        return pop

    # -- o contrato Brain --------------------------------------------- #
    def act_batch(self, obs):
        if self.dispositivo == "cpu":
            return self._act_batch_numpy(obs)
        return self._act_batch_torch(obs)

    def _act_batch_numpy(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        ativacoes = [x]
        ultima = len(self.pop) - 1
        for i, (W, b) in enumerate(self.pop):
            x = np.einsum("poi,pi->po", W, x) + b
            x = _sigmoide(x) if i == ultima else np.tanh(x)
            ativacoes.append(x)
        self._ativ = ativacoes
        return x

    def _atualizar_cache_torch(self):
        """Copia os pesos atuais (numpy) para tensores na GPU/CPU escolhida.

        Chamado ao criar a população, e de novo toda vez que `evoluir()` troca
        `self.pop` — o cache fica preso à geração atual, nunca avalia pesos
        velhos por engano.
        """
        import torch
        self._torch = torch
        self._pop_torch = [
            (torch.as_tensor(W, dtype=torch.float32, device=self.dispositivo),
             torch.as_tensor(b, dtype=torch.float32, device=self.dispositivo))
            for W, b in self.pop
        ]

    def _act_batch_torch(self, obs):
        """A mesma conta de `_act_batch_numpy`, em `torch.einsum`.

        Por que só a rede vai para a GPU, e não a física
        --------------------------------------------------
        Medido neste projeto: com 2000 robôs e rede pequena (12,12), o
        `World.step` (sensores + colisão + movimento) custa ~7,5 ms/passo
        contra ~0,9 ms/passo da rede — a física domina, não vale a viagem até a
        GPU. Mas com uma rede de (128,128), a rede sozinha já custa ~22 ms/passo
        em CPU pura, mais que a física inteira. É aí que a GPU ajuda de
        verdade: `torch.einsum` em lote usa cuBLAS, ordens de grandeza mais
        rápido que o laço do numpy para essas contas. A física fica na CPU de
        propósito — já é rápida o bastante, e portá-la exigiria reescrever o
        raycast inteiro (`robo/geometry.py`, `robo/sensors.py`) para um ganho
        que, medido, não é o gargalo hoje.
        """
        torch = self._torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.dispositivo)
            ativacoes = [x]
            ultima = len(self._pop_torch) - 1
            for i, (W, b) in enumerate(self._pop_torch):
                x = torch.einsum("poi,pi->po", W, x) + b
                x = torch.sigmoid(x) if i == ultima else torch.tanh(x)
                ativacoes.append(x)
            # devolve numpy sempre: o resto do simulador é CPU/numpy do início
            # ao fim e não sabe (nem precisa saber) que uma GPU foi usada aqui.
            self._ativ = [a.cpu().numpy() for a in ativacoes]
            return self._ativ[-1]

    def inspect(self, i=0):
        if self._ativ is None:
            return None
        return {"ativacoes": [a[i] for a in self._ativ],
                "pesos": [W[i] for W, _ in self.pop]}

    # -- extrair um indivíduo, para salvar ou visualizar sozinho -------- #
    def genoma(self, i):
        """As camadas de UM indivíduo, no formato que `robo.persist` espera."""
        return [(W[i].copy(), b[i].copy()) for W, b in self.pop]

    @property
    def ativacoes(self):
        return ["tanh"] * (len(self.pop) - 1) + ["sigmoid"]

    # -- evolução -------------------------------------------------------- #
    def evoluir(self, fitness: np.ndarray):
        """Uma geração: elitismo + seleção por torneio + crossover + mutação.

        `fitness`: um número por indivíduo, maior é melhor. É o que a SUA
        função de fitness devolveu — esta classe não sabe (nem precisa saber)
        como esse número foi calculado.
        """
        hiper = self.hiper
        P = hiper.populacao
        n_elite = max(1, int(P * hiper.elite_frac))
        ordem = np.argsort(fitness)[::-1]
        elite_idx = ordem[:n_elite]
        n_filhos = P - n_elite

        pais_a = self._torneio(fitness, n_filhos)
        pais_b = self._torneio(fitness, n_filhos)

        nova_pop = []
        for W, b in self.pop:
            WA, BA = W[pais_a], b[pais_a]
            WB, BB = W[pais_b], b[pais_b]

            # crossover uniforme: cada peso do filho vem de um dos dois pais
            mascara_w = self.rng.random(WA.shape) < 0.5
            mascara_b = self.rng.random(BA.shape) < 0.5
            filho_w = np.where(mascara_w, WA, WB)
            filho_b = np.where(mascara_b, BA, BB)

            # mutação: só uma fração dos pesos muda, e muda pouco
            mut_w = self.rng.random(filho_w.shape) < hiper.taxa_mutacao
            mut_b = self.rng.random(filho_b.shape) < hiper.taxa_mutacao
            filho_w = filho_w + mut_w * self.rng.normal(0, hiper.forca_mutacao, filho_w.shape)
            filho_b = filho_b + mut_b * self.rng.normal(0, hiper.forca_mutacao, filho_b.shape)

            novo_W = np.concatenate([W[elite_idx], filho_w], axis=0)
            novo_b = np.concatenate([b[elite_idx], filho_b], axis=0)
            nova_pop.append((novo_W, novo_b))

        self.pop = nova_pop
        if self.dispositivo != "cpu":
            self._atualizar_cache_torch()   # o cache da GPU não pode ficar com a geração antiga

    def _torneio(self, fitness, n):
        k = max(2, self.hiper.torneio)
        candidatos = self.rng.integers(0, self.hiper.populacao, size=(n, k))
        return candidatos[np.arange(n), np.argmax(fitness[candidatos], axis=1)]


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def escolher_pista(padrao="zigue-zague") -> str:
    """Menu de pistas antes do treino: as embutidas e as que você desenhou.

    Só aparece quando o terminal é interativo. Rodando por script, tarefa
    agendada ou dentro de um teste, cai no padrão sem travar esperando uma
    digitação que nunca vem.
    """
    itens = catalogo()
    if not sys.stdin.isatty():
        print(f"terminal não interativo; usando a pista padrão ({padrao}). "
              f"Passe --pista para escolher outra.")
        return padrao

    print("\nEscolha a pista de treino:\n")
    origem_atual = None
    for i, it in enumerate(itens, 1):
        if it["origem"] != origem_atual:
            origem_atual = it["origem"]
            titulo = "embutidas" if origem_atual == "embutida" else "suas (tracks/)"
            print(f"  {titulo}")
        if it["erro"]:
            print(f"   {i:2d}) {it['ref']:24s}  não deu para ler: {it['erro'][:40]}")
            continue
        marca = "  <- padrão" if it["ref"] == padrao else ""
        caixas = f"{it['caixas']:2d} caixas" if it["caixas"] else "sem caixas"
        print(f"   {i:2d}) {it['ref']:24s} {it['comprimento']:6.2f} m  {caixas}{marca}")

    validos = [it for it in itens if not it["erro"]]
    while True:
        try:
            resp = input(f"\nnúmero, nome/caminho, ou Enter para {padrao}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return padrao
        if not resp:
            return padrao
        if resp.isdigit():
            n = int(resp)
            if 1 <= n <= len(itens):
                escolhida = itens[n - 1]
                if escolhida["erro"]:
                    print("essa pista não pôde ser lida; escolha outra.")
                    continue
                return escolhida["ref"]
            print(f"número fora da lista (1 a {len(itens)}).")
            continue
        # aceita também digitar o nome ou o caminho direto
        if resp in PISTAS or any(it["ref"] == resp for it in validos):
            return resp
        if os.path.exists(resp):
            return resp
        print("não achei essa pista; use o número, o nome embutido ou um caminho existente.")


def avisar_troca_de_pista(pista_checkpoint: dict, track, pista_ref: str) -> None:
    """Compara a pista do checkpoint com a atual e avisa se mudou.

    Não bloqueia: treinar no fácil e migrar para o difícil é uma tática
    legítima. O que não pode é isso acontecer em silêncio e a queda no fitness
    parecer culpa da rede.
    """
    if not pista_checkpoint:
        print("aviso: este checkpoint é de uma versão anterior e não registrou "
              "a pista; não dá para conferir se é a mesma.")
        return

    atual = descrever(track, pista_ref)
    if atual["hash"] == pista_checkpoint.get("hash"):
        print(f"pista confere: {atual['ref']} ({atual['comprimento']:.2f} m)")
        return

    antes = pista_checkpoint
    if antes.get("ref") == atual["ref"]:
        print(f"ATENÇÃO: a pista '{atual['ref']}' foi EDITADA desde o treino "
              f"({antes.get('comprimento', 0):.2f} m -> {atual['comprimento']:.2f} m). "
              f"A rede foi treinada noutro percurso com o mesmo nome.")
    else:
        print(f"ATENÇÃO: mudando de pista — treinado em '{antes.get('ref')}' "
              f"({antes.get('comprimento', 0):.2f} m), continuando em "
              f"'{atual['ref']}' ({atual['comprimento']:.2f} m). "
              f"Uma queda no fitness aqui é esperada, não é falha da rede.")


def resolver_dispositivo(pedido: str) -> str:
    """'cpu'/'cuda'/'auto' -> 'cpu' ou 'cuda', nunca falha, sempre avisa por quê.

    GPU exige torch instalado com build CUDA — o padrão do `pip install torch`
    às vezes traz uma versão "+cpu" mesmo com placa de vídeo presente (foi o
    caso deste projeto; corrigido reinstalando de
    https://download.pytorch.org/whl/cu124). Por isso `--dispositivo cuda`
    pedido explicitamente e indisponível vira aviso, não erro: o treino
    continua rodando, só que na CPU.
    """
    if pedido == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if pedido == "cuda":
            print("torch não está instalado; caindo para CPU. "
                  "instale com: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return "cpu"
    if not torch.cuda.is_available():
        if pedido == "cuda":
            motivo = ("build de CPU (torch.version.cuda é None)" if torch.version.cuda is None
                      else "GPU não encontrada ou driver ausente")
            print(f"GPU pedida mas indisponível ({motivo}); caindo para CPU.")
        return "cpu"
    return "cuda"


# ======================================================================
# 3. FUNÇÃO DE FITNESS — troque esta função pela sua fórmula
# ======================================================================
# Reaproveitando o exemplo de brains/exemplo.py: avança ganha, demora perde,
# bater perde, chegar dá bônus, e enrolar perde. Se quiser outra coisa, escreva
# a sua função aqui — ela só precisa receber a telemetria do episódio (o
# dicionário que `EpisodeRunner.run()` devolve) e devolver um array (P,) de
# números, maior é melhor. Veja os campos disponíveis no GUIA.md, seção 6.
minha_fitness = fitness_com_antienrolação

MARCOS = MARCOS_PADRAO
"""Em que pontos do episódio o progresso é fotografado.

A regra anti-enrolação precisa saber onde o robô estava no meio do caminho, e
isso a telemetria final não conta. Se você trocar `minha_fitness` por uma
fórmula que não olha marcos, pode deixar `MARCOS = ()` — sem marcos o passo não
faz verificação nenhuma e não custa nada."""


# ======================================================================
# 4. CHECKPOINT — salvar/retomar o estado completo do treino
# ======================================================================
# Isto é diferente de `robo.persist`: aquele formato guarda UM indivíduo
# pronto para competir ou virar firmware. Este aqui guarda a POPULAÇÃO
# inteira, para você conseguir parar o treino (Ctrl+C, queda de luz, PC
# precisou reiniciar) e continuar exatamente de onde parou.
def salvar_checkpoint(caminho, populacao: PopulacaoGA, geracao: int,
                      melhor_fitness_global: float, pista_info: dict = None):
    dados = {
        "geracao": geracao,
        "melhor_fitness_global": melhor_fitness_global,
        "n_sensores": populacao.n_sensores,
        "hiper_json": json.dumps(asdict(populacao.hiper)),
        # em qual pista isto foi treinado. Sem esta marca, retomar um
        # checkpoint noutra pista continuaria em silêncio, e a queda no fitness
        # pareceria culpa da rede. Guarda também um hash da geometria: você
        # pode reeditar `tracks/pista1.json` e o nome continuar o mesmo,
        # enquanto o percurso virou outro.
        "pista_json": json.dumps(pista_info or {}),
        # sem isto, retomar recomeçaria a sequência aleatória do zero: a
        # promessa de "mesma semente = mesma evolução" só se sustenta se o
        # gerador continuar exatamente de onde parou, não se ele reinicia
        # com um novo ponto de partida a cada checkpoint.
        "rng_state_json": json.dumps(populacao.rng.bit_generator.state),
    }
    for i, (W, b) in enumerate(populacao.pop):
        dados[f"W{i}"] = W
        dados[f"b{i}"] = b
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    np.savez(caminho, **dados)


def carregar_checkpoint(caminho):
    d = np.load(caminho, allow_pickle=True)
    campos = json.loads(str(d["hiper_json"]))
    campos["camadas_ocultas"] = tuple(campos["camadas_ocultas"])   # JSON não tem tupla
    hiper = Hiperparametros(**campos)
    n_camadas = len(hiper.camadas_ocultas) + 1
    pop = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n_camadas)]

    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(str(d["rng_state_json"]))

    # checkpoints antigos não têm a marca da pista; devolve {} para quem
    # carregar saber que não dá para comparar, em vez de quebrar
    pista = json.loads(str(d["pista_json"])) if "pista_json" in d.files else {}

    return {
        "pop": pop, "hiper": hiper, "geracao": int(d["geracao"]),
        "melhor_fitness_global": float(d["melhor_fitness_global"]),
        "n_sensores": int(d["n_sensores"]), "rng": rng, "pista": pista,
    }


# ======================================================================
# 5. A FUNÇÃO DE TREINO — o laço de muitas épocas
# ======================================================================
def treinar(hiper: Hiperparametros, track, cfg: SimConfig, pasta=PASTA_PADRAO,
           geracao_inicial=0, pop_inicial=None, melhor_fitness_global=-np.inf,
           rng=None, dispositivo="cpu", ver=False, ver_a_cada=None, ver_populacao=False,
           ver_fracao=1.0, pista_ref=None):
    os.makedirs(pasta, exist_ok=True)
    pista_info = descrever(track, pista_ref)
    log_path = os.path.join(pasta, "historico.csv")
    novo_log = not os.path.exists(log_path)

    # a frequência de exibição mora nos hiperparâmetros; o argumento só existe
    # para quem chama `treinar()` direto e quer sobrescrever pontualmente
    if ver_a_cada is None:
        ver_a_cada = hiper.ver_a_cada

    # `rng` só vem preenchido ao retomar de um checkpoint (com o estado exato
    # de onde a sequência aleatória parou). Num treino novo, nasce da semente.
    if rng is None:
        rng = np.random.default_rng(hiper.seed)
    populacao = PopulacaoGA(cfg.n_sensors, hiper, rng, pop_inicial=pop_inicial,
                            dispositivo=dispositivo)
    runner = EpisodeRunner(track, cfg, n_robots=hiper.populacao, marcos=MARCOS)

    # o gráfico ao vivo lê esta lista enquanto o treino a preenche. Ela também
    # sobrevive à retomada: as gerações anteriores são lidas do historico.csv,
    # senão o gráfico começaria do zero a cada `--continuar`.
    historico = _historico_do_csv(log_path, ate=geracao_inicial)

    demo_runner, viewer = None, None
    if ver_populacao:
        # a população inteira, ao vivo, dentro do próprio treino — não é uma
        # demonstração à parte: é o `runner` de verdade, então mostrar ali tem
        # um custo real (a tela agora faz parte do laço quente). Vale a pena
        # quando você quer assistir; desligue para treinar o mais rápido
        # possível.
        from robo.viewer import Viewer
        viewer = Viewer(track, cfg, brain=populacao,
                        titulo=f"Treino — população de {hiper.populacao}",
                        historico=historico, hiper=hiper, pesos=PESOS,
                        fracao_visivel=ver_fracao)
        print(f"assistindo a população inteira a cada {ver_a_cada} gerações. "
              f"As gerações com janela rodam na velocidade da tela (bem mais "
              f"lentas que as outras) — aumente --ver-a-cada para treinar mais rápido.")
        if ver_fracao < 1.0:
            print(f"desenhando {ver_fracao:.0%} da população por vez (tecla V troca ao vivo); "
                  f"a simulação continua com todo mundo, só o desenho é mais leve.")
    elif ver:
        from robo.training import EpisodeRunner as _Runner
        from robo.viewer import Viewer
        demo_runner = _Runner(track, cfg, n_robots=1, verificar_contrato=False)
        viewer = Viewer(track, cfg, brain=None, titulo="Treino — melhor da geração",
                        historico=historico, hiper=hiper, pesos=PESOS)

    with open(log_path, "a", encoding="utf-8") as log:
        if novo_log:
            log.write("geracao,melhor,media,pior,chegaram,bateram,tempo_s\n")

        t_inicio = time.perf_counter()
        for geracao in range(geracao_inicial, hiper.geracoes):
            t0 = time.perf_counter()
            # desenhar toda geração prenderia o treino inteiro à velocidade da
            # tela; só liga a janela nas gerações de exibição, igual ao modo
            # "--ver" de mostrar só a campeã — a diferença é que aqui é a
            # população inteira, dentro do runner de verdade, não uma demo à parte.
            mostrar_agora = ver_populacao and (geracao % ver_a_cada == 0
                                               or geracao == hiper.geracoes - 1)
            dados = runner.run(populacao, seed=hiper.seed * 100_000 + geracao,
                               on_step=viewer.on_step if mostrar_agora else None)
            fitness = minha_fitness(dados)

            idx_melhor = int(np.argmax(fitness))
            chegaram = int(dados["finished"].sum())
            bateram = int(dados["collided"].sum())
            dt = time.perf_counter() - t0

            # a geração com janela roda na velocidade da tela (o Viewer limita a
            # ~20 fps), então ela demora muito mais que as outras — marcar isso
            # evita a impressão de que o treino travou
            marca = " [na tela]" if mostrar_agora else ""
            print(f"geração {geracao:4d} | melhor {fitness[idx_melhor]:8.1f} | "
                  f"média {fitness.mean():8.1f} | pior {fitness.min():8.1f} | "
                  f"chegaram {chegaram:3d}/{hiper.populacao} | "
                  f"bateram {bateram:3d}/{hiper.populacao} | {dt:5.2f} s{marca}")
            log.write(f"{geracao},{fitness[idx_melhor]:.2f},{fitness.mean():.2f},"
                      f"{fitness.min():.2f},{chegaram},{bateram},{dt:.2f}\n")
            log.flush()

            # alimenta o gráfico ao vivo. `historico` é a mesma lista que o
            # viewer segura, então basta acrescentar aqui.
            historico.append({
                "geracao": geracao,
                "melhor": float(fitness[idx_melhor]),
                "media": float(fitness.mean()),
                "pior": float(fitness.min()),
                "chegaram": chegaram,
                "bateram": bateram,
            })

            if fitness[idx_melhor] > melhor_fitness_global:
                melhor_fitness_global = float(fitness[idx_melhor])
                salvar(os.path.join(pasta, "melhor" + EXT_MODELO),
                      populacao.genoma(idx_melhor), populacao.ativacoes, cfg,
                      geracao=geracao, fitness=melhor_fitness_global)

            if ver and (geracao % ver_a_cada == 0 or geracao == hiper.geracoes - 1):
                _mostrar_campeao(demo_runner, viewer, populacao, idx_melhor,
                                 fitness[idx_melhor], cfg, geracao)

            # o checkpoint precisa da população JÁ EVOLUÍDA: ele guarda a
            # geração que ainda não rodou, pronta para a próxima chamada.
            # Salvar antes de evoluir faria o retomar reavaliar a mesma
            # geração com o número trocado, em vez de continuar de fato.
            populacao.evoluir(fitness)

            if (geracao + 1) % hiper.checkpoint_a_cada == 0 or geracao == hiper.geracoes - 1:
                salvar_checkpoint(
                    os.path.join(pasta, f"geracao_{geracao + 1:04d}{EXT_CHECKPOINT}"),
                    populacao, geracao, melhor_fitness_global, pista_info)

        total = time.perf_counter() - t_inicio
        print(f"\nfim: {hiper.geracoes - geracao_inicial} gerações em {total:.1f} s "
              f"| melhor fitness alcançado: {melhor_fitness_global:.1f}")
        print(f"melhor rede em {os.path.join(pasta, 'melhor' + EXT_MODELO)}")

    if viewer is not None:
        viewer.close()

    return populacao, melhor_fitness_global


def _historico_do_csv(caminho, ate=None):
    """Lê o historico.csv de volta, para o gráfico não recomeçar do zero.

    Sem isto, retomar um treino mostraria um gráfico com uma geração só,
    escondendo justamente a parte que interessa: a curva inteira até aqui.
    Um CSV ausente ou com linha estragada não é erro — devolve o que der.
    """
    historico = []
    if not os.path.exists(caminho):
        return historico
    with open(caminho, encoding="utf-8") as f:
        next(f, None)                       # cabeçalho
        for linha in f:
            partes = linha.strip().split(",")
            if len(partes) < 6:
                continue
            try:
                g = int(partes[0])
                if ate is not None and g >= ate:
                    continue
                historico.append({"geracao": g, "melhor": float(partes[1]),
                                  "media": float(partes[2]), "pior": float(partes[3]),
                                  "chegaram": int(partes[4]), "bateram": int(partes[5])})
            except ValueError:
                continue
    return historico


def _mostrar_campeao(demo_runner, viewer, populacao, idx_melhor, fit, cfg, geracao):
    """Roda o melhor indivíduo sozinho numa janela, para você acompanhar."""
    campeao = RedeSalva(populacao.genoma(idx_melhor), populacao.ativacoes, cfg)
    viewer.brain = campeao
    import pygame
    pygame.display.set_caption(f"Treino — geração {geracao}, fitness {fit:.1f}")
    demo_runner.run(campeao, seed=geracao, on_step=viewer.on_step)


# ======================================================================
# linha de comando
# ======================================================================
def main(argv=None):
    # Os padrões da linha de comando saem da própria classe `Hiperparametros`,
    # e não de números repetidos aqui. Se ficassem duplicados, editar a classe
    # (que é o que o topo deste arquivo manda fazer) não teria efeito nenhum ao
    # rodar pelo terminal ou pelo VSCode — o argparse sobrescreveria em silêncio.
    padrao = Hiperparametros()

    p = argparse.ArgumentParser(
        description="Treino por neuroevolução",
        # mostra no --help o valor efetivo de cada opção, que sai da classe
        # Hiperparametros: assim dá para conferir o que está valendo sem
        # precisar abrir o arquivo
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pista", default=None,
                   help=f"embutida ({', '.join(PISTAS)}) ou caminho de um JSON. "
                        f"Sem isto, abre um menu para escolher (ou usa a padrão, "
                        f"se o terminal não for interativo).")
    p.add_argument("--listar-pistas", action="store_true",
                   help="mostra as pistas disponíveis e sai")
    p.add_argument("--populacao", type=int, default=padrao.populacao)
    p.add_argument("--geracoes", type=int, default=padrao.geracoes)
    p.add_argument("--camadas", default=",".join(str(c) for c in padrao.camadas_ocultas),
                   help="tamanho das camadas ocultas, separadas por vírgula; vazio = sem oculta")
    p.add_argument("--elite", type=float, default=padrao.elite_frac)
    p.add_argument("--torneio", type=int, default=padrao.torneio)
    p.add_argument("--mutacao", type=float, default=padrao.taxa_mutacao,
                   help="taxa de mutação")
    p.add_argument("--forca", type=float, default=padrao.forca_mutacao,
                   help="força da mutação")
    p.add_argument("--checkpoint-a-cada", type=int, default=padrao.checkpoint_a_cada)
    p.add_argument("--pasta", default=PASTA_PADRAO)
    p.add_argument("--seed", type=int, default=padrao.seed)
    p.add_argument("--continuar", default=None,
                   help="caminho de um checkpoint .ga.npz para retomar o treino")
    p.add_argument("--ver", action="store_true", help="mostra o campeão numa janela periodicamente")
    p.add_argument("--ver-populacao", action="store_true",
                   help="mostra a população inteira treinando, não só a campeã")
    p.add_argument("--ver-a-cada", type=int, default=padrao.ver_a_cada,
                   help="de quantas em quantas gerações abre a janela (com --ver ou --ver-populacao)")
    p.add_argument("--ver-fracao", type=float, default=1.0,
                   help="fração da população desenhada de cada vez, com --ver-populacao "
                        "(ex: 0.1 = 10%%). A simulação continua com todo mundo, só o "
                        "desenho fica mais leve. A tecla V troca isso ao vivo, na janela.")
    p.add_argument("--dispositivo", choices=["cpu", "cuda", "auto"], default="cpu",
                   help="onde roda a passada da rede. 'auto' usa GPU se achar uma disponível. "
                        "Só vale a pena com redes grandes (--camadas 64,64 ou mais) — "
                        "veja o GUIA.md, seção sobre GPU.")
    args = p.parse_args(argv)

    if args.listar_pistas:
        for it in catalogo():
            estado = it["erro"] or f"{it['comprimento']:6.2f} m  {it['caixas']} caixas"
            print(f"{it['origem']:9s} {it['ref']:26s} {estado}")
        return 0

    cfg = SimConfig.carregar_ou_padrao()
    # bater atrapalha (a fitness já desconta em `punicao_batida`), mas não
    # elimina: o robô fica livre para recuar, virar e retomar o traçado, em
    # vez de morrer no primeiro toque. Quem não se recupera ainda sai pelo
    # corte de `parado_limite_s` (5 s sem avançar), então o episódio não fica
    # preso a alguém enroscado na parede.
    cfg.collision_ends_episode = False

    # o checkpoint vem primeiro porque ele sabe em que pista treinou: retomar
    # sem dizer nada deve continuar no mesmo percurso, não voltar para o padrão
    estado = None
    if args.continuar:
        try:
            estado = carregar_checkpoint(args.continuar)
        except FileNotFoundError:
            print(f"checkpoint não encontrado: {args.continuar}")
            return 1

    if args.pista:
        pista_ref = args.pista
    elif estado is not None:
        pista_ref = estado["pista"].get("ref") or "zigue-zague"
    else:
        pista_ref = escolher_pista()

    try:
        track = carregar(pista_ref)
    except (FileNotFoundError, KeyError) as e:
        print(f"não consegui carregar a pista: {e}")
        if estado is not None:
            print("a pista do checkpoint sumiu ou foi renomeada; "
                  "passe --pista com o caminho atual.")
        return 1
    print(f"pista: {pista_ref} ({track.length:.2f} m, "
          f"{len(track.obstacles)} caixas)")

    dispositivo = resolver_dispositivo(args.dispositivo)
    if dispositivo == "cuda":
        import torch
        print(f"rede rodando na GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("rede rodando na CPU")

    geracao_inicial, pop_inicial, melhor_global, rng_retomado = 0, None, -np.inf, None

    if estado is not None:
        avisar_troca_de_pista(estado["pista"], track, pista_ref)
        if estado["n_sensores"] != cfg.n_sensors:
            print(f"o checkpoint foi treinado com {estado['n_sensores']} sensores, "
                  f"e a config atual tem {cfg.n_sensors}. Ajuste a calibragem "
                  f"(tecla P no jogo) antes de continuar.")
            return 1
        hiper = estado["hiper"]           # os hiperparâmetros do checkpoint mandam,
        hiper.geracoes = args.geracoes    # exceto o alvo de gerações, que você pode esticar,
        hiper.ver_a_cada = args.ver_a_cada  # e a frequência de exibição, que não afeta o treino
        geracao_inicial = estado["geracao"] + 1
        pop_inicial = estado["pop"]
        melhor_global = estado["melhor_fitness_global"]
        rng_retomado = estado["rng"]
        if geracao_inicial >= hiper.geracoes:
            print(f"este checkpoint já chegou à geração {geracao_inicial - 1}; "
                  f"--geracoes {hiper.geracoes} não pede mais nada. "
                  f"Aumente --geracoes para continuar treinando.")
            return 0
        print(f"retomando do checkpoint: geração {geracao_inicial}, "
              f"melhor fitness até aqui {melhor_global:.1f}")
    else:
        hiper = Hiperparametros(
            populacao=args.populacao, geracoes=args.geracoes,
            camadas_ocultas=tuple(int(c) for c in args.camadas.split(",") if c.strip()),
            elite_frac=args.elite, torneio=args.torneio,
            taxa_mutacao=args.mutacao, forca_mutacao=args.forca,
            checkpoint_a_cada=args.checkpoint_a_cada, seed=args.seed,
            ver_a_cada=args.ver_a_cada,
        )

    treinar(hiper, track, cfg, pasta=args.pasta, geracao_inicial=geracao_inicial,
           pop_inicial=pop_inicial, melhor_fitness_global=melhor_global,
           rng=rng_retomado, dispositivo=dispositivo, ver=args.ver,
           ver_populacao=args.ver_populacao, ver_fracao=args.ver_fracao, pista_ref=pista_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
