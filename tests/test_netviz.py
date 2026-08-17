"""Verificações do painel da rede.

O painel consome o que a *sua* rede devolver, então o que mais importa aqui é
que ele nunca derrube o treino: `inspect` ausente, quebrado ou torto tem que
apenas não desenhar.

    python -m tests.test_netviz
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys

import numpy as np
import pygame

from robo.config import SimConfig
from robo.netviz import LIMITE_CONEXOES, NetPanel
from robo.pistas import zigue_zague
from robo.training import EpisodeRunner
from robo.viewer import Viewer
from brains.exemplo import RedeAleatoria

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def rede_pronta(ocultas=(6, 5), n=3, sensores=4, seed=0):
    b = RedeAleatoria(sensores, n, ocultas=ocultas, seed=seed)
    b.act_batch(np.random.default_rng(0).random((n, sensores)))
    return b


def pintou(surface) -> bool:
    """Alguma coisa foi desenhada na área do painel?"""
    return pygame.surfarray.array3d(surface)[:320, :460].sum() > 0


# ---------------------------------------------------------------------- #
def test_desenha_a_rede():
    print("\ndesenhar a rede")
    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    painel = NetPanel(pygame.Rect(10, 10, 300, 440))
    brain = rede_pronta()
    dados = brain.inspect(0)

    checar("inspect tem o formato do contrato",
           len(dados["ativacoes"]) == 4 and len(dados["pesos"]) == 3,
           f"{[len(a) for a in dados['ativacoes']]}")
    painel.draw(tela, dados)
    checar("pintou alguma coisa", pintou(tela))


def test_inspect_invalido_nao_derruba():
    print("\ninspect quebrado só não desenha")
    painel = NetPanel(pygame.Rect(10, 10, 300, 440))

    casos = {
        "None": None,
        "dicionário vazio": {},
        "sem pesos": {"ativacoes": [np.zeros(4), np.zeros(3)]},
        "sem ativações": {"pesos": [np.zeros((3, 4))]},
        "contagem torta": {"ativacoes": [np.zeros(4), np.zeros(3), np.zeros(4)],
                           "pesos": [np.zeros((3, 4))]},
    }
    for nome, dados in casos.items():
        tela = pygame.Surface((900, 600))
        tela.fill((0, 0, 0))
        try:
            painel.draw(tela, dados)
            ok, detalhe = not pintou(tela), "não desenhou, como esperado"
        except Exception as e:
            ok, detalhe = False, f"levantou {type(e).__name__}: {e}"
        checar(f"{nome}", ok, detalhe)


def test_rede_gigante_e_podada():
    print("\nrede grande poda as conexões")
    brain = rede_pronta(ocultas=(64, 64))
    dados = brain.inspect(0)
    total = sum(w.size for w in dados["pesos"])
    checar("o caso realmente passa do limite", total > LIMITE_CONEXOES,
           f"{total} conexões contra limite de {LIMITE_CONEXOES}")

    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    painel = NetPanel(pygame.Rect(10, 10, 300, 440))
    import time
    t0 = time.perf_counter()
    painel.draw(tela, dados)
    ms = (time.perf_counter() - t0) * 1000
    checar("desenha rápido o bastante para 20 fps", ms < 50, f"{ms:.1f} ms")
    checar("e pintou", pintou(tela))


def test_uma_camada_so():
    print("\nrede sem camada oculta")
    brain = rede_pronta(ocultas=())
    dados = brain.inspect(0)
    checar("entrada direto na saída", len(dados["ativacoes"]) == 2)
    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    NetPanel(pygame.Rect(10, 10, 300, 440)).draw(tela, dados)
    checar("desenha assim mesmo", pintou(tela))


def test_viewer_sobrevive_a_rede_sem_inspect():
    print("\nviewer com rede que não implementa inspect")

    class SemInspect:
        def act_batch(self, obs):
            s = np.zeros((len(obs), 4)); s[:, 0] = 1
            return s

    class InspectQuebrado(SemInspect):
        def inspect(self, i=0):
            raise RuntimeError("estourei de propósito")

    track = zigue_zague()
    cfg = SimConfig()
    for nome, brain in [("sem inspect", SemInspect()), ("inspect que estoura",
                                                        InspectQuebrado())]:
        try:
            v = Viewer(track, cfg, brain=brain, tamanho=(800, 520))
            v.fps = 100000
            runner = EpisodeRunner(track, cfg, n_robots=3)
            runner.run(brain, seed=1, on_step=v.on_step, max_steps=15)
            ok, detalhe = True, "episódio rodou inteiro"
        except Exception as e:
            ok, detalhe = False, f"{type(e).__name__}: {e}"
        checar(nome, ok, detalhe)


def test_viewer_desenha_o_episodio():
    print("\nviewer no laço do runner")
    track = zigue_zague()
    cfg = SimConfig()
    n = 20
    brain = RedeAleatoria(cfg.n_sensors, n, seed=2)
    v = Viewer(track, cfg, brain=brain, tamanho=(1000, 640))
    v.fps = 100000
    runner = EpisodeRunner(track, cfg, n_robots=n)
    dados = runner.run(brain, seed=5, on_step=v.on_step, max_steps=40)

    checar("rodou os passos pedidos", dados["steps"] == 40, f"{dados['steps']}")
    checar("desenhou o painel junto", pintou(v.tela))


def main():
    pygame.init()
    for fn in [test_desenha_a_rede, test_inspect_invalido_nao_derruba,
               test_rede_gigante_e_podada, test_uma_camada_so,
               test_viewer_sobrevive_a_rede_sem_inspect, test_viewer_desenha_o_episodio]:
        fn()
    pygame.quit()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
