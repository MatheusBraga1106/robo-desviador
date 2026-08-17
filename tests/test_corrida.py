"""Verificações da corrida.

    python -m tests.test_corrida
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import tempfile

import numpy as np
import pygame

from robo.config import SimConfig
from robo.corrida import IA, VOCE, Corrida, Piloto, carregar_rede
from robo.persist import salvar
from robo.pistas import zigue_zague
from robo.track import Track

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def nova_corrida(brain=None, track=None, cfg=None):
    cfg = cfg or SimConfig()
    cfg.collision_ends_episode = False
    c = Corrida(track or zigue_zague(), cfg, brain or Piloto(),
                tamanho=(900, 600), mostrar_rede=False)
    return c


def correr(c, passos, teclas=(0, 0, 0, 0)):
    """Roda passos com o teclado fixo, driblando o pygame.key global."""
    import robo.corrida as mod
    original = mod.Corrida._teclado
    mod.Corrida._teclado = staticmethod(lambda: np.array(teclas, dtype=float))
    try:
        for _ in range(passos):
            c.passo()
    finally:
        mod.Corrida._teclado = original


# ---------------------------------------------------------------------- #
def test_largam_separados_e_parados():
    print("\nlargada")
    c = nova_corrida()
    d = float(np.hypot(*(c.world.pos[VOCE] - c.world.pos[IA])))
    checar("os dois não largam um dentro do outro", d > c.cfg.robot.radius,
           f"{d:.3f} m de distância")
    checar("ambos dentro da pista",
           (c.track.distance_to_walls(c.world.pos) >= c.cfg.robot.radius).all())

    antes = c.world.pos.copy()
    correr(c, 10, teclas=(1, 0, 0, 0))
    checar("ninguém anda durante a contagem", np.allclose(c.world.pos, antes),
           f"contagem em {c.contagem:.2f} s")


def test_corredor_apertado_nao_empurra_para_parede():
    print("\nlargada em corredor apertado")
    estreita = Track.from_centerline([(0.5, 0.5), (4.0, 0.5)], width=0.32)
    c = nova_corrida(track=estreita)
    checar("ninguém nasce dentro da parede",
           (c.track.distance_to_walls(c.world.pos) >= c.cfg.robot.radius).all(),
           f"distâncias {np.round(c.track.distance_to_walls(c.world.pos), 3)}")


def test_a_rede_dirige_sozinha():
    print("\na rede dirige o robô dela")
    c = nova_corrida()
    correr(c, int(3.0 / c.cfg.dt) + 40, teclas=(0, 0, 0, 0))   # você fica parado
    checar("a IA saiu do lugar", c.world.distance[IA] > 0.2,
           f"{c.world.distance[IA]:.2f} m")
    checar("você continua parado", c.world.distance[VOCE] < 1e-9,
           f"{c.world.distance[VOCE]:.3f} m")


def test_placar_conta_o_tempo_de_quem_corre():
    print("\ncronômetro")
    c = nova_corrida()
    correr(c, int(3.0 / c.cfg.dt) + 60, teclas=(1, 0, 0, 0))
    checar("o tempo anda depois da contagem", c.tempo[VOCE] > 0.5,
           f"{c.tempo[VOCE]:.2f} s")
    checar("os dois medidos no mesmo relógio",
           abs(c.tempo[VOCE] - c.tempo[IA]) < 1e-9)


def test_vence_quem_chega():
    print("\nvitória por chegada")
    track = zigue_zague()
    # o objetivo precisa estar fora do alcance da largada, senão os dois nascem
    # dentro dele, chegam no passo zero e o resultado é sempre empate
    track.goal = np.array([track.start[0] + 1.0, track.start[1], 0.3])
    partida = float(np.hypot(*(track.goal[:2] - track.start[:2])))
    checar("o objetivo exige dirigir até ele", partida > track.goal[2] + 0.2,
           f"{partida:.2f} m de distância, raio {track.goal[2]:.2f} m")

    class Parado:
        def act_batch(self, obs):
            return np.zeros((len(np.atleast_2d(obs)), 4))

    c = nova_corrida(track=track)
    c.brain = Parado()
    correr(c, int(3.0 / c.cfg.dt) + 90, teclas=(1, 0, 0, 0))
    checar("quem andou venceu", c.vencedor == VOCE, f"vencedor {c.vencedor}")

    c2 = nova_corrida(track=track)
    correr(c2, int(3.0 / c2.cfg.dt) + 90, teclas=(0, 0, 0, 0))
    checar("parado, você perde para a rede", c2.vencedor == IA, f"vencedor {c2.vencedor}")


def test_tempo_esgotado_decide_por_avanco():
    print("\ntempo esgotado")
    cfg = SimConfig()
    cfg.collision_ends_episode = False
    cfg.max_steps = int(3.0 / cfg.dt) + 40
    c = nova_corrida(cfg=cfg)
    correr(c, cfg.max_steps + 5, teclas=(0, 0, 0, 0))
    checar("alguém foi declarado", c.vencedor is not None, f"{c.vencedor}")
    avanco = c.track.progress_along(c.world.pos)
    checar("venceu quem avançou mais",
           c.vencedor == int(np.argmax(avanco)) or c.vencedor == -1,
           f"avanços {np.round(avanco, 2)}, vencedor {c.vencedor}")


def test_reiniciar():
    print("\nreiniciar")
    c = nova_corrida()
    correr(c, int(3.0 / c.cfg.dt) + 50, teclas=(1, 0, 0, 0))
    c.reiniciar()
    checar("zera a contagem", abs(c.contagem - 3.0) < 1e-9)
    checar("zera o cronômetro", np.allclose(c.tempo, 0))
    checar("zera o vencedor", c.vencedor is None)
    checar("volta para a largada separada",
           float(np.hypot(*(c.world.pos[VOCE] - c.world.pos[IA]))) > c.cfg.robot.radius)


def test_carregar_rede_com_config_diferente():
    print("\nrede salva com outra montagem de sensor")
    caminho = os.path.join(tempfile.gettempdir(), "corrida_rede.npz")
    treino = SimConfig()
    treino.sensor.count = 6
    rng = np.random.default_rng(0)
    camadas = [(rng.normal(0, 0.3, (4, 6)), np.zeros(4))]
    salvar(caminho, camadas, ["sigmoid"], treino)

    atual = SimConfig()          # 4 sensores
    brain, cfg = carregar_rede(caminho, atual)
    os.remove(caminho)
    checar("a config do treino prevalece", cfg.n_sensors == 6, f"{cfg.n_sensors}")
    checar("e a rede aceita essa entrada",
           brain.act_batch(np.zeros((1, 6))).shape == (1, 4))

    c = nova_corrida(brain=brain, cfg=cfg)
    correr(c, int(3.0 / cfg.dt) + 20, teclas=(0, 0, 0, 0))
    checar("a corrida roda com a config do treino", np.isfinite(c.world.pos).all())


def test_rede_incompativel_sem_config():
    print("\nrede sem config e com tamanho errado")
    caminho = os.path.join(tempfile.gettempdir(), "corrida_ruim.npz")
    camadas = [(np.zeros((4, 9)), np.zeros(4))]
    salvar(caminho, camadas, ["sigmoid"])          # sem cfg
    try:
        carregar_rede(caminho, SimConfig())
        ok, detalhe = False, "não reclamou"
    except ValueError as e:
        ok, detalhe = "espera 9 sensores" in str(e), f"disse: {str(e)[:60]}"
    os.remove(caminho)
    checar("recusa com o motivo claro", ok, detalhe)


def test_piloto_desvia():
    print("\nadversário de regra fixa")
    piloto = Piloto()
    livre = piloto.act_batch(np.ones((1, 4)))
    checar("caminho livre, só acelera", livre[0, 0] == 1 and livre[0, 2] == 0
           and livre[0, 3] == 0, f"{livre[0]}")

    # obstáculo à direita (sensores de índice baixo) -> tem que virar à esquerda
    obs = np.array([[0.1, 0.2, 1.0, 1.0]])
    acao = piloto.act_batch(obs)
    checar("obstáculo à direita, vira à esquerda", acao[0, 2] == 1 and acao[0, 3] == 0,
           f"{acao[0]}")
    acao = piloto.act_batch(obs[:, ::-1])
    checar("obstáculo à esquerda, vira à direita", acao[0, 3] == 1 and acao[0, 2] == 0,
           f"{acao[0]}")


def test_desenha():
    print("\ndesenhar a corrida")
    c = nova_corrida()
    c.painel = None
    try:
        c.desenhar()
        correr(c, int(3.0 / c.cfg.dt) + 30, teclas=(1, 0, 0, 0))
        c.desenhar()
        pintou = pygame.surfarray.array3d(c.tela).sum() > 0
        ok, detalhe = pintou, "pintou" if pintou else "tela vazia"
    except Exception as e:
        ok, detalhe = False, f"{type(e).__name__}: {e}"
    checar("desenha antes e depois da largada", ok, detalhe)


def main():
    pygame.init()
    for fn in [test_largam_separados_e_parados,
               test_corredor_apertado_nao_empurra_para_parede,
               test_a_rede_dirige_sozinha, test_placar_conta_o_tempo_de_quem_corre,
               test_vence_quem_chega, test_tempo_esgotado_decide_por_avanco,
               test_reiniciar, test_carregar_rede_com_config_diferente,
               test_rede_incompativel_sem_config, test_piloto_desvia, test_desenha]:
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
