"""Verificações da avaliação de generalização (brains/avaliar.py).

O que importa aqui é o veredito não mentir: dizer "generaliza" para uma rede
especializada, ou acusar decoreba onde não há, é pior que não ter a ferramenta.

    python -m tests.test_avaliar
"""

import sys

import numpy as np

from brains.avaliar import avaliar_em, veredito, veredito_largada
from robo.config import SimConfig
from robo.pistas import carregar, curva_u, zigue_zague

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


# ---------------------------------------------------------------- espelho
def test_espelhar_preserva_a_dificuldade():
    print("\npista espelhada")
    t = curva_u()
    e = t.espelhar()

    checar("mesmo comprimento", abs(t.length - e.length) < 1e-9,
           f"{t.length:.2f} contra {e.length:.2f} m")
    checar("mesmo número de paredes", len(t.walls) == len(e.walls))
    checar("mesma largura de corredor", abs(t.width - e.width) < 1e-9)
    checar("espelhar duas vezes volta ao original",
           np.allclose(t.centerline, e.espelhar().centerline))
    checar("o ângulo de largada inverte",
           abs(e.start[2] + t.start[2]) < 1e-9, f"{t.start[2]:.2f} -> {e.start[2]:.2f}")
    checar("o nome avisa que é espelhada", "espelhada" in e.name, e.name)


def test_espelhar_com_caixas():
    print("\nespelhar leva as caixas junto")
    t = curva_u()
    t.add_obstacle(2.0, 0.4, 0.3, 0.15, angulo=0.6)
    e = t.espelhar()
    checar("a caixa veio", len(e.obstacles) == 1)
    checar("o giro da caixa inverte",
           abs(e.obstacles[0, 4] + t.obstacles[0, 4]) < 1e-9,
           f"{t.obstacles[0,4]:.2f} -> {e.obstacles[0,4]:.2f}")
    checar("as medidas não mudam",
           np.allclose(e.obstacles[0, 2:4], t.obstacles[0, 2:4]))


def test_espelhar_pista_sem_tracado():
    print("\nespelhar pista feita de paredes soltas")
    from robo.track import Track
    t = Track(extra_walls=[[0, 0, 2, 0], [0, 1, 2, 1]], start=[0.5, 0.5, 0.0],
              goal=[1.8, 0.5, 0.2])
    e = t.espelhar()
    checar("não quebra sem linha central", len(e.walls) == len(t.walls))
    checar("as paredes foram refletidas", not np.allclose(e.walls, t.walls))


# ---------------------------------------------------------------- jitter
class Reto:
    def act_batch(self, obs):
        s = np.zeros((len(obs), 4))
        s[:, 0] = 1.0
        return s


def test_jitter_espalha_a_largada():
    print("\nlargada sorteada")
    from robo.world import World

    cfg = SimConfig()
    track = zigue_zague()

    sem = World(track, cfg, 12, jitter_largada=None)
    checar("sem jitter, todos largam no mesmo ponto",
           np.allclose(sem.pos, sem.pos[0]) and np.allclose(sem.theta, sem.theta[0]))

    com = World(track, cfg, 12, rng=np.random.default_rng(1),
                jitter_largada=(0.12, 0.35))
    checar("com jitter, as poses diferem", not np.allclose(com.pos, com.pos[0]))
    checar("ninguém nasce dentro da parede",
           bool((track.distance_to_walls(com.pos) >= cfg.robot.radius).all()),
           f"mínimo {track.distance_to_walls(com.pos).min():.3f} m")
    checar("o deslocamento respeita o limite pedido",
           float(np.abs(com.pos - track.start[:2]).max()) <= 0.12 + 1e-9,
           f"{np.abs(com.pos - track.start[:2]).max():.3f} m")


def test_avaliar_em_devolve_fracoes():
    print("\navaliar_em")
    cfg = SimConfig()
    r = avaliar_em(zigue_zague(), Reto(), cfg, n=8, seed=1)
    for campo in ("chegou", "bateu", "desistiu", "avanco", "avanco_melhor"):
        checar(f"{campo} é uma fração entre 0 e 1", 0.0 <= r[campo] <= 1.0,
               f"{r[campo]:.2f}")
    checar("o melhor é >= a média", r["avanco_melhor"] >= r["avanco"] - 1e-9)
    checar("andar reto no zigue-zague bate", r["bateu"] > 0.5, f"{r['bateu']:.0%}")


# ---------------------------------------------------------------- vereditos
def _r(chegou=0.0, avanco=0.0, pista="x"):
    return {"pista": pista, "chegou": chegou, "avanco": avanco,
            "avanco_melhor": avanco, "bateu": 0.0, "desistiu": 0.0}


def test_veredito_decorou_o_lado():
    print("\nveredito: viés de curva")
    bom = _r(chegou=1.0, avanco=0.95, pista="treino")
    ruim = _r(chegou=0.0, avanco=0.30, pista="espelho")

    texto = " ".join(veredito(bom, ruim, []))
    checar("acusa quando cai muito no espelho", "DECOROU O LADO" in texto, texto[:60])

    igual = _r(chegou=0.9, avanco=0.90, pista="espelho")
    texto = " ".join(veredito(bom, igual, []))
    checar("não acusa quando vai bem nos dois",
           "DECOROU" not in texto and "não é viés" in texto, texto[:60])

    fraco = _r(chegou=0.0, avanco=0.10, pista="treino")
    texto = " ".join(veredito(fraco, ruim, []))
    checar("não fala de viés se nem na pista de treino foi bem",
           "DECOROU" not in texto, texto[:60])


def test_veredito_especializou():
    print("\nveredito: especialização")
    bom = _r(chegou=1.0, avanco=0.94, pista="treino")
    outras_ruins = [_r(avanco=0.30, pista="a"), _r(avanco=0.40, pista="b")]
    texto = " ".join(veredito(bom, None, outras_ruins))
    checar("acusa quando desaba fora da pista de treino",
           "ESPECIALIZOU" in texto, texto[:70])

    outras_boas = [_r(avanco=0.80, pista="a"), _r(avanco=0.75, pista="b")]
    texto = " ".join(veredito(bom, None, outras_boas))
    checar("elogia quando se segura fora dela",
           "generaliza" in texto and "ESPECIALIZOU" not in texto, texto[:70])


def test_veredito_largada():
    print("\nveredito: fragilidade à largada")
    exato = _r(chegou=1.0, avanco=0.94)
    sorteado = _r(chegou=0.08, avanco=0.26)
    texto = " ".join(veredito_largada(exato, sorteado))
    checar("acusa quando só funciona da pose exata",
           "FRÁGIL À LARGADA" in texto, texto[:70])

    robusto = _r(chegou=0.85, avanco=0.90)
    texto = " ".join(veredito_largada(exato, robusto))
    checar("aprova quando aguenta variação",
           "aguenta" in texto and "FRÁGIL" not in texto, texto[:70])

    checar("sem dados, não inventa veredito", veredito_largada(None, sorteado) == [])


def test_achar_a_rede_mais_recente():
    print("\nachar a última rede treinada")
    import os
    import shutil
    import tempfile
    import time

    from robo.persist import rede_mais_recente, salvar

    base = os.path.join(tempfile.gettempdir(), "achar_rede")
    shutil.rmtree(base, ignore_errors=True)
    a = os.path.join(base, "a")
    b = os.path.join(base, "b")
    os.makedirs(a); os.makedirs(b)

    checar("sem nenhuma rede, devolve None", rede_mais_recente([a, b]) is None)

    camadas = [(np.zeros((4, 4)), np.zeros(4))]
    salvar(os.path.join(a, "velha.npz"), camadas, ["sigmoid"])
    time.sleep(0.05)
    salvar(os.path.join(b, "melhor.npz"), camadas, ["sigmoid"])

    achado = rede_mais_recente([a, b])
    checar("pega a mais recente, não a primeira encontrada",
           achado.endswith("melhor.npz"), achado)

    # o .ga.npz é estado de população, não uma rede: carregá-lo daria erro obscuro
    np.savez(os.path.join(b, "geracao_0099.ga.npz"), qualquer=np.zeros(3))
    achado = rede_mais_recente([a, b])
    checar("ignora checkpoint de população (.ga.npz)",
           achado.endswith("melhor.npz"), achado)

    checar("pasta inexistente não quebra",
           rede_mais_recente([os.path.join(base, "nao_existe")]) is None)

    shutil.rmtree(base, ignore_errors=True)


def main():
    for fn in [test_espelhar_preserva_a_dificuldade, test_espelhar_com_caixas,
               test_espelhar_pista_sem_tracado, test_jitter_espalha_a_largada,
               test_avaliar_em_devolve_fracoes, test_veredito_decorou_o_lado,
               test_veredito_especializou, test_veredito_largada,
               test_achar_a_rede_mais_recente]:
        fn()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
