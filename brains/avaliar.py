"""Testa uma rede treinada em pistas que ela nunca viu.

    python -m brains.avaliar --rede checkpoints/melhor.npz
    python -m brains.avaliar --rede checkpoints/melhor.npz --treinada-em curva-u
    python -m brains.avaliar --rede checkpoints/melhor.npz --ver

Que tipo de decoreba é possível aqui
------------------------------------
A rede deste projeto vê **só as distâncias dos sensores** e é uma MLP sem
memória: nada de posição, contador de passos ou estado interno. Ela não tem como
decorar "no passo 300 vire à esquerda" — a informação simplesmente não chega
até ela.

O que ela consegue é aprender um **reflexo enviesado**. "Na dúvida, vire à
esquerda" resolve uma pista cujas curvas são todas à esquerda, e desmonta em
qualquer outra. É overfitting de verdade, só que na forma que esta arquitetura
permite.

Por isso o teste mais afiado aqui é a **pista espelhada**: mesmo comprimento,
mesma dificuldade, curvas invertidas. Quem vai bem no original e mal no espelho
decorou o lado, não aprendeu a desviar.

O segundo teste é a **largada sorteada**: se a rede só funciona partindo do
ponto exato de sempre, ela se apegou à pose inicial em vez da leitura dos
sensores.
"""

import argparse
import os
import sys

import numpy as np

from robo.config import SimConfig
from robo.persist import RedeSalva, rede_mais_recente
from robo.pistas import PISTAS, carregar, catalogo
from robo.training import EpisodeRunner

JITTER_PADRAO = (0.12, 0.35)     # metros, radianos


def avaliar_em(track, brain, cfg, n=24, jitter=JITTER_PADRAO, seed=0, ver=False):
    """Roda a rede `n` vezes na pista, com largadas levemente diferentes.

    Devolve um resumo. As largadas sorteadas existem porque a simulação é
    determinística: rodar a mesma rede na mesma pose daria sempre o mesmo
    resultado, e um único número não diz se aquilo foi robusto ou sorte.
    """
    runner = EpisodeRunner(track, cfg, n_robots=n, verificar_contrato=False,
                           jitter_largada=jitter)
    viewer, on_step = (None, None)
    if ver:
        viewer, on_step = _abrir_janela(track, cfg, brain, f"Avaliando — {track.name}")
    try:
        d = runner.run(brain, seed=seed, on_step=on_step)
    finally:
        if viewer is not None:
            viewer.close()
    comprimento = max(d["track_length"], 1e-9)
    return {
        "pista": track.name,
        "chegou": float(d["finished"].mean()),
        "bateu": float(d["collided"].mean()),
        "desistiu": float(d["desistiu"].mean()),
        "avanco": float((d["progress"] / comprimento).mean()),
        "avanco_melhor": float((d["progress"] / comprimento).max()),
    }


def _abrir_janela(track, cfg, brain, titulo):
    """Janela para acompanhar a avaliação. Devolve (viewer, on_step)."""
    from robo.viewer import Viewer
    viewer = Viewer(track, cfg, brain=brain, titulo=titulo)
    return viewer, viewer.on_step


def linha_resumo(r):
    return (f"{r['pista'][:26]:26s} "
            f"chegou {r['chegou']:5.0%}  "
            f"avanço médio {r['avanco']:5.0%}  melhor {r['avanco_melhor']:5.0%}  "
            f"bateu {r['bateu']:4.0%}  parou {r['desistiu']:4.0%}")


def veredito_largada(exato, sorteado):
    """Compara a pose exata com largadas levemente diferentes.

    É a checagem mais rápida de decoreba, e a que passa despercebida com mais
    facilidade: a simulação é determinística, então avaliar sempre da mesma pose
    devolve o mesmo número bonito e esconde que a política não tolera 10 cm de
    diferença. Um robô real nunca larga duas vezes do mesmo ponto.
    """
    if not exato or not sorteado:
        return []
    queda = exato["chegou"] - sorteado["chegou"]
    if exato["chegou"] > 0.5 and queda > 0.35:
        return [f"FRÁGIL À LARGADA: chega {exato['chegou']:.0%} saindo da pose exata, "
                f"mas só {sorteado['chegou']:.0%} com a largada variando alguns "
                f"centímetros. A política depende do ponto de partida, não do que "
                f"os sensores mostram — no robô real isso não se sustenta."]
    if exato["chegou"] > 0.5:
        return [f"aguenta variação na largada ({sorteado['chegou']:.0%} contra "
                f"{exato['chegou']:.0%} saindo do ponto exato)."]
    return []


def veredito(treino, espelho, outras):
    """Interpreta os números — a parte que evita conclusão errada.

    Comparar só "foi bem" ou "foi mal" esconde o caso interessante: ir bem no
    treino e mal no espelho é assinatura de viés de curva, não de incompetência.
    """
    linhas = []
    if treino and espelho:
        queda = treino["avanco"] - espelho["avanco"]
        if treino["avanco"] > 0.5 and queda > 0.25:
            linhas.append(
                f"DECOROU O LADO: avança {treino['avanco']:.0%} no original e só "
                f"{espelho['avanco']:.0%} no espelho. As curvas são as mesmas, "
                f"invertidas — ela aprendeu 'vire para tal lado', não a desviar.")
        elif treino["avanco"] > 0.5:
            linhas.append(
                f"passa no espelho ({espelho['avanco']:.0%} contra "
                f"{treino['avanco']:.0%}): não é viés de curva.")

    if outras:
        media = float(np.mean([o["avanco"] for o in outras]))
        melhor = max(outras, key=lambda o: o["avanco"])
        pior = min(outras, key=lambda o: o["avanco"])
        linhas.append(f"em pistas novas, avança {media:.0%} em média "
                      f"(melhor: {melhor['pista']} {melhor['avanco']:.0%}; "
                      f"pior: {pior['pista']} {pior['avanco']:.0%}).")
        if treino and treino["avanco"] - media > 0.35:
            linhas.append(
                f"ESPECIALIZOU: {treino['avanco']:.0%} na pista de treino contra "
                f"{media:.0%} nas outras. Treine em mais de uma pista, ou com "
                f"largada sorteada, para forçar uma política mais geral.")
        elif media > 0.5:
            linhas.append("generaliza razoavelmente: o desempenho não desaba fora "
                          "da pista de treino.")
    return linhas


# --------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Avalia uma rede treinada em várias pistas")
    p.add_argument("--rede", default=None,
                   help="arquivo .npz salvo com robo.persist. "
                        "Sem isto, usa a rede treinada mais recente.")
    p.add_argument("--treinada-em", default=None,
                   help="pista usada no treino, para comparar com as demais")
    p.add_argument("--pistas", default=None,
                   help="lista separada por vírgula; por padrão, todas as disponíveis")
    p.add_argument("--repeticoes", type=int, default=24,
                   help="largadas sorteadas por pista")
    p.add_argument("--sem-jitter", action="store_true",
                   help="larga todo mundo na pose exata (sem variação)")
    p.add_argument("--ver", action="store_true", help="mostra cada pista numa janela")
    args = p.parse_args(argv)

    cfg = SimConfig.carregar_ou_padrao()
    caminho = args.rede or rede_mais_recente()
    if caminho is None:
        print("nenhuma rede treinada encontrada. Rode um treino primeiro "
              "(python -m brains.treino_ga) ou passe --rede com o caminho.")
        return 1
    try:
        brain = RedeSalva.carregar(caminho)
    except FileNotFoundError:
        print(f"rede não encontrada: {caminho}")
        return 1

    if brain.cfg is not None and brain.cfg.n_sensors != cfg.n_sensors:
        print(f"aviso: a rede foi treinada com {brain.cfg.n_sensors} sensores e a "
              f"config atual tem {cfg.n_sensors}. Usando a do treino.")
        cfg = brain.cfg
    elif brain.n_sensores != cfg.n_sensors:
        print(f"a rede espera {brain.n_sensores} sensores, mas a config tem "
              f"{cfg.n_sensors}. Ajuste a calibragem antes de avaliar.")
        return 1

    if args.pistas:
        refs = [r.strip() for r in args.pistas.split(",") if r.strip()]
    else:
        refs = [it["ref"] for it in catalogo() if not it["erro"]]

    jitter = None if args.sem_jitter else JITTER_PADRAO
    print(f"rede: {caminho}")
    print(f"{args.repeticoes} largadas por pista"
          f"{'' if jitter else ' (pose exata, sem variação)'}\n")

    treino = espelho = exato = None
    outras = []

    if args.treinada_em:
        try:
            t = carregar(args.treinada_em)
        except (FileNotFoundError, KeyError) as e:
            print(f"não consegui carregar a pista de treino: {e}")
            return 1
        treino = avaliar_em(t, brain, cfg, args.repeticoes, jitter, ver=args.ver)
        espelho = avaliar_em(t.espelhar(), brain, cfg, args.repeticoes, jitter, ver=args.ver)
        # a pose exata roda de graça e é a referência que revela fragilidade:
        # sem ela, uma queda por causa da largada parece falta de habilidade
        exato = (avaliar_em(t, brain, cfg, 1, None) if jitter else None)  # sempre sem janela
        if exato:
            print("  [ exata  ]  " + linha_resumo(exato))
        print("  [ treino ]  " + linha_resumo(treino))
        print("  [ espelho]  " + linha_resumo(espelho))
        print()
        refs = [r for r in refs if r != args.treinada_em]

    for ref in refs:
        try:
            t = carregar(ref)
        except Exception as e:
            print(f"  pulei {ref}: {e}")
            continue
        r = avaliar_em(t, brain, cfg, args.repeticoes, jitter, ver=args.ver)
        outras.append(r)
        print("  [ nova   ]  " + linha_resumo(r))

    print()
    for linha in veredito_largada(exato, treino) + veredito(treino, espelho, outras):
        print("  " + linha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
