"""Modo de jogar: você dirige o robô pelo teclado.

    python -m robo.game
    python -m robo.game --pista caracol
    python -m robo.game --pista tracks/minha.json

Setas ou WASD dirigem. As mesmas 4 entradas que a rede vai produzir — nada aqui
tem atalho que a rede não tenha.
"""

import argparse
import sys

import numpy as np
import pygame

from .config import SimConfig
from .physics import keys_to_buttons
from .calibra import PainelCalibragem
from .pistas import PISTAS, carregar
from .render import Camera, Renderer
from .world import World

AJUDA = [
    "setas / wasd  dirigir",
    "r             reiniciar",
    "c             câmera: seguir ou pista inteira",
    "p             calibrar motores e sensores",
    "esc           sair",
]


def abrir_janela(tamanho=(1180, 760), titulo="Robô desviador"):
    pygame.init()
    pygame.display.set_caption(titulo)
    return pygame.display.set_mode(tamanho)


def jogar(track, cfg: SimConfig, tamanho=(1180, 760), zoom=190.0):
    """Abre a janela, joga, e fecha tudo no fim. Uso avulso."""
    tela = abrir_janela(tamanho, f"Robô desviador — {track.name}")
    rodar(track, cfg, tela, zoom)
    pygame.quit()


def rodar(track, cfg: SimConfig, tela, zoom=190.0):
    """Roda o laço do jogo numa janela que já existe, e devolve o controle ao sair.

    Separado do `jogar` porque o editor entra aqui com a pista que você está
    desenhando e precisa **voltar** para a edição no Esc — se esta função
    chamasse `pygame.quit()`, levaria o editor junto.
    """
    tamanho = tela.get_size()
    relogio = pygame.time.Clock()

    world = World(track, cfg, n_robots=1)
    render = Renderer(tela)
    cam = Camera(tamanho)
    painel = PainelCalibragem(cfg, tamanho, render.font, render.font_peq)
    seguindo = True

    fps = int(round(1.0 / cfg.dt))
    rodando = True
    while rodando:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                rodando = False
                continue
            # o painel tem prioridade: com ele aberto, "s" grava em vez de dirigir
            if painel.evento(ev):
                continue
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    rodando = False
                elif ev.key == pygame.K_r:
                    world.reset()
                elif ev.key == pygame.K_c:
                    seguindo = not seguindo
                elif ev.key == pygame.K_p:
                    painel.alternar()

        if painel.consumir_remontagem():
            world.reconfigure_sensors()

        k = pygame.key.get_pressed()
        # com o painel aberto, WASD viraria atalho de calibragem sem querer;
        # as setas continuam dirigindo para você sentir o efeito do ajuste
        wasd = not painel.aberto
        botoes = keys_to_buttons(
            up=k[pygame.K_UP] or (wasd and k[pygame.K_w]),
            down=k[pygame.K_DOWN] or (wasd and k[pygame.K_s]),
            left=k[pygame.K_LEFT] or (wasd and k[pygame.K_a]),
            right=k[pygame.K_RIGHT] or (wasd and k[pygame.K_d]),
        )
        world.step(botoes)

        if seguindo:
            cam.follow(world.pos[0], zoom)
        else:
            cam.fit(track.bounds)

        render.fundo()
        render.desenhar_pista(track, cam)
        render.desenhar_robos(world, cam, detalhe_em=0)
        render.hud(world, extra_linhas=AJUDA)
        painel.desenhar(tela)

        if world.finished[0]:
            render.aviso(f"CHEGOU!  {world.distance[0]:.2f} m em "
                         f"{world.steps_alive[0] * cfg.dt:.1f} s  —  R para repetir")
        elif not world.alive[0]:
            render.aviso("BATEU  —  R para repetir", (250, 120, 120))

        pygame.display.flip()
        relogio.tick(fps)


def main(argv=None):
    p = argparse.ArgumentParser(description="Jogo do robô desviador de obstáculos")
    p.add_argument("--pista", default="zigue-zague",
                   help=f"embutida ({', '.join(PISTAS)}) ou caminho de um JSON")
    p.add_argument("--sensores", type=int, default=None, help="quantidade de HC-SR04")
    p.add_argument("--zoom", type=float, default=190.0, help="pixels por metro")
    p.add_argument("--fim-ao-bater", action="store_true",
                   help="bater encerra a tentativa (padrão: dá para dar ré e continuar)")
    args = p.parse_args(argv)

    cfg = SimConfig.carregar_ou_padrao()
    cfg.collision_ends_episode = args.fim_ao_bater
    # os cortes por falta de progresso são regra de treino, para o episódio não
    # ficar preso a uma rede que gira em falso. Jogando, parar para pensar (ou
    # manobrar devagar) é legítimo — eliminar o jogador por isso seria absurdo.
    cfg.parado_limite_s = 0.0
    cfg.corte_progresso_em = 0.0
    if args.sensores:
        cfg.sensor.count = args.sensores

    try:
        track = carregar(args.pista)
    except (FileNotFoundError, KeyError) as e:
        print(f"não consegui carregar a pista: {e}")
        return 1

    jogar(track, cfg, zoom=args.zoom)
    return 0


if __name__ == "__main__":
    sys.exit(main())
