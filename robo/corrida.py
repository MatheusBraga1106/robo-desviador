"""Corrida: você contra a sua rede, na mesma pista.

    python -m robo.corrida --rede modelos/v1.npz
    python -m robo.corrida --rede modelos/v1.npz --pista tracks/minha.json
    python -m robo.corrida                        # contra o adversário embutido

Os dois robôs largam lado a lado e apertam os **mesmos 4 botões** — você pelo
teclado, a rede pela saída dela. Não há atalho para nenhum dos dois: mesma
física, mesmos sensores, mesma latência. É essa a graça de comparar.

Vence quem chega primeiro. Se o tempo acabar, vence quem avançou mais no traçado.
"""

import argparse
import sys

import numpy as np
import pygame

from .brain import as_buttons
from .config import SimConfig
from .game import abrir_janela
from .netviz import NetPanel
from .physics import keys_to_buttons
from .pistas import PISTAS, carregar
from .render import Camera, Renderer
from .world import World

VOCE, IA = 0, 1
COR_VOCE = (60, 220, 190)
COR_IA = (245, 150, 60)
COR_TEXTO = (232, 236, 245)
COR_FRACO = (141, 151, 171)
CONTAGEM_S = 3.0


class Piloto:
    """Adversário embutido, para a corrida funcionar antes de você ter uma rede.

    É uma regra fixa, não uma rede neural: vai para frente e vira para o lado
    mais livre quando algo aparece na frente. Serve de linha de base — se a sua
    rede não ganhar dela, ela ainda não aprendeu nada.
    """

    nome = "regra fixa"

    def __init__(self, limiar=0.35):
        self.limiar = limiar

    def act_batch(self, obs):
        obs = np.atleast_2d(obs)
        n = obs.shape[1]
        meio = n // 2
        esquerda = obs[:, meio:].min(axis=1)      # sensores de ângulo positivo
        direita = obs[:, :n - meio].min(axis=1)
        perto = obs.min(axis=1) < self.limiar

        saida = np.zeros((len(obs), 4))
        saida[:, 0] = 1.0                                    # acelerar sempre
        saida[:, 2] = (perto & (esquerda >= direita)).astype(float)
        saida[:, 3] = (perto & (esquerda < direita)).astype(float)
        return saida


class Corrida:
    def __init__(self, track, cfg: SimConfig, brain, tamanho=(1180, 760),
                 mostrar_rede=True):
        self.track = track
        self.cfg = cfg
        self.brain = brain
        self.tamanho = tamanho

        self.tela = abrir_janela(tamanho, f"Corrida — {track.name}")
        self.render = Renderer(self.tela)
        self.cam = Camera(tamanho)
        self.cam.fit(track.bounds)
        self.relogio = pygame.time.Clock()
        self.painel = (NetPanel(pygame.Rect(12, 12, 280, min(tamanho[1] - 24, 420)))
                       if mostrar_rede else None)

        self.world = World(track, cfg, n_robots=2)
        self.reiniciar()

    # ------------------------------------------------------------------ #
    def reiniciar(self):
        self.world.reset()
        self._separar_largada()
        self.contagem = CONTAGEM_S
        self.vencedor = None
        self.tempo = np.zeros(2)

    def _separar_largada(self):
        """Afasta os dois lados a lado, para não largarem um dentro do outro."""
        s = self.track.start
        lateral = np.array([-np.sin(s[2]), np.cos(s[2])])
        for i, sinal in ((VOCE, +1), (IA, -1)):
            tentativa = self.world.pos[i] + lateral * sinal * self.cfg.robot.radius * 1.6
            # só desloca se couber: em corredor apertado, sobrepor é melhor que
            # nascer dentro da parede
            if self.track.distance_to_walls(tentativa[None, :])[0] > self.cfg.robot.radius:
                self.world.pos[i] = tentativa
        self.world.reconfigure_sensors()

    # ------------------------------------------------------------------ #
    def loop(self):
        rodando = True
        while rodando:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    rodando = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        rodando = False
                    elif ev.key == pygame.K_r:
                        self.reiniciar()
                    elif ev.key == pygame.K_n and self.painel is not None:
                        self.painel = None
            self.passo()
            self.desenhar()
            self.relogio.tick(int(round(1.0 / self.cfg.dt)))
        pygame.quit()

    def passo(self):
        if self.vencedor is not None:
            return
        if self.contagem > 0:
            self.contagem -= self.cfg.dt
            self.world.step(np.zeros((2, 4)))     # ninguém anda antes do sinal
            return

        obs = self.world.observation()
        botoes = np.zeros((2, 4))
        botoes[VOCE] = self._teclado()
        botoes[IA] = np.asarray(self.brain.act_batch(obs[IA:IA + 1])).reshape(4)

        self.world.step(botoes)
        ativo = self.world.alive & ~self.world.finished
        self.tempo += ativo * self.cfg.dt
        self._julgar()

    @staticmethod
    def _teclado():
        k = pygame.key.get_pressed()
        return keys_to_buttons(
            up=k[pygame.K_UP] or k[pygame.K_w],
            down=k[pygame.K_DOWN] or k[pygame.K_s],
            left=k[pygame.K_LEFT] or k[pygame.K_a],
            right=k[pygame.K_RIGHT] or k[pygame.K_d],
        )[0]

    def _julgar(self):
        fim = self.world.finished
        if fim.any():
            # os dois no mesmo passo: desempata o tempo, e só então é empate
            if fim.all():
                self.vencedor = (VOCE if self.tempo[VOCE] < self.tempo[IA]
                                 else IA if self.tempo[IA] < self.tempo[VOCE] else -1)
            else:
                self.vencedor = int(np.argmax(fim))
        elif self.world.steps >= self.cfg.max_steps or not self.world.alive.any():
            avanco = self.track.progress_along(self.world.pos)
            self.vencedor = (int(np.argmax(avanco))
                             if abs(avanco[0] - avanco[1]) > 0.05 else -1)

    # ------------------------------------------------------------------ #
    def desenhar(self):
        r, cam = self.render, self.cam
        r.fundo()
        r.desenhar_pista(self.track, cam)
        self._robos()
        if self.painel is not None:
            self.painel.draw(self.tela, self._inspecionar())
        self._placar()
        self._aviso()
        pygame.display.flip()

    def _robos(self):
        raio = self.cam.px(self.cfg.robot.radius)
        for i, cor in ((IA, COR_IA), (VOCE, COR_VOCE)):
            p = self.cam.to_screen(self.world.pos[i])
            centro = (int(p[0]), int(p[1]))
            vivo = self.world.alive[i]
            pygame.draw.circle(self.tela, cor if vivo else (90, 94, 106), centro, raio)
            pygame.draw.circle(self.tela, (12, 20, 24), centro, raio, 2)
            th = self.world.theta[i]
            frente = self.cam.to_screen(
                self.world.pos[i] + np.array([np.cos(th), np.sin(th)])
                * self.cfg.robot.radius * 1.7)
            pygame.draw.line(self.tela, (255, 255, 255), centro, frente, 2)

    def _inspecionar(self):
        if not hasattr(self.brain, "inspect"):
            return None
        try:
            return self.brain.inspect(0)
        except Exception:
            return None

    def _placar(self):
        avanco = self.track.progress_along(self.world.pos)
        total = max(self.track.length, 1e-9)
        f, fp = self.render.font, self.render.font_peq

        larg, alt = 300, 116
        caixa = pygame.Surface((larg, alt), pygame.SRCALPHA)
        caixa.fill((10, 13, 22, 220))
        pygame.draw.rect(caixa, (57, 64, 79), caixa.get_rect(), 1, border_radius=6)

        nome_ia = getattr(self.brain, "nome", "sua rede")
        for k, (rotulo, cor, i) in enumerate((("você", COR_VOCE, VOCE),
                                              (nome_ia, COR_IA, IA))):
            y = 12 + k * 46
            caixa.blit(f.render(rotulo[:18], True, cor), (14, y))
            estado = ("chegou" if self.world.finished[i]
                      else "bateu" if not self.world.alive[i] else "")
            info = f"{avanco[i]:4.1f} m   {self.tempo[i]:5.1f} s   {estado}"
            caixa.blit(fp.render(info, True, COR_TEXTO), (14, y + 18))

            trilha = pygame.Rect(14, y + 34, larg - 28, 5)
            pygame.draw.rect(caixa, (34, 40, 56), trilha, border_radius=3)
            cheio = int(trilha.w * np.clip(avanco[i] / total, 0, 1))
            if cheio:
                pygame.draw.rect(caixa, cor, (trilha.x, trilha.y, cheio, 5),
                                 border_radius=3)

        self.tela.blit(caixa, (self.tamanho[0] - larg - 12, 12))
        dica = fp.render("r reinicia · n esconde a rede · esc sai", True, COR_FRACO)
        self.tela.blit(dica, (self.tamanho[0] - larg - 12, alt + 18))

    def _aviso(self):
        if self.contagem > 0:
            n = int(np.ceil(self.contagem))
            self.render.aviso(f"{n}...", COR_TEXTO)
        elif self.vencedor == VOCE:
            self.render.aviso(f"VOCÊ VENCEU  —  {self.tempo[VOCE]:.1f} s  ·  R repete",
                              COR_VOCE)
        elif self.vencedor == IA:
            self.render.aviso(f"A REDE VENCEU  —  {self.tempo[IA]:.1f} s  ·  R repete",
                              COR_IA)
        elif self.vencedor == -1:
            self.render.aviso("EMPATE  —  R repete", COR_FRACO)


# --------------------------------------------------------------------- #
def carregar_rede(caminho, cfg):
    """Carrega a rede salva. Devolve (brain, cfg) — a config pode mudar.

    Se o arquivo trouxe a `SimConfig` do treino e ela não bate com a atual, a do
    treino ganha: rede treinada com 4 sensores a 90° não sabe o que fazer com 6
    leituras a 150°, e correr assim mediria o robô errado.
    """
    from .persist import RedeSalva

    brain = RedeSalva.carregar(caminho)
    brain.nome = caminho.split("/")[-1].split("\\")[-1]

    if brain.cfg is not None and brain.cfg.n_sensors != cfg.n_sensors:
        print(f"aviso: a rede foi treinada com {brain.cfg.n_sensors} sensores e a "
              f"config atual tem {cfg.n_sensors}. Usando a do treino.")
        return brain, brain.cfg
    if brain.n_sensores != cfg.n_sensors:
        raise ValueError(f"a rede espera {brain.n_sensores} sensores, "
                         f"mas a config tem {cfg.n_sensors}")
    return brain, cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="Corrida: você contra a sua rede")
    p.add_argument("--rede", default=None, help="arquivo .npz salvo com robo.persist")
    p.add_argument("--ultima", action="store_true",
                   help="corre contra a rede treinada mais recente, sem digitar o caminho")
    p.add_argument("--pista", default="zigue-zague",
                   help=f"embutida ({', '.join(PISTAS)}) ou caminho de um JSON")
    p.add_argument("--sem-painel", action="store_true", help="não desenhar a rede")
    args = p.parse_args(argv)

    cfg = SimConfig.carregar_ou_padrao()
    cfg.collision_ends_episode = False      # bater atrapalha, mas não elimina
    # numa corrida ninguém é eliminado por parar: os cortes de treino existem
    # para não travar o episódio numa rede que gira em falso, e aqui há um
    # humano do outro lado
    cfg.parado_limite_s = 0.0
    cfg.corte_progresso_em = 0.0

    try:
        track = carregar(args.pista)
    except (FileNotFoundError, KeyError) as e:
        print(f"não consegui carregar a pista: {e}")
        return 1

    caminho_rede = args.rede
    if args.ultima and not caminho_rede:
        from .persist import rede_mais_recente
        caminho_rede = rede_mais_recente()
        if caminho_rede is None:
            print("nenhuma rede treinada encontrada; correndo contra a regra fixa.")
    if caminho_rede:
        try:
            brain, cfg = carregar_rede(caminho_rede, cfg)
        except (FileNotFoundError, ValueError) as e:
            print(f"não consegui carregar a rede: {e}")
            return 1
    else:
        brain = Piloto()
        print("sem --rede: correndo contra o adversário de regra fixa embutido")

    Corrida(track, cfg, brain, mostrar_rede=not args.sem_painel).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
