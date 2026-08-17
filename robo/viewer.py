"""Janela para assistir a população treinando.

Feito para entrar direto no `EpisodeRunner` sem que ele saiba o que é uma tela::

    viewer = Viewer(track, cfg, brain=minha_rede)
    dados = runner.run(minha_rede, on_step=viewer.on_step)
    viewer.close()

Fechar a janela ou apertar Esc faz o `on_step` devolver False, e o runner
encerra o episódio — nenhum dos dois lados precisa conhecer o outro.
"""

import numpy as np
import pygame

from .config import SimConfig
from .game import abrir_janela
from .grafico import PainelGrafico, PainelPesos
from .netviz import NetPanel
from .render import Camera, Renderer

LARGURA_PAINEL_REDE = 300


class Viewer:
    def __init__(self, track, cfg: SimConfig = None, brain=None,
                 tamanho=(1180, 760), titulo="Treinando", fps=None,
                 mostrar_rede=True, detalhe_em=0, historico=None, hiper=None, pesos=None):
        self.track = track
        self.cfg = cfg or SimConfig()
        self.brain = brain
        self.tamanho = tamanho
        self.detalhe_em = detalhe_em

        # `historico` é a MESMA lista que o laço de treino vai preenchendo — o
        # viewer só lê. Guardar a referência (em vez de uma cópia) é o que faz
        # o gráfico acompanhar as gerações sem ninguém precisar avisar ninguém.
        self.historico = historico
        self.hiper = hiper
        self.pesos = pesos

        self.tela = abrir_janela(tamanho, titulo)
        self.render = Renderer(self.tela)
        self.cam = Camera(tamanho)
        self.cam.fit(track.bounds)
        self.relogio = pygame.time.Clock()
        self.fps = fps or int(round(1.0 / self.cfg.dt))

        altura = min(tamanho[1] - 24, 480)
        self.painel = (NetPanel(pygame.Rect(12, 12, LARGURA_PAINEL_REDE, altura))
                       if mostrar_rede else None)

        larg_g, alt_g = 420, 240
        self.grafico = PainelGrafico(
            pygame.Rect(12, tamanho[1] - alt_g - 12, larg_g, alt_g),
            self.render.font, self.render.font_peq)
        self.mostrar_grafico = historico is not None

        # gerações em que os pesos mudaram, para o gráfico marcar. Sem isso
        # você compararia gerações medidas com réguas diferentes sem perceber.
        self.marcas_pesos = []
        self.painel_pesos = PainelPesos(
            pygame.Rect(tamanho[0] - 292, tamanho[1] - 200, 280, 190),
            pesos, self.render.font, self.render.font_peq,
            ao_mudar=self._anotar_mudanca_de_pesos)

        self.turbo = False
        self.fechou = False

    # ------------------------------------------------------------------ #
    def on_step(self, world, passo) -> bool:
        """Passe isto como `on_step` do runner. False encerra o episódio."""
        if not self._eventos():
            self.fechou = True
            return False
        if self.turbo and passo % 12:
            return True          # em turbo, desenha só de vez em quando

        self.desenhar(world, passo)
        if not self.turbo:
            self.relogio.tick(self.fps)
        return True

    def _anotar_mudanca_de_pesos(self):
        g = len(self.historico) if self.historico is not None else 0
        if not self.marcas_pesos or self.marcas_pesos[-1] != g:
            self.marcas_pesos.append(g)

    def _eventos(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if self.painel_pesos.evento(ev):
                continue
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_SPACE:
                    self.turbo = not self.turbo
                if ev.key == pygame.K_r:
                    self.painel = None if self.painel else \
                        NetPanel(pygame.Rect(12, 12, LARGURA_PAINEL_REDE,
                                             min(self.tamanho[1] - 24, 480)))
                if ev.key == pygame.K_g and self.historico is not None:
                    self.mostrar_grafico = not self.mostrar_grafico
                if ev.key == pygame.K_p and self.pesos is not None:
                    self.painel_pesos.alternar_edicao()
        return True

    # ------------------------------------------------------------------ #
    def desenhar(self, world, passo=0):
        self.render.fundo()
        self.render.desenhar_pista(self.track, self.cam)
        self.render.desenhar_robos(world, self.cam, detalhe_em=self.detalhe_em)

        if self.painel is not None and self.brain is not None:
            self.painel.draw(self.tela, self._inspecionar())

        if self.mostrar_grafico:
            self.grafico.draw(self.tela, self.historico, self.hiper, self.marcas_pesos)
        if self.pesos is not None:
            alt = self.painel_pesos.altura_necessaria()
            self.painel_pesos.rect.top = self.tamanho[1] - alt - 12
            self.painel_pesos.draw(self.tela, self.pesos)

        self._hud(world, passo)
        pygame.display.flip()

    def _inspecionar(self):
        """A rede pode não implementar inspect — e isso é permitido."""
        if not hasattr(self.brain, "inspect"):
            return None
        try:
            return self.brain.inspect(self.detalhe_em)
        except Exception:
            return None

    def _hud(self, world, passo):
        vivos = int((world.alive & ~world.finished).sum())
        chegou = int(world.finished.sum())
        prog = self.track.progress_along(world.pos)
        linhas = [
            f"passo {passo}/{self.cfg.max_steps}   {world.steps * self.cfg.dt:5.1f} s",
            f"vivos {vivos}/{world.P}   chegaram {chegou}",
            f"avanço melhor {prog.max():5.2f} m de {self.track.length:.2f} m",
            "espaço=turbo  r=rede  g=gráfico  p=editar pesos  esc=sair",
        ]
        f = self.render.font_peq
        larg = max(f.size(t)[0] for t in linhas) + 24
        alt = len(linhas) * (f.get_linesize() + 2) + 16

        caixa = pygame.Surface((larg, alt), pygame.SRCALPHA)
        caixa.fill((10, 13, 22, 214))
        pygame.draw.rect(caixa, (57, 64, 79), caixa.get_rect(), 1, border_radius=6)
        for i, t in enumerate(linhas):
            cor = (141, 151, 171) if i == len(linhas) - 1 else (232, 236, 245)
            caixa.blit(f.render(t, True, cor), (12, 8 + i * (f.get_linesize() + 2)))
        self.tela.blit(caixa, (self.tamanho[0] - larg - 12, 12))

    def close(self):
        pygame.quit()
