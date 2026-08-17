"""Painel de calibragem: barras para ajustar o robô enquanto você dirige.

Isto existe por um motivo só — fechar a distância entre a simulação e o robô que
você vai montar. Você mede a velocidade do seu no chão, arrasta a barra até bater,
e sente na hora se a resposta ficou parecida. Depois `s` grava em `config.json`,
e o treino lê o mesmo arquivo.

Sem gravar não adiantaria nada: o treino roda noutro processo.

    no jogo:  p abre e fecha  ·  s grava  ·  z volta ao de fábrica
"""

import numpy as np
import pygame

from .config import CAMINHO_PADRAO, SimConfig

COR_PAINEL = (10, 13, 22, 232)
COR_BORDA = (57, 64, 79)
COR_TEXTO = (232, 236, 245)
COR_FRACO = (141, 151, 171)
COR_TITULO = (127, 180, 240)
COR_TRILHA = (34, 40, 56)
COR_BARRA = (77, 163, 255)
COR_BARRA_MEXENDO = (120, 200, 255)
COR_AVISO = (239, 159, 39)
COR_OK = (93, 202, 165)

LARGURA = 330
ALTURA_LINHA = 30


class Barra:
    """Uma barra ligada direto a um campo do config, por nome."""

    def __init__(self, alvo, campo, rotulo, minimo, maximo, unidade="",
                 casas=2, inteiro=False, remonta=False):
        self.alvo, self.campo = alvo, campo
        self.rotulo, self.unidade = rotulo, unidade
        self.min, self.max = minimo, maximo
        self.casas, self.inteiro = casas, inteiro
        self.remonta = remonta
        """True quando mexer nisto exige remontar o array de sensores — o painel
        avisa o mundo, em vez de recriar tudo e perder a pose do robô."""

    @property
    def valor(self):
        return getattr(self.alvo, self.campo)

    @valor.setter
    def valor(self, v):
        v = float(np.clip(v, self.min, self.max))
        setattr(self.alvo, self.campo, int(round(v)) if self.inteiro else v)

    @property
    def fracao(self) -> float:
        return (float(self.valor) - self.min) / max(self.max - self.min, 1e-9)

    def texto(self) -> str:
        v = self.valor
        n = f"{v:d}" if self.inteiro else f"{v:.{self.casas}f}"
        return f"{n}{(' ' + self.unidade) if self.unidade else ''}"


class PainelCalibragem:
    def __init__(self, cfg: SimConfig, tamanho_tela, fonte=None, fonte_peq=None):
        self.cfg = cfg
        self.tela_w, self.tela_h = tamanho_tela
        self.font = fonte or pygame.font.SysFont("consolas,menlo,monospace", 14)
        self.font_peq = fonte_peq or pygame.font.SysFont("consolas,menlo,monospace", 12)

        self.aberto = False
        self.arrastando = None
        self.recado = ""
        self.precisa_remontar = False
        self.grupos = self._montar(cfg)
        self.rect = pygame.Rect(self.tela_w - LARGURA - 12, 12, LARGURA, self._altura())

    def _montar(self, cfg):
        r, s = cfg.robot, cfg.sensor
        return [
            ("MOTORES", [
                Barra(r, "max_wheel_speed", "velocidade máx", 0.05, 1.5, "m/s"),
                Barra(r, "pwm_deadzone", "zona morta PWM", 0.0, 0.6),
                Barra(r, "accel_time", "tempo p/ acelerar", 0.02, 1.5, "s"),
                Barra(r, "motor_bias", "assimetria", -0.25, 0.25),
                Barra(r, "turn_gain", "força da curva", 0.2, 1.5),
            ]),
            ("CHASSI", [
                Barra(r, "wheel_base", "entre-eixos", 0.05, 0.45, "m"),
                Barra(r, "radius", "raio do corpo", 0.03, 0.30, "m", remonta=True),
            ]),
            ("SENSORES", [
                Barra(s, "count", "quantidade", 1, 8, inteiro=True, remonta=True),
                Barra(s, "fov_deg", "leque", 20, 300, "°", casas=0, remonta=True),
                Barra(s, "cone_deg", "cone", 2, 60, "°", casas=0, remonta=True),
                Barra(s, "rays_per_sensor", "sub-raios", 1, 11, inteiro=True, remonta=True),
                Barra(s, "max_range", "alcance", 0.2, 4.0, "m"),
                Barra(s, "reading_time", "tempo de leitura", 0.005, 0.2, "s", casas=3),
                Barra(s, "max_incidence_deg", "limite de eco", 20, 89, "°", casas=0,
                      remonta=True),
                Barra(s, "noise_std", "ruído", 0.0, 0.2, "m", casas=3),
            ]),
        ]

    @property
    def barras(self):
        return [b for _, bs in self.grupos for b in bs]

    def _altura(self):
        linhas = sum(len(bs) for _, bs in self.grupos)
        return 42 + linhas * ALTURA_LINHA + len(self.grupos) * 22 + 46

    # ------------------------------------------------------------------ #
    def alternar(self):
        self.aberto = not self.aberto
        self.arrastando = None
        if self.aberto:
            self.recado = "arraste as barras · s grava · z restaura"

    def evento(self, ev) -> bool:
        """Devolve True se o painel consumiu o evento."""
        if not self.aberto:
            return False

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if not self.rect.collidepoint(ev.pos):
                return False
            self.arrastando = self._barra_em(ev.pos)
            if self.arrastando:
                self._mover(ev.pos)
            return True

        if ev.type == pygame.MOUSEMOTION and self.arrastando:
            self._mover(ev.pos)
            return True

        if ev.type == pygame.MOUSEBUTTONUP and self.arrastando:
            self.arrastando = None
            return True

        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_s:
                self.gravar()
                return True
            if ev.key == pygame.K_z:
                self.restaurar()
                return True
        return False

    def _linhas_com_posicao(self):
        y = self.rect.y + 38
        for titulo, barras in self.grupos:
            y += 22
            for b in barras:
                yield b, y
                y += ALTURA_LINHA

    def _barra_em(self, pos):
        for b, y in self._linhas_com_posicao():
            if y + 12 <= pos[1] <= y + 30:
                return b
        return None

    def _mover(self, pos):
        b = self.arrastando
        x0 = self.rect.x + 16
        largura = self.rect.w - 32
        f = (pos[0] - x0) / max(largura, 1)
        b.valor = self.min_max_para(b, f)
        self.recado = f"{b.rotulo} = {b.texto()}"
        if b.remonta:
            self.precisa_remontar = True

    @staticmethod
    def min_max_para(b, fracao):
        return b.min + float(np.clip(fracao, 0.0, 1.0)) * (b.max - b.min)

    # ------------------------------------------------------------------ #
    def gravar(self):
        self.cfg.save()
        self.recado = f"gravado em {CAMINHO_PADRAO} — o treino lê daqui"

    def restaurar(self):
        padrao = SimConfig()
        for campo, origem in (("robot", padrao.robot), ("sensor", padrao.sensor)):
            atual = getattr(self.cfg, campo)
            for nome in origem.__dataclass_fields__:
                setattr(atual, nome, getattr(origem, nome))
        self.precisa_remontar = True
        self.recado = "valores de fábrica restaurados (não gravados)"

    def consumir_remontagem(self) -> bool:
        """O jogo pergunta isto para saber se precisa reconfigurar os sensores."""
        pendente, self.precisa_remontar = self.precisa_remontar, False
        return pendente

    # ------------------------------------------------------------------ #
    def desenhar(self, surface):
        if not self.aberto:
            return
        self.rect.height = self._altura()
        painel = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        painel.fill(COR_PAINEL)
        pygame.draw.rect(painel, COR_BORDA, painel.get_rect(), 1, border_radius=6)
        painel.blit(self.font.render("CALIBRAGEM", True, COR_TITULO), (16, 12))
        surface.blit(painel, self.rect.topleft)

        y = self.rect.y + 38
        for titulo, barras in self.grupos:
            surface.blit(self.font_peq.render(titulo, True, COR_FRACO),
                         (self.rect.x + 16, y + 4))
            y += 22
            for b in barras:
                self._desenhar_barra(surface, b, y)
                y += ALTURA_LINHA

        cor = COR_OK if "gravado" in self.recado else (
            COR_AVISO if "fábrica" in self.recado else COR_FRACO)
        surface.blit(self.font_peq.render(self.recado[:44], True, cor),
                     (self.rect.x + 16, self.rect.bottom - 34))
        surface.blit(self.font_peq.render("p fecha · s grava · z restaura", True, COR_FRACO),
                     (self.rect.x + 16, self.rect.bottom - 18))

    def _desenhar_barra(self, surface, b, y):
        x = self.rect.x + 16
        largura = self.rect.w - 32

        surface.blit(self.font_peq.render(b.rotulo, True, COR_TEXTO), (x, y))
        valor = self.font_peq.render(b.texto(), True, COR_TITULO)
        surface.blit(valor, (x + largura - valor.get_width(), y))

        pygame.draw.rect(surface, COR_TRILHA, (x, y + 18, largura, 6), border_radius=3)
        cheio = int(largura * b.fracao)
        cor = COR_BARRA_MEXENDO if b is self.arrastando else COR_BARRA
        if cheio > 0:
            pygame.draw.rect(surface, cor, (x, y + 18, cheio, 6), border_radius=3)
        pygame.draw.circle(surface, cor, (x + cheio, y + 21), 6)
        pygame.draw.circle(surface, (12, 16, 24), (x + cheio, y + 21), 6, 1)
