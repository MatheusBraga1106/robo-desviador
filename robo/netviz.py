"""Painel de visualização da rede.

Desenha o que a sua rede devolve em `inspect()` — neurônios por camada, coloridos
pela ativação, e as conexões coloridas pelo peso. Não sabe nada sobre a sua
arquitetura: se `inspect` entrega 3 camadas ou 8, ele desenha 3 ou 8.

    painel = NetPanel(pygame.Rect(0, 0, 300, 420))
    painel.draw(tela, brain.inspect(0))

Devolver None em `inspect` simplesmente não desenha nada.
"""

import numpy as np
import pygame

from .brain import ROTULOS

COR_FUNDO = (10, 13, 22, 214)
COR_BORDA = (57, 64, 79)
COR_ROTULO = (120, 230, 140)
COR_NEURONIO_BORDA = (26, 22, 18)
LIMITE_CONEXOES = 1200
"""Acima disso só as conexões mais fortes são desenhadas. Uma camada 64x64 tem
4096 linhas: desenhar todas custa caro e vira borrão preto, que não informa nada."""


def _cor_ativacao(v: float):
    """Ativação -> calor. Escuro é inativo, laranja e claro é ativo."""
    t = float(np.clip(abs(v), 0.0, 1.0))
    return (int(30 + 215 * t), int(20 + 130 * t ** 1.4), int(18 + 40 * t ** 3))


def _cor_peso(w: float, escala: float):
    """Peso -> cor e opacidade. Vermelho excita, azul inibe, forte fica opaco."""
    t = float(np.clip(abs(w) / escala, 0.0, 1.0)) if escala > 0 else 0.0
    alfa = int(20 + 200 * t ** 0.7)
    return ((225, 70, 55, alfa) if w >= 0 else (70, 140, 235, alfa))


class NetPanel:
    def __init__(self, rect: pygame.Rect, fonte=None):
        self.rect = pygame.Rect(rect)
        self.fonte = fonte or pygame.font.SysFont("consolas,menlo,monospace", 12)

    # ------------------------------------------------------------------ #
    def draw(self, surface, dados, rotulos=ROTULOS):
        """`dados` é o que `Brain.inspect()` devolveu, ou None."""
        if not self._valido(dados):
            return
        ativacoes = [np.asarray(a, dtype=np.float64).ravel() for a in dados["ativacoes"]]
        pesos = [np.asarray(w, dtype=np.float64) for w in dados["pesos"]]

        painel = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        painel.fill(COR_FUNDO)
        pygame.draw.rect(painel, COR_BORDA, painel.get_rect(), 1, border_radius=6)

        centros = self._posicoes(ativacoes, rotulos)
        self._conexoes(painel, centros, pesos)
        self._neuronios(painel, centros, ativacoes)
        self._rotulos(painel, centros[-1], ativacoes[-1], rotulos)

        surface.blit(painel, self.rect.topleft)

    @staticmethod
    def _valido(dados) -> bool:
        if not dados:
            return False
        a, w = dados.get("ativacoes"), dados.get("pesos")
        return bool(a) and w is not None and len(w) == len(a) - 1

    # ------------------------------------------------------------------ #
    def _posicoes(self, ativacoes, rotulos):
        """Centro de cada neurônio, em coordenadas do painel."""
        margem_x, margem_y = 22, 18
        # a última coluna precisa de espaço à direita para os rótulos
        largura_rotulo = max(self.fonte.size(r)[0] for r in rotulos) + 12
        util_x = self.rect.w - 2 * margem_x - largura_rotulo
        util_y = self.rect.h - 2 * margem_y

        n_camadas = len(ativacoes)
        col = (util_x / max(n_camadas - 1, 1)) if n_camadas > 1 else 0

        centros = []
        for c, a in enumerate(ativacoes):
            x = margem_x + c * col
            n = len(a)
            passo = util_y / max(n, 1)
            centros.append([(x, margem_y + passo * (i + 0.5)) for i in range(n)])
        return centros

    def _raio(self, centros):
        maior = max(len(c) for c in centros)
        vao = (self.rect.h - 36) / max(maior, 1)
        return int(np.clip(vao * 0.34, 2, 9))

    def _conexoes(self, painel, centros, pesos):
        total = sum(w.size for w in pesos)
        # com muita conexão, só as mais fortes: o resto vira borrão
        corte = 0.0
        if total > LIMITE_CONEXOES:
            todos = np.abs(np.concatenate([w.ravel() for w in pesos]))
            corte = float(np.partition(todos, -LIMITE_CONEXOES)[-LIMITE_CONEXOES])

        for c, w in enumerate(pesos):
            escala = float(np.abs(w).max()) or 1.0
            origem, destino = centros[c], centros[c + 1]
            saidas, entradas = w.shape
            for j in range(min(saidas, len(destino))):
                for i in range(min(entradas, len(origem))):
                    peso = w[j, i]
                    if abs(peso) < corte:
                        continue
                    pygame.draw.line(painel, _cor_peso(peso, escala),
                                     origem[i], destino[j], 1)

    def _neuronios(self, painel, centros, ativacoes):
        r = self._raio(centros)
        for camada, pontos in enumerate(centros):
            valores = ativacoes[camada]
            for i, p in enumerate(pontos):
                pygame.draw.circle(painel, _cor_ativacao(valores[i]), p, r)
                pygame.draw.circle(painel, COR_NEURONIO_BORDA, p, r, 1)

    def _rotulos(self, painel, saida, valores, rotulos):
        """Nomes dos botões, destacando os que estão sendo apertados."""
        for i, p in enumerate(saida):
            if i >= len(rotulos):
                break
            apertado = valores[i] > 0.5
            cor = COR_ROTULO if apertado else (86, 110, 92)
            texto = self.fonte.render(rotulos[i], True, cor)
            painel.blit(texto, (p[0] + 14, p[1] - texto.get_height() / 2))
