"""Gráficos do treino, desenhados ao vivo em pygame.

Sobre "loss" e "learning rate"
------------------------------
Num algoritmo genético não existe nenhum dos dois no sentido literal: não há
função de perda sendo minimizada por gradiente, nem passo de otimizador. Os
equivalentes honestos são:

* **loss  ->  fitness.** Mesma leitura de uma curva de treino, com o sinal
  invertido: aqui subir é bom. É o painel de cima.
* **learning rate  ->  força da mutação** (o tamanho do passo que a evolução dá
  a cada geração) e a **taxa de melhora** (quanto o melhor indivíduo ganhou em
  relação à geração anterior). É o painel de baixo.

A taxa de melhora é o que responde "ainda está aprendendo?". Quando ela oscila
em torno de zero por muitas gerações, a evolução estagnou — e aí mexer na força
da mutação ou na pressão de seleção costuma ser mais útil do que só esperar.
"""

import numpy as np
import pygame

COR_PAINEL = (10, 13, 22, 243)
COR_BORDA = (57, 64, 79)
COR_EIXO = (48, 56, 72)
COR_TEXTO = (232, 236, 245)
COR_FRACO = (141, 151, 171)

COR_MELHOR = (93, 202, 165)
COR_MEDIA = (77, 163, 255)
COR_PIOR = (150, 70, 78)
COR_MELHORA = (239, 159, 39)
COR_ZERO = (70, 78, 96)
COR_MARCA = (200, 140, 60)


ALTURA_BARRA = 30


class PainelPesos:
    """Os pesos do fitness, na tela, com opção de editar durante o treino.

    Existe porque calibrar às cegas é o que mais custa tempo aqui: você olha a
    curva subindo (ou não) e precisa saber com que regras aquilo foi produzido.

    Editando ao vivo, um cuidado
    ----------------------------
    Mudar um peso no meio do treino torna as gerações **incomparáveis**: as de
    antes foram medidas com outra régua, e a curva do gráfico ao lado passaria a
    misturar duas coisas diferentes sem avisar. Por isso o painel chama
    `ao_mudar` a cada ajuste, e quem escuta marca a geração — o gráfico desenha
    uma linha vertical ali, deixando visível onde a régua mudou.
    """

    def __init__(self, rect, pesos=None, fonte=None, fonte_peq=None, ao_mudar=None):
        self.rect = pygame.Rect(rect)
        self.font = fonte or pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_peq = fonte_peq or pygame.font.SysFont("consolas,menlo,monospace", 11)
        self.ao_mudar = ao_mudar

        self.editando = False
        self.arrastando = None
        self.pesos = pesos
        self.barras = self._montar(pesos) if pesos is not None else []

    @staticmethod
    def _montar(p):
        from .calibra import Barra
        return [
            Barra(p, "peso_avanco", "por metro avançado", 0.0, 30.0, casas=1),
            Barra(p, "punicao_batida", "bateu", 0.0, 200.0, casas=0),
            Barra(p, "bonus_chegada", "chegou", 0.0, 500.0, casas=0),
            Barra(p, "peso_tempo", "por segundo (quem chega)", 0.0, 3.0, casas=2),
            Barra(p, "enrolar_peso", "enrolar", 0.0, 200.0, casas=0),
            Barra(p, "enrolar_fracao_passos", "cobra em", 0.05, 0.95, casas=2),
            Barra(p, "enrolar_fracao_rota", "meta da rota", 0.0, 0.9, casas=2),
        ]

    # ------------------------------------------------------------------ #
    def altura_necessaria(self, pesos=None):
        p = pesos if pesos is not None else self.pesos
        if p is None:
            return 0
        cab = self.font_peq.get_linesize() + 14
        if self.editando:
            return cab + len(self.barras) * ALTURA_BARRA + 20
        return cab + (len(p.linhas())) * (self.font_peq.get_linesize() + 2) + 14

    def alternar_edicao(self):
        self.editando = not self.editando
        self.arrastando = None

    def evento(self, ev) -> bool:
        """Devolve True se consumiu o evento. Só age no modo de edição."""
        if not self.editando or self.pesos is None:
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
        return False

    def _posicoes(self):
        y = self.rect.y + self.font_peq.get_linesize() + 14
        for b in self.barras:
            yield b, y
            y += ALTURA_BARRA

    def _barra_em(self, pos):
        for b, y in self._posicoes():
            if y <= pos[1] <= y + ALTURA_BARRA:
                return b
        return None

    def _mover(self, pos):
        b = self.arrastando
        x0, larg = self.rect.x + 14, self.rect.w - 28
        antes = b.valor
        b.valor = b.min + float(np.clip((pos[0] - x0) / max(larg, 1), 0, 1)) * (b.max - b.min)
        if b.valor != antes and self.ao_mudar is not None:
            self.ao_mudar()

    # ------------------------------------------------------------------ #
    def draw(self, surface, pesos=None):
        p = pesos if pesos is not None else self.pesos
        if p is None:
            return
        self.rect.height = self.altura_necessaria(p)

        painel = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        painel.fill(COR_PAINEL)
        borda = COR_MELHORA if self.editando else COR_BORDA
        pygame.draw.rect(painel, borda, painel.get_rect(), 1, border_radius=6)
        surface.blit(painel, self.rect.topleft)

        x, y = self.rect.x + 12, self.rect.y + 7
        titulo = "PESOS — editando" if self.editando else "PESOS DO FITNESS"
        surface.blit(self.font_peq.render(titulo, True,
                                          COR_MELHORA if self.editando else COR_TEXTO), (x, y))

        if self.editando:
            for b, yb in self._posicoes():
                self._desenhar_barra(surface, b, yb)
        else:
            y += self.font_peq.get_linesize() + 6
            for rotulo, valor in p.linhas():
                surface.blit(self.font_peq.render(rotulo, True, COR_FRACO), (x, y))
                img = self.font_peq.render(valor, True, COR_TEXTO)
                surface.blit(img, (self.rect.right - 12 - img.get_width(), y))
                y += self.font_peq.get_linesize() + 2

    def _desenhar_barra(self, surface, b, y):
        x, larg = self.rect.x + 14, self.rect.w - 28
        surface.blit(self.font_peq.render(b.rotulo, True, COR_TEXTO), (x, y))
        img = self.font_peq.render(b.texto(), True, COR_MELHORA)
        surface.blit(img, (x + larg - img.get_width(), y))

        trilha_y = y + self.font_peq.get_linesize() + 3
        pygame.draw.rect(surface, (34, 40, 56), (x, trilha_y, larg, 5), border_radius=3)
        cheio = int(larg * b.fracao)
        if cheio > 0:
            pygame.draw.rect(surface, COR_MELHORA, (x, trilha_y, cheio, 5), border_radius=3)
        pygame.draw.circle(surface, COR_MELHORA, (x + cheio, trilha_y + 2), 5)


class PainelGrafico:
    """Dois gráficos empilhados: fitness por geração, e taxa de melhora."""

    def __init__(self, rect, fonte=None, fonte_peq=None, janela_media=5):
        self.rect = pygame.Rect(rect)
        self.font = fonte or pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.font_peq = fonte_peq or pygame.font.SysFont("consolas,menlo,monospace", 11)
        self.janela_media = janela_media
        """Quantas gerações entram na média móvel da taxa de melhora. A taxa
        crua salta demais de uma geração para outra (o fitness depende de qual
        cenário caiu no sorteio), e o que interessa é a tendência."""

    # ------------------------------------------------------------------ #
    def draw(self, surface, historico, hiper=None, marcas_pesos=()):
        """`historico`: lista de dicts com geracao/melhor/media/pior.

        `marcas_pesos`: índices de geração em que os pesos do fitness mudaram.
        São desenhados como linha vertical, porque dali para trás a curva foi
        medida com outra régua — comparar os dois lados sem saber disso é o
        jeito mais fácil de tirar a conclusão errada de um treino.
        """
        painel = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        painel.fill(COR_PAINEL)
        pygame.draw.rect(painel, COR_BORDA, painel.get_rect(), 1, border_radius=6)
        surface.blit(painel, self.rect.topleft)

        if not historico:
            self._aviso(surface, "aguardando a primeira geração...")
            return

        # layout medido, não chutado: cada rótulo ocupa uma faixa própria, e os
        # gráficos ficam com o que sobra. Escrever por cima da moldura (ou para
        # fora do painel) é o jeito mais fácil de deixar isto ilegível.
        m = 10
        alt_rotulo = self.font_peq.get_linesize()
        larg = self.rect.w - 2 * m
        x = self.rect.x + m

        sobra = self.rect.h - 2 * m - 4 * alt_rotulo - 6
        alto_fit = int(sobra * 0.60)
        alto_taxa = sobra - alto_fit

        y = self.rect.y + m
        titulo_fit = y
        area_fit = pygame.Rect(x, y + alt_rotulo, larg, alto_fit)
        legenda_y = area_fit.bottom
        titulo_taxa = legenda_y + alt_rotulo + 6
        area_taxa = pygame.Rect(x, titulo_taxa + alt_rotulo, larg, alto_taxa)
        rodape_y = area_taxa.bottom

        self._fitness(surface, area_fit, titulo_fit, legenda_y, historico)
        self._taxa(surface, area_taxa, titulo_taxa, rodape_y, historico, hiper)
        for area in (area_fit, area_taxa):
            self._marcas(surface, area, historico, marcas_pesos)

    def _marcas(self, surface, area, historico, marcas):
        """Linha vertical onde os pesos mudaram."""
        n = len(historico)
        if n < 2:
            return
        for g in marcas:
            if not (0 <= g < n):
                continue
            x = area.x + area.w * g / (n - 1)
            for yy in range(area.y, area.bottom, 4):    # tracejada
                surface.set_at((int(x), yy), COR_MARCA)

    def _aviso(self, surface, texto):
        img = self.font_peq.render(texto, True, COR_FRACO)
        surface.blit(img, (self.rect.x + 12, self.rect.y + 12))

    def _escrever(self, surface, texto, x, y, cor=COR_TEXTO):
        """Escreve cortando o que não couber na largura do painel."""
        limite = self.rect.right - 10 - x
        while texto and self.font_peq.size(texto)[0] > limite:
            texto = texto[:-1]
        surface.blit(self.font_peq.render(texto, True, cor), (x, y))

    # ------------------------------------------------------------------ #
    def _fitness(self, surface, area, y_titulo, y_legenda, historico):
        melhor = np.array([h["melhor"] for h in historico], dtype=float)
        media = np.array([h["media"] for h in historico], dtype=float)
        pior = np.array([h["pior"] for h in historico], dtype=float)

        lo, hi = self._escala(np.concatenate([melhor, media, pior]))
        self._moldura(surface, area, lo, hi)

        self._linha(surface, area, pior, lo, hi, COR_PIOR, 1)
        self._linha(surface, area, media, lo, hi, COR_MEDIA, 2)
        self._linha(surface, area, melhor, lo, hi, COR_MELHOR, 2)

        self._escrever(surface, f"FITNESS (o 'loss' daqui — subir é bom)",
                       area.x, y_titulo)
        self._legenda(surface, area, y_legenda, melhor, media, pior, len(historico))

    def _legenda(self, surface, area, y, melhor, media, pior, n_ger):
        itens = [(f"melhor {melhor[-1]:.0f}", COR_MELHOR),
                 (f"média {media[-1]:.0f}", COR_MEDIA),
                 (f"pior {pior[-1]:.0f}", COR_PIOR)]
        self._escrever(surface, f"{n_ger} ger", area.x, y, COR_FRACO)
        x = area.right
        for rotulo, cor in reversed(itens):
            img = self.font_peq.render(rotulo, True, cor)
            x -= img.get_width()
            surface.blit(img, (x, y))
            x -= 12

    # ------------------------------------------------------------------ #
    def _taxa(self, surface, area, y_titulo, y_rodape, historico, hiper):
        """Quanto o melhor indivíduo ganhou por geração — o 'está aprendendo?'."""
        melhor = np.array([h["melhor"] for h in historico], dtype=float)
        if len(melhor) < 2:
            self._moldura(surface, area, -1, 1)
            self._escrever(surface, "TAXA DE MELHORA  ·  precisa de 2+ gerações",
                           area.x, y_titulo, COR_FRACO)
            return

        ganho = np.diff(melhor)                      # quanto subiu a cada geração
        suave = self._media_movel(ganho, self.janela_media)

        lo, hi = self._escala(suave, simetrico_no_zero=True)
        self._moldura(surface, area, lo, hi)

        # a linha do zero é a referência que importa: abaixo dela, estagnou
        if lo < 0 < hi:
            y = self._y(0.0, area, lo, hi)
            pygame.draw.line(surface, COR_ZERO, (area.x, y), (area.right, y), 1)

        self._linha(surface, area, suave, lo, hi, COR_MELHORA, 2)

        recentes = suave[-self.janela_media:]
        estagnou = abs(recentes.mean()) < 1e-6
        self._escrever(surface,
                       f"TAXA DE MELHORA  ·  {suave[-1]:+.2f}/ger  "
                       f"{'estagnado' if estagnou else 'aprendendo'}",
                       area.x, y_titulo, COR_FRACO if estagnou else COR_TEXTO)

        if hiper is not None:
            # o "learning rate" desta evolução: o tamanho do passo que ela dá
            self._escrever(surface,
                           f"passo da evolução: mutação {hiper.taxa_mutacao:.2f} "
                           f"× força {hiper.forca_mutacao:.2f}",
                           area.x, y_rodape, COR_FRACO)

    @staticmethod
    def _media_movel(v, n):
        if len(v) < n or n <= 1:
            return v
        nucleo = np.ones(n) / n
        # 'same' mantém o comprimento, o que alinha o eixo x com as gerações
        return np.convolve(v, nucleo, mode="same")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _escala(valores, simetrico_no_zero=False):
        lo, hi = float(np.min(valores)), float(np.max(valores))
        if simetrico_no_zero:
            m = max(abs(lo), abs(hi), 1e-6)
            lo, hi = -m, m
        if hi - lo < 1e-9:          # tudo igual: abre uma faixa para não dividir por zero
            lo, hi = lo - 1.0, hi + 1.0
        folga = (hi - lo) * 0.08
        return lo - folga, hi + folga

    @staticmethod
    def _y(valor, area, lo, hi):
        t = (valor - lo) / (hi - lo)
        return area.bottom - t * area.h

    def _moldura(self, surface, area, lo, hi):
        pygame.draw.rect(surface, COR_EIXO, area, 1)
        for valor in (lo, (lo + hi) / 2, hi):
            y = self._y(valor, area, lo, hi)
            img = self.font_peq.render(f"{valor:.0f}", True, COR_FRACO)
            surface.blit(img, (area.x + 3, y - img.get_height() / 2))

    def _linha(self, surface, area, valores, lo, hi, cor, espessura):
        n = len(valores)
        if n == 1:
            y = self._y(valores[0], area, lo, hi)
            pygame.draw.circle(surface, cor, (area.x + area.w // 2, int(y)), 3)
            return
        passo = area.w / (n - 1)
        pontos = [(area.x + i * passo, self._y(v, area, lo, hi))
                  for i, v in enumerate(valores)]
        pygame.draw.lines(surface, cor, False, pontos, espessura)
