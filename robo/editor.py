"""Editor de pista.

    python -m robo.editor
    python -m robo.editor --abrir tracks/minha.json

Você desenha a linha do meio do corredor e o resto sai dela: as duas paredes, o
chão, os checkpoints. Depois espalha caixas, marca onde o robô nasce e onde é a
chegada, e testa dirigindo sem sair do programa.

Um gesto só, quatro modos
-------------------------
Em todo modo, **apertar e arrastar** quer dizer "coloca e dimensiona":

    1  linha central   clique acrescenta ponto no fim;
                       arrastar em cima de um ponto existente move ele
    2  caixas          arrasta de um canto ao outro do retângulo
    3  largada         clique põe o robô, arrastar escolhe para onde ele olha
    4  objetivo        clique põe o centro, arrastar escolhe o raio

Botão direito apaga o que estiver sob o cursor no modo atual.

A roda do mouse é zoom em todos os modos, porque é o que o dedo espera. Os
escalares ficam em `[` e `]`, que mudam de significado conforme o modo.
"""

import argparse
import os
import sys

import numpy as np
import pygame

from .config import SimConfig
from .game import abrir_janela, rodar
from .pistas import PISTAS, carregar
from .render import Camera, Renderer
from .track import Track

MODO_LINHA, MODO_CAIXA, MODO_LARGADA, MODO_OBJETIVO = range(4)
NOMES = ["linha central", "caixas", "largada", "objetivo"]

COR_LINHA = (125, 92, 200)
COR_PONTO = (168, 140, 240)
COR_AVISO = (239, 159, 39)
COR_OK = (93, 202, 165)
COR_TEXTO = (232, 236, 245)
COR_FRACO = (141, 151, 171)
COR_PAINEL = (18, 21, 29)
COR_BORDA = (57, 64, 79)

LARGURA_PAINEL = 186
PASTA_PISTAS = "tracks"


class Editor:
    def __init__(self, track=None, tamanho=(1180, 760)):
        self.tamanho = tamanho
        self.tela = abrir_janela(tamanho, "Editor de pista")
        self.render = Renderer(self.tela)
        self.relogio = pygame.time.Clock()

        self.cam = Camera(tamanho)
        self.cam.scale = 110.0
        self.cam.center = np.array([3.0, 2.0])

        self.modo = MODO_LINHA
        self.grade = True
        self.passo_grade = 0.05
        self.arrastando = None      # ('caixa'|'largada'|'objetivo'|'ponto', dados)
        self.pan = None
        # o cursor vem dos próprios eventos, não de pygame.mouse.get_pos(): o
        # estado global do mouse não é confiável fora de uma janela de verdade,
        # e assim a lógica de "o que está sob o cursor" fica testável
        self.cursor = (0, 0)
        self.recado = ""
        self.nome_arquivo = None

        self.track = track or Track(centerline=None, width=0.9, name="nova")
        self.pontos = ([] if self.track.centerline is None
                       else [tuple(p) for p in self.track.centerline])
        # com pista carregada, largada e objetivo já vieram definidos
        self.tem_largada = track is not None
        self.tem_objetivo = track is not None

    # ------------------------------------------------------------------ #
    def mundo(self, pos_tela):
        p = np.asarray(pos_tela, dtype=np.float64)
        x = (p[0] - self.tamanho[0] / 2.0) / self.cam.scale + self.cam.center[0]
        y = self.cam.center[1] - (p[1] - self.tamanho[1] / 2.0) / self.cam.scale
        if self.grade:
            x = round(x / self.passo_grade) * self.passo_grade
            y = round(y / self.passo_grade) * self.passo_grade
        return np.array([x, y])

    def no_painel(self, pos) -> bool:
        return pos[0] < LARGURA_PAINEL

    def regerar(self):
        """Reconstrói a pista a partir dos pontos, preservando o que é manual."""
        self.track.centerline = (np.array(self.pontos, dtype=np.float64)
                                 if len(self.pontos) >= 2 else None)
        # rebuild() só preenche largada/objetivo quando estão vazios, então o
        # que você posicionou à mão sobrevive a qualquer mudança na linha
        self.track.rebuild()

    # ------------------------------------------------------------------ #
    def curvas_apertadas(self):
        """Vértices que a largura atual não comporta.

        Duas coisas dão errado ao gerar o corredor a partir da linha central:

        1. **Curva fechada demais.** O corredor nasce deslocando a linha para os
           dois lados; se o ângulo fecha mais do que a meia-largura permite, as
           paredes se cruzam e o corredor vira um nó.
        2. **Trecho curto demais.** Um segmento menor que a largura faz a
           esquadria invadir a tampa da ponta, e o corredor sai com uma aba torta.

        Nos dois casos o resultado sai errado em silêncio, então é melhor marcar
        na hora do que você descobrir dirigindo.
        """
        ruins = set()
        if len(self.pontos) < 2:
            return []
        pts = np.array(self.pontos)
        meia = self.track.width / 2.0

        comprimentos = np.hypot(*(pts[1:] - pts[:-1]).T)
        for i, c in enumerate(comprimentos):
            if c < self.track.width:
                ruins.update((i, i + 1))

        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            v1, v2 = b - a, c - b
            n1, n2 = np.hypot(*v1), np.hypot(*v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            virada = np.arccos(cos)                      # 0 = reta, pi = volta
            # o raio interno da esquadria some quando a curva aperta
            if virada > 1e-6 and meia / np.tan(max(np.pi - virada, 1e-6) / 2) > min(n1, n2):
                ruins.add(i)
        return sorted(ruins)

    def pendencias(self):
        # marcadores em ASCII: a fonte monoespaçada do sistema não tem "✓" e
        # desenharia um quadrado vazio no lugar
        p = []
        if len(self.pontos) < 2:
            p.append(("!", "linha com 2+ pontos", COR_AVISO))
        p.append(("+" if self.tem_largada else "!", "largada",
                  COR_OK if self.tem_largada else COR_AVISO))
        p.append(("+" if self.tem_objetivo else "!", "objetivo",
                  COR_OK if self.tem_objetivo else COR_AVISO))
        n = len(self.curvas_apertadas())
        if n:
            p.append(("!", f"{n} ponto{'s' if n > 1 else ''} apertado{'s' if n > 1 else ''}",
                      COR_AVISO))
        return p

    def pronta(self) -> bool:
        return len(self.pontos) >= 2 and self.tem_largada and self.tem_objetivo

    # ------------------------------------------------------------------ #
    def loop(self):
        rodando = True
        while rodando:
            for ev in pygame.event.get():
                rodando = self.evento(ev)
                if not rodando:
                    break
            self.desenhar()
            self.relogio.tick(60)
        pygame.quit()

    def evento(self, ev) -> bool:
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            return self.tecla(ev)
        if ev.type == pygame.MOUSEWHEEL:
            self.cam.scale = float(np.clip(self.cam.scale * (1.1 ** ev.y), 25, 600))
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_desce(ev)
        elif ev.type == pygame.MOUSEMOTION:
            self.mouse_move(ev)
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.mouse_sobe(ev)
        return True

    def tecla(self, ev) -> bool:
        k = ev.key
        if k == pygame.K_ESCAPE:
            return False
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self.modo = k - pygame.K_1
            self.recado = ""
        elif k == pygame.K_g:
            self.grade = not self.grade
        elif k == pygame.K_z:
            self.desfazer()
        elif k == pygame.K_a:
            self.auto_largada_objetivo()
        elif k == pygame.K_n:
            self.nova()
        elif k == pygame.K_s:
            self.salvar()
        elif k == pygame.K_l:
            self.abrir_ultima()
        elif k == pygame.K_t:
            self.testar()
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            self.escalar(-1 if k == pygame.K_LEFTBRACKET else +1)
        return True

    def escalar(self, sinal):
        """`[` e `]` mudam de significado conforme o modo."""
        if self.modo == MODO_LINHA:
            self.track.width = float(np.clip(self.track.width + sinal * 0.05, 0.3, 4.0))
            self.regerar()
        elif self.modo == MODO_CAIXA:
            i = self.caixa_sob_cursor()
            if i is not None:
                self.track.obstacles[i, 4] += sinal * np.deg2rad(15)
                self.track.rebuild()
        elif self.modo == MODO_OBJETIVO and self.tem_objetivo:
            self.track.goal[2] = float(np.clip(self.track.goal[2] + sinal * 0.05, 0.05, 2.0))

    def caixa_sob_cursor(self):
        p = self.mundo(self.cursor)
        dentro = self.track.contains_obstacle(p[0], p[1])
        if not len(dentro) or not dentro.any():
            return None
        area = self.track.obstacles[:, 2] * self.track.obstacles[:, 3]
        return int(np.argmin(np.where(dentro, area, np.inf)))

    # ------------------------------------------------------------------ #
    def mouse_desce(self, ev):
        self.cursor = ev.pos
        if self.no_painel(ev.pos):
            return
        if ev.button == 2:
            self.pan = (np.array(ev.pos), self.cam.center.copy())
            return
        p = self.mundo(ev.pos)

        if ev.button == 3:
            self.apagar_em(p)
            return
        if ev.button != 1:
            return

        if self.modo == MODO_LINHA:
            i = self.ponto_perto(p)
            if i is not None:
                self.arrastando = ("ponto", i)
            else:
                self.pontos.append(tuple(p))
                self.regerar()
        elif self.modo == MODO_CAIXA:
            self.arrastando = ("caixa", p)
        elif self.modo == MODO_LARGADA:
            self.track.start = np.array([p[0], p[1], self.track.start[2] if self.tem_largada else 0.0])
            self.tem_largada = True
            self.arrastando = ("largada", p)
        elif self.modo == MODO_OBJETIVO:
            raio = self.track.goal[2] if self.tem_objetivo else self.track.width * 0.35
            self.track.goal = np.array([p[0], p[1], raio])
            self.tem_objetivo = True
            self.arrastando = ("objetivo", p)

    def mouse_move(self, ev):
        self.cursor = ev.pos
        if self.pan is not None:
            inicio, centro = self.pan
            d = (np.array(ev.pos) - inicio) / self.cam.scale
            self.cam.center = centro - np.array([d[0], -d[1]])
            return
        if self.arrastando is None:
            return

        tipo, dado = self.arrastando
        p = self.mundo(ev.pos)
        if tipo == "ponto":
            self.pontos[dado] = tuple(p)
            self.regerar()
        elif tipo == "largada":
            d = p - self.track.start[:2]
            if np.hypot(*d) > 0.03:
                self.track.start[2] = np.arctan2(d[1], d[0])
        elif tipo == "objetivo":
            r = float(np.hypot(*(p - self.track.goal[:2])))
            if r > 0.03:
                self.track.goal[2] = min(r, 2.0)

    def mouse_sobe(self, ev):
        if ev.button == 2:
            self.pan = None
            return
        if self.arrastando is None:
            return
        tipo, dado = self.arrastando
        if tipo == "caixa":
            p = self.mundo(ev.pos)
            larg, comp = abs(p[0] - dado[0]), abs(p[1] - dado[1])
            if larg >= 0.03 and comp >= 0.03:
                self.track.add_obstacle((p[0] + dado[0]) / 2, (p[1] + dado[1]) / 2, larg, comp)
            else:
                self.recado = "caixa pequena demais, arraste mais"
        self.arrastando = None

    def ponto_perto(self, p, raio_tela=10):
        if not self.pontos:
            return None
        pts = np.array(self.pontos)
        d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) * self.cam.scale
        i = int(np.argmin(d))
        return i if d[i] <= raio_tela else None

    def apagar_em(self, p):
        if self.modo == MODO_CAIXA:
            if not self.track.remove_obstacle_at(p[0], p[1]):
                self.recado = "nenhuma caixa aqui"
        elif self.modo == MODO_LINHA:
            i = self.ponto_perto(p)
            if i is not None:
                self.pontos.pop(i)
                self.regerar()

    def desfazer(self):
        if self.modo == MODO_CAIXA and len(self.track.obstacles):
            self.track.obstacles = self.track.obstacles[:-1]
            self.track.rebuild()
        elif self.pontos:
            self.pontos.pop()
            self.regerar()

    # ------------------------------------------------------------------ #
    def auto_largada_objetivo(self):
        """Preenche os dois a partir da linha central — atalho, não obrigação."""
        if len(self.pontos) < 2:
            self.recado = "desenhe a linha central primeiro"
            return
        self.track.start = None
        self.track.goal = None
        self.regerar()
        self.tem_largada = self.tem_objetivo = True
        self.recado = "largada e objetivo pelo traçado"

    def nova(self):
        self.pontos = []
        self.track = Track(centerline=None, width=self.track.width, name="nova")
        self.tem_largada = self.tem_objetivo = False
        self.nome_arquivo = None
        self.recado = "pista nova"

    def salvar(self):
        if not self.pronta():
            self.recado = "falta largada ou objetivo (a = automático)"
            return
        os.makedirs(PASTA_PISTAS, exist_ok=True)
        if self.nome_arquivo is None:
            n = 1
            while os.path.exists(os.path.join(PASTA_PISTAS, f"pista{n}.json")):
                n += 1
            self.nome_arquivo = os.path.join(PASTA_PISTAS, f"pista{n}.json")
            self.track.name = f"pista{n}"
        self.track.save(self.nome_arquivo)
        self.recado = f"salvo em {self.nome_arquivo}"

    def abrir_ultima(self):
        if not os.path.isdir(PASTA_PISTAS):
            self.recado = "nenhuma pista salva ainda"
            return
        arquivos = sorted(f for f in os.listdir(PASTA_PISTAS) if f.endswith(".json"))
        if not arquivos:
            self.recado = "nenhuma pista salva ainda"
            return
        caminho = os.path.join(PASTA_PISTAS, arquivos[-1])
        self.carregar_track(Track.load(caminho), caminho)
        self.recado = f"aberto {arquivos[-1]}"

    def carregar_track(self, track, caminho=None):
        self.track = track
        self.pontos = ([] if track.centerline is None
                       else [tuple(p) for p in track.centerline])
        self.tem_largada = self.tem_objetivo = True
        self.nome_arquivo = caminho

    def testar(self):
        """Entra no jogo com a pista atual e volta para a edição no Esc."""
        if not self.pronta():
            self.recado = "falta largada ou objetivo (a = automático)"
            return
        cfg = SimConfig()
        cfg.collision_ends_episode = False
        pygame.display.set_caption("Testando — Esc volta para o editor")
        rodar(self.track, cfg, self.tela)
        pygame.display.set_caption("Editor de pista")
        self.recado = "de volta ao editor"

    # ------------------------------------------------------------------ #
    def desenhar(self):
        r, cam = self.render, self.cam
        r.fundo()
        if self.grade:
            r.grade(cam, 0.5, self._limites_visiveis())

        if len(self.pontos) >= 2:
            r.desenhar_pista(self.track, cam, mostrar_checkpoints=False)
        else:
            r.desenhar_caixas(self.track, cam)

        self._traco()
        self._marcadores()
        self._previa()
        self._painel()
        pygame.display.flip()

    def _limites_visiveis(self):
        meia = np.array(self.tamanho) / 2.0 / self.cam.scale
        c = self.cam.center
        return (c[0] - meia[0], c[1] - meia[1], c[0] + meia[0], c[1] + meia[1])

    def _traco(self):
        """A linha central e seus pontos, para você ver o que está editando."""
        if len(self.pontos) >= 2:
            pts = self.cam.to_screen(np.array(self.pontos))
            pygame.draw.lines(self.tela, COR_LINHA, False, pts, 2)
        apertadas = set(self.curvas_apertadas())
        for i, p in enumerate(self.pontos):
            centro = self.cam.to_screen(p)
            ruim = i in apertadas
            pygame.draw.circle(self.tela, COR_AVISO if ruim else COR_PONTO, centro, 5)
            if ruim:
                pygame.draw.circle(self.tela, COR_AVISO, centro, 14, 2)

    def _marcadores(self):
        if self.tem_largada:
            s = self.track.start
            c = self.cam.to_screen(s[:2])
            pygame.draw.circle(self.tela, (60, 220, 190), c, max(5, self.cam.px(0.09)))
            ponta = self.cam.to_screen(
                s[:2] + np.array([np.cos(s[2]), np.sin(s[2])]) * 0.28)
            pygame.draw.line(self.tela, (255, 255, 255), c, ponta, 3)
        if self.tem_objetivo:
            self.render._objetivo(self.track.goal, self.cam)

    def _previa(self):
        """Enquanto arrasta uma caixa, mostra o retângulo e a medida em cm."""
        if self.arrastando is None or self.arrastando[0] != "caixa":
            return
        inicio = self.arrastando[1]
        atual = self.mundo(self.cursor)
        cantos = np.array([[inicio[0], inicio[1]], [atual[0], inicio[1]],
                           [atual[0], atual[1]], [inicio[0], atual[1]]])
        pygame.draw.polygon(self.tela, COR_AVISO, self.cam.to_screen(cantos), 2)
        larg, comp = abs(atual[0] - inicio[0]) * 100, abs(atual[1] - inicio[1]) * 100
        txt = self.render.font.render(f"{larg:.0f} x {comp:.0f} cm", True, COR_TEXTO)
        self.tela.blit(txt, self.cam.to_screen(atual) + np.array([12, 12]))

    def _painel(self):
        tela, f, fp = self.tela, self.render.font, self.render.font_peq
        pygame.draw.rect(tela, COR_PAINEL, (0, 0, LARGURA_PAINEL, self.tamanho[1]))
        pygame.draw.line(tela, COR_BORDA, (LARGURA_PAINEL, 0),
                         (LARGURA_PAINEL, self.tamanho[1]))
        y = 12

        def linha(txt, cor=COR_TEXTO, fonte=None, dx=8):
            nonlocal y
            tela.blit((fonte or fp).render(txt, True, cor), (12 + dx, y))
            y += (fonte or fp).get_linesize() + 3

        def regua():
            nonlocal y
            y += 6
            pygame.draw.line(tela, COR_BORDA, (12, y), (LARGURA_PAINEL - 12, y))
            y += 8

        linha("MODO", COR_FRACO, dx=0)
        for i, nome in enumerate(NOMES):
            if i == self.modo:
                pygame.draw.rect(tela, (32, 48, 74), (10, y - 2, LARGURA_PAINEL - 22,
                                                      fp.get_linesize() + 4), border_radius=4)
            linha(f"{i+1} · {nome}", (127, 180, 240) if i == self.modo else COR_TEXTO)

        regua()
        linha("PISTA", COR_FRACO, dx=0)
        linha(f"largura   {self.track.width:.2f} m")
        linha(f"pontos    {len(self.pontos)}")
        linha(f"extensao  {self._extensao():.1f} m")
        linha(f"caixas    {len(self.track.obstacles)}")
        linha(f"grade     {'5 cm' if self.grade else 'desligada'}")

        regua()
        linha("FALTA" if not self.pronta() else "ESTADO", COR_FRACO, dx=0)
        for marca, texto, cor in self.pendencias():
            linha(f"{marca} {texto}", cor)

        regua()
        for texto in [
            "[ ]  largura/giro/raio",
            "roda  zoom",
            "meio  arrastar tela",
            "dir   apagar sob o cursor",
            "a  largada+objetivo auto",
            "t testar    s salvar",
            "l abrir     n nova",
            "g grade     z desfazer",
            "esc sair",
        ]:
            linha(texto, COR_FRACO)

        if self.recado:
            img = fp.render(self.recado, True, COR_AVISO)
            tela.blit(img, (12, self.tamanho[1] - 24))

    def _extensao(self):
        if len(self.pontos) < 2:
            return 0.0
        pts = np.array(self.pontos)
        return float(np.hypot(*(pts[1:] - pts[:-1]).T).sum())


def main(argv=None):
    p = argparse.ArgumentParser(description="Editor de pista do robô desviador")
    p.add_argument("--abrir", default=None,
                   help=f"JSON existente ou pista embutida ({', '.join(PISTAS)})")
    args = p.parse_args(argv)

    track = None
    if args.abrir:
        try:
            track = carregar(args.abrir)
        except (FileNotFoundError, KeyError) as e:
            print(f"não consegui abrir: {e}")
            return 1

    Editor(track).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
