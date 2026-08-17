"""Verificações do editor, sem abrir janela.

Roda com o driver de vídeo "dummy": o pygame funciona, desenha numa superfície
de mentira, e dá para exercitar cliques e teclas de verdade.

    python -m tests.test_editor
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import tempfile

import numpy as np
import pygame

from robo.editor import (Editor, MODO_LINHA, MODO_CAIXA, MODO_LARGADA,
                         MODO_OBJETIVO, LARGURA_PAINEL)
from robo.pistas import zigue_zague
from robo.track import Track

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def novo_editor(track=None):
    ed = Editor(track, tamanho=(900, 600))
    ed.grade = False           # sem encaixe, para as coordenadas baterem exatas
    return ed


def tela_de(ed, x, y):
    """Ponto do mundo -> pixel, para simular cliques onde eu quero."""
    p = ed.cam.to_screen([x, y])
    return (int(round(p[0])), int(round(p[1])))


def clicar(ed, x, y, botao=1):
    pos = tela_de(ed, x, y)
    ed.mouse_desce(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=botao))
    ed.mouse_sobe(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=botao))


def arrastar(ed, x0, y0, x1, y1, botao=1):
    ini, fim = tela_de(ed, x0, y0), tela_de(ed, x1, y1)
    ed.mouse_desce(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=ini, button=botao))
    ed.mouse_move(pygame.event.Event(pygame.MOUSEMOTION, pos=fim, buttons=(1, 0, 0)))
    ed.mouse_sobe(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=fim, button=botao))


def teclar(ed, key):
    return ed.tecla(pygame.event.Event(pygame.KEYDOWN, key=key))


# ---------------------------------------------------------------------- #
def test_desenhar_linha():
    print("\ndesenhar a linha central")
    ed = novo_editor()
    for x, y in [(1, 1), (3, 1), (3, 3)]:
        clicar(ed, x, y)
    checar("três cliques, três pontos", len(ed.pontos) == 3, f"{len(ed.pontos)}")
    checar("corredor gerado", ed.track.centerline is not None and len(ed.track.walls) > 0,
           f"{len(ed.track.walls)} paredes")

    teclar(ed, pygame.K_z)
    checar("z desfaz o último ponto", len(ed.pontos) == 2)

    # clicar em cima de um ponto existente move, não acrescenta
    arrastar(ed, 1, 1, 1.5, 1.4)
    checar("arrastar move o ponto, não cria outro", len(ed.pontos) == 2)
    checar("o ponto foi mesmo para onde arrastei",
           np.allclose(ed.pontos[0], (1.5, 1.4), atol=0.02), f"{np.round(ed.pontos[0],2)}")


def test_painel_nao_desenha():
    print("\ncliques no painel")
    ed = novo_editor()
    ed.mouse_desce(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                      pos=(LARGURA_PAINEL - 20, 300), button=1))
    checar("clique no painel não vira ponto", len(ed.pontos) == 0)


def test_caixa_por_arrasto():
    print("\ncaixa por arrasto")
    ed = novo_editor()
    ed.modo = MODO_CAIXA
    arrastar(ed, 2.0, 2.0, 2.3, 2.2)
    checar("uma caixa criada", len(ed.track.obstacles) == 1)
    if len(ed.track.obstacles):
        cx, cy, larg, comp, _ = ed.track.obstacles[0]
        checar("centro entre os dois cantos", abs(cx - 2.15) < 0.02 and abs(cy - 2.1) < 0.02,
               f"({cx:.2f}, {cy:.2f})")
        checar("medidas do arrasto", abs(larg - 0.3) < 0.02 and abs(comp - 0.2) < 0.02,
               f"{larg:.2f} x {comp:.2f} m")

    arrastar(ed, 4.0, 4.0, 4.005, 4.005)
    checar("arrasto minúsculo é recusado", len(ed.track.obstacles) == 1, ed.recado)

    ed.modo = MODO_CAIXA
    clicar(ed, 2.15, 2.1, botao=3)
    checar("botão direito apaga a caixa", len(ed.track.obstacles) == 0)


def test_giro_da_caixa():
    print("\ngirar caixa com [ ]")
    ed = novo_editor()
    ed.modo = MODO_CAIXA
    arrastar(ed, 2.0, 2.0, 2.4, 2.2)
    ed.cursor = tela_de(ed, 2.2, 2.1)     # cursor em cima da caixa
    antes = float(ed.track.obstacles[0, 4])
    ed.escalar(+1)
    depois = float(ed.track.obstacles[0, 4])
    checar("] gira 15°", abs(np.rad2deg(depois - antes) - 15) < 1e-6,
           f"{np.rad2deg(depois-antes):.1f}°")
    checar("as paredes da caixa acompanharam o giro",
           not np.allclose(ed.track.walls[-4:], _paredes_sem_giro(ed)))

    ed.cursor = tela_de(ed, 5.0, 5.0)     # cursor longe de qualquer caixa
    ed.escalar(+1)
    checar("fora da caixa, não gira nada",
           abs(float(ed.track.obstacles[0, 4]) - depois) < 1e-12)


def _paredes_sem_giro(ed):
    from robo.track import _box_walls
    sem = ed.track.obstacles.copy()
    sem[:, 4] = 0.0
    return _box_walls(sem)


def test_largura_do_corredor():
    print("\nlargura com [ ]")
    ed = novo_editor()
    for x, y in [(1, 1), (3, 1)]:
        clicar(ed, x, y)
    ed.escalar(+1)
    checar("] aumenta 5 cm", abs(ed.track.width - 0.95) < 1e-9, f"{ed.track.width:.2f} m")
    checar("as paredes acompanharam",
           abs(abs(ed.track.walls[0][1] - 1.0) - 0.475) < 1e-6)


def test_largada_e_objetivo():
    print("\nlargada e objetivo por arrasto")
    ed = novo_editor()
    ed.modo = MODO_LARGADA
    arrastar(ed, 1.0, 1.0, 1.0, 2.0)      # arrasta para cima = olhando +y
    checar("posição da largada", np.allclose(ed.track.start[:2], (1, 1), atol=0.02))
    checar("arrastar define para onde olha",
           abs(np.rad2deg(ed.track.start[2]) - 90) < 2,
           f"{np.rad2deg(ed.track.start[2]):.0f}°")

    ed.modo = MODO_OBJETIVO
    arrastar(ed, 3.0, 3.0, 3.4, 3.0)
    checar("posição do objetivo", np.allclose(ed.track.goal[:2], (3, 3), atol=0.02))
    checar("arrastar define o raio", abs(ed.track.goal[2] - 0.4) < 0.02,
           f"raio {ed.track.goal[2]:.2f} m")


def test_bloqueio_ate_estar_pronta():
    print("\nsalvar exige largada e objetivo")
    ed = novo_editor()
    for x, y in [(1, 1), (3, 1)]:
        clicar(ed, x, y)
    checar("ainda não está pronta", not ed.pronta())
    ed.salvar()
    checar("salvar recusa e avisa", ed.nome_arquivo is None and "falta" in ed.recado,
           f"'{ed.recado}'")

    teclar(ed, pygame.K_a)
    checar("a preenche os dois", ed.pronta() and ed.tem_largada and ed.tem_objetivo)
    checar("largada saiu no começo do traçado",
           ed.track.start[0] > 1.0 and abs(ed.track.start[1] - 1.0) < 0.01,
           f"{np.round(ed.track.start, 2)}")


def test_avisos_de_geometria():
    print("\navisos de geometria")
    ed = novo_editor()
    ed.track.width = 1.0
    for x, y in [(1, 1), (3, 1), (3.05, 1.4)]:   # vira quase 180° num espaço curto
        clicar(ed, x, y)
    checar("acusa a curva impossível", len(ed.curvas_apertadas()) >= 1,
           f"vértices ruins: {ed.curvas_apertadas()}")

    ed2 = novo_editor()
    ed2.track.width = 0.6
    for x, y in [(1, 1), (3, 1), (3, 3)]:        # 90° com trechos longos
        clicar(ed2, x, y)
    checar("curva de 90° com folga não acusa", ed2.curvas_apertadas() == [],
           f"{ed2.curvas_apertadas()}")

    # trecho menor que a largura enviesa a tampa da ponta
    ed3 = novo_editor()
    ed3.track.width = 0.9
    for x, y in [(1, 1), (3, 1), (3.5, 1)]:      # último trecho de 0.50 m
        clicar(ed3, x, y)
    checar("acusa trecho curto demais", set(ed3.curvas_apertadas()) >= {1, 2},
           f"{ed3.curvas_apertadas()}")

    # estreitar abaixo dos 0.50 m do trecho limpa o aviso.
    # o passo é de 5 cm e o `escalar` trava em 0.30 m, então isso termina.
    for _ in range(12):
        if ed3.track.width < 0.5:
            break
        ed3.escalar(-1)
    checar("estreitar o corredor resolve", ed3.curvas_apertadas() == [],
           f"largura {ed3.track.width:.2f} m < trecho 0.50 m, ruins {ed3.curvas_apertadas()}")


def test_salvar_e_reabrir():
    print("\nsalvar e reabrir mantém tudo")
    ed = novo_editor()
    for x, y in [(1, 1), (3, 1), (3, 2.5)]:
        clicar(ed, x, y)
    ed.modo = MODO_CAIXA
    arrastar(ed, 2.0, 0.9, 2.3, 1.1)
    teclar(ed, pygame.K_a)

    caminho = os.path.join(tempfile.gettempdir(), "editor_teste.json")
    ed.nome_arquivo = caminho
    ed.salvar()
    checar("salvou", os.path.exists(caminho), ed.recado)

    ed2 = novo_editor(Track.load(caminho))
    os.remove(caminho)
    checar("pontos voltaram editáveis", len(ed2.pontos) == 3, f"{len(ed2.pontos)}")
    checar("caixa voltou", len(ed2.track.obstacles) == 1)
    checar("largada e objetivo voltaram", ed2.tem_largada and ed2.tem_objetivo)

    # e o ponto do refactor do Track: dá para continuar editando
    ed2.escalar(+1)
    checar("largura ainda ajustável depois de reabrir",
           abs(ed2.track.width - (ed.track.width + 0.05)) < 1e-9)


def test_abrir_pista_embutida():
    print("\nabrir pista embutida")
    ed = novo_editor(zigue_zague())
    checar("linha central disponível para editar", len(ed.pontos) == 6, f"{len(ed.pontos)}")
    ed.pontos[-1] = (5.5, 3.5)
    ed.regerar()
    checar("editar a linha regera o corredor",
           np.allclose(ed.track.centerline[-1], (5.5, 3.5)))


def test_desenha_sem_estourar():
    print("\ndesenhar em cada modo")
    ed = novo_editor()
    for x, y in [(1, 1), (3, 1), (3, 3)]:
        clicar(ed, x, y)
    ed.modo = MODO_CAIXA
    arrastar(ed, 2.0, 0.9, 2.3, 1.1)
    teclar(ed, pygame.K_a)
    try:
        for modo in (MODO_LINHA, MODO_CAIXA, MODO_LARGADA, MODO_OBJETIVO):
            ed.modo = modo
            ed.desenhar()
        # e com a pista ainda vazia, que é o estado da primeira tela
        vazio = novo_editor()
        vazio.desenhar()
        ok, erro = True, ""
    except Exception as e:
        ok, erro = False, f"{type(e).__name__}: {e}"
    checar("desenha nos 4 modos e na pista vazia", ok, erro)


def main():
    pygame.init()
    for fn in [test_desenhar_linha, test_painel_nao_desenha, test_caixa_por_arrasto,
               test_giro_da_caixa, test_largura_do_corredor, test_largada_e_objetivo,
               test_bloqueio_ate_estar_pronta, test_avisos_de_geometria,
               test_salvar_e_reabrir, test_abrir_pista_embutida, test_desenha_sem_estourar]:
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
