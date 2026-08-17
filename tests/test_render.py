"""Verificações do desenho (robo/render.py).

Dois bugs reais foram achados montando o modo "assistir a população inteira"
e ficam travados aqui:

1. `pygame.draw.circle` com uma cor RGBA (alfa) numa Surface sem `SRCALPHA`
   ignora o alfa e desenha opaco — testado diretamente, pedir alfa 120
   devolve um pixel com alfa 255. Por isso os robôs "de fundo" usam uma cor
   sólida escurecida, não transparência.
2. Numa população que ainda não aprendeu a se espalhar, é comum muitos
   indivíduos ficarem exatamente empilhados. Desenhar o robô destacado
   primeiro e os outros por cima o fazia sumir debaixo da pilha.

    python -m tests.test_render
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys

import numpy as np
import pygame

from robo.config import SimConfig
from robo.pistas import zigue_zague
from robo.render import COR_ROBO, COR_ROBO_FUNDO, COR_ROBO_MORTO, Camera, Renderer
from robo.world import World

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def contar_pixels(surface, cor):
    arr = pygame.surfarray.array3d(surface)
    return int(np.all(arr == np.array(cor), axis=-1).sum())


# ---------------------------------------------------------------------- #
def test_pygame_ignora_alfa_em_superficie_comum():
    print("\npremissa: pygame ignora alfa numa Surface sem SRCALPHA")
    tela = pygame.Surface((60, 60))
    tela.fill((0, 0, 0))
    pygame.draw.circle(tela, (60, 220, 190, 120), (30, 30), 15)
    pixel = tela.get_at((30, 30))
    checar("o pixel sai opaco (alfa 255), não translúcido", pixel.a == 255, f"alfa={pixel.a}")
    checar("por isso COR_ROBO_FUNDO é uma cor sólida diferente, não a mesma com alfa",
           tuple(COR_ROBO_FUNDO) != COR_ROBO, f"{COR_ROBO_FUNDO} vs {COR_ROBO}")


def test_robo_destacado_visivel_com_populacao_empilhada():
    print("\nrobô destacado continua visível quando outros se empilham em cima")
    cfg = SimConfig()
    N = 40
    track = zigue_zague()
    w = World(track, cfg, N)
    w.reset()
    w.theta[:] = w.theta[0]
    # metade empilhada exatamente sobre o destacado (o pior caso: cobre tudo
    # que estiver embaixo), metade espalhada — assim o teste sabe, sem
    # depender de sorte, que os dois tons têm que aparecer em algum lugar
    metade = N // 2
    w.pos[:metade] = w.pos[0]
    for i in range(metade, N):
        w.pos[i] = w.pos[0] + np.array([0.15 * (i - metade + 1), 0.0])

    tela = pygame.Surface((900, 600))
    cam = Camera((900, 600))
    cam.fit(track.bounds)
    r = Renderer(tela)
    r.fundo()
    r.desenhar_pista(track, cam)
    r.desenhar_robos(w, cam, detalhe_em=0)

    checar("existe pelo menos um pixel da cor cheia (o destacado apareceu)",
           contar_pixels(tela, COR_ROBO) > 0)
    checar("e também existem pixels da cor de fundo (os espalhados aparecem)",
           contar_pixels(tela, COR_ROBO_FUNDO) > 0)


def test_destacado_no_topo_mesmo_desenhado_por_ultimo_na_ordem_dos_indices():
    print("\no destaque vence mesmo quando é o primeiro da lista (pior ordem)")
    cfg = SimConfig()
    N = 10
    track = zigue_zague()
    w = World(track, cfg, N)
    w.reset()
    w.pos[:] = w.pos[0]   # todos empilhados, incluindo o robô 0 (o destacado)

    tela = pygame.Surface((900, 600))
    cam = Camera((900, 600))
    cam.fit(track.bounds)
    r = Renderer(tela)
    r.fundo()
    r.desenhar_robos(w, cam, detalhe_em=0)
    checar("o robô 0 aparece por cima mesmo estando embaixo na ordem de índices",
           contar_pixels(tela, COR_ROBO) > 0)


def test_robo_morto_fica_com_cor_diferente_do_vivo():
    print("\nrobô que colidiu usa cor diferente do vivo")
    cfg = SimConfig()
    N = 5
    track = zigue_zague()
    w = World(track, cfg, N)
    w.reset()
    w.alive[:] = True
    w.alive[0] = False       # simula o destacado como já tendo colidido
    w.pos[:] = w.pos[0] + np.array([0.3, 0.0])   # afasta da pilha, sem sobrepor

    tela = pygame.Surface((900, 600))
    cam = Camera((900, 600))
    cam.fit(track.bounds)
    r = Renderer(tela)
    r.fundo()
    r.desenhar_robos(w, cam, detalhe_em=0)
    checar("cor de morto aparece, não a de vivo", contar_pixels(tela, COR_ROBO_MORTO) > 0
           and contar_pixels(tela, COR_ROBO) == 0)


def test_um_robo_so_nao_fica_esmaecido():
    print("\ncom um único robô (P=1), ele sempre usa a cor cheia")
    cfg = SimConfig()
    track = zigue_zague()
    w = World(track, cfg, 1)
    w.reset()

    tela = pygame.Surface((900, 600))
    cam = Camera((900, 600))
    cam.fit(track.bounds)
    r = Renderer(tela)
    r.fundo()
    r.desenhar_robos(w, cam, detalhe_em=0)
    checar("aparece com a cor cheia, não a de fundo", contar_pixels(tela, COR_ROBO) > 0)


def test_grafico_casos_degenerados():
    print("\ngráfico do treino: casos que quebrariam")
    from robo.grafico import PainelGrafico
    from brains.treino_ga import Hiperparametros

    painel = PainelGrafico(pygame.Rect(10, 10, 420, 240))
    hiper = Hiperparametros()

    def desenhar(historico, com_hiper=True):
        tela = pygame.Surface((900, 600))
        tela.fill((0, 0, 0))
        painel.draw(tela, historico, hiper if com_hiper else None)
        return tela

    casos = {
        "histórico vazio": [],
        "uma geração só": [{"geracao": 0, "melhor": 1.0, "media": 0.0, "pior": -1.0}],
        "duas gerações": [{"geracao": i, "melhor": 1.0 * i, "media": 0.0, "pior": -1.0}
                          for i in range(2)],
        "tudo igual (escala degenerada)": [
            {"geracao": i, "melhor": 5.0, "media": 5.0, "pior": 5.0} for i in range(8)],
        "sem hiperparâmetros": [
            {"geracao": i, "melhor": float(i), "media": 0.0, "pior": -1.0}
            for i in range(6)],
        "valores enormes": [
            {"geracao": i, "melhor": 1e6, "media": -1e6, "pior": -1e9} for i in range(5)],
    }
    for nome, hist in casos.items():
        try:
            desenhar(hist, com_hiper=(nome != "sem hiperparâmetros"))
            ok, detalhe = True, ""
        except Exception as e:
            ok, detalhe = False, f"{type(e).__name__}: {e}"
        checar(nome, ok, detalhe)

    tela = desenhar([{"geracao": i, "melhor": float(i), "media": 0.0, "pior": -2.0}
                     for i in range(30)])
    checar("com dados de verdade, pinta alguma coisa",
           pygame.surfarray.array3d(tela)[10:430, 10:250].sum() > 0)


def test_grafico_nao_vaza_do_painel():
    print("\ngráfico não escreve fora do próprio painel")
    from robo.grafico import PainelGrafico
    from brains.treino_ga import Hiperparametros

    rect = pygame.Rect(10, 10, 420, 240)
    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    hist = [{"geracao": i, "melhor": 100.0 - i, "media": 13.0, "pior": -54.0}
            for i in range(40)]
    PainelGrafico(rect).draw(tela, hist, Hiperparametros())

    arr = pygame.surfarray.array3d(tela)
    fora = arr.copy()
    fora[rect.x:rect.right, rect.y:rect.bottom] = 0     # apaga o painel
    checar("nada foi desenhado fora do retângulo", int(fora.sum()) == 0,
           f"{int(fora.sum())} de intensidade vazou")


def test_painel_de_pesos():
    print("\npainel de pesos do fitness")
    from robo.grafico import PainelPesos
    from brains.exemplo import PESOS, PesosFitness

    rect = pygame.Rect(10, 10, 260, 190)
    painel = PainelPesos(rect)

    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    painel.draw(tela, PESOS)
    checar("desenha os pesos", pygame.surfarray.array3d(tela).sum() > 0)

    # todo peso do dataclass tem que aparecer em `linhas()`, senão você calibra
    # um valor que a tela não mostra e fica sem saber o que está valendo
    from dataclasses import fields
    rotulos = " ".join(r for r, _ in PESOS.linhas()).lower()
    valores = " ".join(v for _, v in PESOS.linhas())
    p = PesosFitness(peso_avanco=7.5, punicao_batida=33.0, bonus_chegada=222.0,
                     peso_tempo=0.25, enrolar_peso=44.0)
    mostrado = " ".join(v for _, v in p.linhas())
    for esperado in ("7.5", "33", "222", "0.25", "44"):
        checar(f"o valor {esperado} aparece na tela", esperado in mostrado, mostrado)

    checar("a altura acompanha o número de linhas",
           painel.altura_necessaria(PESOS) > len(PESOS.linhas()) * 10)

    tela2 = pygame.Surface((900, 600))
    tela2.fill((0, 0, 0))
    painel.draw(tela2, None)
    checar("sem pesos, não desenha nada", pygame.surfarray.array3d(tela2).sum() == 0)


def test_editar_pesos_muda_o_objeto_de_verdade():
    print("\neditar pesos pelo painel")
    from robo.grafico import PainelPesos
    from brains.exemplo import PesosFitness

    pesos = PesosFitness()
    mudancas = []
    painel = PainelPesos(pygame.Rect(100, 100, 280, 250), pesos,
                         ao_mudar=lambda: mudancas.append(1))

    clique = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(200, 150), button=1)
    checar("fechado, ignora o mouse", not painel.evento(clique))
    checar("e nada mudou", not mudancas)

    painel.alternar_edicao()
    checar("a tecla abre a edição", painel.editando)

    barra = painel.barras[0]
    y = next(yy for b, yy in painel._posicoes() if b is barra)
    antes = barra.valor

    # arrasta até a direita: o valor tem que ir ao máximo da barra
    painel.evento(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                     pos=(painel.rect.right - 15, y + 10), button=1))
    checar("arrastar muda o peso", pesos.peso_avanco != antes,
           f"{antes} -> {pesos.peso_avanco:.1f}")
    checar("respeita o teto da barra", pesos.peso_avanco <= barra.max + 1e-9)
    checar("avisou quem escuta", len(mudancas) >= 1, f"{len(mudancas)} avisos")

    # e para a esquerda vai ao mínimo
    painel.evento(pygame.event.Event(pygame.MOUSEMOTION, pos=(painel.rect.x - 50, y + 10)))
    checar("respeita o piso da barra", pesos.peso_avanco >= barra.min - 1e-9,
           f"{pesos.peso_avanco:.2f}")
    painel.evento(pygame.event.Event(pygame.MOUSEBUTTONUP,
                                     pos=(painel.rect.x, y + 10), button=1))
    checar("soltar encerra o arrasto", painel.arrastando is None)

    fora = pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(5, 5), button=1)
    checar("clique fora do painel não é consumido", not painel.evento(fora))


def test_viewer_marca_a_geracao_em_que_os_pesos_mudaram():
    print("\ngráfico marca onde a régua mudou")
    from robo.config import SimConfig
    from robo.viewer import Viewer
    from brains.exemplo import PesosFitness

    hist = [{"geracao": i, "melhor": float(i), "media": 0.0, "pior": -5.0}
            for i in range(10)]
    v = Viewer(zigue_zague(), SimConfig(), brain=None, tamanho=(900, 600),
               historico=hist, pesos=PesosFitness())

    checar("começa sem marcas", v.marcas_pesos == [])
    v._anotar_mudanca_de_pesos()
    checar("anota a geração atual", v.marcas_pesos == [10], f"{v.marcas_pesos}")
    v._anotar_mudanca_de_pesos()
    checar("dois ajustes na mesma geração viram uma marca só",
           v.marcas_pesos == [10], f"{v.marcas_pesos}")

    hist.append({"geracao": 10, "melhor": 11.0, "media": 1.0, "pior": -5.0})
    v._anotar_mudanca_de_pesos()
    checar("mas numa geração nova, marca de novo", v.marcas_pesos == [10, 11],
           f"{v.marcas_pesos}")

    w = World(zigue_zague(), SimConfig(), 3)
    w.reset()
    try:
        v.desenhar(w, 1)
        ok, detalhe = True, ""
    except Exception as e:
        ok, detalhe = False, f"{type(e).__name__}: {e}"
    checar("e o gráfico desenha com as marcas", ok, detalhe)


def test_painel_de_pesos_nao_vaza():
    print("\npainel de pesos não escreve fora do retângulo")
    from robo.grafico import PainelPesos
    from brains.exemplo import PesosFitness

    rect = pygame.Rect(30, 40, 260, 190)
    tela = pygame.Surface((900, 600))
    tela.fill((0, 0, 0))
    painel = PainelPesos(rect)
    painel.draw(tela, PesosFitness(bonus_chegada=999999.0, peso_avanco=123.456))

    arr = pygame.surfarray.array3d(tela).copy()
    arr[rect.x:rect.right, rect.y:rect.bottom] = 0
    checar("nada vazou", int(arr.sum()) == 0, f"{int(arr.sum())} de intensidade fora")


def test_viewer_sem_historico_nao_desenha_grafico():
    print("\nviewer sem histórico não tenta desenhar gráfico")
    from robo.config import SimConfig
    from robo.viewer import Viewer

    track = zigue_zague()
    v = Viewer(track, SimConfig(), brain=None, tamanho=(800, 520))
    checar("gráfico desligado quando não há histórico", not v.mostrar_grafico)

    v2 = Viewer(track, SimConfig(), brain=None, tamanho=(800, 520), historico=[])
    checar("ligado quando o histórico existe, mesmo vazio", v2.mostrar_grafico)

    w = World(SimConfig() and track, SimConfig(), 3)
    w.reset()
    try:
        v.desenhar(w, 1)
        v2.desenhar(w, 1)
        ok, detalhe = True, ""
    except Exception as e:
        ok, detalhe = False, f"{type(e).__name__}: {e}"
    checar("os dois desenham sem estourar", ok, detalhe)


def main():
    pygame.init()
    for fn in [test_pygame_ignora_alfa_em_superficie_comum,
               test_robo_destacado_visivel_com_populacao_empilhada,
               test_destacado_no_topo_mesmo_desenhado_por_ultimo_na_ordem_dos_indices,
               test_robo_morto_fica_com_cor_diferente_do_vivo,
               test_um_robo_so_nao_fica_esmaecido,
               test_grafico_casos_degenerados, test_grafico_nao_vaza_do_painel,
               test_painel_de_pesos, test_editar_pesos_muda_o_objeto_de_verdade,
               test_viewer_marca_a_geracao_em_que_os_pesos_mudaram,
               test_painel_de_pesos_nao_vaza,
               test_viewer_sem_historico_nao_desenha_grafico]:
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
