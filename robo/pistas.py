"""Percursos prontos, para ter o que jogar antes do editor existir.

Todos saem de uma linha central e uma largura — exatamente o que o editor vai
produzir. Medidas em metros, pensadas para um robô de ~18 cm num corredor de
90 cm, que é algo montável numa sala com caixas ou placas de papelão.
"""

import hashlib
import os

import numpy as np

from .track import Track

LARGURA_PADRAO = 0.9


def zigue_zague(largura=LARGURA_PADRAO) -> Track:
    """Três retas ligadas por curvas de 90°, ida e volta."""
    return Track.from_centerline(
        [(0.5, 0.5), (4.5, 0.5), (4.5, 2.0), (0.5, 2.0), (0.5, 3.5), (4.5, 3.5)],
        width=largura, name="zigue-zague",
    )


def curva_u(largura=LARGURA_PADRAO) -> Track:
    """Percurso curto em U — bom para os primeiros testes."""
    return Track.from_centerline(
        [(0.5, 0.5), (3.5, 0.5), (3.5, 2.5), (0.5, 2.5)],
        width=largura, name="curva-u",
    )


def caracol(largura=LARGURA_PADRAO) -> Track:
    """Espiral para dentro: as curvas vão ficando mais fechadas."""
    return Track.from_centerline(
        [(0.5, 0.5), (5.5, 0.5), (5.5, 4.5), (0.5, 4.5), (0.5, 1.7),
         (4.2, 1.7), (4.2, 3.2), (1.8, 3.2)],
        width=largura, name="caracol",
    )


def diagonais(largura=LARGURA_PADRAO) -> Track:
    """Paredes em ângulo — é aqui que o eco perdido do ultrassom aparece."""
    return Track.from_centerline(
        [(0.5, 0.5), (2.5, 1.6), (4.5, 0.6), (6.0, 2.4), (4.0, 3.8), (1.2, 3.2)],
        width=largura, name="diagonais",
    )


PISTAS = {
    "curva-u": curva_u,
    "zigue-zague": zigue_zague,
    "caracol": caracol,
    "diagonais": diagonais,
}


def carregar(nome_ou_caminho: str) -> Track:
    """Aceita o nome de uma pista embutida ou o caminho de um JSON."""
    if nome_ou_caminho in PISTAS:
        return PISTAS[nome_ou_caminho]()
    return Track.load(nome_ou_caminho)


# --------------------------------------------------------------------- #
PASTA_PISTAS = "tracks"


def pistas_salvas(pasta=PASTA_PISTAS):
    """Caminhos das pistas que você desenhou no editor, em ordem alfabética.

    Devolve lista vazia se a pasta não existe — quem nunca abriu o editor não
    tem pista salva, e isso não é erro.
    """
    if not os.path.isdir(pasta):
        return []
    return [os.path.join(pasta, f).replace("\\", "/")
            for f in sorted(os.listdir(pasta)) if f.endswith(".json")]


def catalogo(pasta=PASTA_PISTAS):
    """Todas as pistas disponíveis, embutidas e suas, com um resumo de cada uma.

    Cada item: {"ref", "origem", "nome", "comprimento", "caixas", "erro"}. `ref`
    é o que se passa em `--pista`. Uma pista salva com JSON corrompido entra na
    lista com `erro` preenchido, em vez de derrubar a listagem inteira — assim
    um arquivo ruim não impede você de escolher os outros.
    """
    itens = []
    for nome in PISTAS:
        t = PISTAS[nome]()
        itens.append({"ref": nome, "origem": "embutida", "nome": t.name,
                      "comprimento": t.length, "caixas": len(t.obstacles), "erro": None})

    for caminho in pistas_salvas(pasta):
        try:
            t = Track.load(caminho)
            itens.append({"ref": caminho, "origem": "sua", "nome": t.name,
                          "comprimento": t.length, "caixas": len(t.obstacles),
                          "erro": None})
        except Exception as e:
            itens.append({"ref": caminho, "origem": "sua", "nome": os.path.basename(caminho),
                          "comprimento": 0.0, "caixas": 0,
                          "erro": f"{type(e).__name__}: {e}"})
    return itens


def impressao_digital(track: Track) -> str:
    """Resumo curto da geometria da pista, para detectar que ela mudou.

    Só o nome não basta: você pode editar `tracks/pista1.json` no editor e o
    nome continuar o mesmo, enquanto o percurso virou outro. O hash pega isso.
    """
    partes = [np.asarray(track.walls, dtype=np.float64).round(6).tobytes(),
              np.asarray(track.start, dtype=np.float64).round(6).tobytes(),
              np.asarray(track.goal, dtype=np.float64).round(6).tobytes()]
    h = hashlib.sha1()
    for p in partes:
        h.update(p)
    return h.hexdigest()[:12]


def descrever(track: Track, ref: str = None) -> dict:
    """Identidade da pista, para guardar no checkpoint do treino."""
    return {"ref": ref or track.name, "nome": track.name,
            "hash": impressao_digital(track), "comprimento": round(track.length, 3)}
