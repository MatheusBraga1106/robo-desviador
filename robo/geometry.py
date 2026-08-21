"""Primitivas geométricas vetorizadas.

Tudo aqui opera em lote sobre todos os robôs e todos os raios de uma vez — é o
que permite simular uma população inteira sem laço em Python.
"""

import numpy as np


def ray_segment_hits(origins, dirs, seg_a, seg_b, edge=None, normal=None):
    """Interseção de raios com segmentos de parede.

    origins : (..., 2) origem de cada raio
    dirs    : (..., 2) direção unitária de cada raio
    seg_a   : (S, 2) início de cada parede — **ou** um lote com prefixo igual
              ao de `origins`/`dirs` até a dimensão S (ver abaixo)
    seg_b   : mesma forma de `seg_a`
    edge    : opcional, já `seg_b - seg_a`, mesma forma de `seg_a`
    normal  : opcional, já a normal normalizada de cada parede, mesma forma

    Duas formas de usar:

    1. **Segmentos globais** — `seg_a`/`seg_b` em (S, 2), os mesmos para toda
       consulta. É o caso simples: broadcasting comum do numpy contra
       `origins`/`dirs` de qualquer forma (..., 2).
    2. **Candidatos por consulta** — `seg_a`/`seg_b` já filtrados por
       proximidade (veja `Track.candidatos_proximos`), com um prefixo de forma
       que faça broadcasting contra `origins[..., None, :]`. Por exemplo, com
       `origins` em (P, m, k, 2), passe os candidatos em (P, 1, 1, K, 2) — o
       chamador insere os eixos que faltam, esta função não presume nada sobre
       onde fica o eixo P.

    `edge`/`normal` só dependem de `seg_a`/`seg_b`, não do raio — passe-os
    pré-computados (veja `Track._precomputar_geometria_paredes`) quando os
    mesmos segmentos forem usados em muitas chamadas seguidas, o caso comum de
    uma pista que não muda dentro do episódio. Sem eles, calcula na hora, como
    sempre foi.

    Devolve (t, cos_incidencia), ambos com forma (..., S):
      t              distância ao longo do raio até o ponto de impacto (inf se não bate)
      cos_incidencia |cos| do ângulo entre o raio e a normal da parede. 1 = batida
                     de frente, 0 = raio rasante. É isso que decide se o eco volta.
    """
    if edge is None:
        edge = seg_b - seg_a

    d = dirs[..., None, :]                                # (..., 1, 2)
    o = origins[..., None, :]                             # (..., 1, 2)
    ao = seg_a - o                                        # (..., S, 2)

    # produto vetorial 2D (escalar): u x v = u.x*v.y - u.y*v.x
    denom = d[..., 0] * edge[..., 1] - d[..., 1] * edge[..., 0]      # (..., S)
    paralelo = np.abs(denom) < 1e-12
    safe = np.where(paralelo, 1.0, denom)

    t = (ao[..., 0] * edge[..., 1] - ao[..., 1] * edge[..., 0]) / safe
    u = (ao[..., 0] * d[..., 1] - ao[..., 1] * d[..., 0]) / safe

    valido = (~paralelo) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(valido, t, np.inf)

    if normal is None:
        # normal da parede, normalizada
        comp = np.hypot(edge[..., 0], edge[..., 1])
        comp = np.where(comp < 1e-12, 1.0, comp)
        normal = np.stack([-edge[..., 1] / comp, edge[..., 0] / comp], axis=-1)
    nx, ny = normal[..., 0], normal[..., 1]

    cos_inc = np.abs(d[..., 0] * nx + d[..., 1] * ny)
    cos_inc = np.where(valido, cos_inc, 0.0)
    return t, cos_inc


def point_segment_distance(points, seg_a, seg_b, edge=None, comp2=None):
    """Distância de cada ponto a cada segmento.

    points : (..., 2)
    seg_a  : (S, 2) segmentos globais, **ou** um lote (..., S, 2) com prefixo
             igual ao de `points` — mesma ideia de `ray_segment_hits`, ver lá
    edge   : opcional, já `seg_b - seg_a`
    comp2  : opcional, já o comprimento² de cada segmento

    Mesma ideia de `ray_segment_hits`: passe `edge`/`comp2` pré-computados para
    não recalcular a cada chamada quando os segmentos são sempre os mesmos.

    devolve: (..., S)
    """
    if edge is None:
        edge = seg_b - seg_a
    if comp2 is None:
        comp2 = np.sum(edge * edge, axis=-1)
        comp2 = np.where(comp2 < 1e-12, 1.0, comp2)

    ap = points[..., None, :] - seg_a                     # (..., S, 2)
    u = np.clip(np.sum(ap * edge, axis=-1) / comp2, 0.0, 1.0)

    proj = seg_a + u[..., None] * edge                    # (..., S, 2)
    delta = points[..., None, :] - proj
    return np.hypot(delta[..., 0], delta[..., 1])


def project_polyline(points, poly):
    """Projeta pontos numa polilinha. Devolve (distância, comprimento de arco).

    O arco é quanto do traçado já foi percorrido até o ponto mais próximo — é a
    medida certa de "o quanto avancei", diferente da distância em linha reta até
    o fim, que num corredor sinuoso mede o lugar errado.

    points : (..., 2)
    poly   : (N, 2)
    """
    poly = np.asarray(poly, dtype=np.float64)
    a, b = poly[:-1], poly[1:]
    edge = b - a
    comp = np.hypot(edge[:, 0], edge[:, 1])
    comp2 = np.where(comp < 1e-12, 1.0, comp ** 2)
    acumulado = np.concatenate([[0.0], np.cumsum(comp)])

    ap = np.asarray(points)[..., None, :] - a                # (..., S, 2)
    u = np.clip(np.einsum("...sk,sk->...s", ap, edge) / comp2, 0.0, 1.0)
    proj = a + u[..., None] * edge
    delta = np.asarray(points)[..., None, :] - proj
    d = np.hypot(delta[..., 0], delta[..., 1])               # (..., S)

    melhor = np.argmin(d, axis=-1)
    dist = np.take_along_axis(d, melhor[..., None], axis=-1)[..., 0]
    arco = (acumulado[melhor]
            + np.take_along_axis(u, melhor[..., None], axis=-1)[..., 0] * comp[melhor])
    return dist, arco


def segments_cross(p1, p2, q1, q2) -> np.ndarray:
    """Dois segmentos se cruzam? Todos (..., 2), devolve (...,) booleano.

    Usado para saber se o robô atravessou um checkpoint no passo — testar só a
    posição atual deixaria passar batido quando o robô é rápido.
    """
    def cruz(a, b, c):
        return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    d1, d2 = cruz(q1, q2, p1), cruz(q1, q2, p2)
    d3, d4 = cruz(p1, p2, q1), cruz(p1, p2, q2)
    return ((d1 * d2) < 0) & ((d3 * d4) < 0)


def offset_polyline(points, distance):
    """Desloca uma polilinha lateralmente, com junta em esquadria (miter).

    Usado para gerar as duas paredes de um corredor a partir da linha central.
    `distance` positivo desloca para a esquerda de quem percorre a linha.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("a linha central precisa de pelo menos 2 pontos")

    seg = pts[1:] - pts[:-1]
    comp = np.hypot(seg[:, 0], seg[:, 1])[:, None]
    comp = np.where(comp < 1e-12, 1.0, comp)
    dir_ = seg / comp
    normal = np.stack([-dir_[:, 1], dir_[:, 0]], axis=1)   # normal à esquerda

    out = np.empty_like(pts)
    out[0] = pts[0] + normal[0] * distance
    out[-1] = pts[-1] + normal[-1] * distance

    for i in range(1, len(pts) - 1):
        n0, n1 = normal[i - 1], normal[i]
        m = n0 + n1
        comp_m = np.hypot(m[0], m[1])
        if comp_m < 1e-9:                      # curva de 180°, sem esquadria possível
            out[i] = pts[i] + n1 * distance
            continue
        m = m / comp_m
        # alonga a esquadria para os dois lados se encontrarem no vértice
        escala = 1.0 / max(np.dot(m, n1), 0.35)
        out[i] = pts[i] + m * distance * escala

    return out


def polyline_segments(points):
    """Polilinha (N,2) -> segmentos (N-1, 4) no formato [x1, y1, x2, y2]."""
    pts = np.asarray(points, dtype=np.float64)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)
