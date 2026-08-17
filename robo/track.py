"""O percurso: a receita e as paredes que saem dela.

Por que guardar a receita em vez das paredes prontas
----------------------------------------------------
Uma pista podia ser só uma lista de segmentos. O problema aparece na hora de
reabrir: com as paredes prontas na mão, não dá mais para mudar a largura do
corredor nem mexer na linha central, porque a informação que gerou aquilo se
perdeu. Sobrou o bolo, sumiu a receita.

Então o `Track` guarda os **ingredientes** — linha central, largura, obstáculos —
e as paredes são **derivadas** deles por `rebuild()`. Editar vira mudar um
ingrediente e regerar. O JSON salva a receita, então uma pista salva continua
editável para sempre.

Pistas feitas à mão (paredes soltas, sem linha central) continuam funcionando:
elas entram como `extra_walls` e simplesmente não têm receita para regerar.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .geometry import (offset_polyline, polyline_segments, point_segment_distance,
                       project_polyline)

VAZIO_4 = np.zeros((0, 4))   # segmentos de parede [x1, y1, x2, y2]
VAZIO_5 = np.zeros((0, 5))   # caixas [cx, cy, largura, comprimento, ângulo]


@dataclass
class Track:
    # --- ingredientes (é isto que o editor mexe e o JSON guarda) ---
    centerline: Optional[np.ndarray] = None   # (N, 2) linha do meio do corredor
    width: float = 0.9                        # largura do corredor
    obstacles: np.ndarray = field(default_factory=lambda: VAZIO_5.copy())
    """(M, 5) -> [cx, cy, largura, comprimento, ângulo]. Caixas retangulares.

    `largura` é a medida no eixo local x da caixa (a face que olha para o ângulo)
    e `comprimento` a do eixo y. Retângulo e não quadrado porque o que você vai
    espalhar na sala tem forma: caixa de papelão deitada, tábua fina em pé, e
    cada uma se comporta diferente para o sensor.

    Caixa e não círculo porque uma caixa é só 4 segmentos de parede: reusa o
    raycast, a colisão e o teste de eco perdido que já existem. E é mais honesto
    com o HC-SR04, que precisa de face plana para o som voltar — objeto redondo
    espalha o eco e o sensor não enxerga direito."""

    extra_walls: np.ndarray = field(default_factory=lambda: VAZIO_4.copy())
    """Paredes soltas, para pistas montadas à mão sem linha central."""

    start: Optional[np.ndarray] = None        # (3,) [x, y, theta]
    goal: Optional[np.ndarray] = None         # (3,) [x, y, raio]
    checkpoint_spacing: float = 0.5
    close_ends: bool = True
    name: str = "sem nome"

    # --- derivados ---
    # `walls` e `floor` saem sempre do rebuild(), então nem aparecem no
    # construtor: aceitar `walls=` seria uma armadilha, porque o valor passado
    # seria sobrescrito em silêncio. Para paredes soltas, use `extra_walls`.
    walls: np.ndarray = field(default=None, init=False, repr=False)
    floor: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    checkpoints: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.centerline is not None:
            self.centerline = np.asarray(self.centerline, dtype=np.float64).reshape(-1, 2)
        self.obstacles = _normalizar_caixas(self.obstacles)
        self.extra_walls = np.asarray(self.extra_walls, dtype=np.float64).reshape(-1, 4)
        if self.start is not None:
            self.start = np.asarray(self.start, dtype=np.float64).reshape(3)
        if self.goal is not None:
            self.goal = np.asarray(self.goal, dtype=np.float64).reshape(3)
        self.rebuild()

    # ------------------------------------------------------------------ #
    def rebuild(self):
        """Regera paredes, chão e checkpoints a partir dos ingredientes.

        Chame depois de mexer em `centerline`, `width` ou `obstacles`. É barato:
        são algumas dezenas de segmentos, não vale a pena otimizar.
        """
        corredor, chao = self._corridor()
        partes = [p for p in (corredor, self.extra_walls, _box_walls(self.obstacles))
                  if len(p)]
        self.walls = np.concatenate(partes, axis=0) if partes else VAZIO_4.copy()
        self.floor = chao

        if self.checkpoints is None or self.centerline is not None:
            self.checkpoints = (_checkpoints_ao_longo(self.centerline, self.width,
                                                      self.checkpoint_spacing)
                                if self._tem_linha() else VAZIO_4.copy())

        # largada e objetivo têm padrão automático, mas o editor sobrescreve:
        # lá você escolhe onde o robô nasce e onde é a chegada.
        if self._tem_linha():
            if self.start is None:
                self.start = self._largada_automatica()
            if self.goal is None:
                self.goal = self._objetivo_automatico()
        if self.start is None:
            self.start = np.zeros(3)
        if self.goal is None:
            self.goal = np.array([0.0, 0.0, 0.2])

    def _tem_linha(self) -> bool:
        return self.centerline is not None and len(self.centerline) >= 2

    def _corridor(self):
        if not self._tem_linha():
            return VAZIO_4.copy(), None
        pts = self.centerline
        esquerda = offset_polyline(pts, +self.width / 2.0)
        direita = offset_polyline(pts, -self.width / 2.0)

        paredes = [polyline_segments(esquerda), polyline_segments(direita)]
        if self.close_ends:
            paredes.append(np.array([[*esquerda[0], *direita[0]]]))
            paredes.append(np.array([[*esquerda[-1], *direita[-1]]]))

        chao = np.concatenate([esquerda, direita[::-1]], axis=0)
        return np.concatenate(paredes, axis=0), chao

    def _largada_automatica(self):
        pts = self.centerline
        d = pts[1] - pts[0]
        d = d / max(np.hypot(*d), 1e-9)
        # afasta da parede que fecha o início: nascer em cima dela trava o robô
        origem = pts[0] + d * (self.width * 0.4) if self.close_ends else pts[0]
        return np.array([origem[0], origem[1], np.arctan2(d[1], d[0])])

    def _objetivo_automatico(self):
        pts = self.centerline
        d = pts[-1] - pts[-2]
        d = d / max(np.hypot(*d), 1e-9)
        alvo = pts[-1] - d * (self.width * 0.4) if self.close_ends else pts[-1]
        return np.array([alvo[0], alvo[1], self.width * 0.35])

    # ------------------------------------------------------------------ #
    def add_obstacle(self, cx, cy, largura=0.25, comprimento=None, angulo=0.0):
        """Acrescenta uma caixa. Sem `comprimento`, sai quadrada."""
        if comprimento is None:
            comprimento = largura
        self.obstacles = np.vstack([self.obstacles, [cx, cy, largura, comprimento, angulo]])
        self.rebuild()

    def contains_obstacle(self, x, y) -> np.ndarray:
        """Quais caixas contêm o ponto. Teste exato de retângulo.

        Leva o ponto para o referencial da caixa (desfazendo a rotação) e compara
        com as meias-medidas. Aproximar por um raio erraria bastante numa caixa
        comprida — que é justamente a que faz sentido usar como tábua.
        """
        if not len(self.obstacles):
            return np.zeros(0, dtype=bool)
        cx, cy, larg, comp, ang = self.obstacles.T
        dx, dy = x - cx, y - cy
        c, s = np.cos(-ang), np.sin(-ang)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        return (np.abs(lx) <= larg / 2.0) & (np.abs(ly) <= comp / 2.0)

    def remove_obstacle_at(self, x, y) -> bool:
        """Remove a caixa sob o ponto. Devolve se removeu alguma."""
        dentro = self.contains_obstacle(x, y)
        if not dentro.any():
            return False
        # com caixas sobrepostas, tira a menor: é a que o clique provavelmente mirava
        area = self.obstacles[:, 2] * self.obstacles[:, 3]
        alvo = int(np.argmin(np.where(dentro, area, np.inf)))
        self.obstacles = np.delete(self.obstacles, alvo, axis=0)
        self.rebuild()
        return True

    # ------------------------------------------------------------------ #
    @classmethod
    def from_centerline(cls, points, width=0.9, checkpoint_spacing=0.5,
                        close_start=True, close_end=True, name="sem nome",
                        obstacles=None):
        """Corredor a partir da linha do meio — o mesmo caminho do editor."""
        return cls(centerline=points, width=width,
                   checkpoint_spacing=checkpoint_spacing,
                   close_ends=close_start and close_end,
                   obstacles=obstacles if obstacles is not None else VAZIO_5.copy(),
                   name=name)

    # -- acesso conveniente para as rotinas vetorizadas -------------------- #
    @property
    def boundary_walls(self) -> np.ndarray:
        """Só as paredes do percurso, sem as caixas.

        A física não distingue as duas — caixa é parede como qualquer outra. Mas
        o desenho distingue: parede do corredor sai zebrada, caixa sai como bloco
        sólido, senão uma tábua de 8 cm vira um tracinho vermelho ilegível.
        """
        n = 4 * len(self.obstacles)
        return self.walls[:-n] if n else self.walls

    @property
    def seg_a(self) -> np.ndarray:
        return self.walls[:, :2]

    @property
    def seg_b(self) -> np.ndarray:
        return self.walls[:, 2:]

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax) englobando paredes e objetivo."""
        if not len(self.walls):
            g = self.goal
            return float(g[0] - 1), float(g[1] - 1), float(g[0] + 1), float(g[1] + 1)
        xs = np.concatenate([self.walls[:, 0], self.walls[:, 2], [self.goal[0]]])
        ys = np.concatenate([self.walls[:, 1], self.walls[:, 3], [self.goal[1]]])
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    # -- consultas --------------------------------------------------------- #
    def distance_to_walls(self, points) -> np.ndarray:
        """Menor distância de cada ponto até alguma parede. points (..., 2)."""
        if not len(self.walls):
            return np.full(np.shape(points)[:-1], np.inf)
        return point_segment_distance(np.asarray(points), self.seg_a, self.seg_b).min(axis=-1)

    @property
    def length(self) -> float:
        """Comprimento do traçado, em metros. 0 se a pista não tem linha central."""
        if not self._tem_linha():
            return 0.0
        d = self.centerline[1:] - self.centerline[:-1]
        return float(np.hypot(d[:, 0], d[:, 1]).sum())

    def progress_along(self, points) -> np.ndarray:
        """Quanto do traçado cada ponto já percorreu, em metros.

        Projeta na linha central e devolve o comprimento de arco. É esta a medida
        de "o quanto cheguei perto da chegada" — a distância em linha reta até o
        objetivo mede outra coisa: num corredor sinuoso o robô pode estar a 1 m
        do fim em linha reta e a 5 m de percurso, com uma parede no meio.

        Sem linha central (pista de paredes soltas), não há traçado para medir e
        o resultado é zero — aí use `distance_to_goal`.
        """
        if not self._tem_linha():
            return np.zeros(np.shape(points)[:-1])
        return project_polyline(points, self.centerline)[1]

    def reached_goal(self, points) -> np.ndarray:
        d = np.hypot(points[..., 0] - self.goal[0], points[..., 1] - self.goal[1])
        return d <= self.goal[2]

    # -- variações, para testar generalização ------------------------------ #
    def espelhar(self, nome=None) -> "Track":
        """Cópia refletida no eixo horizontal: as mesmas curvas, ao contrário.

        É o teste mais direto de decoreba neste projeto. A rede não tem memória
        nem posição — só as distâncias dos sensores — então ela não consegue
        decorar "no passo 300 vire à esquerda". O que ela consegue é aprender um
        **reflexo enviesado**: "na dúvida, vire à esquerda" resolve uma pista que
        só tem curvas à esquerda, e falha em qualquer outra.

        A pista espelhada tem exatamente a mesma dificuldade e o mesmo
        comprimento, só com as curvas invertidas. Quem vai bem no original e mal
        no espelho aprendeu o lado, não o desvio.
        """
        eixo = 2.0 * float(np.mean([self.bounds[1], self.bounds[3]]))

        def refletir_y(v):
            return eixo - v

        linha = None
        if self.centerline is not None:
            linha = self.centerline.copy()
            linha[:, 1] = refletir_y(linha[:, 1])

        caixas = self.obstacles.copy()
        if len(caixas):
            caixas[:, 1] = refletir_y(caixas[:, 1])
            caixas[:, 4] = -caixas[:, 4]         # o giro inverte junto

        paredes = self.extra_walls.copy()
        if len(paredes):
            paredes[:, 1] = refletir_y(paredes[:, 1])
            paredes[:, 3] = refletir_y(paredes[:, 3])

        start = self.start.copy()
        start[1] = refletir_y(start[1])
        start[2] = -start[2]                     # olhar para cima vira olhar para baixo
        goal = self.goal.copy()
        goal[1] = refletir_y(goal[1])

        return Track(centerline=linha, width=self.width, obstacles=caixas,
                     extra_walls=paredes, start=start, goal=goal,
                     checkpoint_spacing=self.checkpoint_spacing,
                     close_ends=self.close_ends,
                     name=nome or f"{self.name} (espelhada)")

    # -- persistência ------------------------------------------------------ #
    def save(self, path):
        """Guarda a receita, não o resultado — por isso continua editável."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "centerline": None if self.centerline is None else self.centerline.tolist(),
                "width": self.width,
                "obstacles": self.obstacles.tolist(),
                "extra_walls": self.extra_walls.tolist(),
                "start": self.start.tolist(),
                "goal": self.goal.tolist(),
                "checkpoint_spacing": self.checkpoint_spacing,
                "close_ends": self.close_ends,
            }, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        # formato antigo: só as paredes prontas, sem receita
        if "walls" in d and "centerline" not in d:
            return cls(extra_walls=d["walls"], start=d["start"], goal=d["goal"],
                       checkpoints=np.asarray(d.get("checkpoints", VAZIO_4)).reshape(-1, 4),
                       name=d.get("name", "sem nome"))

        return cls(
            centerline=d.get("centerline"),
            width=d.get("width", 0.9),
            obstacles=d.get("obstacles", VAZIO_5),
            extra_walls=d.get("extra_walls", VAZIO_4),
            start=d.get("start"), goal=d.get("goal"),
            checkpoint_spacing=d.get("checkpoint_spacing", 0.5),
            close_ends=d.get("close_ends", True),
            name=d.get("name", "sem nome"),
        )


# ---------------------------------------------------------------------- #
def _normalizar_caixas(obstacles) -> np.ndarray:
    """Aceita (M,5) e também (M,4) do formato quadrado antigo."""
    arr = np.asarray(obstacles, dtype=np.float64)
    if arr.size == 0:
        return VAZIO_5.copy()
    arr = arr.reshape(len(arr), -1)
    if arr.shape[1] == 4:   # [cx, cy, lado, ângulo] -> largura = comprimento
        arr = np.insert(arr, 3, arr[:, 2], axis=1)
    return arr.reshape(-1, 5)


def box_corners(obstacles) -> np.ndarray:
    """Cantos de cada caixa, (M, 4, 2), em ordem ao redor do perímetro."""
    cx, cy, larg, comp, ang = obstacles.T
    c, s = np.cos(ang), np.sin(ang)

    locais = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    cantos = np.empty((len(obstacles), 4, 2))
    for k, (sx, sy) in enumerate(locais):
        px, py = sx * larg / 2.0, sy * comp / 2.0
        cantos[:, k, 0] = cx + px * c - py * s
        cantos[:, k, 1] = cy + px * s + py * c
    return cantos


def _box_walls(obstacles) -> np.ndarray:
    """Cada caixa vira 4 segmentos de parede, iguais a qualquer outra parede."""
    if not len(obstacles):
        return VAZIO_4.copy()
    cantos = box_corners(obstacles)
    return np.concatenate(
        [np.concatenate([cantos[:, k], cantos[:, (k + 1) % 4]], axis=1) for k in range(4)],
        axis=0)


def _checkpoints_ao_longo(pts, width, spacing):
    """Segmentos perpendiculares à linha central, para medir progresso."""
    if pts is None or spacing <= 0 or len(pts) < 2:
        return VAZIO_4.copy()

    seg = pts[1:] - pts[:-1]
    comp = np.hypot(seg[:, 0], seg[:, 1])
    total = comp.sum()
    if total < spacing:
        return VAZIO_4.copy()

    acumulado = np.concatenate([[0.0], np.cumsum(comp)])
    saida = []
    for s in np.arange(spacing, total, spacing):
        i = int(np.searchsorted(acumulado, s) - 1)
        i = max(0, min(i, len(seg) - 1))
        u = (s - acumulado[i]) / max(comp[i], 1e-9)
        p = pts[i] + u * seg[i]
        d = seg[i] / max(comp[i], 1e-9)
        n = np.array([-d[1], d[0]]) * (width / 2.0)
        saida.append([p[0] + n[0], p[1] + n[1], p[0] - n[0], p[1] - n[1]])
    return np.array(saida) if saida else VAZIO_4.copy()
