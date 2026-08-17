"""Array de sensores ultrassônicos HC-SR04.

O que separa isto de um raycast comum é a fidelidade ao sensor real:

* **Cone, não raio.** O HC-SR04 devolve o obstáculo mais próximo dentro de um
  cone de ~30°. Simulado com sub-raios espalhados pelo cone, pegando o menor.
* **Eco perdido.** Parede muito inclinada reflete o som para longe, e nenhum eco
  volta. Como "sem eco" e "caminho livre" produzem a mesma leitura, o robô entra
  de frente numa parede em ângulo achando que está livre. É a falha clássica do
  ultrassom, e a rede precisa aprender a conviver com ela.
* **Rodízio.** Disparar os sensores juntos causa crosstalk (um escuta o eco do
  outro), então na prática eles vão um de cada vez. Com 4 sensores a 60 ms, uma
  varredura completa leva 240 ms — e a rede sempre decide com dado velho.
"""

import numpy as np

from .config import SensorConfig
from .geometry import ray_segment_hits


class UltrasonicArray:
    def __init__(self, cfg: SensorConfig, n_robots: int, robot_radius: float, rng=None):
        self.cfg = cfg
        self.P = n_robots
        self.radius = robot_radius
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.reconfigure()

    def reconfigure(self, robot_radius=None):
        """Recalcula o que depende da montagem dos sensores.

        Alcance, ruído e tempo de leitura são lidos direto do `cfg` a cada uso,
        então mudam sozinhos. Já quantidade, leque e cone viram tabelas aqui —
        por isso o painel de calibragem chama isto quando você mexe neles.
        """
        cfg = self.cfg
        if robot_radius is not None:
            self.radius = robot_radius

        self.mount = cfg.resolved_angles()                 # (n,)
        self.n = len(self.mount)

        k = max(1, cfg.rays_per_sensor)
        meio = np.deg2rad(cfg.cone_deg) / 2.0
        espalhamento = np.zeros(1) if k == 1 else np.linspace(-meio, meio, k)
        self.cone = self.mount[:, None] + espalhamento[None, :]   # (n, k)

        self.cos_limite = np.cos(np.deg2rad(cfg.max_incidence_deg))
        self.reset()

    def reset(self):
        """Começa com tudo lendo 'livre', como o sensor antes do primeiro pulso."""
        self.last = np.full((self.P, self.n), self.cfg.max_range, dtype=np.float64)
        self.age = np.zeros((self.P, self.n), dtype=np.float64)
        self._proximo = 0
        self._orcamento = 0.0

    # ------------------------------------------------------------------ #
    def update(self, pos, theta, track, dt):
        """Avança o tempo e dispara os sensores cuja vez chegou.

        pos   : (P, 2)
        theta : (P,)
        """
        self.age += dt

        if not self.cfg.round_robin:
            self._disparar(np.arange(self.n), pos, theta, track)
            return

        self._orcamento += dt
        # o hardware não consegue mais que uma leitura por `reading_time`
        limite = self.n
        while self._orcamento >= self.cfg.reading_time and limite > 0:
            self._orcamento -= self.cfg.reading_time
            self._disparar(np.array([self._proximo]), pos, theta, track)
            self._proximo = (self._proximo + 1) % self.n
            limite -= 1

    def _disparar(self, indices, pos, theta, track):
        cfg = self.cfg
        angulos = theta[:, None, None] + self.cone[None, indices, :]   # (P, m, k)

        dirs = np.stack([np.cos(angulos), np.sin(angulos)], axis=-1)   # (P, m, k, 2)
        # o sensor fica na borda do corpo, não no centro
        origens = pos[:, None, None, :] + dirs * self.radius

        t, cos_inc = ray_segment_hits(origens, dirs, track.seg_a, track.seg_b)

        eco = (
            np.isfinite(t)
            & (t <= cfg.max_range)
            & (t >= cfg.min_range)          # abaixo da zona cega o eco não é lido
            & (cos_inc >= self.cos_limite)  # rasante demais: o som reflete para longe
        )
        t = np.where(eco, t, np.inf)
        leitura = t.min(axis=-1).min(axis=-1)          # menor entre paredes e sub-raios
        leitura = np.where(np.isfinite(leitura), leitura, cfg.max_range)

        if cfg.noise_std > 0:
            leitura = leitura + self.rng.normal(0, cfg.noise_std, leitura.shape)
        if cfg.dropout_prob > 0:
            perdida = self.rng.random(leitura.shape) < cfg.dropout_prob
            leitura = np.where(perdida, cfg.max_range, leitura)

        self.last[:, indices] = np.clip(leitura, 0.0, cfg.max_range)
        self.age[:, indices] = 0.0

    # ------------------------------------------------------------------ #
    def observation(self) -> np.ndarray:
        """(P, n_sensores) em [0, 1]. 0 = obstáculo colado, 1 = livre ou sem eco."""
        return (self.last / self.cfg.max_range).astype(np.float32)

    def cone_angles(self, theta) -> np.ndarray:
        """Ângulos absolutos dos sub-raios, para desenhar. (P, n, k)"""
        return theta[:, None, None] + self.cone[None, :, :]

    def origins(self, pos, theta) -> np.ndarray:
        """Onde cada sensor está montado no corpo. (P, n, 2)"""
        ang = theta[:, None] + self.mount[None, :]
        return pos[:, None, :] + np.stack([np.cos(ang), np.sin(ang)], axis=-1) * self.radius
