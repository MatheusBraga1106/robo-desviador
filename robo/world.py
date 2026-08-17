"""O mundo: P robôs num percurso, avançando juntos.

Um único `World` serve os três modos. Jogando, P = 1 e os botões vêm do teclado;
treinando, P = tamanho da população e os botões vêm da sua rede. O simulador não
sabe a diferença.

O que sai daqui é **telemetria crua**, nunca fitness: distância percorrida,
checkpoints passados, se bateu, se chegou. O que conta como "bom" é decisão sua.
"""

import numpy as np

from .config import SimConfig
from .geometry import segments_cross
from .physics import DifferentialDrive, N_BOTOES
from .sensors import UltrasonicArray
from .track import Track


class World:
    def __init__(self, track: Track, cfg: SimConfig = None, n_robots: int = 1, rng=None,
                 marcos=None, jitter_largada=None):
        self.track = track
        self.cfg = cfg or SimConfig()
        self.P = n_robots
        self.rng = rng if rng is not None else np.random.default_rng(self.cfg.seed)

        # o corte de progresso é um marco como outro qualquer: para decidir se
        # elimina, precisa fotografar o progresso naquele instante
        extras = ([self.cfg.corte_progresso_em]
                  if getattr(self.cfg, "corte_progresso_em", 0) > 0 else [])
        self.marcos = sorted(set(list(marcos or ()) + extras))
        """Frações do episódio (0 a 1) em que o progresso é fotografado.

        Serve para regras de fitness que perguntam "onde ele estava na metade
        do caminho?" — algo que a telemetria final não responde. Sem marcos,
        não custa nada: a lista fica vazia e o passo não faz verificação alguma."""

        self.jitter_largada = jitter_largada
        """(metros, radianos) de variação na pose inicial, ou None.

        Cada robô larga num ponto e ângulo levemente diferentes. Serve para dois
        fins: testar se a rede depende da largada exata (decoreba), e treinar
        com um pouco de variação para ela não se apegar a uma pose específica.
        Quem cair dentro de parede fica na pose original."""

        self.drive = DifferentialDrive(self.cfg.robot, n_robots)
        self.sensors = UltrasonicArray(self.cfg.sensor, n_robots,
                                       self.cfg.robot.radius, self.rng)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        t = self.track
        self.pos = np.tile(t.start[:2], (self.P, 1)).astype(np.float64)
        self.theta = np.full(self.P, t.start[2], dtype=np.float64)
        self._espalhar_largada()
        self.v = np.zeros(self.P)
        self.omega = np.zeros(self.P)

        self.drive.reset()
        self.sensors.reset()
        self.sensors.update(self.pos, self.theta, t, self.cfg.sensor.reading_time * self.sensors.n)

        self.alive = np.ones(self.P, dtype=bool)
        self.collided = np.zeros(self.P, dtype=bool)
        self.desistiu = np.zeros(self.P, dtype=bool)
        self.finished = np.zeros(self.P, dtype=bool)
        self.bumps = np.zeros(self.P, dtype=np.int32)
        self.distance = np.zeros(self.P)
        self.checkpoint = np.zeros(self.P, dtype=np.int32)
        self.steps_alive = np.zeros(self.P, dtype=np.int32)
        self.steps = 0

        self._progresso = self.track.progress_along(self.pos)
        self._progresso_ref = self._progresso.copy()   # último avanço reconhecido
        self._passo_ref = np.zeros(self.P, dtype=np.int64)
        self.step_info = self._info_parada()

        # fotos do progresso nos marcos pedidos, preenchidas ao longo do episódio
        self.progresso_marcos = {}
        self.vivos_marcos = {}
        self._passos_marcos = {f: max(1, int(round(f * self.cfg.max_steps)))
                               for f in self.marcos}
        return self.observation()

    def _espalhar_largada(self):
        """Aplica a variação de pose inicial, sem colocar ninguém dentro da parede."""
        if not self.jitter_largada:
            return
        desloc, giro = self.jitter_largada
        if self.P == 1 and not desloc and not giro:
            return

        alvo = self.pos + self.rng.uniform(-desloc, desloc, (self.P, 2))
        cabe = self.track.distance_to_walls(alvo) >= self.cfg.robot.radius
        self.pos = np.where(cabe[:, None], alvo, self.pos)
        self.theta = self.theta + np.where(cabe, self.rng.uniform(-giro, giro, self.P), 0.0)

    # ------------------------------------------------------------------ #
    def step(self, buttons):
        """Avança um passo de controle. `buttons`: (P, 4) em [0,1] ou booleano."""
        cfg = self.cfg
        dt = cfg.dt

        buttons = np.asarray(buttons, dtype=np.float64).reshape(self.P, N_BOTOES)
        parado = ~(self.alive & ~self.finished)
        buttons = np.where(parado[:, None], 0.0, buttons)

        v, omega = self.drive.step(buttons, dt)
        self.v, self.omega = v, omega

        theta_novo = (self.theta + omega * dt + np.pi) % (2 * np.pi) - np.pi
        avanco = v * dt
        pos_novo = self.pos + np.stack([np.cos(theta_novo), np.sin(theta_novo)], axis=1) * avanco[:, None]

        # o robô não atravessa parede: se o passo levaria para dentro dela, fica
        bateu = self.track.distance_to_walls(pos_novo) < cfg.robot.radius
        bateu &= ~parado
        andou = ~bateu & ~parado

        pos_anterior = self.pos.copy()
        self.pos = np.where(andou[:, None], pos_novo, self.pos)
        self.theta = np.where(andou, theta_novo, self.theta)
        self.drive.stop(bateu)

        self.distance += np.where(andou, np.abs(avanco), 0.0)
        self.bumps += bateu.astype(np.int32)
        self.collided |= bateu
        if cfg.collision_ends_episode:
            self.alive &= ~bateu

        self._avancar_checkpoints(pos_anterior)
        chegou_agora = self.track.reached_goal(self.pos) & self.alive & ~self.finished
        self.finished |= chegou_agora

        self.sensors.update(self.pos, self.theta, self.track, dt)

        self.steps_alive += (self.alive & ~self.finished).astype(np.int32)
        self.steps += 1

        obs = self.observation()
        self._registrar_passo(obs, bateu, chegou_agora, ~parado, v)
        self._fotografar_marcos()
        self._eliminar_parados()
        return obs

    def _eliminar_parados(self):
        """Tira de campo quem não avança há tempo demais.

        Diferente do corte por fração, este roda o tempo todo — e é o que
        realmente encurta a geração, já que o episódio só acaba quando o último
        sobrevivente sai. Quem está progredindo nunca é afetado: qualquer avanço
        acima de `parado_avanco_minimo` zera o contador.
        """
        limite = self.cfg.parado_limite_s
        if limite <= 0:
            return

        avancou = self._progresso > self._progresso_ref + self.cfg.parado_avanco_minimo
        self._progresso_ref = np.where(avancou, self._progresso, self._progresso_ref)
        self._passo_ref = np.where(avancou, self.steps, self._passo_ref)

        limite_passos = max(1, int(round(limite / self.cfg.dt)))
        travados = ((self.steps - self._passo_ref) >= limite_passos) \
            & self.alive & ~self.finished
        if travados.any():
            self.desistiu |= travados
            self.alive &= ~travados
            self.drive.stop(travados)

    def _fotografar_marcos(self):
        """Guarda o progresso (e quem estava vivo) ao cruzar cada marco."""
        for fracao, passo_alvo in self._passos_marcos.items():
            if self.steps == passo_alvo:
                self.progresso_marcos[fracao] = self._progresso.copy()
                self.vivos_marcos[fracao] = (self.alive & ~self.finished).copy()
                if fracao == self.cfg.corte_progresso_em:
                    self._aplicar_corte()

    def _aplicar_corte(self):
        """Elimina quem não avançou o mínimo exigido até aqui.

        Marcado como `desistiu`, não como `collided`: são coisas diferentes, e
        misturá-las esconderia de você qual dos dois problemas a rede tem.
        """
        meta = self.cfg.corte_progresso_minimo * self.track.length
        if meta <= 0:                      # pista sem traçado: nada a exigir
            return
        parados = (self._progresso < meta) & self.alive & ~self.finished
        self.desistiu |= parados
        self.alive &= ~parados
        self.drive.stop(parados)

    def _registrar_passo(self, obs, bateu, chegou, ativo, v):
        """Fatos deste passo, para quem treina com recompensa por passo.

        São fatos, não recompensa: avanço, batida, chegada, proximidade. Montar
        o número que o otimizador maximiza continua sendo sua conta — inclusive
        porque o peso relativo dessas coisas é a parte que você vai querer mexer.
        """
        progresso = self.track.progress_along(self.pos)
        self.step_info = {
            "delta_progress": np.where(ativo, progresso - self._progresso, 0.0),
            "collided_now": bateu.copy(),
            "finished_now": chegou.copy(),
            "alive": self.alive & ~self.finished,
            "active": ativo,
            "min_reading": obs.min(axis=1).astype(np.float64),
            "forward_speed": np.where(ativo, v, 0.0),
            "progress": progresso,
            "dt": self.cfg.dt,
        }
        self._progresso = progresso

    def _info_parada(self) -> dict:
        z = np.zeros(self.P)
        return {
            "delta_progress": z.copy(),
            "collided_now": np.zeros(self.P, dtype=bool),
            "finished_now": np.zeros(self.P, dtype=bool),
            "alive": self.alive.copy(),
            "active": self.alive.copy(),
            "min_reading": self.observation().min(axis=1).astype(np.float64),
            "forward_speed": z.copy(),
            "progress": self._progresso.copy(),
            "dt": self.cfg.dt,
        }

    def _avancar_checkpoints(self, pos_anterior):
        cps = self.track.checkpoints
        if len(cps) == 0:
            return
        idx = np.clip(self.checkpoint, 0, len(cps) - 1)
        alvo = cps[idx]
        pendente = self.checkpoint < len(cps)
        cruzou = segments_cross(pos_anterior, self.pos, alvo[:, :2], alvo[:, 2:]) & pendente
        self.checkpoint += cruzou.astype(np.int32)

    def reconfigure_sensors(self):
        """Aplica mudanças de montagem sem perder a pose do robô.

        Recriar o `World` inteiro resetaria a posição, o que atrapalha justamente
        quando você quer comparar dois arranjos de sensor no mesmo lugar da pista.
        """
        self.sensors.reconfigure(self.cfg.robot.radius)
        self.sensors.update(self.pos, self.theta, self.track,
                            self.cfg.sensor.reading_time * self.sensors.n)

    # ------------------------------------------------------------------ #
    def observation(self) -> np.ndarray:
        """(P, n_sensores) em [0,1] — a única entrada que a rede recebe."""
        return self.sensors.observation()

    @property
    def done(self) -> bool:
        return self.steps >= self.cfg.max_steps or not (self.alive & ~self.finished).any()

    def telemetry(self) -> dict:
        """Dados crus por robô. O fitness é seu — aqui não há juízo de valor.

        Para "o quanto ele chegou perto da chegada", use `progress` (ou
        `remaining`), não `distance_to_goal`: a distância em linha reta ignora as
        paredes, e num corredor sinuoso o robô pode estar pertíssimo do objetivo
        em linha reta e longe em percurso, do outro lado de uma parede. Premiar
        isso ensina o robô a encostar na parede certa, não a progredir.
        """
        alvo = self.track.goal[:2]
        progresso = self.track.progress_along(self.pos)
        comprimento = self.track.length
        return {
            "alive": self.alive.copy(),
            "collided": self.collided.copy(),
            "desistiu": self.desistiu.copy(),
            "finished": self.finished.copy(),
            "bumps": self.bumps.copy(),
            "distance": self.distance.copy(),
            "progress": progresso,
            "remaining": np.maximum(comprimento - progresso, 0.0),
            "track_length": comprimento,
            "checkpoints": self.checkpoint.copy(),
            "total_checkpoints": len(self.track.checkpoints),
            "steps_alive": self.steps_alive.copy(),
            "time": self.steps_alive * self.cfg.dt,
            "distance_to_goal": np.hypot(self.pos[:, 0] - alvo[0], self.pos[:, 1] - alvo[1]),
            "position": self.pos.copy(),
            "readings": self.sensors.last.copy(),
            "progress_marcos": self._marcos_com_fallback(progresso),
            "alive_marcos": {f: self.vivos_marcos.get(f, self.alive & ~self.finished).copy()
                             for f in self.marcos},
        }

    def _marcos_com_fallback(self, progresso_final):
        """Progresso em cada marco, com o final valendo se o marco não chegou.

        O episódio pode acabar antes do marco (todos bateram, ou alguém chegou
        e ninguém mais está vivo). Nesse caso o progresso final É o valor no
        marco — parar antes não significa "sem dado", significa que dali para
        frente nada mudaria.
        """
        return {f: self.progresso_marcos.get(f, progresso_final).copy()
                for f in self.marcos}
