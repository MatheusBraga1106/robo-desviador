"""Roda episódios. O aprendizado é seu.

O controle é invertido de propósito: eu rodo a simulação e devolvo **telemetria
crua**; você decide o que é bom e como a próxima geração nasce. Não existe
fitness aqui, nem laço de gerações, nem otimizador::

    runner = EpisodeRunner(track, cfg, n_robots=100)
    while True:
        cerebros = voce_monta_a_populacao()
        dados = runner.run(cerebros)
        voce_evolui(dados)            # o fitness é sua conta

Se um robô que bate deve morrer ou continuar penalizando é escolha sua também,
em `cfg.collision_ends_episode`. O padrão encerra para ele, que é o que
algoritmo evolutivo espera; deixando vivo, ele fica parado contra a parede
acumulando tempo, o que dá gradiente mais denso.
"""

import time
from typing import Optional

import numpy as np

from .brain import check_brain
from .config import SimConfig
from .track import Track
from .world import World


class EpisodeRunner:
    def __init__(self, track: Track, cfg: SimConfig = None, n_robots: int = 1,
                 verificar_contrato: bool = True, marcos=None, jitter_largada=None):
        self.track = track
        self.cfg = cfg or SimConfig()
        self.n_robots = n_robots
        self.verificar = verificar_contrato
        self._ja_verifiquei = False
        # `marcos`: frações do episódio em que o progresso é fotografado, para
        # regras de fitness do tipo "onde ele estava na metade do caminho?"
        self.world = World(track, self.cfg, n_robots, marcos=marcos,
                           jitter_largada=jitter_largada)

    # ------------------------------------------------------------------ #
    def run(self, brain, seed: Optional[int] = None, on_step=None,
            max_steps: Optional[int] = None) -> dict:
        """Roda um episódio inteiro com a população.

        brain    : qualquer objeto com act_batch(obs) -> (P, 4)
        seed     : fixa o sorteio do mundo, para comparar cérebros de forma justa
        on_step  : chamado como on_step(world, passo) a cada passo — é por aqui
                   que a visualização ao vivo entra, sem o runner saber de tela
        Devolve a telemetria crua do `World`, mais tempo e passos.
        """
        if self.verificar and not self._ja_verifiquei:
            check_brain(brain, self.cfg.n_sensors, self.n_robots)
            self._ja_verifiquei = True

        if seed is not None:
            self.world.rng = np.random.default_rng(seed)
            self.world.sensors.rng = self.world.rng

        teto = max_steps or self.cfg.max_steps
        obs = self.world.reset()
        t0 = time.perf_counter()
        passos = 0

        while passos < teto and not self._todos_parados():
            acao = brain.act_batch(obs)
            obs = self.world.step(acao)
            passos += 1
            if on_step is not None and on_step(self.world, passos) is False:
                break   # a visualização pediu para parar (você fechou a janela)

        dados = self.world.telemetry()
        dados["steps"] = passos
        dados["wall_time"] = time.perf_counter() - t0
        return dados

    def _todos_parados(self) -> bool:
        w = self.world
        return not (w.alive & ~w.finished).any()

    # ------------------------------------------------------------------ #
    # Laço passo a passo, para quem treina com recompensa por passo (PPO, DQN).
    # O `run` acima entrega o episódio inteiro de uma vez, que é o que a
    # neuroevolução quer; aqui você fica no controle de cada transição.
    #
    #     obs = runner.reset(seed=0)
    #     while True:
    #         acao = politica(obs)
    #         obs, info, fim = runner.step(acao)
    #         r = minha_recompensa(info)      # a fórmula é sua
    #         if fim: break
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.world.rng = np.random.default_rng(seed)
            self.world.sensors.rng = self.world.rng
        self._passos = 0
        return self.world.reset()

    def step(self, action):
        """Devolve (obs, info, fim).

        `info` traz os fatos deste passo por robô — `delta_progress` (avanço no
        traçado), `collided_now`, `finished_now`, `alive`, `min_reading`,
        `forward_speed` e `dt`. Nenhum deles é recompensa: o número que o seu
        otimizador maximiza sai daqui pela sua fórmula.
        """
        obs = self.world.step(action)
        self._passos = getattr(self, "_passos", 0) + 1
        fim = self._todos_parados() or self._passos >= self.cfg.max_steps
        return obs, self.world.step_info, fim

    # ------------------------------------------------------------------ #
    def run_many(self, brain, cenarios, on_step=None) -> list:
        """Roda o mesmo cérebro em vários mundos, para reduzir sorte.

        `cenarios` é uma lista de seeds. Devolve uma lista de telemetrias — a
        média (ou o pior caso, se você preferir) é conta sua.
        """
        return [self.run(brain, seed=s, on_step=on_step) for s in cenarios]


def resumo(dados: dict) -> str:
    """Uma linha legível de telemetria, para acompanhar o treino no terminal.

    Note que não há fitness aqui: são fatos, não juízo.
    """
    n = len(dados["alive"])
    chegou = int(dados["finished"].sum())
    bateu = int(dados["collided"].sum())
    cps = dados["checkpoints"]
    return (f"{n} robôs · {chegou} chegaram · {bateu} bateram · "
            f"checkpoint médio {cps.mean():.1f}/{dados['total_checkpoints']} "
            f"(melhor {int(cps.max())}) · "
            f"percurso médio {dados['distance'].mean():.2f} m · "
            f"{dados['steps']} passos em {dados['wall_time']:.2f} s")
