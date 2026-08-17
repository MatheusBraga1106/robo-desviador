"""Tração diferencial: 4 botões -> PWM dos dois motores -> movimento.

Este é o único caminho de controle que existe. O teclado e a rede neural passam
pelos mesmos 4 botões, então o que você sente jogando é exatamente o que a rede
enfrenta — nada de atalho para um lado ou para o outro.

Convenção do mundo: y para cima, ângulo em radianos crescendo no anti-horário.
"""

import numpy as np

from .config import RobotConfig

# índices dos botões
ACELERAR, RE, ESQUERDA, DIREITA = 0, 1, 2, 3
N_BOTOES = 4


class DifferentialDrive:
    def __init__(self, cfg: RobotConfig, n_robots: int):
        self.cfg = cfg
        self.P = n_robots
        self.reset()

    def reset(self):
        self.wheel = np.zeros((self.P, 2), dtype=np.float64)   # m/s: [esquerda, direita]

    # ------------------------------------------------------------------ #
    def buttons_to_pwm(self, buttons) -> np.ndarray:
        """(P, 4) -> (P, 2) PWM em [-1, 1] para [roda esquerda, roda direita].

        Combinações funcionam como esperado: acelerar sozinho anda reto, virar
        sozinho gira no lugar, e acelerar + virar faz curva aberta.
        """
        b = np.asarray(buttons, dtype=np.float64).reshape(-1, N_BOTOES) > 0.5

        base = b[:, ACELERAR].astype(float) - b[:, RE].astype(float)
        giro = b[:, DIREITA].astype(float) - b[:, ESQUERDA].astype(float)

        esq = base + giro * self.cfg.turn_gain
        dir_ = base - giro * self.cfg.turn_gain

        # satura preservando a proporção do diferencial, senão a curva entorta
        pico = np.maximum(np.maximum(np.abs(esq), np.abs(dir_)), 1.0)
        return np.stack([esq / pico, dir_ / pico], axis=1)

    def _pwm_para_velocidade(self, pwm) -> np.ndarray:
        """Aplica assimetria e zona morta, devolvendo a velocidade pedida (m/s)."""
        cfg = self.cfg

        # um motor puxa mais que o outro: o robô desvia sozinho em linha reta
        ganho = np.array([1.0 - cfg.motor_bias, 1.0 + cfg.motor_bias])
        pwm = np.clip(pwm * ganho, -1.0, 1.0)

        # abaixo da zona morta o motor não vence o atrito estático
        dz = cfg.pwm_deadzone
        mag = np.abs(pwm)
        efetivo = np.where(mag > dz, (mag - dz) / max(1.0 - dz, 1e-9), 0.0)
        return np.sign(pwm) * efetivo * cfg.max_wheel_speed

    def step(self, buttons, dt):
        """Avança um passo de controle. Devolve (v, omega) por robô."""
        alvo = self._pwm_para_velocidade(self.buttons_to_pwm(buttons))

        # inércia do motor e do chassi: a roda leva tempo para chegar no alvo
        tau = max(self.cfg.accel_time / 3.0, 1e-6)
        alfa = 1.0 - np.exp(-dt / tau)
        self.wheel += (alvo - self.wheel) * alfa

        vl, vr = self.wheel[:, 0], self.wheel[:, 1]
        v = (vl + vr) / 2.0
        omega = (vr - vl) / self.cfg.wheel_base
        return v, omega

    def stop(self, mask):
        """Zera as rodas dos robôs marcados (usado ao colidir)."""
        self.wheel[mask] = 0.0


def keys_to_buttons(up, down, left, right) -> np.ndarray:
    """Teclas do jogador no mesmo formato que a rede produz."""
    return np.array([[float(up), float(down), float(left), float(right)]])
