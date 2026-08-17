"""Parâmetros do robô, dos sensores e da simulação.

Os valores padrão descrevem um robô pequeno de mesa: chassi de ~18 cm com dois
motores DC e roda boba, e HC-SR04 na frente. Meça os seus e ajuste — quanto mais
perto daqui do robô real, melhor a rede treinada transfere.

Todas as unidades em metros, segundos e radianos.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

CAMINHO_PADRAO = "config.json"
"""Onde a calibragem feita no jogo é gravada, e de onde o treino a lê."""


@dataclass
class SensorConfig:
    """Um array de HC-SR04 montados na borda do robô."""

    count: int = 4
    fov_deg: float = 90.0
    """Leque total onde os sensores ficam montados (0° = frente do robô).

    Cuidado com a combinação leque x quantidade: com número PAR de sensores
    nenhum aponta para frente. Medido num corredor de 0,9 m, a 1 m de uma parede
    frontal, o menor valor lido foi:

        4 sensores em 150° (±25, ±75)  ->  0.61 m  (é a parede lateral; a frontal
                                                    só aparece por volta de 0,6 m)
        4 sensores em  90° (±15, ±45)  ->  0.81 m
        4 sensores em  60° (±10, ±30)  ->  0.97 m
        5 sensores em 150° (±75, ±38, 0) -> 1.00 m  (o sensor central a enxerga inteira)

    Leque largo enxerga melhor os lados e pior a frente. Um sensor central (número
    ímpar) resolve a frente. E num corredor estreito o cone de 30° bate na parede
    lateral por volta de 1 m, então nenhum arranjo enxerga muito além disso."""

    angles_deg: Optional[List[float]] = None
    """Ângulos explícitos de montagem. Se None, distribui `count` sensores no fov."""

    cone_deg: float = 30.0
    """Abertura do cone do HC-SR04. Ele devolve o obstáculo mais próximo dentro
    do cone inteiro, não de um raio fino — por isso a simulação usa sub-raios."""

    rays_per_sensor: int = 5
    """Sub-raios que aproximam o cone. 1 vira um laser (irreal para ultrassom)."""

    max_range: float = 2.0
    """Alcance útil. O HC-SR04 anuncia 4 m, mas em ambiente fechado com alvo
    pequeno o retorno confiável costuma ficar bem abaixo disso."""

    min_range: float = 0.03
    """Zona cega: abaixo disso o eco volta antes do fim do pulso e não é lido."""

    reading_time: float = 0.06
    """Tempo de uma leitura. O pulso de 40 kHz precisa ir e voltar, e o timeout
    domina quando não há eco."""

    round_robin: bool = True
    """Dispara um sensor por vez. Disparar todos juntos causa crosstalk: um
    sensor escuta o eco do outro e mede distância errada."""

    max_incidence_deg: float = 65.0
    """Acima deste ângulo em relação à normal da parede, o som reflete para longe
    e nenhum eco volta. É a falha clássica do ultrassom: parede inclinada lida
    como caminho livre."""

    noise_std: float = 0.0
    """Ruído gaussiano na distância (m). Desligado por padrão."""

    dropout_prob: float = 0.0
    """Chance de uma leitura simplesmente falhar. Desligado por padrão."""

    def resolved_angles(self) -> np.ndarray:
        """Ângulos de montagem em radianos, 0 = frente."""
        if self.angles_deg is not None:
            return np.deg2rad(np.asarray(self.angles_deg, dtype=np.float64))
        if self.count == 1:
            return np.zeros(1)
        half = np.deg2rad(self.fov_deg) / 2.0
        return np.linspace(-half, half, self.count)


@dataclass
class RobotConfig:
    """Chassi diferencial: duas rodas motorizadas e uma roda boba."""

    radius: float = 0.09
    """Raio do corpo, usado para colisão e para posicionar os sensores."""

    wheel_base: float = 0.14
    """Distância entre as duas rodas. Menor = gira mais rápido no mesmo PWM."""

    max_wheel_speed: float = 0.35
    """Velocidade de cada roda com PWM cheio. É esta a 'velocidade limitada' do
    robô real — meça cronometrando o seu no chão."""

    accel_time: float = 0.25
    """Tempo para a roda chegar perto da velocidade pedida. Modela a inércia do
    motor e do chassi; sem isso a simulação responde rápido demais."""

    pwm_deadzone: float = 0.15
    """Abaixo deste PWM o motor não vence o atrito estático e nada acontece."""

    motor_bias: float = 0.0
    """Assimetria entre os motores: 0.05 = direito 5% mais forte que o esquerdo,
    então o robô puxa para um lado sozinho. Meça mandando os dois a PWM igual."""

    turn_gain: float = 0.7
    """Quanto de diferencial os botões de virar aplicam. 1.0 gira no próprio eixo
    ao virar parado; valores menores fazem curvas mais abertas."""

    @property
    def max_turn_rate(self) -> float:
        """Giro máximo (rad/s), com uma roda para frente e outra para trás."""
        return 2.0 * self.max_wheel_speed / self.wheel_base


@dataclass
class SimConfig:
    dt: float = 0.05
    """Período do laço de controle. Deve casar com o `delay()` do firmware."""

    max_steps: int = 2000
    """Teto de passos por episódio (0.05 s * 2000 = 100 s)."""

    collision_ends_episode: bool = True

    corte_progresso_em: float = 0.4
    """Fração do episódio em que o robô é cobrado por ter avançado. `0` desliga.

    Punir com pontos quem fica girando não encurta nada: o robô continua vivo
    até o último passo, e o episódio inteiro tem que rodar por causa dele. Este
    corte elimina de fato — e como o episódio acaba quando todos morrem ou
    chegam, tirar os enroladores do caminho encurta muito as gerações."""

    corte_progresso_minimo: float = 0.2
    """Quanto da rota ele precisa ter avançado no corte para continuar vivo."""

    parado_limite_s: float = 5.0
    """Segundos sem avançar antes de eliminar o robô. `0` desliga.

    O corte por fração pega quem enrolou *até certo ponto*, mas não encurta o
    episódio: quem sobrevive continua rodando até o último passo. Este aqui é
    contínuo — o robô que para de progredir sai na hora, em qualquer momento.
    É o que de fato encurta a geração, porque o episódio acaba quando o último
    sobrevivente sai. Medido numa curva-u com 60 robôs por 20 gerações: passos
    por geração caíram de 2000 para 514 (treino 3,8x mais rápido) sem perder
    quem chega ao fim.

    5 s é folgado de propósito: a 0,35 m/s dá quase 2 m de manobra, então dar ré
    para contornar um obstáculo não dispara o corte. Baixe se quiser mais
    agressivo, mas cuidado para não matar quem está manobrando."""

    parado_avanco_minimo: float = 0.05
    """Metros de avanço no traçado que contam como "saiu do lugar"."""

    robot: RobotConfig = field(default_factory=RobotConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)

    seed: int = 0

    @property
    def n_sensors(self) -> int:
        return len(self.sensor.resolved_angles())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        d = dict(d)
        robot = RobotConfig(**d.pop("robot", {}))
        sensor = SensorConfig(**d.pop("sensor", {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known}, robot=robot, sensor=sensor)

    # ------------------------------------------------------------------ #
    def save(self, path=None):
        """Grava a calibragem. Sem isso, ajustar os números não serviria de nada:
        o treino roda noutro processo e precisa dos mesmos valores do jogo."""
        with open(path or CAMINHO_PADRAO, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path=None) -> "SimConfig":
        with open(path or CAMINHO_PADRAO, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def carregar_ou_padrao(cls, path=None) -> "SimConfig":
        """Usa a sua calibragem se ela existir; senão, os valores de fábrica."""
        try:
            return cls.load(path)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()
