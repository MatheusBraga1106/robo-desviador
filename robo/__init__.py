"""Simulador do robô desviador de obstáculos.

O jogo, a física e os sensores moram aqui. A rede neural é sua — ela conversa
com este pacote só pelos 4 botões (acelerar, ré, esquerda, direita) e pelas
leituras normalizadas dos sensores.
"""

from .config import SimConfig, RobotConfig, SensorConfig
from .track import Track
from .world import World
from .physics import ACELERAR, RE, ESQUERDA, DIREITA, N_BOTOES

__all__ = [
    "SimConfig", "RobotConfig", "SensorConfig",
    "Track", "World",
    "ACELERAR", "RE", "ESQUERDA", "DIREITA", "N_BOTOES",
]
