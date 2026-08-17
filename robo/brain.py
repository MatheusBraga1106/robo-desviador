"""O contrato entre o simulador e a sua rede.

Tudo que o jogo precisa saber sobre a sua rede está aqui, e é pouco de propósito:
ela recebe as leituras dos sensores e devolve os 4 botões. Nenhuma suposição
sobre framework — numpy, torch, jax, o que você quiser — e nenhuma suposição
sobre como você treina. O laço de aprendizado é seu.

O mínimo
--------
Só `act_batch` é obrigatório::

    class MinhaRede:
        def act_batch(self, obs):          # (P, n_sensores) -> (P, 4)
            ...

`obs` vem em [0, 1], onde 0 é obstáculo colado e 1 é livre (ou sem eco, que o
sensor real não distingue de livre). A saída vai em [0, 1] por botão, e acima de
0,5 conta como tecla apertada — os mesmos 4 botões que o teclado produz.

A ordem dos botões é `[acelerar, ré, esquerda, direita]`, e está em
`robo.physics` como ACELERAR, RE, ESQUERDA, DIREITA.

O opcional
----------
`inspect(i)` alimenta o painel de visualização. Devolver None desliga o painel,
e o resto continua funcionando igual.
"""

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .physics import N_BOTOES

LIMIAR_BOTAO = 0.5
ROTULOS = ("acelerar", "ré", "esquerda", "direita")


@runtime_checkable
class Brain(Protocol):
    """O que o simulador espera da sua rede."""

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """Decide para toda a população de uma vez.

        obs : (P, n_sensores) float32 em [0, 1]
        ret : (P, 4) em [0, 1] — acima de LIMIAR_BOTAO conta como apertado

        Vem em lote porque é assim que dá para assistir 100 robôs em tempo real.
        Se a sua rede só sabe pensar num robô por vez, `PorRobo` faz a ponte.
        """

    def inspect(self, i: int = 0) -> Optional[dict]:
        """Opcional. Estado interno do robô `i`, só para o painel::

            {"ativacoes": [array por camada, da entrada à saída],
             "pesos":     [matriz (saída, entrada) por conexão]}

        `ativacoes` tem L arrays e `pesos` tem L-1 matrizes. Devolva None (ou
        nem implemente) para não desenhar painel nenhum.
        """


def as_buttons(action: np.ndarray) -> np.ndarray:
    """Saída contínua da rede -> botões apertados. (P, 4) booleano."""
    return np.asarray(action, dtype=np.float64).reshape(-1, N_BOTOES) > LIMIAR_BOTAO


def describe_action(action) -> str:
    """Ação de um robô em texto, para depurar: 'acelerar+direita'."""
    apertados = [r for r, on in zip(ROTULOS, as_buttons(action)[0]) if on]
    return "+".join(apertados) if apertados else "parado"


class PorRobo:
    """Adapta uma rede que decide um robô por vez para o formato em lote.

    Útil para começar, mas é um laço em Python: com população grande vale a pena
    escrever `act_batch` de verdade, vetorizado.

        brain = PorRobo(lambda obs: minha_rede(obs))
    """

    def __init__(self, fn, inspect_fn=None):
        self.fn = fn
        self.inspect_fn = inspect_fn

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return np.array([np.asarray(self.fn(o), dtype=np.float64).reshape(N_BOTOES)
                         for o in obs])

    def inspect(self, i: int = 0):
        return self.inspect_fn(i) if self.inspect_fn else None


def check_brain(brain, n_sensores: int, n_robos: int = 4) -> str:
    """Confere o contrato antes de o treino começar.

    Uma forma errada some dentro do numpy e vira comportamento estranho lá na
    frente. Melhor falhar aqui, com o motivo escrito.
    """
    if not hasattr(brain, "act_batch"):
        raise TypeError("a rede precisa de act_batch(obs) -> (P, 4). "
                        "Se ela decide um robô por vez, embrulhe em robo.brain.PorRobo")

    obs = np.random.default_rng(0).random((n_robos, n_sensores)).astype(np.float32)
    saida = np.asarray(brain.act_batch(obs), dtype=np.float64)

    if saida.shape != (n_robos, N_BOTOES):
        raise ValueError(
            f"act_batch devolveu {saida.shape}, esperado ({n_robos}, {N_BOTOES}). "
            f"A saída é [acelerar, ré, esquerda, direita] por robô.")
    if not np.isfinite(saida).all():
        raise ValueError("act_batch devolveu NaN ou infinito")

    fora = (saida < -1e-6) | (saida > 1 + 1e-6)
    aviso = ""
    if fora.any():
        aviso = (f" (atenção: {fora.sum()} valores fora de [0,1]; "
                 f"o limiar de {LIMIAR_BOTAO} ainda se aplica, mas confira a ativação de saída)")

    detalhes = _detalhes_do_inspect(brain, n_sensores)
    return f"contrato ok: {n_sensores} sensores -> {N_BOTOES} botões{aviso}{detalhes}"


def _detalhes_do_inspect(brain, n_sensores) -> str:
    if not hasattr(brain, "inspect"):
        return " · sem inspect, painel desligado"
    try:
        d = brain.inspect(0)
    except Exception as e:
        return f" · inspect falhou ({type(e).__name__}), painel desligado"
    if d is None:
        return " · inspect devolveu None, painel desligado"

    ativ, pesos = d.get("ativacoes"), d.get("pesos")
    if ativ is None or pesos is None:
        return " · inspect sem 'ativacoes'/'pesos', painel desligado"
    if len(pesos) != len(ativ) - 1:
        return (f" · inspect inconsistente: {len(ativ)} camadas de ativação "
                f"exigem {len(ativ)-1} matrizes de peso, vieram {len(pesos)}")
    formato = "-".join(str(len(a)) for a in ativ)
    return f" · painel ligado, rede {formato}"
