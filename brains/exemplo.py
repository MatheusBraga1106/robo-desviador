"""Cérebro de exemplo — existe só para provar que o contrato fecha.

É uma MLP de pesos aleatórios, sem nenhum aprendizado. **Você substitui isto.**
O que vale a pena copiar daqui é o formato: `act_batch` recebe (P, n_sensores) e
devolve (P, 4), e `inspect` entrega as ativações para o painel.

    python -m brains.exemplo                 # roda um episódio e mostra a telemetria
    python -m brains.exemplo --ver           # assiste na tela
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

from robo.config import SimConfig
from robo.pistas import carregar
from robo.training import EpisodeRunner, resumo


class RedeAleatoria:
    """MLP com pesos sorteados. Uma por robô, avaliadas todas de uma vez.

    Os pesos ficam num tensor (P, saída, entrada) por camada, então a população
    inteira passa num punhado de einsums em vez de um laço em Python.
    """

    def __init__(self, n_sensores, n_robos, ocultas=(8, 8), seed=0):
        self.rng = np.random.default_rng(seed)
        self.tamanhos = [n_sensores, *ocultas, 4]
        self.camadas = []
        for entrada, saida in zip(self.tamanhos, self.tamanhos[1:]):
            escala = np.sqrt(2.0 / (entrada + saida))
            self.camadas.append((
                self.rng.normal(0, escala, (n_robos, saida, entrada)),
                np.zeros((n_robos, saida)),
            ))
        self._ativacoes = None

    def act_batch(self, obs):
        """(P, n_sensores) -> (P, 4) em [0, 1]."""
        x = np.asarray(obs, dtype=np.float64)
        ativacoes = [x]
        ultima = len(self.camadas) - 1
        for i, (W, b) in enumerate(self.camadas):
            x = np.einsum("poi,pi->po", W, x) + b
            # sigmoide na saída para cair em [0,1], que é onde o limiar de 0.5 vive
            x = _sigmoide(x) if i == ultima else np.tanh(x)
            ativacoes.append(x)
        self._ativacoes = ativacoes
        return x

    def inspect(self, i=0):
        """Estado interno do robô `i`, para o painel. Opcional no contrato."""
        if self._ativacoes is None:
            return None
        return {
            "ativacoes": [a[i] for a in self._ativacoes],
            "pesos": [W[i] for W, _ in self.camadas],
        }


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ===================================================================== #
# PESOS DO FITNESS — é aqui que você calibra o que vale ponto
# ===================================================================== #
@dataclass
class PesosFitness:
    """Todos os pesos num lugar só, para calibrar sem caçar valores no código.

    Cada campo é um "quanto isso importa". A escala relativa entre eles é o que
    manda: dobrar todos não muda nada, mas dobrar um só muda o comportamento
    que a evolução vai perseguir.
    """

    peso_avanco: float = 5.0
    """Pontos por metro avançado no traçado. É o sinal principal — se o robô
    fica parado demais, subir isto é o botão mais direto."""

    punicao_batida: float = 30.0
    """Perda por bater. Alto demais faz a rede virar medrosa e não sair do
    lugar; baixo demais faz ela ignorar as paredes."""

    bonus_chegada: float = 100.0
    """Prêmio por completar o percurso. Precisa ser bem maior que o avanço
    máximo (`peso_avanco × comprimento`), senão chegar não compensa."""

    peso_tempo: float = 0.5
    """Desconto por segundo, **aplicado só a quem chega**. Serve de desempate
    entre os que completaram. Cobrar isto de todo mundo inverte o treino: veja
    a explicação em `fitness_exemplo`."""

    enrolar_fracao_passos: float = 0.4
    """Em que ponto do episódio o robô é cobrado por ter avançado."""

    enrolar_fracao_rota: float = 0.2
    """Quanto da rota ele precisava ter feito até lá."""

    enrolar_peso: float = 40.0
    """Perda máxima de quem não avançou nada até o marco. Sobe isto se ainda
    aparecerem robôs girando em falso."""

    def linhas(self):
        """Pares (rótulo, valor) para mostrar na tela do treino."""
        return [
            ("por metro avançado", f"+{self.peso_avanco:.1f}"),
            ("bateu", f"-{self.punicao_batida:.0f}"),
            ("chegou", f"+{self.bonus_chegada:.0f}"),
            ("por segundo (só quem chega)", f"-{self.peso_tempo:.2f}"),
            (f"enrolar até {self.enrolar_fracao_passos:.0%}",
             f"-{self.enrolar_peso:.0f}"),
            ("meta nesse ponto", f"{self.enrolar_fracao_rota:.0%} da rota"),
        ]


PESOS = PesosFitness()
"""Os pesos em uso. Mude os campos da classe acima, ou este objeto."""


# --------------------------------------------------------------------- #
def fitness_exemplo(dados, peso_avanco=5.0, punicao_batida=30.0, bonus_chegada=100.0,
                    peso_tempo=0.5):
    """Um fitness possível — o seu, escrito. Mora aqui, fora do simulador.

    Isto é de propósito: o `World` devolve fatos (bateu, quanto avançou, quanto
    demorou) e quem decide o que vale ponto é você. Trocar esta função não exige
    tocar em nada do resto.

        avanço no traçado   quanto mais longe chega, mais ponto — o sinal principal
        - batida            bateu, perdeu
        + chegada           bônus por completar, menos o tempo que levou

    O tempo só conta para quem CHEGA
    --------------------------------
    Uma versão anterior cobrava `peso_tempo * time` de todo mundo, e isso
    invertia o treino: sobreviver custava até -50, o mesmo que bater, então
    ficar parado (que morre cedo e acumula pouco tempo) pontuava melhor do que
    andar. Medido: parado -2,1 contra -51,3 de quem andava 3 m. A evolução
    obedecia — e produzia robôs imóveis.

    Aqui o tempo entra só no bônus de chegada, como desempate entre quem
    completou. Quem ainda está aprendendo a não bater não é punido por demorar.
    """
    pontos = peso_avanco * dados["progress"]
    pontos -= punicao_batida * dados["collided"]
    pontos += np.maximum(bonus_chegada - peso_tempo * dados["time"], 0.0) * dados["finished"]
    return pontos


MARCOS_PADRAO = (0.4,)
"""Frações do episódio que `punicao_por_enrolar` precisa fotografar.

Passe isto ao `EpisodeRunner(..., marcos=MARCOS_PADRAO)`, senão a telemetria não
terá o progresso no meio do caminho e a regra não tem como funcionar."""


def punicao_por_enrolar(dados, fracao_passos=0.4, fracao_rota=0.2, peso=40.0,
                        so_vivos=True, proporcional=True):
    """Pune quem gastou boa parte do episódio sem sair do lugar.

    A ideia: passados `fracao_passos` do tempo, quem não avançou pelo menos
    `fracao_rota` do traçado estava andando em círculo ou indo e voltando.

    Duas decisões que valem explicar, porque mudam o comportamento:

    **Proporcional, não tudo-ou-nada.** Um corte seco daria a mesma punição a
    quem avançou 0% e a quem avançou 19%, e evolução não consegue escalar um
    degrau: sem diferença de nota, não há para onde subir. Aqui a punição cresce
    conforme a distância até a meta, então melhorar de 5% para 15% já rende
    ponto. Passe `proporcional=False` se quiser o corte seco mesmo assim.

    **Isenta só quem bateu.** Quem bateu no passo 50 também "não avançou", mas
    já levou a punição de batida; cobrar de novo misturaria dois sinais na mesma
    nota. Mas quem foi *eliminado por não progredir* (`desistiu`) continua sendo
    cobrado — ele é exatamente o alvo desta regra. Numa versão anterior a
    isenção olhava "estava vivo no marco", e o resultado era o oposto do
    pretendido: o corte por inatividade matava o enrolador antes do marco, e aí
    ele escapava da punição inteira. Com `so_vivos=False`, cobra de todos.
    """
    marcos = dados.get("progress_marcos") or {}
    if fracao_passos not in marcos:
        raise KeyError(
            f"o progresso em {fracao_passos:.0%} do episódio não foi registrado. "
            f"Crie o runner com marcos={{{fracao_passos}}} — por exemplo "
            f"EpisodeRunner(track, cfg, n_robots=P, marcos=MARCOS_PADRAO)."
        )

    avanco = marcos[fracao_passos]
    meta = fracao_rota * dados["track_length"]
    if meta <= 0:                      # pista sem traçado: não há o que exigir
        return np.zeros_like(avanco)

    falta = np.clip(meta - avanco, 0.0, meta)
    punicao = peso * (falta / meta) if proporcional else peso * (falta > 0)

    if so_vivos:
        # isenta quem bateu (já punido por isso), mas cobra de quem desistiu
        # por não progredir — esse é o alvo da regra
        punicao = punicao * ~np.asarray(dados["collided"], dtype=bool)
    return punicao


def fitness_com_antienrolação(dados, pesos: "PesosFitness" = None):
    """`fitness_exemplo` mais a punição por enrolar. Exige `marcos` no runner.

    Tudo vem de um `PesosFitness`, e os repasses são por **nome**. Passar por
    posição já causou um bug silencioso aqui: bastou mudar a ordem dos
    parâmetros de `fitness_exemplo` para o peso do tempo virar peso do avanço,
    sem erro nenhum — só um treino que aprendia a coisa errada.
    """
    p = pesos or PESOS
    return fitness_exemplo(dados,
                           peso_avanco=p.peso_avanco,
                           punicao_batida=p.punicao_batida,
                           bonus_chegada=p.bonus_chegada,
                           peso_tempo=p.peso_tempo) \
        - punicao_por_enrolar(dados,
                              fracao_passos=p.enrolar_fracao_passos,
                              fracao_rota=p.enrolar_fracao_rota,
                              peso=p.enrolar_peso)


# --------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="Exemplo do contrato Brain")
    p.add_argument("--pista", default="zigue-zague")
    p.add_argument("--robos", type=int, default=60)
    p.add_argument("--ver", action="store_true", help="assiste na tela")
    args = p.parse_args(argv)

    track = carregar(args.pista)
    cfg = SimConfig()
    runner = EpisodeRunner(track, cfg, n_robots=args.robos)
    brain = RedeAleatoria(cfg.n_sensors, args.robos, seed=1)

    viewer = None
    if args.ver:
        from robo.viewer import Viewer
        viewer = Viewer(track, cfg, brain=brain, titulo="Exemplo — rede aleatória")

    dados = runner.run(brain, seed=7, on_step=viewer.on_step if viewer else None)
    if viewer:
        viewer.close()

    print(resumo(dados))
    pontos = fitness_exemplo(dados)
    melhor = int(np.argmax(pontos))
    print(f"\nfitness de exemplo: melhor {pontos[melhor]:.1f} · média {pontos.mean():.1f}")
    print(f"o campeão avançou {dados['progress'][melhor]:.2f} m de "
          f"{dados['track_length']:.2f} m em {dados['time'][melhor]:.1f} s"
          f"{' e bateu' if dados['collided'][melhor] else ''}")
    print("\ncom pesos aleatórios isso é ruim mesmo — o aprendizado é seu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
