"""Salvar e carregar a rede treinada, num formato que sobrevive ao PyTorch.

Por que não guardar só o `state_dict`
-------------------------------------
Porque o destino final desta rede é um Arduino, e lá não existe torch. O
`state_dict` também amarra o arquivo à classe Python que o criou: renomeie a
classe e o arquivo vira lixo.

Então o formato aqui é neutro — só as matrizes, os vieses, o nome de cada
ativação e a `SimConfig` usada no treino. Isso vira `Brain` com dois einsum,
vira C com um `for` aninhado, e continua legível daqui a um ano.

Guarde o `state_dict` também se quiser retomar o treino; mas é este arquivo que
vai virar firmware.

    from robo.persist import de_torch, salvar, RedeSalva

    salvar("modelos/v1.npz", *de_torch(minha_rede), cfg)
    brain = RedeSalva.carregar("modelos/v1.npz")     # já é um Brain
"""

import os

import numpy as np

from .config import SimConfig

ATIVACOES = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "linear": lambda x: x,
}

# como cada ativação se chama no torch, para reconhecê-la ao extrair
_DO_TORCH = {"Tanh": "tanh", "ReLU": "relu", "Sigmoid": "sigmoid",
             "LeakyReLU": "leaky_relu", "Identity": "linear"}


def salvar(caminho, camadas, ativacoes, cfg: SimConfig = None, **extra):
    """camadas: lista de (W, b) com W em (saída, entrada). ativacoes: um nome por camada."""
    camadas = [(np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32))
               for W, b in camadas]
    _validar(camadas, ativacoes)

    dados = {"n_camadas": len(camadas),
             "ativacoes": np.array(list(ativacoes), dtype=object)}
    for i, (W, b) in enumerate(camadas):
        dados[f"W{i}"], dados[f"b{i}"] = W, b
    if cfg is not None:
        # a config vai junto porque rede treinada com 4 sensores a 90° é lixo
        # num robô com 5 a 150°, e sem essa marca você descobre isso no firmware
        dados["config"] = np.array(cfg.to_dict(), dtype=object)
    for k, v in extra.items():
        dados[f"extra_{k}"] = np.array(v, dtype=object)

    np.savez(caminho, **dados)
    return caminho


def carregar(caminho) -> dict:
    d = np.load(caminho, allow_pickle=True)
    if "n_camadas" not in d.files:
        if "hiper_json" in d.files or "geracao" in d.files:
            raise ValueError(
                f"{caminho} é um checkpoint de população (.ga.npz), não uma rede "
                f"pronta. Isso guarda TODOS os indivíduos do treino, para retomar "
                f"com 'python -m brains.treino_ga --continuar {caminho}'. "
                f"Para avaliar, use a rede campeã (ex.: checkpoints/melhor.npz).")
        raise ValueError(f"{caminho} não parece uma rede salva por robo.persist "
                         f"(falta a chave 'n_camadas').")
    n = int(d["n_camadas"])
    camadas = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
    ativacoes = [str(a) for a in d["ativacoes"]]
    cfg = SimConfig.from_dict(d["config"].item()) if "config" in d else None
    extra = {k[6:]: d[k].item() for k in d.files if k.startswith("extra_")}
    return {"camadas": camadas, "ativacoes": ativacoes, "cfg": cfg, "extra": extra}


PASTAS_DE_MODELO = ("checkpoints", "modelos", "checkpoints_teste")


def rede_mais_recente(pastas=PASTAS_DE_MODELO):
    """Caminho da última rede treinada, ou None se não houver nenhuma.

    Procura os `.npz` implantáveis e devolve o mais recente pela data de
    modificação. Ignora os `.ga.npz`: aqueles são estado de população para
    retomar o treino, não uma rede que se possa rodar ou embarcar — confundir os
    dois daria um erro obscuro na hora de carregar.
    """
    candidatos = []
    for pasta in pastas:
        if not os.path.isdir(pasta):
            continue
        for nome in os.listdir(pasta):
            if not nome.endswith(".npz") or nome.endswith(".ga.npz"):
                continue
            caminho = os.path.join(pasta, nome)
            candidatos.append((os.path.getmtime(caminho), caminho))
    if not candidatos:
        return None
    return max(candidatos)[1].replace("\\", "/")


def _validar(camadas, ativacoes):
    if len(ativacoes) != len(camadas):
        raise ValueError(f"{len(camadas)} camadas exigem {len(camadas)} ativações, "
                         f"vieram {len(ativacoes)}")
    for nome in ativacoes:
        if nome not in ATIVACOES:
            raise ValueError(f"ativação desconhecida: {nome!r}. "
                             f"conhecidas: {', '.join(ATIVACOES)}")
    for i, (W, b) in enumerate(camadas):
        if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
            raise ValueError(f"camada {i}: W {W.shape} e b {b.shape} não combinam; "
                             f"esperado W (saída, entrada) e b (saída,)")
        if i and camadas[i - 1][0].shape[0] != W.shape[1]:
            raise ValueError(f"camada {i}: entra {W.shape[1]}, mas a anterior "
                             f"sai {camadas[i-1][0].shape[0]}")


# --------------------------------------------------------------------- #
def de_torch(modulo):
    """Extrai (camadas, ativacoes) de uma rede torch de camadas lineares.

    Funciona com `nn.Sequential` de Linear + ativação, que é o formato de 99%
    das políticas pequenas. Rede com bloco exótico não é reconhecida — aí monte
    a lista à mão, que é literalmente `[(W, b), ...]`.

    Não importa torch: só olha os nomes das classes e os tensores, então roda
    mesmo num ambiente sem torch instalado.
    """
    camadas, ativacoes = [], []
    for sub in _percorrer(modulo):
        nome = type(sub).__name__
        if nome == "Linear":
            if camadas and len(ativacoes) < len(camadas):
                ativacoes.append("linear")     # duas Linear seguidas, sem ativação
            W = sub.weight.detach().cpu().numpy()
            b = (sub.bias.detach().cpu().numpy() if sub.bias is not None
                 else np.zeros(W.shape[0], dtype=np.float32))
            camadas.append((W, b))
        elif nome in _DO_TORCH and camadas and len(ativacoes) < len(camadas):
            ativacoes.append(_DO_TORCH[nome])

    if not camadas:
        raise ValueError("nenhuma camada Linear encontrada; "
                         "monte a lista [(W, b), ...] à mão")
    while len(ativacoes) < len(camadas):
        ativacoes.append("linear")
    return camadas, ativacoes


def _percorrer(modulo):
    """Percorre os submódulos em ordem, sem depender de torch estar instalado."""
    filhos = list(getattr(modulo, "children", lambda: [])())
    if not filhos:
        yield modulo
        return
    for f in filhos:
        yield from _percorrer(f)


# --------------------------------------------------------------------- #
class RedeSalva:
    """Um `Brain` pronto a partir do arquivo salvo. Numpy puro, sem torch.

    É com isto que a corrida contra a IA e a conferência do firmware rodam: se
    esta classe reproduz o que você treinou, o C do Arduino também vai.
    """

    def __init__(self, camadas, ativacoes, cfg: SimConfig = None, extra=None):
        _validar([(np.asarray(W), np.asarray(b)) for W, b in camadas], ativacoes)
        self.camadas = [(np.asarray(W, dtype=np.float64),
                         np.asarray(b, dtype=np.float64)) for W, b in camadas]
        self.ativacoes = list(ativacoes)
        self.cfg = cfg
        self.extra = extra or {}
        self._ativ = None

    @classmethod
    def carregar(cls, caminho) -> "RedeSalva":
        d = carregar(caminho)
        return cls(d["camadas"], d["ativacoes"], d["cfg"], d["extra"])

    @property
    def n_sensores(self) -> int:
        return int(self.camadas[0][0].shape[1])

    def act_batch(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        registro = [x]
        for (W, b), nome in zip(self.camadas, self.ativacoes):
            x = ATIVACOES[nome](x @ W.T + b)
            registro.append(x)
        self._ativ = registro
        return x

    def inspect(self, i: int = 0):
        if self._ativ is None:
            return None
        return {"ativacoes": [a[i] for a in self._ativ],
                "pesos": [W for W, _ in self.camadas]}
