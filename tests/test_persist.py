"""Verificações do laço passo a passo e do formato de salvar.

    python -m tests.test_persist
"""

import os
import sys
import tempfile

import numpy as np

from robo.config import SimConfig
from robo.persist import ATIVACOES, RedeSalva, carregar, de_torch, salvar
from robo.pistas import zigue_zague
from robo.training import EpisodeRunner

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def rede_falsa(tamanhos=(4, 6, 4), seed=0):
    rng = np.random.default_rng(seed)
    camadas = [(rng.normal(0, 0.4, (b, a)), rng.normal(0, 0.1, b))
               for a, b in zip(tamanhos, tamanhos[1:])]
    ativacoes = ["tanh"] * (len(camadas) - 1) + ["sigmoid"]
    return camadas, ativacoes


class Reto:
    def act_batch(self, obs):
        s = np.zeros((len(obs), 4)); s[:, 0] = 1.0
        return s


# ------------------------------------------------------- passo a passo
def test_laco_passo_a_passo():
    print("\nlaço passo a passo")
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=3)
    obs = runner.reset(seed=1)
    checar("reset devolve observação", obs.shape == (3, 4), f"{obs.shape}")

    brain, passos, fim = Reto(), 0, False
    campos = None
    while not fim and passos < 60:
        obs, info, fim = runner.step(brain.act_batch(obs))
        campos = info
        passos += 1

    esperados = {"delta_progress", "collided_now", "finished_now", "alive",
                 "active", "min_reading", "forward_speed", "progress", "dt"}
    checar("info tem os fatos do passo", esperados <= set(campos), f"{sorted(campos)}")
    checar("um valor por robô", len(campos["delta_progress"]) == 3)
    checar("nada de recompensa embutida",
           not any("reward" in k or "recompensa" in k or "fitness" in k for k in campos))


def test_delta_progress_acompanha():
    print("\ndelta de avanço por passo")
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=2)
    obs = runner.reset(seed=1)
    # a largada nasce afastada da parede que fecha o início, então o robô já
    # começa alguns centímetros adiante no traçado
    inicial = runner.world.step_info["progress"].copy()
    checar("largada já começa adiante no traçado", inicial[0] > 0.1,
           f"{inicial[0]:.2f} m")

    soma = np.zeros(2)
    for _ in range(40):
        obs, info, fim = runner.step(Reto().act_batch(obs))
        soma += info["delta_progress"]
        if fim:
            break

    checar("andar para frente dá delta positivo", soma.max() > 0.3, f"{soma.max():.2f} m")
    andou = info["progress"][0] - inicial[0]
    checar("a soma dos deltas bate com o avanço",
           abs(soma[0] - andou) < 1e-9,
           f"soma {soma[0]:.4f} m contra avanço {andou:.4f} m")

    parado = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=1)
    o = parado.reset(seed=1)
    o, info, _ = parado.step(np.zeros((1, 4)))
    checar("robô parado não avança", abs(info["delta_progress"][0]) < 1e-9)


def test_batida_e_chegada_sinalizadas_uma_vez():
    print("\nbatida e chegada sinalizadas no passo certo")
    cfg = SimConfig()
    runner = EpisodeRunner(zigue_zague(), cfg, n_robots=1)
    obs = runner.reset(seed=2)
    batidas, fim = 0, False
    while not fim:
        obs, info, fim = runner.step(Reto().act_batch(obs))
        batidas += int(info["collided_now"].sum())
    checar("bateu exatamente uma vez", batidas == 1, f"{batidas} sinalizações")

    # objetivo em cima da largada: chega no primeiro passo, e só uma vez
    track = zigue_zague()
    track.goal = np.array([track.start[0] + 0.05, track.start[1], 0.4])
    r2 = EpisodeRunner(track, cfg, n_robots=1)
    o = r2.reset(seed=1)
    chegadas = 0
    for _ in range(20):
        o, info, fim = r2.step(Reto().act_batch(o))
        chegadas += int(info["finished_now"].sum())
        if fim:
            break
    checar("chegou sinalizado uma vez só", chegadas == 1, f"{chegadas} sinalizações")


def test_recompensa_do_usuario_e_montavel():
    print("\ndá para montar a recompensa que ele descreveu")

    def minha_recompensa(info, peso_tempo=0.5, punicao=50.0, bonus=100.0):
        r = info["delta_progress"] * 10.0          # mais perto da chegada, mais ponto
        r -= peso_tempo * info["dt"] * info["active"]   # demorar custa
        r -= punicao * info["collided_now"]        # bateu, perdeu
        r += bonus * info["finished_now"]
        return r

    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=2)
    obs = runner.reset(seed=2)
    total, fim = np.zeros(2), False
    while not fim:
        obs, info, fim = runner.step(Reto().act_batch(obs))
        total += minha_recompensa(info)
    checar("soma finita e negativa (ele bate)", np.isfinite(total).all() and total.max() < 0,
           f"{np.round(total, 1)}")


# ------------------------------------------------------- persistência
def test_salvar_e_carregar():
    print("\nsalvar e carregar")
    caminho = os.path.join(tempfile.gettempdir(), "rede_teste.npz")
    camadas, ativacoes = rede_falsa()
    cfg = SimConfig()
    cfg.sensor.count = 4
    salvar(caminho, camadas, ativacoes, cfg, nota="geração 42")

    d = carregar(caminho)
    checar("camadas voltaram", len(d["camadas"]) == len(camadas))
    checar("pesos idênticos",
           all(np.allclose(a[0], b[0].astype(np.float32)) for a, b in zip(d["camadas"], camadas)))
    checar("ativações voltaram", d["ativacoes"] == ativacoes, f"{d['ativacoes']}")
    checar("a config veio junto", d["cfg"] is not None and d["cfg"].sensor.count == 4)
    checar("extras vieram", d["extra"].get("nota") == "geração 42", f"{d['extra']}")

    rede = RedeSalva.carregar(caminho)
    os.remove(caminho)
    obs = np.random.default_rng(0).random((5, 4))
    saida = rede.act_batch(obs)
    checar("carregada já é um Brain", saida.shape == (5, 4), f"{saida.shape}")
    checar("saída em [0,1] com sigmoide no fim",
           (saida >= 0).all() and (saida <= 1).all())
    checar("inspect alimenta o painel",
           len(rede.inspect(0)["ativacoes"]) == len(camadas) + 1)


def test_formatos_errados_falham_cedo():
    print("\nformato errado falha na hora de salvar")
    caminho = os.path.join(tempfile.gettempdir(), "ruim.npz")
    camadas, ativacoes = rede_falsa()

    casos = [
        ("ativações a menos", camadas, ativacoes[:-1], "exigem"),
        ("ativação inexistente", camadas, ["tanh", "banana"], "desconhecida"),
        ("camadas que não encaixam",
         [(np.zeros((6, 4)), np.zeros(6)), (np.zeros((4, 99)), np.zeros(4))],
         ["tanh", "tanh"], "a anterior sai"),
    ]
    for nome, c, a, trecho in casos:
        try:
            salvar(caminho, c, a)
            checar(nome, False, "não reclamou")
        except ValueError as e:
            checar(nome, trecho in str(e), f"disse: {str(e)[:64]}")
    if os.path.exists(caminho):
        os.remove(caminho)


def test_extrair_do_torch():
    print("\nextrair de uma rede torch")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        checar("torch não instalado, pulei", True, "instale com pip install torch")
        return

    rede = nn.Sequential(nn.Linear(4, 8), nn.Tanh(),
                         nn.Linear(8, 6), nn.ReLU(),
                         nn.Linear(6, 4), nn.Sigmoid())
    camadas, ativacoes = de_torch(rede)
    checar("achou as 3 camadas", len(camadas) == 3, f"{[W.shape for W, _ in camadas]}")
    checar("achou as ativações", ativacoes == ["tanh", "relu", "sigmoid"], f"{ativacoes}")

    caminho = os.path.join(tempfile.gettempdir(), "do_torch.npz")
    salvar(caminho, camadas, ativacoes, SimConfig())
    neutra = RedeSalva.carregar(caminho)
    os.remove(caminho)

    obs = np.random.default_rng(3).random((7, 4))
    do_torch = rede(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
    do_numpy = neutra.act_batch(obs)
    erro = float(np.max(np.abs(do_torch - do_numpy)))
    checar("numpy reproduz o torch", erro < 1e-5, f"erro máximo {erro:.2e}")

    # rede aninhada, como sai de uma política com extrator + cabeça
    aninhada = nn.Sequential(nn.Sequential(nn.Linear(4, 5), nn.Tanh()), nn.Linear(5, 4))
    c2, a2 = de_torch(aninhada)
    checar("percorre módulos aninhados", len(c2) == 2 and a2 == ["tanh", "linear"],
           f"{len(c2)} camadas, {a2}")


def test_rede_salva_roda_no_runner():
    print("\nrede carregada roda no simulador")
    caminho = os.path.join(tempfile.gettempdir(), "rede_runner.npz")
    cfg = SimConfig()
    camadas, ativacoes = rede_falsa((cfg.n_sensors, 6, 4))
    salvar(caminho, camadas, ativacoes, cfg)
    rede = RedeSalva.carregar(caminho)
    os.remove(caminho)

    checar("sabe quantos sensores espera", rede.n_sensores == cfg.n_sensors,
           f"{rede.n_sensores}")
    runner = EpisodeRunner(zigue_zague(), cfg, n_robots=5)
    dados = runner.run(rede, seed=4, max_steps=80)
    checar("passa no check_brain e roda", dados["steps"] > 0, f"{dados['steps']} passos")
    checar("telemetria saiu", np.isfinite(dados["progress"]).all())


def main():
    for fn in [test_laco_passo_a_passo, test_delta_progress_acompanha,
               test_batida_e_chegada_sinalizadas_uma_vez,
               test_recompensa_do_usuario_e_montavel, test_salvar_e_carregar,
               test_formatos_errados_falham_cedo, test_extrair_do_torch,
               test_rede_salva_roda_no_runner]:
        fn()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
