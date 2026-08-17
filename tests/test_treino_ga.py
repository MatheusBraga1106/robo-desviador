"""Verificações do treino por neuroevolução (brains/treino_ga.py).

O que mais importa aqui: retomar um checkpoint tem que continuar a evolução
exatamente de onde parou — mesma população, mesma sequência aleatória — e não
só "parecido". Um bug nisso é silencioso: o treino roda, os números parecem
razoáveis, e ninguém percebe que na verdade reiniciou a sorte a cada checkpoint.

    python -m tests.test_treino_ga
"""

import os
import sys
import tempfile

import numpy as np

from brains.treino_ga import (Hiperparametros, PopulacaoGA, carregar_checkpoint,
                              salvar_checkpoint, treinar)
from robo.config import SimConfig
from robo.persist import RedeSalva
from robo.pistas import zigue_zague
from robo.training import EpisodeRunner

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def hiper_pequeno(**over):
    base = dict(populacao=16, geracoes=4, camadas_ocultas=(6,), checkpoint_a_cada=2,
               seed=3)
    base.update(over)
    return Hiperparametros(**base)


# ---------------------------------------------------------------------- #
def test_populacao_e_um_brain_valido():
    print("\nPopulacaoGA satisfaz o contrato Brain")
    hiper = hiper_pequeno()
    rng = np.random.default_rng(0)
    pop = PopulacaoGA(n_sensores=4, hiper=hiper, rng=rng)

    obs = rng.random((hiper.populacao, 4)).astype(np.float32)
    saida = pop.act_batch(obs)
    checar("forma da saída", saida.shape == (hiper.populacao, 4), f"{saida.shape}")
    checar("saída em [0,1] (sigmoide no fim)", (saida >= 0).all() and (saida <= 1).all())

    d = pop.inspect(0)
    checar("inspect tem o formato certo",
           len(d["ativacoes"]) == len(hiper.camadas_ocultas) + 2
           and len(d["pesos"]) == len(hiper.camadas_ocultas) + 1,
           f"{[len(a) for a in d['ativacoes']]}")

    genoma = pop.genoma(3)
    checar("genoma de um indivíduo tem o número certo de camadas",
           len(genoma) == len(hiper.camadas_ocultas) + 1)
    checar("as camadas do genoma batem com a população",
           all(np.array_equal(genoma[i][0], pop.pop[i][0][3]) for i in range(len(genoma))))


def test_roda_no_episode_runner():
    print("\nroda dentro do EpisodeRunner de verdade")
    hiper = hiper_pequeno()
    rng = np.random.default_rng(1)
    pop = PopulacaoGA(4, hiper, rng)
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=hiper.populacao)
    dados = runner.run(pop, seed=0, max_steps=50)
    checar("telemetria saiu", dados["steps"] > 0 and len(dados["alive"]) == hiper.populacao)


def test_elite_preservada_sem_mudanca():
    print("\nelitismo preserva os melhores exatamente")
    hiper = hiper_pequeno(populacao=30, elite_frac=0.2)
    rng = np.random.default_rng(2)
    pop = PopulacaoGA(4, hiper, rng)
    antes = [(W.copy(), b.copy()) for W, b in pop.pop]

    fitness = rng.uniform(-10, 10, hiper.populacao)
    n_elite = max(1, int(hiper.populacao * hiper.elite_frac))
    melhores = np.argsort(fitness)[::-1][:n_elite]

    pop.evoluir(fitness)

    checar("o melhor de todos sobrevive sem mudar um peso",
           all(np.array_equal(antes[c][0][melhores[0]], pop.pop[c][0][0])
               for c in range(len(pop.pop))))
    checar("a população continua do mesmo tamanho",
           pop.pop[0][0].shape[0] == hiper.populacao)


def test_evoluir_muda_o_resto():
    print("\nfora da elite, a população muda")
    hiper = hiper_pequeno(populacao=30, elite_frac=0.1, taxa_mutacao=0.5, forca_mutacao=0.5)
    rng = np.random.default_rng(4)
    pop = PopulacaoGA(4, hiper, rng)
    antes = pop.pop[0][0].copy()
    pop.evoluir(rng.uniform(-10, 10, hiper.populacao))
    checar("os pesos não são todos idênticos aos anteriores",
           not np.allclose(antes, pop.pop[0][0]))


def test_checkpoint_ida_e_volta():
    print("\ncheckpoint: salvar e carregar reproduz o estado exato")
    hiper = hiper_pequeno()
    rng = np.random.default_rng(5)
    pop = PopulacaoGA(4, hiper, rng)
    pop.evoluir(rng.uniform(-10, 10, hiper.populacao))   # avança o rng um pouco

    caminho = os.path.join(tempfile.gettempdir(), "checkpoint_teste.ga.npz")
    salvar_checkpoint(caminho, pop, geracao=7, melhor_fitness_global=-12.5)
    estado = carregar_checkpoint(caminho)
    os.remove(caminho)

    checar("geração preservada", estado["geracao"] == 7)
    checar("melhor fitness preservado", abs(estado["melhor_fitness_global"] - (-12.5)) < 1e-9)
    checar("n_sensores preservado", estado["n_sensores"] == 4)
    checar("hiperparâmetros preservados", estado["hiper"] == hiper, f"{estado['hiper']}")
    checar("pesos preservados exatamente",
           all(np.array_equal(estado["pop"][i][0], pop.pop[i][0]) for i in range(len(pop.pop))))

    # o ponto crítico: o rng recuperado continua a MESMA sequência, não uma nova
    esperado = pop.rng.integers(0, 1_000_000, size=5)
    obtido = estado["rng"].integers(0, 1_000_000, size=5)
    checar("o gerador aleatório retomado dá os mesmos próximos números",
           np.array_equal(esperado, obtido), f"{esperado} vs {obtido}")


def test_retomar_e_bit_a_bit_identico_ao_continuo():
    print("\nretomar == não ter parado (o teste que mais importa)")
    track, cfg = zigue_zague(), SimConfig()

    direto = os.path.join(tempfile.gettempdir(), "ga_direto")
    partes = os.path.join(tempfile.gettempdir(), "ga_partes")
    for pasta in (direto, partes):
        if os.path.exists(os.path.join(pasta, "historico.csv")):
            os.remove(os.path.join(pasta, "historico.csv"))

    hiper = hiper_pequeno(populacao=14, geracoes=6, checkpoint_a_cada=3, seed=9)
    treinar(hiper, track, cfg, pasta=direto)

    hiper_1a_parte = hiper_pequeno(populacao=14, geracoes=3, checkpoint_a_cada=3, seed=9)
    _, melhor_ate_aqui = treinar(hiper_1a_parte, track, cfg, pasta=partes)

    ck = os.path.join(partes, "geracao_0003.ga.npz")
    checar("o checkpoint da metade foi criado", os.path.exists(ck))
    estado = carregar_checkpoint(ck)
    hiper_retomado = estado["hiper"]
    hiper_retomado.geracoes = 6
    treinar(hiper_retomado, track, cfg, pasta=partes,
           geracao_inicial=estado["geracao"] + 1, pop_inicial=estado["pop"],
           melhor_fitness_global=estado["melhor_fitness_global"], rng=estado["rng"])

    with open(os.path.join(direto, "historico.csv")) as f:
        log_direto = f.read()
    with open(os.path.join(partes, "historico.csv")) as f:
        log_partes = f.read()

    # a coluna de tempo de parede varia entre execuções; compara tudo, menos ela
    def sem_tempo(log):
        linhas = log.strip().splitlines()
        return [",".join(l.split(",")[:-1]) for l in linhas]

    checar("os dois históricos são idênticos, geração a geração",
           sem_tempo(log_direto) == sem_tempo(log_partes),
           "divergiram" if sem_tempo(log_direto) != sem_tempo(log_partes) else "")

    import shutil
    shutil.rmtree(direto, ignore_errors=True)
    shutil.rmtree(partes, ignore_errors=True)


def test_modelo_salvo_carrega_como_brain():
    print("\na 'melhor' rede salva carrega e roda")
    pasta = os.path.join(tempfile.gettempdir(), "ga_modelo")
    if os.path.exists(pasta):
        import shutil
        shutil.rmtree(pasta)

    hiper = hiper_pequeno(populacao=12, geracoes=2, checkpoint_a_cada=2)
    treinar(hiper, zigue_zague(), SimConfig(), pasta=pasta)

    caminho = os.path.join(pasta, "melhor.npz")
    checar("o arquivo foi criado", os.path.exists(caminho))
    rede = RedeSalva.carregar(caminho)
    checar("é um Brain de verdade", hasattr(rede, "act_batch"))
    checar("sabe quantos sensores espera", rede.n_sensores == SimConfig().n_sensors)

    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=3)
    dados = runner.run(rede, seed=0, max_steps=40)
    checar("roda no simulador", dados["steps"] > 0)

    import shutil
    shutil.rmtree(pasta, ignore_errors=True)


def test_cli_usa_os_padroes_da_classe():
    print("\neditar Hiperparametros muda o padrão da linha de comando")
    import argparse
    from dataclasses import fields
    from unittest.mock import patch

    import brains.treino_ga as mod

    # captura os defaults efetivos do parser sem rodar treino nenhum
    capturado = {}
    original = argparse.ArgumentParser.parse_args

    def espiao(self, argv=None, namespace=None):
        capturado.update(vars(original(self, [])))
        raise SystemExit(0)

    def defaults_do_cli(hiper):
        capturado.clear()
        with patch.object(mod, "Hiperparametros", lambda: hiper), \
             patch.object(argparse.ArgumentParser, "parse_args", espiao):
            try:
                mod.main([])
            except SystemExit:
                pass
        return dict(capturado)

    padrao = mod.Hiperparametros()
    d = defaults_do_cli(padrao)
    checar("população do CLI vem da classe", d["populacao"] == padrao.populacao,
           f"CLI {d['populacao']} vs classe {padrao.populacao}")
    checar("gerações também", d["geracoes"] == padrao.geracoes)
    checar("camadas também",
           d["camadas"] == ",".join(str(c) for c in padrao.camadas_ocultas),
           f"{d['camadas']}")
    checar("força da mutação também", d["forca"] == padrao.forca_mutacao)
    checar("checkpoint_a_cada também", d["checkpoint_a_cada"] == padrao.checkpoint_a_cada)
    checar("frequência de exibição também", d["ver_a_cada"] == padrao.ver_a_cada,
           f"CLI {d['ver_a_cada']} vs classe {padrao.ver_a_cada}")

    # o teste que importa: mexer na classe tem que mudar o CLI junto. Antes
    # desta correção, o argparse tinha números fixos próprios e ignorava a
    # classe em silêncio — você editava e não acontecia nada.
    outro = mod.Hiperparametros(populacao=7777, geracoes=13, camadas_ocultas=(5, 6, 7),
                                forca_mutacao=0.99)
    d2 = defaults_do_cli(outro)
    checar("mudar a classe muda o CLI junto",
           d2["populacao"] == 7777 and d2["geracoes"] == 13
           and d2["camadas"] == "5,6,7" and d2["forca"] == 0.99,
           f"populacao={d2['populacao']} geracoes={d2['geracoes']} camadas={d2['camadas']}")

    # e nenhum campo da classe pode ficar sem par no CLI sem querer
    no_cli = {"populacao": "populacao", "geracoes": "geracoes",
              "elite_frac": "elite", "torneio": "torneio",
              "taxa_mutacao": "mutacao", "forca_mutacao": "forca",
              "checkpoint_a_cada": "checkpoint_a_cada", "seed": "seed",
              "camadas_ocultas": "camadas", "ver_a_cada": "ver_a_cada"}
    faltando = [f.name for f in fields(mod.Hiperparametros) if f.name not in no_cli]
    checar("todo hiperparâmetro tem opção equivalente no CLI", not faltando,
           f"sem opção: {faltando}" if faltando else "")


def test_catalogo_de_pistas():
    print("\ncatálogo de pistas")
    from robo.pistas import PISTAS, catalogo, impressao_digital, pistas_salvas

    itens = catalogo()
    embutidas = [i for i in itens if i["origem"] == "embutida"]
    checar("lista todas as embutidas", len(embutidas) == len(PISTAS), f"{len(embutidas)}")
    checar("cada item tem o que o menu precisa",
           all({"ref", "origem", "nome", "comprimento", "caixas", "erro"} <= set(i)
               for i in itens))
    checar("comprimentos são reais", all(i["comprimento"] > 0 for i in embutidas))

    vazio = catalogo(pasta=os.path.join(tempfile.gettempdir(), "pasta_que_nao_existe"))
    checar("pasta de pistas inexistente não quebra, só não lista nada seu",
           all(i["origem"] == "embutida" for i in vazio))
    checar("pistas_salvas devolve lista vazia nesse caso",
           pistas_salvas(os.path.join(tempfile.gettempdir(), "nada")) == [])


def test_catalogo_tolera_json_corrompido():
    print("\npista salva corrompida não derruba a listagem")
    from robo.pistas import catalogo

    pasta = os.path.join(tempfile.gettempdir(), "pistas_ruins")
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, "quebrada.json"), "w") as f:
        f.write("{isto nao e json valido")

    itens = catalogo(pasta=pasta)
    ruim = [i for i in itens if i["ref"].endswith("quebrada.json")]
    checar("a pista ruim aparece na lista", len(ruim) == 1)
    checar("e vem marcada com o erro, em vez de estourar",
           bool(ruim and ruim[0]["erro"]), f"{ruim[0]['erro'][:40] if ruim else ''}")
    checar("as embutidas continuam listadas",
           any(i["origem"] == "embutida" for i in itens))

    import shutil
    shutil.rmtree(pasta, ignore_errors=True)


def test_impressao_digital_detecta_edicao():
    print("\nimpressão digital pega pista editada com o mesmo nome")
    from robo.pistas import carregar, impressao_digital

    a = carregar("curva-u")
    b = carregar("curva-u")
    checar("mesma pista, mesmo hash", impressao_digital(a) == impressao_digital(b))
    checar("pistas diferentes, hashes diferentes",
           impressao_digital(a) != impressao_digital(carregar("caracol")))

    editada = carregar("curva-u")
    editada.centerline = editada.centerline[:-1]      # como se você mexesse no editor
    editada.rebuild()
    checar("mesmo nome, geometria mudada -> hash muda",
           impressao_digital(editada) != impressao_digital(a),
           f"{editada.name} continua se chamando '{a.name}'")


def test_checkpoint_registra_a_pista():
    print("\ncheckpoint guarda em que pista treinou")
    from robo.pistas import carregar, descrever

    hiper = hiper_pequeno()
    pop = PopulacaoGA(4, hiper, np.random.default_rng(0))
    track = carregar("caracol")
    info = descrever(track, "caracol")

    caminho = os.path.join(tempfile.gettempdir(), "ck_pista.ga.npz")
    salvar_checkpoint(caminho, pop, geracao=3, melhor_fitness_global=1.0, pista_info=info)
    estado = carregar_checkpoint(caminho)
    os.remove(caminho)

    checar("a referência da pista voltou", estado["pista"].get("ref") == "caracol",
           f"{estado['pista']}")
    checar("o hash voltou igual", estado["pista"].get("hash") == info["hash"])
    checar("o comprimento voltou", abs(estado["pista"].get("comprimento", 0) - 24.4) < 0.01)


def test_checkpoint_antigo_sem_pista_nao_quebra():
    print("\ncheckpoint de versão anterior (sem a marca da pista)")
    hiper = hiper_pequeno()
    pop = PopulacaoGA(4, hiper, np.random.default_rng(0))
    caminho = os.path.join(tempfile.gettempdir(), "ck_sem_pista.ga.npz")

    # grava e depois remove a chave, imitando um arquivo salvo antes desta versão
    salvar_checkpoint(caminho, pop, geracao=1, melhor_fitness_global=0.0)
    d = dict(np.load(caminho, allow_pickle=True))
    d.pop("pista_json", None)
    np.savez(caminho, **d)

    estado = carregar_checkpoint(caminho)
    os.remove(caminho)
    checar("carrega mesmo assim", estado["geracao"] == 1)
    checar("e devolve pista vazia, sinalizando 'não dá para comparar'",
           estado["pista"] == {}, f"{estado['pista']}")


def test_avisar_troca_de_pista():
    print("\naviso de troca de pista")
    import io
    from contextlib import redirect_stdout

    from brains.treino_ga import avisar_troca_de_pista
    from robo.pistas import carregar, descrever

    def falar(info, track, ref):
        buf = io.StringIO()
        with redirect_stdout(buf):
            avisar_troca_de_pista(info, track, ref)
        return buf.getvalue()

    curva = carregar("curva-u")
    info_curva = descrever(curva, "curva-u")

    saida = falar(info_curva, curva, "curva-u")
    checar("mesma pista: confirma, sem alarme",
           "confere" in saida and "ATENÇÃO" not in saida, saida.strip()[:60])

    caracol = carregar("caracol")
    saida = falar(info_curva, caracol, "caracol")
    checar("pista diferente: avisa e explica a queda de fitness",
           "ATENÇÃO" in saida and "curva-u" in saida and "caracol" in saida,
           saida.strip()[:80])

    editada = carregar("curva-u")
    editada.centerline = editada.centerline[:-1]
    editada.rebuild()
    saida = falar(info_curva, editada, "curva-u")
    checar("mesmo nome mas editada: avisa que foi EDITADA",
           "EDITADA" in saida, saida.strip()[:80])

    saida = falar({}, curva, "curva-u")
    checar("checkpoint antigo: diz que não dá para conferir",
           "não registrou" in saida, saida.strip()[:60])


def test_escolher_pista_sem_terminal_interativo():
    print("\nmenu não trava em script/tarefa automática")
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch

    from brains.treino_ga import escolher_pista

    with patch("sys.stdin.isatty", return_value=False):
        buf = io.StringIO()
        with redirect_stdout(buf):
            escolhida = escolher_pista(padrao="diagonais")
    checar("cai no padrão sem esperar digitação", escolhida == "diagonais", escolhida)
    checar("e explica por quê", "não interativo" in buf.getvalue())


def test_escolher_pista_interativo():
    print("\nmenu interativo aceita número, nome e caminho")
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch

    from brains.treino_ga import escolher_pista
    from robo.pistas import catalogo

    itens = catalogo()

    def escolher(respostas):
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", side_effect=respostas):
            with redirect_stdout(io.StringIO()):
                return escolher_pista()

    checar("número escolhe o item certo", escolher(["1"]) == itens[0]["ref"],
           f"{escolher(['1'])}")
    checar("Enter usa o padrão", escolher([""]) == "zigue-zague")
    checar("nome embutido direto funciona", escolher(["caracol"]) == "caracol")
    checar("número fora da lista pede de novo, não quebra",
           escolher(["999", "3"]) == itens[2]["ref"])
    checar("texto sem sentido pede de novo",
           escolher(["nao existe essa pista", "1"]) == itens[0]["ref"])
    checar("Ctrl+C / fim de entrada cai no padrão",
           escolher(EOFError()) == "zigue-zague")


def _tem_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def test_resolver_dispositivo():
    print("\nresolver_dispositivo nunca falha")
    from brains.treino_ga import resolver_dispositivo
    checar("'cpu' pedido continua 'cpu'", resolver_dispositivo("cpu") == "cpu")

    d = resolver_dispositivo("cuda")
    if _tem_gpu():
        checar("com GPU disponível, 'cuda' pedido dá 'cuda'", d == "cuda", f"{d}")
    else:
        checar("sem GPU, 'cuda' pedido cai para 'cpu' sem quebrar", d == "cpu", f"{d}")

    d = resolver_dispositivo("auto")
    checar("'auto' sempre devolve algo válido", d in ("cpu", "cuda"), f"{d}")


def test_gpu_reproduz_a_mesma_conta_da_cpu():
    print("\nforward na GPU bate com o forward na CPU")
    if not _tem_gpu():
        checar("sem GPU nesta máquina, pulei", True, "(nada a comparar)")
        return

    hiper = hiper_pequeno(populacao=25, camadas_ocultas=(10, 8))
    rng_cpu = np.random.default_rng(11)
    rng_gpu = np.random.default_rng(11)
    pop_cpu = PopulacaoGA(6, hiper, rng_cpu, dispositivo="cpu")
    pop_gpu = PopulacaoGA(6, hiper, rng_gpu, dispositivo="cuda")

    checar("os pesos iniciais são idênticos (mesma seed)",
           all(np.allclose(a[0], b[0]) for a, b in zip(pop_cpu.pop, pop_gpu.pop)))

    obs = np.random.default_rng(0).random((25, 6)).astype(np.float32)
    saida_cpu = pop_cpu.act_batch(obs)
    saida_gpu = pop_gpu.act_batch(obs)
    erro = float(np.max(np.abs(saida_cpu - saida_gpu)))
    checar("saída da GPU bate com a da CPU", erro < 1e-4, f"erro máximo {erro:.2e}")

    d_cpu, d_gpu = pop_cpu.inspect(3), pop_gpu.inspect(3)
    erro_ativ = max(float(np.max(np.abs(a - b)))
                    for a, b in zip(d_cpu["ativacoes"], d_gpu["ativacoes"]))
    checar("inspect() também bate (painel funciona igual)", erro_ativ < 1e-4,
           f"erro máximo {erro_ativ:.2e}")


def test_gpu_evoluir_preserva_o_cache():
    print("\nevoluir() na GPU não deixa o cache com a geração antiga")
    if not _tem_gpu():
        checar("sem GPU nesta máquina, pulei", True, "(nada a testar)")
        return

    hiper = hiper_pequeno(populacao=20)
    rng = np.random.default_rng(6)
    pop = PopulacaoGA(4, hiper, rng, dispositivo="cuda")
    obs = np.random.default_rng(0).random((20, 4)).astype(np.float32)

    antes = pop.act_batch(obs).copy()
    pop.evoluir(rng.uniform(-10, 10, hiper.populacao))
    depois = pop.act_batch(obs)

    checar("a saída muda depois de evoluir (não ficou presa à geração antiga)",
           not np.allclose(antes, depois))
    checar("e a saída bate com os pesos numpy atuais (não com os antigos)",
           np.allclose(depois, pop._act_batch_numpy(obs), atol=1e-4))


def test_treino_completo_na_gpu():
    print("\num treino pequeno de ponta a ponta na GPU")
    if not _tem_gpu():
        checar("sem GPU nesta máquina, pulei", True, "(nada a treinar)")
        return

    pasta = os.path.join(tempfile.gettempdir(), "ga_gpu")
    if os.path.exists(pasta):
        import shutil
        shutil.rmtree(pasta)

    hiper = hiper_pequeno(populacao=20, geracoes=3, checkpoint_a_cada=2)
    populacao, melhor = treinar(hiper, zigue_zague(), SimConfig(), pasta=pasta,
                                dispositivo="cuda")
    checar("treinou e devolveu um fitness real", np.isfinite(melhor))
    checar("a população ficou marcada como 'cuda'", populacao.dispositivo == "cuda")

    rede = RedeSalva.carregar(os.path.join(pasta, "melhor.npz"))
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=3)
    dados = runner.run(rede, seed=0, max_steps=30)
    checar("a rede salva (numpy puro) roda fora da GPU sem problema", dados["steps"] > 0)

    import shutil
    shutil.rmtree(pasta, ignore_errors=True)


def test_sensores_incompativeis_ao_continuar_sao_detectados():
    print("\nretomar com config de sensores incompatível")
    pasta = os.path.join(tempfile.gettempdir(), "ga_sensores")
    if os.path.exists(pasta):
        import shutil
        shutil.rmtree(pasta)

    cfg6 = SimConfig()
    cfg6.sensor.count = 6
    hiper = hiper_pequeno(populacao=10, geracoes=1, checkpoint_a_cada=1)
    treinar(hiper, zigue_zague(), cfg6, pasta=pasta)

    estado = carregar_checkpoint(os.path.join(pasta, "geracao_0001.ga.npz"))
    checar("o checkpoint sabe que foi treinado com 6 sensores",
           estado["n_sensores"] == 6, f"{estado['n_sensores']}")
    checar("bate com a config usada", estado["n_sensores"] == cfg6.n_sensors)
    checar("e não bate com a config de fábrica (4 sensores)",
           estado["n_sensores"] != SimConfig().n_sensors)

    import shutil
    shutil.rmtree(pasta, ignore_errors=True)


def main():
    for fn in [test_populacao_e_um_brain_valido, test_roda_no_episode_runner,
               test_elite_preservada_sem_mudanca, test_evoluir_muda_o_resto,
               test_checkpoint_ida_e_volta, test_retomar_e_bit_a_bit_identico_ao_continuo,
               test_modelo_salvo_carrega_como_brain,
               test_cli_usa_os_padroes_da_classe, test_catalogo_de_pistas, test_catalogo_tolera_json_corrompido,
               test_impressao_digital_detecta_edicao, test_checkpoint_registra_a_pista,
               test_checkpoint_antigo_sem_pista_nao_quebra, test_avisar_troca_de_pista,
               test_escolher_pista_sem_terminal_interativo, test_escolher_pista_interativo,
               test_resolver_dispositivo,
               test_gpu_reproduz_a_mesma_conta_da_cpu, test_gpu_evoluir_preserva_o_cache,
               test_treino_completo_na_gpu,
               test_sensores_incompativeis_ao_continuar_sao_detectados]:
        fn()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
