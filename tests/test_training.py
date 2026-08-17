"""Verificações do contrato Brain, do EpisodeRunner e da medida de progresso.

    python -m tests.test_training
"""

import sys

import numpy as np

from robo.brain import PorRobo, as_buttons, check_brain, describe_action
from robo.config import SimConfig
from robo.geometry import project_polyline
from robo.pistas import zigue_zague
from robo.track import Track
from robo.training import EpisodeRunner, resumo

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


class Reto:
    """Cérebro burro que só acelera. Útil como referência previsível."""
    def act_batch(self, obs):
        saida = np.zeros((len(obs), 4))
        saida[:, 0] = 1.0
        return saida


# ------------------------------------------------------- progresso
def test_progresso_ao_longo_do_tracado():
    print("\nprogresso medido no traçado, não em linha reta")
    linha = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])   # L de 8 m

    d, arco = project_polyline(np.array([[2.0, 0.0]]), linha)
    checar("meio do primeiro trecho", abs(arco[0] - 2.0) < 1e-9, f"arco {arco[0]:.3f}")

    d, arco = project_polyline(np.array([[4.0, 2.0]]), linha)
    checar("meio do segundo trecho", abs(arco[0] - 6.0) < 1e-9, f"arco {arco[0]:.3f}")

    d, arco = project_polyline(np.array([[3.0, 0.7]]), linha)
    checar("fora da linha projeta certo", abs(arco[0] - 3.0) < 1e-9 and abs(d[0] - 0.7) < 1e-9,
           f"arco {arco[0]:.2f}, distância {d[0]:.2f}")

    # o caso que motivou tudo: perto do objetivo em linha reta, longe em percurso
    fim = linha[-1]
    perto_reto = np.array([[0.1, 3.9]])      # quase no início do traçado
    meio = np.array([[4.0, 0.0]])            # metade do caminho andado
    _, arco_perto = project_polyline(perto_reto, linha)
    _, arco_meio = project_polyline(meio, linha)
    reta_perto = np.hypot(*(perto_reto[0] - fim))
    reta_meio = np.hypot(*(meio[0] - fim))

    checar("em linha reta o ponto errado parece mais perto", reta_perto < reta_meio,
           f"{reta_perto:.2f} m < {reta_meio:.2f} m")
    checar("no traçado a ordem se inverte, como deve ser", arco_perto[0] < arco_meio[0],
           f"avançou {arco_perto[0]:.2f} m contra {arco_meio[0]:.2f} m")


def test_telemetria_tem_progresso():
    print("\ntelemetria de progresso")
    track = zigue_zague()
    runner = EpisodeRunner(track, SimConfig(), n_robots=3)
    dados = runner.run(Reto(), seed=1)

    checar("track_length bate com o traçado",
           abs(dados["track_length"] - track.length) < 1e-9,
           f"{dados['track_length']:.2f} m")
    checar("progresso dentro do traçado",
           np.all(dados["progress"] >= 0) and np.all(dados["progress"] <= track.length + 1e-9),
           f"{np.round(dados['progress'], 2)}")
    checar("progress + remaining = comprimento",
           np.allclose(dados["progress"] + dados["remaining"], track.length))
    checar("andar reto avança alguma coisa", dados["progress"].max() > 0.5,
           f"máximo {dados['progress'].max():.2f} m")


# ------------------------------------------------------- contrato
def test_check_brain_aceita_o_certo():
    print("\ncheck_brain com rede correta")
    msg = check_brain(Reto(), n_sensores=4, n_robos=5)
    checar("aprova e descreve", "contrato ok" in msg, msg)
    checar("avisa que não há painel", "painel desligado" in msg)


def test_check_brain_pega_erros():
    print("\ncheck_brain com rede errada")

    class SemAct:
        pass

    class FormaErrada:
        def act_batch(self, obs):
            return np.zeros((len(obs), 2))       # 2 saídas em vez de 4

    class ComNaN:
        def act_batch(self, obs):
            s = np.zeros((len(obs), 4)); s[0, 0] = np.nan
            return s

    for nome, rede, esperado in [
        ("falta act_batch", SemAct(), "act_batch"),
        ("forma errada", FormaErrada(), "esperado"),
        ("NaN na saída", ComNaN(), "NaN"),
    ]:
        try:
            check_brain(rede, 4, 3)
            checar(nome, False, "não reclamou")
        except (TypeError, ValueError) as e:
            checar(nome, esperado in str(e), f"disse: {str(e)[:70]}")


def test_inspect_inconsistente_e_apontado():
    print("\ncheck_brain com inspect torto")

    class Torto:
        def act_batch(self, obs):
            s = np.zeros((len(obs), 4)); s[:, 0] = 1
            return s

        def inspect(self, i=0):
            return {"ativacoes": [np.zeros(4), np.zeros(3), np.zeros(4)],
                    "pesos": [np.zeros((3, 4))]}      # falta uma matriz

    msg = check_brain(Torto(), 4, 3)
    checar("acusa o número de matrizes", "exigem 2" in msg and "vieram 1" in msg, msg)


def test_botoes_e_adaptador():
    print("\nbotões e adaptador de um robô por vez")
    a = np.array([[0.9, 0.1, 0.6, 0.4]])
    b = as_buttons(a)[0]
    checar("limiar de 0.5", list(b) == [True, False, True, False], f"{b}")
    checar("descrição legível", describe_action(a) == "acelerar+esquerda", describe_action(a))
    checar("nada apertado vira 'parado'", describe_action(np.zeros((1, 4))) == "parado")

    rede = PorRobo(lambda obs: [float(obs[0] > 0.5), 0.0, 0.0, 0.0])
    saida = rede.act_batch(np.array([[0.9, 0, 0, 0], [0.1, 0, 0, 0]]))
    checar("PorRobo monta o lote", saida.shape == (2, 4), f"{saida.shape}")
    checar("e chama a função por robô", saida[0, 0] == 1.0 and saida[1, 0] == 0.0)
    checar("PorRobo passa no check_brain", "contrato ok" in check_brain(rede, 4, 3))


# ------------------------------------------------------- runner
def test_seed_torna_justo():
    print("\nmesma seed, mesmo mundo")
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=4)
    a = runner.run(Reto(), seed=42)
    b = runner.run(Reto(), seed=42)
    checar("dois episódios com a mesma seed são idênticos",
           np.allclose(a["progress"], b["progress"]) and np.allclose(a["position"], b["position"]))


def test_on_step_pode_interromper():
    print("\non_step interrompendo")
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=2)
    vistos = []

    def parar_no_dez(world, passo):
        vistos.append(passo)
        return passo < 10

    dados = runner.run(Reto(), on_step=parar_no_dez)
    checar("parou quando pedi", dados["steps"] == 10, f"{dados['steps']} passos")
    checar("on_step viu cada passo", vistos == list(range(1, 11)))


def test_batida_encerra_por_padrao():
    print("\nbater encerra o episódio")
    cfg = SimConfig()
    runner = EpisodeRunner(zigue_zague(), cfg, n_robots=2)
    dados = runner.run(Reto(), seed=3)
    checar("padrão é encerrar ao bater", cfg.collision_ends_episode)
    checar("andar reto num corredor com curva bate", dados["collided"].all(),
           f"{int(dados['collided'].sum())}/2")
    checar("episódio parou antes do teto", dados["steps"] < cfg.max_steps,
           f"{dados['steps']} de {cfg.max_steps}")

    cfg2 = SimConfig()
    cfg2.collision_ends_episode = False
    # os cortes por falta de progresso desligados: este teste é sobre colisão,
    # e com eles ligados o robô encostado na parede sairia por não avançar —
    # que é o comportamento certo deles, mas mediria outra coisa aqui
    cfg2.parado_limite_s = 0.0
    cfg2.corte_progresso_em = 0.0
    r2 = EpisodeRunner(zigue_zague(), cfg2, n_robots=2)
    d2 = r2.run(Reto(), seed=3)
    checar("desligado, ele fica vivo acumulando tempo", d2["steps"] == cfg2.max_steps,
           f"{d2['steps']} passos, batidas {int(d2['bumps'].max())}")

    # e com os cortes ligados (o padrão do treino), ele sai bem antes
    cfg3 = SimConfig()
    cfg3.collision_ends_episode = False
    r3 = EpisodeRunner(zigue_zague(), cfg3, n_robots=2)
    d3 = r3.run(Reto(), seed=3)
    checar("mas o corte por falta de progresso tira ele mesmo assim",
           d3["steps"] < cfg3.max_steps and bool(d3["desistiu"].any()),
           f"{d3['steps']} passos, {int(d3['desistiu'].sum())} desistiram")


def test_run_many():
    print("\nvários cenários")
    runner = EpisodeRunner(zigue_zague(), SimConfig(), n_robots=3)
    saidas = runner.run_many(Reto(), [1, 2, 3])
    checar("um resultado por cenário", len(saidas) == 3)
    checar("resumo não quebra", "robôs" in resumo(saidas[0]), resumo(saidas[0])[:60])


def test_sem_linha_central():
    print("\npista sem traçado")
    track = Track(extra_walls=[[-5, -5, -4, -5]], start=[0, 0, 0], goal=[3, 0, 0.2])
    runner = EpisodeRunner(track, SimConfig(), n_robots=2)
    dados = runner.run(Reto(), max_steps=30)
    checar("progresso é zero, sem traçado para medir", np.all(dados["progress"] == 0))
    checar("mas distance_to_goal continua servindo", np.all(dados["distance_to_goal"] > 0),
           f"{np.round(dados['distance_to_goal'], 2)}")


# ------------------------------------------------------- anti-enrolação
class Girando:
    """Acelera e vira ao mesmo tempo: roda em círculo sem sair do lugar."""
    def act_batch(self, obs):
        s = np.zeros((len(obs), 4))
        s[:, 0] = 1.0
        s[:, 2] = 1.0
        return s


def test_marcos_fotografam_o_meio_do_episodio():
    print("\nmarcos: progresso no meio do episódio")
    cfg = SimConfig()
    cfg.parado_limite_s = 0.0          # isolando o mecanismo dos marcos
    cfg.corte_progresso_em = 0.0
    runner = EpisodeRunner(zigue_zague(), cfg, n_robots=3, marcos=[0.25, 0.5])
    dados = runner.run(Reto(), seed=1)

    marcos = dados["progress_marcos"]
    checar("os dois marcos foram registrados", set(marcos) == {0.25, 0.5}, f"{sorted(marcos)}")
    checar("um valor por robô", len(marcos[0.25]) == 3)
    checar("o marco mais tarde tem progresso >= o mais cedo",
           bool((marcos[0.5] >= marcos[0.25] - 1e-9).all()),
           f"{marcos[0.25][0]:.2f} -> {marcos[0.5][0]:.2f} m")
    checar("o progresso final é >= o do último marco",
           bool((dados["progress"] >= marcos[0.5] - 1e-9).all()))
    checar("alive_marcos veio junto", set(dados["alive_marcos"]) == {0.25, 0.5})


def test_marco_depois_do_fim_usa_o_valor_final():
    print("\nmarco além do fim do episódio")
    cfg = SimConfig()
    cfg.parado_limite_s = 0.0
    cfg.corte_progresso_em = 0.0
    runner = EpisodeRunner(zigue_zague(), cfg, n_robots=2, marcos=[0.9])
    # todos batem cedo: o episódio acaba muito antes de 90% dos passos
    dados = runner.run(Reto(), seed=3)
    checar("o episódio acabou antes do marco", dados["steps"] < 0.9 * cfg.max_steps,
           f"{dados['steps']} passos de {cfg.max_steps}")
    checar("o marco caiu para o progresso final, em vez de faltar",
           np.allclose(dados["progress_marcos"][0.9], dados["progress"]))


def test_punicao_por_enrolar():
    print("\npunição por enrolar")
    from brains.exemplo import MARCOS_PADRAO, punicao_por_enrolar

    cfg = SimConfig()
    cfg.parado_limite_s = 0.0          # deixa o girador vivo, para ser punido
    cfg.corte_progresso_em = 0.0
    track = zigue_zague()
    runner = EpisodeRunner(track, cfg, n_robots=2, marcos=MARCOS_PADRAO)

    d_girando = runner.run(Girando(), seed=1)
    pun_girando = punicao_por_enrolar(d_girando)
    checar("quem gira em círculo é punido", float(pun_girando[0]) > 1.0,
           f"punição {pun_girando[0]:.1f}, avanço "
           f"{d_girando['progress_marcos'][0.4][0]:.2f} m")

    # quem já passou da meta não leva nada
    d_falso = dict(d_girando)
    d_falso["progress_marcos"] = {0.4: np.full(2, track.length)}
    checar("quem cumpriu a meta não é punido",
           float(punicao_por_enrolar(d_falso).max()) == 0.0)

    # proporcional: metade do caminho até a meta = metade da punição
    meta = 0.2 * track.length
    d_meio = dict(d_girando)
    d_meio["progress_marcos"] = {0.4: np.full(2, meta / 2)}
    d_meio["alive_marcos"] = {0.4: np.ones(2, dtype=bool)}
    pun_meio = punicao_por_enrolar(d_meio, peso=40.0)
    checar("punição é proporcional à distância da meta",
           abs(float(pun_meio[0]) - 20.0) < 1e-6, f"{pun_meio[0]:.1f} (esperado 20)")

    d_zero = dict(d_meio)
    d_zero["progress_marcos"] = {0.4: np.zeros(2)}
    checar("corte seco dá tudo ou nada",
           float(punicao_por_enrolar(d_zero, peso=40.0, proporcional=False)[0]) == 40.0)

    checar("sem o marco, o erro explica o que fazer",
           _erro_de(lambda: punicao_por_enrolar({"progress_marcos": {},
                                                 "track_length": 10.0})).count("marcos") > 0)


def _erro_de(fn):
    try:
        fn()
        return ""
    except Exception as e:
        return str(e)


def test_quem_anda_mais_pontua_mais():
    print("\nquem anda mais tem que ser o melhor")
    from brains.exemplo import (MARCOS_PADRAO, fitness_com_antienrolação,
                                punicao_por_enrolar)

    class Varios:
        """robô 0 anda reto; 1 gira em círculo; 2 fica parado."""
        def act_batch(self, obs):
            s = np.zeros((len(obs), 4))
            s[0] = [1, 0, 0, 0]
            s[1] = [1, 0, 1, 0]
            s[2] = [0, 0, 0, 0]
            return s

    cfg = SimConfig()          # cortes ligados, como no treino
    runner = EpisodeRunner(carregar_curva_u(), cfg, n_robots=3, marcos=MARCOS_PADRAO)
    d = runner.run(Varios(), seed=1)
    f = fitness_com_antienrolação(d)

    checar("o que anda reto avança bem mais",
           d["progress"][0] > 5 * d["progress"][1],
           f'{d["progress"][0]:.2f} m contra {d["progress"][1]:.2f} m')
    checar("e é ele o melhor da turma", int(np.argmax(f)) == 0,
           f"fitness: reto {f[0]:.1f}, gira {f[1]:.1f}, parado {f[2]:.1f}")

    # o bug que motivou este teste: o corte por inatividade matava o enrolador
    # antes do marco, e a isenção "só vivos" o livrava da punição inteira
    pun = punicao_por_enrolar(d)
    checar("quem desistiu por não progredir É punido, não isento",
           float(pun[1]) > 1.0 and float(pun[2]) > 1.0,
           f"gira {pun[1]:.1f}, parado {pun[2]:.1f}")
    checar("mas quem bateu continua isento (já pagou pela batida)",
           float(pun[0]) == 0.0 and bool(d["collided"][0]))


def test_tempo_nao_pune_quem_ainda_nao_chegou():
    print("\ntempo só conta para quem chega")
    from brains.exemplo import fitness_exemplo

    n = 2
    base = {
        "progress": np.array([4.0, 4.0]),
        "collided": np.zeros(n, dtype=bool),
        "finished": np.zeros(n, dtype=bool),
        "time": np.array([5.0, 90.0]),        # um rápido, um lento
        "track_length": 8.0,
    }
    f = fitness_exemplo(base)
    checar("sem chegar, demorar não tira ponto", abs(f[0] - f[1]) < 1e-9,
           f"{f[0]:.1f} contra {f[1]:.1f}")

    chegou = dict(base, finished=np.ones(n, dtype=bool))
    g = fitness_exemplo(chegou)
    checar("tendo chegado, quem foi mais rápido pontua mais", g[0] > g[1],
           f"{g[0]:.1f} contra {g[1]:.1f}")
    checar("e chegar vale mais que só avançar", g[0] > f[0])


def carregar_curva_u():
    from robo.pistas import curva_u
    return curva_u()


def test_eliminar_parados_encurta_o_episodio():
    print("\neliminar quem parou de avançar")
    track = zigue_zague()

    def rodar(limite):
        cfg = SimConfig()
        cfg.corte_progresso_em = 0.0
        cfg.parado_limite_s = limite
        r = EpisodeRunner(track, cfg, n_robots=4)
        return r.run(Girando(), seed=1)

    solto = rodar(0.0)
    cortado = rodar(2.0)

    checar("sem o corte, quem gira sobrevive até o fim",
           solto["steps"] == SimConfig().max_steps, f"{solto['steps']} passos")
    checar("com o corte, o episódio acaba muito antes",
           cortado["steps"] < solto["steps"] / 2,
           f"{cortado['steps']} contra {solto['steps']} passos")
    checar("e eles ficam marcados como desistiram", bool(cortado["desistiu"].all()),
           f"{int(cortado['desistiu'].sum())}/4")
    checar("desistir é diferente de bater",
           not bool(cortado["collided"].any()), "nenhum bateu, como esperado")


def test_quem_avanca_nunca_e_cortado():
    print("\nquem avança não é afetado pelo corte")
    cfg = SimConfig()
    cfg.corte_progresso_em = 0.0
    cfg.parado_limite_s = 1.0        # bem agressivo de propósito
    # corredor reto e longo: dá para andar sem bater
    track = Track.from_centerline([(0.5, 0.5), (12.0, 0.5)], width=1.2)
    runner = EpisodeRunner(track, cfg, n_robots=2)
    dados = runner.run(Reto(), seed=1)
    checar("andando reto, ninguém desiste", not bool(dados["desistiu"].any()),
           f"{int(dados['desistiu'].sum())} desistiram")
    checar("e o progresso é real", float(dados["progress"].max()) > 2.0,
           f"{dados['progress'].max():.2f} m")


def main():
    for fn in [test_progresso_ao_longo_do_tracado, test_telemetria_tem_progresso,
               test_check_brain_aceita_o_certo, test_check_brain_pega_erros,
               test_inspect_inconsistente_e_apontado, test_botoes_e_adaptador,
               test_seed_torna_justo, test_on_step_pode_interromper,
               test_batida_encerra_por_padrao, test_run_many, test_sem_linha_central,
               test_marcos_fotografam_o_meio_do_episodio,
               test_marco_depois_do_fim_usa_o_valor_final, test_punicao_por_enrolar,
               test_quem_anda_mais_pontua_mais, test_tempo_nao_pune_quem_ainda_nao_chegou,
               test_eliminar_parados_encurta_o_episodio, test_quem_avanca_nunca_e_cortado]:
        fn()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
