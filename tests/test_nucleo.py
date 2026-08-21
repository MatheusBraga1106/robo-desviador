"""Verificações do núcleo: geometria, sensores e física.

    python -m tests.test_nucleo
"""

import sys
import time

import numpy as np

from robo.config import SimConfig, SensorConfig, RobotConfig
from robo.geometry import ray_segment_hits, point_segment_distance, segments_cross
from robo.physics import DifferentialDrive
from robo.pistas import zigue_zague, diagonais
from robo.sensors import UltrasonicArray
from robo.track import Track
from robo.world import World

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


# ---------------------------------------------------------------- geometria
def test_raycast_contra_forca_bruta():
    print("\nraycast vetorizado x força bruta")
    track = zigue_zague()
    rng = np.random.default_rng(3)

    n, alcance = 20000, 3.0
    passo = alcance / n

    def bruto(o, d):
        """Caminha pelo raio e acha o primeiro trecho que *cruza* uma parede.

        Testar proximidade em vez de cruzamento daria erro proporcional a
        1/sen(ângulo) — enorme em raios rasantes — e mediria o teste, não o código.
        """
        ts = np.linspace(0.0, alcance, n + 1)
        pts = o[None, :] + d[None, :] * ts[:, None]
        cruza = segments_cross(pts[:-1, None, :], pts[1:, None, :],
                               track.seg_a[None, :, :], track.seg_b[None, :, :])
        atingidos = np.flatnonzero(cruza.any(axis=1))
        return ts[atingidos[0] + 1] if len(atingidos) else np.inf

    erro = 0.0
    comparados = 0
    for _ in range(120):
        o = rng.uniform([0.6, 0.6], [4.4, 3.4])
        if track.distance_to_walls(o[None, :])[0] < 0.15:
            continue
        a = rng.uniform(-np.pi, np.pi)
        d = np.array([np.cos(a), np.sin(a)])

        t, _ = ray_segment_hits(o[None, :], d[None, :], track.seg_a, track.seg_b)
        analitico = t.min()
        if not np.isfinite(analitico) or analitico > 3.0:
            continue
        erro = max(erro, abs(analitico - bruto(o, d)))
        comparados += 1

    tol = 2 * passo
    checar("bate com a força bruta", erro < tol,
           f"erro {erro:.2e} < tolerância {tol:.2e} em {comparados} raios")


def test_distancia_ponto_segmento():
    print("\ndistância ponto-segmento")
    a = np.array([[0.0, 0.0]])
    b = np.array([[2.0, 0.0]])
    d = point_segment_distance(np.array([[1.0, 0.5], [-1.0, 0.0], [3.0, 0.0]]), a, b)
    checar("perpendicular no meio", abs(d[0, 0] - 0.5) < 1e-9)
    checar("além da ponta usa a ponta", abs(d[1, 0] - 1.0) < 1e-9 and abs(d[2, 0] - 1.0) < 1e-9)


def test_cruzamento_de_segmentos():
    print("\ncruzamento de segmentos (checkpoints)")
    p1 = np.array([[0.0, 0.0]]); p2 = np.array([[2.0, 2.0]])
    q1 = np.array([[0.0, 2.0]]); q2 = np.array([[2.0, 0.0]])
    checar("em X cruzam", bool(segments_cross(p1, p2, q1, q2)[0]))
    checar("paralelos não cruzam",
           not bool(segments_cross(p1, p2, q1 + np.array([[5.0, 0.0]]),
                                   q2 + np.array([[5.0, 0.0]]))[0]))


# ---------------------------------------------------------------- sensores
def _pista_parede_frontal(distancia=1.0, angulo_graus=0.0):
    """Robô na origem olhando para +x, com uma parede à frente."""
    ang = np.deg2rad(angulo_graus)
    # parede centrada em (distancia, 0), girada `angulo` em torno desse ponto
    meio = np.array([distancia, 0.0])
    d = np.array([-np.sin(ang), np.cos(ang)]) * 3.0
    parede = [[*(meio - d), *(meio + d)]]
    return Track(extra_walls=parede, start=[0.0, 0.0, 0.0], goal=[distancia, 0.0, 0.05])


def test_leitura_conhecida():
    print("\nleitura de distância conhecida")
    cfg = SimConfig()
    cfg.sensor = SensorConfig(count=1, angles_deg=[0.0], cone_deg=30.0, rays_per_sensor=5,
                              max_range=2.0, round_robin=False)
    world = World(_pista_parede_frontal(1.0), cfg, 1)
    world.step(np.zeros((1, 4)))

    esperado = 1.0 - cfg.robot.radius   # o sensor fica na borda do corpo
    lido = float(world.sensors.last[0, 0])
    checar("parede reta a 1 m", abs(lido - esperado) < 0.01,
           f"leu {lido:.3f} m, esperado {esperado:.3f} m (sensor na borda)")

    obs = world.observation()
    checar("observação normalizada", abs(float(obs[0, 0]) - lido / 2.0) < 1e-6,
           f"obs {float(obs[0,0]):.3f}")


def test_eco_perdido_em_parede_inclinada():
    print("\neco perdido em parede inclinada")
    cfg = SimConfig()
    cfg.sensor = SensorConfig(count=1, angles_deg=[0.0], cone_deg=2.0, rays_per_sensor=1,
                              max_range=2.0, round_robin=False, max_incidence_deg=65.0)

    w = World(_pista_parede_frontal(1.0, angulo_graus=0.0), cfg, 1)
    w.step(np.zeros((1, 4)))
    reta = float(w.sensors.last[0, 0])

    w = World(_pista_parede_frontal(1.0, angulo_graus=75.0), cfg, 1)
    w.step(np.zeros((1, 4)))
    inclinada = float(w.sensors.last[0, 0])

    checar("parede de frente devolve eco", reta < 1.9, f"{reta:.2f} m")
    checar("parede a 75° não devolve eco", inclinada >= 1.999,
           f"leu {inclinada:.2f} m = 'livre', que é a falha real do ultrassom")


def test_cone_enxerga_mais_que_um_raio():
    print("\ncone x raio fino")
    # obstáculo deslocado lateralmente: fora do raio central, dentro do cone
    parede = [[0.55, 0.10, 0.55, 0.60]]
    track = Track(extra_walls=parede, start=[0, 0, 0], goal=[2, 0, 0.1])

    def ler(cone, k):
        cfg = SimConfig()
        cfg.sensor = SensorConfig(count=1, angles_deg=[0.0], cone_deg=cone,
                                  rays_per_sensor=k, max_range=2.0, round_robin=False)
        w = World(track, cfg, 1)
        w.step(np.zeros((1, 4)))
        return float(w.sensors.last[0, 0])

    fino, largo = ler(0.1, 1), ler(40.0, 9)
    checar("raio fino não vê o obstáculo lateral", fino >= 1.999, f"{fino:.2f} m")
    checar("cone de 40° vê", largo < 1.5, f"{largo:.2f} m")


def test_rodizio_e_latencia():
    print("\nrodízio e latência dos sensores")
    cfg = SimConfig()
    cfg.dt = 0.05
    cfg.sensor = SensorConfig(count=4, reading_time=0.06, round_robin=True, max_range=2.0)
    world = World(diagonais(), cfg, 1)
    world.sensors.reset()

    atualizados = []
    for _ in range(20):
        antes = world.sensors.age.copy()
        world.step(np.zeros((1, 4)))
        atualizados.append(int(np.sum(world.sensors.age[0] < antes[0])))

    checar("no máximo um sensor por passo", max(atualizados) <= 1,
           f"máximo observado {max(atualizados)}")

    idade_maxima = float(world.sensors.age.max())
    esperado = cfg.sensor.count * cfg.sensor.reading_time
    checar("varredura completa custa ~n x tempo de leitura",
           idade_maxima <= esperado + cfg.dt + 1e-9,
           f"idade máxima {idade_maxima*1000:.0f} ms, varredura {esperado*1000:.0f} ms")


# ---------------------------------------------------------------- física
def test_zona_morta():
    print("\nzona morta do motor")
    cfg = RobotConfig(pwm_deadzone=0.3, turn_gain=0.7)
    d = DifferentialDrive(cfg, 1)
    baixo = d._pwm_para_velocidade(np.array([[0.2, 0.2]]))
    alto = d._pwm_para_velocidade(np.array([[0.9, 0.9]]))
    checar("PWM abaixo da zona morta não move", np.allclose(baixo, 0.0))
    checar("PWM acima move", alto[0, 0] > 0.1, f"{alto[0,0]:.3f} m/s")


def test_inercia():
    print("\ninércia do motor")
    cfg = RobotConfig(accel_time=0.25, pwm_deadzone=0.0)
    d = DifferentialDrive(cfg, 1)
    frente = np.array([[1.0, 0.0, 0.0, 0.0]])

    v1, _ = d.step(frente, 0.05)
    checar("não pula direto para a velocidade final",
           v1[0] < cfg.max_wheel_speed * 0.5, f"após 1 passo: {v1[0]:.3f} m/s")

    for _ in range(int(cfg.accel_time / 0.05)):
        v, _ = d.step(frente, 0.05)
    checar("chega perto do alvo em accel_time",
           v[0] > cfg.max_wheel_speed * 0.9, f"após {cfg.accel_time}s: {v[0]:.3f} m/s")


def test_assimetria_dos_motores():
    print("\nassimetria dos motores")
    track = Track(extra_walls=[[-50, -50, -49, -50]], start=[0, 0, 0], goal=[99, 99, 0.1])

    def desvio(bias):
        cfg = SimConfig()
        cfg.robot = RobotConfig(motor_bias=bias, pwm_deadzone=0.0)
        w = World(track, cfg, 1)
        for _ in range(60):
            w.step(np.array([[1.0, 0.0, 0.0, 0.0]]))
        return float(w.pos[0, 1])

    reto, torto = desvio(0.0), desvio(0.06)
    checar("sem assimetria anda reto", abs(reto) < 1e-6, f"desvio {reto:.4f} m")
    checar("com assimetria desvia sozinho", abs(torto) > 0.02,
           f"desvio {torto:+.3f} m em 3 s")


def test_giro_no_lugar():
    print("\ngiro no próprio eixo")
    cfg = SimConfig()
    cfg.robot = RobotConfig(pwm_deadzone=0.0, turn_gain=1.0)
    track = Track(extra_walls=[[-50, -50, -49, -50]], start=[0, 0, 0], goal=[99, 99, 0.1])
    w = World(track, cfg, 1)
    # poucos passos de propósito: a 5 rad/s o ângulo daria a volta e o sinal
    # medido passaria a ser o do wrap, não o do giro
    for _ in range(4):
        w.step(np.array([[0.0, 0.0, 0.0, 1.0]]))    # só "direita"
    checar("virar sozinho não desloca", float(np.hypot(*w.pos[0])) < 1e-6,
           f"deslocou {float(np.hypot(*w.pos[0])):.5f} m")
    checar("virar direita gira no horário", w.omega[0] < 0 and -np.pi < w.theta[0] < -0.1,
           f"{np.rad2deg(w.omega[0]):.0f} °/s, ângulo {np.rad2deg(w.theta[0]):.1f}°")


# ---------------------------------------------------------------- mundo
def test_nao_atravessa_parede():
    print("\ncolisão")
    cfg = SimConfig()
    cfg.collision_ends_episode = False
    w = World(zigue_zague(), cfg, 1)
    for _ in range(400):
        w.step(np.array([[1.0, 0.0, 0.0, 0.0]]))     # reto até bater
    dist = w.track.distance_to_walls(w.pos)[0]
    checar("robô fica fora da parede", dist >= cfg.robot.radius - 1e-9,
           f"distância até a parede {dist:.3f} m >= raio {cfg.robot.radius}")
    checar("a batida foi registrada", w.bumps[0] > 0, f"{w.bumps[0]} passos em contato")


def test_checkpoints_em_ordem():
    print("\ncheckpoints")
    cfg = SimConfig()
    cfg.collision_ends_episode = False
    track = zigue_zague()
    w = World(track, cfg, 1)
    for _ in range(120):
        w.step(np.array([[1.0, 0.0, 0.0, 0.0]]))
    checar("avança na ordem", 0 < w.checkpoint[0] <= len(track.checkpoints),
           f"{w.checkpoint[0]} de {len(track.checkpoints)}")


def test_telemetria_sem_juizo_de_valor():
    print("\ncontrato da telemetria")
    w = World(zigue_zague(), SimConfig(), 4)
    w.step(np.zeros((4, 4)))
    t = w.telemetry()
    esperadas = {"alive", "collided", "finished", "bumps", "distance", "checkpoints",
                 "steps_alive", "time", "distance_to_goal", "position", "readings"}
    checar("todas as chaves presentes", esperadas <= set(t), f"{sorted(set(t))}")
    checar("nada de fitness embutido", not any("fit" in k or "reward" in k for k in t))


def test_desempenho_populacao():
    print("\ndesempenho com população")
    cfg = SimConfig()
    w = World(zigue_zague(), cfg, 100)
    botoes = np.tile([1.0, 0.0, 0.0, 0.0], (100, 1))
    w.step(botoes)

    t0 = time.perf_counter()
    n = 300
    for _ in range(n):
        w.step(botoes)
    por_passo = (time.perf_counter() - t0) / n
    fps = 1.0 / por_passo
    checar("100 robôs rodam acima de 60 passos/s", fps > 60,
           f"{fps:.0f} passos/s ({por_passo*1000:.2f} ms por passo)")


# ---------------------------------------------------------------- pista
def test_caixa_vira_quatro_paredes():
    print("\nobstáculo em caixa")
    reta = [(0.5, 0.5), (4.0, 0.5)]
    limpa = Track.from_centerline(reta, width=0.9)
    com_caixa = Track.from_centerline(reta, width=0.9,
                                      obstacles=[[2.0, 0.5, 0.3, 0.3, 0.0]])

    checar("caixa acrescenta 4 segmentos",
           len(com_caixa.walls) == len(limpa.walls) + 4,
           f"{len(limpa.walls)} -> {len(com_caixa.walls)}")

    # o robô deve parar antes da caixa, não atravessá-la
    cfg = SimConfig()
    cfg.collision_ends_episode = False
    w = World(com_caixa, cfg, 1)
    for _ in range(200):
        w.step(np.array([[1.0, 0, 0, 0]]))
    checar("robô é barrado pela caixa", w.pos[0, 0] < 2.0,
           f"parou em x={w.pos[0,0]:.2f}, caixa em x=2.00")


def test_caixa_retangular_girada():
    print("\ncaixa retangular girada")
    larg, comp = 0.6, 0.2
    t = Track.from_centerline([(0, 0), (4, 0)], width=2.0,
                              obstacles=[[2.0, 0.0, larg, comp, np.pi / 4]])
    caixa = t.walls[-4:]
    lados = sorted(np.hypot(s[2] - s[0], s[3] - s[1]) for s in caixa)
    checar("dois lados de cada medida",
           np.allclose(lados, [comp, comp, larg, larg]), f"lados {np.round(lados,3)}")

    dist = np.hypot(caixa[:, 0] - 2.0, caixa[:, 1])
    esperado = np.hypot(larg / 2, comp / 2)
    checar("cantos na meia-diagonal", np.allclose(dist, esperado),
           f"{np.round(dist,3)}, esperado {esperado:.3f}")


def test_teste_de_contencao_e_exato():
    print("\nclique dentro da caixa comprida")
    # tábua fina e comprida deitada: o teste por raio erraria muito aqui
    t = Track.from_centerline([(0, 0), (4, 0)], width=3.0,
                              obstacles=[[2.0, 0.0, 1.2, 0.1, 0.0]])
    checar("no centro, dentro", bool(t.contains_obstacle(2.0, 0.0)[0]))
    checar("na ponta do comprimento, dentro", bool(t.contains_obstacle(2.55, 0.0)[0]))
    checar("logo além da ponta, fora", not bool(t.contains_obstacle(2.65, 0.0)[0]))
    # a meia-diagonal é 0.60; um teste por raio acharia que este ponto está dentro
    checar("na direção fina, fora mesmo perto do centro",
           not bool(t.contains_obstacle(2.0, 0.35)[0]))


def test_remocao_pega_a_menor():
    print("\nremover com caixas sobrepostas")
    t = Track.from_centerline([(0, 0), (4, 0)], width=3.0, obstacles=[
        [2.0, 0.0, 1.0, 1.0, 0.0],     # grande
        [2.0, 0.0, 0.2, 0.2, 0.0],     # pequena, por cima
    ])
    t.remove_obstacle_at(2.0, 0.0)
    checar("tira a menor primeiro", len(t.obstacles) == 1
           and np.allclose(t.obstacles[0, 2:4], [1.0, 1.0]),
           f"sobrou {np.round(t.obstacles[0], 2)}")


def test_receita_sobrevive_ao_salvar():
    print("\nsalvar guarda a receita, não só o resultado")
    import os
    import tempfile

    orig = Track.from_centerline([(0.5, 0.5), (3.0, 0.5), (3.0, 2.5)],
                                 width=0.8, obstacles=[[1.5, 0.5, 0.25, 0.4, 0.3]],
                                 name="teste")
    caminho = os.path.join(tempfile.gettempdir(), "pista_teste.json")
    orig.save(caminho)
    lida = Track.load(caminho)
    os.remove(caminho)

    checar("linha central preservada",
           lida.centerline is not None and np.allclose(lida.centerline, orig.centerline))
    checar("largura preservada", abs(lida.width - 0.8) < 1e-12, f"{lida.width}")
    checar("obstáculos preservados", np.allclose(lida.obstacles, orig.obstacles))
    checar("paredes idênticas às originais", np.allclose(lida.walls, orig.walls))

    # e o ponto do refactor: continua editável depois de reabrir
    lida.width = 1.4
    lida.rebuild()
    checar("dá para mudar a largura depois de reabrir",
           not np.allclose(lida.walls[:len(orig.walls)], orig.walls))


def test_rebuild_regenera():
    print("\nrebuild após editar")
    t = Track.from_centerline([(0, 0), (3, 0)], width=0.9)
    n_antes = len(t.walls)
    t.add_obstacle(1.5, 0.0, 0.3)
    checar("add_obstacle regera sozinho", len(t.walls) == n_antes + 4)
    checar("remove_obstacle_at acha a caixa", t.remove_obstacle_at(1.55, 0.05))
    checar("e as paredes voltam", len(t.walls) == n_antes)
    checar("remover no vazio não faz nada", not t.remove_obstacle_at(9.0, 9.0))


def test_largada_e_objetivo_manuais():
    print("\nlargada e objetivo definidos à mão")
    t = Track(centerline=[(0, 0), (3, 0)], width=0.9,
              start=[1.0, 0.2, 1.57], goal=[2.5, -0.1, 0.3])
    checar("largada respeitada", np.allclose(t.start, [1.0, 0.2, 1.57]), f"{t.start}")
    checar("objetivo respeitado", np.allclose(t.goal, [2.5, -0.1, 0.3]), f"{t.goal}")
    t.width = 1.2
    t.rebuild()
    checar("rebuild não sobrescreve os manuais", np.allclose(t.start, [1.0, 0.2, 1.57]))


# ---------------------------------------------------------------- grade espacial
def _pista_grande():
    """Pista sintética grande o bastante para ligar a grade espacial.

    Curva senoidal comprida com caixas espalhadas: passa bem de
    `GRADE_MIN_SEGMENTOS`, e a curvatura garante trechos onde paredes de
    "voltas" diferentes ficam geometricamente perto sem estar perto no
    traçado — o caso que quebraria uma filtragem ingênua por progresso ao
    longo da linha central, mas não quebra a grade por células (que olha
    posição real, não posição no traçado).
    """
    n = 220
    xs = np.linspace(0.0, n * 0.3, n)
    ys = np.sin(xs * 0.6) * 1.4
    centerline = np.stack([xs, ys], axis=1)
    obstaculos = np.array([[xs[i], ys[i] + 0.3, 0.15, 0.15, 0.3] for i in range(5, n, 7)])
    return Track.from_centerline(centerline, width=0.9, obstacles=obstaculos, name="grande")


def test_grade_liga_sozinha_pelo_tamanho_da_pista():
    print("\ngrade espacial: liga sozinha conforme o tamanho da pista")
    pequena = zigue_zague()
    checar(f"pista pequena ({len(pequena.walls)} segmentos) não usa grade",
           not pequena.usar_grade)
    grande = _pista_grande()
    checar(f"pista grande ({len(grande.walls)} segmentos) usa grade", grande.usar_grade)


def test_grade_bate_com_forca_bruta_na_distancia():
    print("\ngrade espacial: distance_to_walls bate com força bruta")
    track = _pista_grande()
    rng = np.random.default_rng(11)
    x0, y0, x1, y1 = track.bounds
    # inclui pontos fora dos limites também, para exercitar quem não acha
    # nada perto (cai no recálculo exato, ver `_distancia_com_grade`)
    pontos = rng.uniform([x0 - 1.0, y0 - 1.0], [x1 + 1.0, y1 + 1.0], (800, 2))

    com_grade = track.distance_to_walls(pontos)
    forca_bruta = point_segment_distance(pontos, track.seg_a, track.seg_b).min(axis=-1)

    erro = float(np.abs(com_grade - forca_bruta).max())
    checar("distâncias idênticas à força bruta", erro < 1e-9, f"erro máximo {erro:.2e}")


def test_grade_bate_com_forca_bruta_no_raycast():
    print("\ngrade espacial: leitura dos sensores bate com força bruta")
    track = _pista_grande()
    cfg = SimConfig()
    cfg.sensor.count = 4
    rng = np.random.default_rng(12)
    P = 60
    x0, y0, x1, y1 = track.bounds
    pos = rng.uniform([x0, y0], [x1, y1], (P, 2))
    theta = rng.uniform(-np.pi, np.pi, P)
    dt_dispara_todos = cfg.sensor.reading_time * cfg.sensor.count

    assert track.usar_grade
    com_grade_arr = UltrasonicArray(cfg.sensor, P, cfg.robot.radius, np.random.default_rng(0))
    com_grade_arr.update(pos, theta, track, dt_dispara_todos)
    com_grade = com_grade_arr.last.copy()

    track.usar_grade = False
    try:
        sem_grade_arr = UltrasonicArray(cfg.sensor, P, cfg.robot.radius, np.random.default_rng(0))
        sem_grade_arr.update(pos, theta, track, dt_dispara_todos)
        sem_grade = sem_grade_arr.last.copy()
    finally:
        track.usar_grade = True     # não vaza estado para o próximo teste

    erro = float(np.abs(com_grade - sem_grade).max())
    checar("leituras idênticas à força bruta", erro < 1e-9, f"erro máximo {erro:.2e}")


def main():
    for fn in [
        test_raycast_contra_forca_bruta, test_distancia_ponto_segmento,
        test_cruzamento_de_segmentos, test_leitura_conhecida,
        test_eco_perdido_em_parede_inclinada, test_cone_enxerga_mais_que_um_raio,
        test_rodizio_e_latencia, test_zona_morta, test_inercia,
        test_assimetria_dos_motores, test_giro_no_lugar, test_nao_atravessa_parede,
        test_checkpoints_em_ordem, test_telemetria_sem_juizo_de_valor,
        test_caixa_vira_quatro_paredes, test_caixa_retangular_girada,
        test_teste_de_contencao_e_exato, test_remocao_pega_a_menor,
        test_receita_sobrevive_ao_salvar, test_rebuild_regenera,
        test_largada_e_objetivo_manuais,
        test_grade_liga_sozinha_pelo_tamanho_da_pista,
        test_grade_bate_com_forca_bruta_na_distancia,
        test_grade_bate_com_forca_bruta_no_raycast,
        test_desempenho_populacao,
    ]:
        fn()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
