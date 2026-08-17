"""Verificações do painel de calibragem.

O que importa: mexer numa barra tem que mudar o robô de verdade, gravar tem que
sobreviver a reabrir, e mudar a montagem dos sensores não pode teleportar o robô.

    python -m tests.test_calibra
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import sys
import tempfile

import numpy as np
import pygame

from robo.calibra import PainelCalibragem
from robo.config import SimConfig
from robo.pistas import zigue_zague
from robo.world import World

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  {'OK  ' if ok else 'FALHA'}  {nome}{'  ' + detalhe if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def novo_painel(cfg=None):
    cfg = cfg or SimConfig()
    p = PainelCalibragem(cfg, (1180, 760))
    p.aberto = True
    return cfg, p


def barra(painel, campo):
    return next(b for b in painel.barras if b.campo == campo)


def arrastar_para(painel, campo, fracao):
    """Simula o arrasto até uma fração da trilha, como o mouse faria."""
    b = barra(painel, campo)
    y = next(yy for bb, yy in painel._linhas_com_posicao() if bb is b)
    x = painel.rect.x + 16 + int((painel.rect.w - 32) * fracao)
    painel.evento(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y + 20), button=1))
    painel.evento(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y + 20), button=1))
    return b


# ---------------------------------------------------------------------- #
def test_arrastar_muda_o_robo():
    print("\nbarra muda o robô de verdade")
    cfg, painel = novo_painel()
    world = World(zigue_zague(), cfg, 1)

    b = arrastar_para(painel, "max_wheel_speed", 1.0)
    checar("velocidade foi ao máximo", abs(cfg.robot.max_wheel_speed - b.max) < 1e-9,
           f"{cfg.robot.max_wheel_speed:.2f} m/s")

    for _ in range(40):
        world.step(np.array([[1.0, 0, 0, 0]]))
    rapido = float(world.v[0])

    arrastar_para(painel, "max_wheel_speed", 0.0)
    world.reset()
    for _ in range(40):
        world.step(np.array([[1.0, 0, 0, 0]]))
    devagar = float(world.v[0])

    checar("o robô realmente anda mais devagar", devagar < rapido / 3,
           f"{rapido:.2f} -> {devagar:.2f} m/s")


def test_zona_morta_pela_barra():
    print("\nzona morta pela barra")

    def girar(deadzone_frac, gain_frac):
        cfg, painel = novo_painel()
        arrastar_para(painel, "pwm_deadzone", deadzone_frac)
        arrastar_para(painel, "turn_gain", gain_frac)
        world = World(zigue_zague(), cfg, 1)
        for _ in range(30):
            world.step(np.array([[0, 0, 1.0, 0]]))   # só "esquerda"
        return cfg, float(world.omega[0])

    cfg_livre, solto = girar(0.0, 1.0)
    checar("sem zona morta, virar gira", abs(solto) > 1.0, f"{solto:.2f} rad/s")

    # virar sozinho gera PWM = turn_gain; com a força da curva baixa e a zona
    # morta alta, esse PWM não vence o atrito estático e o robô fica parado
    cfg_travado, travado = girar(1.0, 0.0)
    pwm = cfg_travado.robot.turn_gain
    checar("o caso montado é mesmo PWM abaixo da zona morta",
           pwm < cfg_travado.robot.pwm_deadzone,
           f"PWM {pwm:.2f} < zona morta {cfg_travado.robot.pwm_deadzone:.2f}")
    checar("e aí o robô não sai do lugar", abs(travado) < 1e-9, f"{travado:.3f} rad/s")


def test_limites_respeitados():
    print("\nlimites das barras")
    cfg, painel = novo_painel()
    for campo, fracao in [("motor_bias", 0.0), ("motor_bias", 1.0),
                          ("count", 1.0), ("cone_deg", 0.0)]:
        b = arrastar_para(painel, campo, fracao)
        v = b.valor
        checar(f"{campo} em {'máx' if fracao else 'mín'}",
               b.min - 1e-9 <= v <= b.max + 1e-9, f"{v}")

    b = barra(painel, "count")
    checar("quantidade de sensores é inteira", isinstance(b.valor, int), f"{b.valor!r}")


def test_remontagem_preserva_a_pose():
    print("\nmudar sensores não teleporta o robô")
    cfg, painel = novo_painel()
    world = World(zigue_zague(), cfg, 1)
    for _ in range(30):
        world.step(np.array([[1.0, 0, 0, 0]]))
    pos, theta = world.pos.copy(), world.theta.copy()
    antes = world.sensors.n

    arrastar_para(painel, "count", 1.0)
    checar("o painel avisa que precisa remontar", painel.consumir_remontagem())
    world.reconfigure_sensors()

    checar("quantidade de sensores mudou", world.sensors.n != antes,
           f"{antes} -> {world.sensors.n}")
    checar("observação tem o novo tamanho", world.observation().shape[1] == world.sensors.n)
    checar("a pose foi preservada",
           np.allclose(world.pos, pos) and np.allclose(world.theta, theta))
    checar("o aviso não fica pendurado", not painel.consumir_remontagem())

    world.step(np.array([[1.0, 0, 0, 0]]))
    checar("continua rodando depois de remontar", np.isfinite(world.observation()).all())


def test_gravar_e_reabrir():
    print("\ngravar e reabrir")
    caminho = os.path.join(tempfile.gettempdir(), "config_teste.json")
    cfg, painel = novo_painel()
    arrastar_para(painel, "max_wheel_speed", 0.8)
    arrastar_para(painel, "count", 0.9)
    esperado_v = cfg.robot.max_wheel_speed
    esperado_n = cfg.sensor.count

    cfg.save(caminho)
    lido = SimConfig.load(caminho)
    os.remove(caminho)

    checar("velocidade sobreviveu", abs(lido.robot.max_wheel_speed - esperado_v) < 1e-9,
           f"{lido.robot.max_wheel_speed:.3f} m/s")
    checar("quantidade de sensores sobreviveu", lido.sensor.count == esperado_n,
           f"{lido.sensor.count}")
    checar("n_sensors bate com a montagem", lido.n_sensors == esperado_n)


def test_sem_arquivo_usa_padrao():
    print("\nsem config.json")
    cfg = SimConfig.carregar_ou_padrao(os.path.join(tempfile.gettempdir(), "nao_existe.json"))
    checar("cai nos valores de fábrica", cfg.robot.max_wheel_speed == SimConfig().robot.max_wheel_speed)

    ruim = os.path.join(tempfile.gettempdir(), "config_ruim.json")
    with open(ruim, "w") as f:
        f.write("{isto nao e json")
    cfg = SimConfig.carregar_ou_padrao(ruim)
    os.remove(ruim)
    checar("arquivo corrompido também não derruba", cfg.robot.pwm_deadzone == 0.15)


def test_restaurar():
    print("\nrestaurar de fábrica")
    cfg, painel = novo_painel()
    arrastar_para(painel, "max_wheel_speed", 1.0)
    arrastar_para(painel, "cone_deg", 1.0)
    painel.consumir_remontagem()

    painel.restaurar()
    padrao = SimConfig()
    checar("motores voltaram", cfg.robot.max_wheel_speed == padrao.robot.max_wheel_speed)
    checar("sensores voltaram", cfg.sensor.cone_deg == padrao.sensor.cone_deg)
    checar("e pede remontagem", painel.consumir_remontagem())


def test_painel_fechado_ignora():
    print("\npainel fechado não rouba eventos")
    cfg, painel = novo_painel()
    painel.aberto = False
    ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)
    checar("tecla s passa direto para o jogo", not painel.evento(ev))

    painel.aberto = True
    checar("aberto, o s é do painel", painel.evento(ev))
    checar("clique fora do painel passa direto",
           not painel.evento(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                                pos=(30, 400), button=1)))


def test_desenha():
    print("\ndesenhar o painel")
    cfg, painel = novo_painel()
    tela = pygame.Surface((1180, 760))
    tela.fill((0, 0, 0))
    try:
        painel.desenhar(tela)
        pintou = pygame.surfarray.array3d(tela).sum() > 0
        ok, detalhe = pintou, "pintou" if pintou else "não pintou nada"
    except Exception as e:
        ok, detalhe = False, f"{type(e).__name__}: {e}"
    checar("desenha sem estourar", ok, detalhe)


def main():
    pygame.init()
    for fn in [test_arrastar_muda_o_robo, test_zona_morta_pela_barra,
               test_limites_respeitados, test_remontagem_preserva_a_pose,
               test_gravar_e_reabrir, test_sem_arquivo_usa_padrao, test_restaurar,
               test_painel_fechado_ignora, test_desenha]:
        fn()
    pygame.quit()

    print()
    if falhas:
        print(f"{len(falhas)} FALHA(S): {', '.join(falhas)}")
        return 1
    print("tudo passou")
    return 0


if __name__ == "__main__":
    sys.exit(main())
