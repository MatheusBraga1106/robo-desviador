"""Desenho do mundo em pygame.

O mundo usa y para cima; a tela usa y para baixo. A câmera resolve isso num
lugar só, então o resto do código não precisa pensar nisso.
"""

import numpy as np
import pygame

from .track import box_corners

COR_FUNDO = (8, 10, 18)
COR_CHAO = (58, 60, 66)
COR_GRADE = (27, 32, 43)
COR_CAIXA = (150, 46, 50)
COR_CAIXA_REALCE = (206, 74, 78)
COR_CAIXA_BORDA = (232, 232, 238)
COR_PAREDE_A = (198, 40, 44)
COR_PAREDE_B = (238, 238, 242)
COR_ROBO = (60, 220, 190)
COR_ROBO_MORTO = (86, 90, 102)


def _esmaecer(cor, fator=0.4, fundo=COR_FUNDO):
    """Versão escurecida e sólida de uma cor, misturada em direção ao fundo.

    Usada para os robôs "de fundo" da população — não é transparência de
    verdade: `pygame.draw.circle` numa Surface comum (sem `SRCALPHA`) ignora o
    canal alfa e desenha opaco mesmo que você passe uma cor RGBA (testado: pedir
    alfa 120 devolve um pixel com alfa 255 na tela). Misturar a cor com o fundo
    de antemão dá o mesmo efeito visual de "apagado", sem depender de alfa que
    a superfície não suporta.
    """
    return tuple(int(c * fator + f * (1 - fator)) for c, f in zip(cor, fundo))


COR_ROBO_FUNDO = _esmaecer(COR_ROBO)
COR_ROBO_MORTO_FUNDO = _esmaecer(COR_ROBO_MORTO)
COR_RAIO = (90, 200, 255)
COR_CONE = (70, 180, 235)
COR_OBJETIVO = (80, 230, 120)
COR_CHECKPOINT = (60, 70, 95)
COR_TEXTO = (232, 236, 245)
COR_APAGADO = (140, 150, 170)
COR_ALERTA = (250, 190, 70)


class Camera:
    def __init__(self, size):
        self.w, self.h = size
        self.center = np.zeros(2)
        self.scale = 100.0        # pixels por metro

    def fit(self, bounds, margem=0.6):
        x0, y0, x1, y1 = bounds
        largura = max(x1 - x0 + 2 * margem, 1e-6)
        altura = max(y1 - y0 + 2 * margem, 1e-6)
        self.scale = min(self.w / largura, self.h / altura)
        self.center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

    def follow(self, pos, scale=None):
        self.center = np.asarray(pos, dtype=np.float64).reshape(2).copy()
        if scale is not None:
            self.scale = scale

    def to_screen(self, pts):
        p = np.asarray(pts, dtype=np.float64)
        x = (p[..., 0] - self.center[0]) * self.scale + self.w / 2.0
        y = self.h / 2.0 - (p[..., 1] - self.center[1]) * self.scale
        return np.stack([x, y], axis=-1)

    def px(self, metros) -> int:
        return max(1, int(round(metros * self.scale)))


class Renderer:
    def __init__(self, surface, fonte=None, fonte_peq=None):
        self.surf = surface
        self.font = fonte or pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.font_peq = fonte_peq or pygame.font.SysFont("consolas,menlo,monospace", 13)
        self._estrelas = None
        self._cache_cenario = None
        self._cache_chave = None

    # ------------------------------------------------------------------ #
    def desenhar_cenario_estatico(self, track, cam: Camera, mostrar_checkpoints=True):
        """`fundo()` + `desenhar_pista()`, cacheados numa Surface à parte.

        Paredes, caixas, checkpoints, chão e o objetivo não se movem sozinhos —
        só mudam se a pista for editada ou a câmera se mexer. Numa pista grande
        redesenhar tudo isso pixel a pixel a cada quadro (a zebra das paredes
        sozinha pode passar de 300 chamadas a `draw.line`) é trabalho jogado
        fora quando a câmera está parada, que é o caso comum de assistir o
        treino da população inteira. Aqui o desenho só é refeito quando a
        pista, a câmera ou o tamanho da tela mudam; do contrário é um blit só.

        Para câmera que persegue o robô (o modo "jogar"), isto invalida a cada
        quadro e não ajuda — por isso só é usado no `Viewer`, não no jogo.
        """
        chave = (id(track), track.walls.shape[0], tuple(cam.center), cam.scale,
                 self.surf.get_size(), mostrar_checkpoints)
        if self._cache_cenario is None or self._cache_chave != chave:
            cache = pygame.Surface(self.surf.get_size())
            aux = Renderer(cache, self.font, self.font_peq)
            aux._estrelas = self._estrelas
            aux.fundo()
            aux.desenhar_pista(track, cam, mostrar_checkpoints)
            self._estrelas = aux._estrelas
            self._cache_cenario = cache
            self._cache_chave = chave
        self.surf.blit(self._cache_cenario, (0, 0))

    def fundo(self):
        self.surf.fill(COR_FUNDO)
        if self._estrelas is None:
            rng = np.random.default_rng(7)
            w, h = self.surf.get_size()
            n = int(w * h / 4200)
            self._estrelas = [
                (int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(40, 190)))
                for _ in range(n)
            ]
        for x, y, brilho in self._estrelas:
            self.surf.set_at((x, y), (brilho, brilho, min(255, brilho + 25)))

    def grade(self, cam: Camera, passo=0.5, limites=None):
        """Grade em metros, para dar noção de escala no editor."""
        x0, y0, x1, y1 = limites or (-1.0, -1.0, 12.0, 12.0)
        for x in np.arange(np.floor(x0), x1 + passo, passo):
            a, b = cam.to_screen([x, y0]), cam.to_screen([x, y1])
            pygame.draw.line(self.surf, COR_GRADE, a, b, 1)
        for y in np.arange(np.floor(y0), y1 + passo, passo):
            a, b = cam.to_screen([x0, y]), cam.to_screen([x1, y])
            pygame.draw.line(self.surf, COR_GRADE, a, b, 1)

    def desenhar_pista(self, track, cam: Camera, mostrar_checkpoints=True):
        if track.floor is not None and len(track.floor) >= 3:
            pygame.draw.polygon(self.surf, COR_CHAO, cam.to_screen(track.floor))

        # checkpoints primeiro, para ficarem por baixo
        if mostrar_checkpoints:
            for cp in track.checkpoints:
                a, b = cam.to_screen(cp[:2]), cam.to_screen(cp[2:])
                pygame.draw.line(self.surf, COR_CHECKPOINT, a, b, 1)

        espessura = max(3, cam.px(0.05))
        for parede in track.boundary_walls:
            self._parede_zebrada(parede, cam, espessura)

        self.desenhar_caixas(track, cam)
        self._objetivo(track.goal, cam)

    def desenhar_caixas(self, track, cam: Camera, destaque=None):
        """Caixas como blocos sólidos — legíveis mesmo com 8 cm de lado."""
        if not len(track.obstacles):
            return
        for i, cantos in enumerate(box_corners(track.obstacles)):
            pts = cam.to_screen(cantos)
            realce = destaque is not None and i == destaque
            pygame.draw.polygon(self.surf, COR_CAIXA_REALCE if realce else COR_CAIXA, pts)
            pygame.draw.polygon(self.surf, COR_CAIXA_BORDA, pts, 2)

    def _parede_zebrada(self, parede, cam, espessura):
        """Paredes em vermelho e branco alternados, como zebra de pista."""
        a = np.array(parede[:2])
        b = np.array(parede[2:])
        comp = float(np.hypot(*(b - a)))
        if comp < 1e-9:
            return
        n = max(1, int(round(comp / 0.25)))
        for i in range(n):
            p0 = cam.to_screen(a + (b - a) * (i / n))
            p1 = cam.to_screen(a + (b - a) * ((i + 1) / n))
            cor = COR_PAREDE_A if i % 2 == 0 else COR_PAREDE_B
            pygame.draw.line(self.surf, cor, p0, p1, espessura)

    def _objetivo(self, goal, cam):
        centro = cam.to_screen(goal[:2])
        raio = cam.px(goal[2])
        for i, r in enumerate(range(raio, 0, -max(3, raio // 4))):
            cor = COR_OBJETIVO if i % 2 == 0 else (28, 90, 50)
            pygame.draw.circle(self.surf, cor, centro, r, 0 if r < raio // 3 else 3)

    # ------------------------------------------------------------------ #
    def desenhar_robos(self, world, cam: Camera, detalhe_em=0, indices=None):
        """Desenha os robôs; o de índice `detalhe_em` ganha os sensores.

        O destacado é desenhado por **último**, de propósito. Numa população
        que ainda não aprendeu a se espalhar, é comum muitos indivíduos
        ficarem empilhados na mesma posição (verificado: numa leva de 80,
        chegaram a 52 sobrepostos no mesmo pixel do robô 0). Desenhar o
        destacado primeiro e os demais por cima faria ele sumir debaixo da
        pilha justamente quando você mais quer vê-lo.

        `indices`: quais robôs "de fundo" desenhar, além do destacado (que é
        sempre desenhado). `None` desenha todo mundo; passe um subconjunto
        para aliviar o quadro numa população grande — a simulação continua
        rodando com todos, só o desenho é que fica mais leve.
        """
        pos = cam.to_screen(world.pos)
        raio = cam.px(world.cfg.robot.radius)

        alvo = range(world.P) if indices is None else indices
        for i in alvo:
            if world.P > 1 and i == detalhe_em:
                continue
            vivo = bool(world.alive[i]) and not world.finished[i]
            centro = (int(pos[i, 0]), int(pos[i, 1]))
            cor_fundo = COR_ROBO_FUNDO if vivo else COR_ROBO_MORTO_FUNDO
            pygame.draw.circle(self.surf, cor_fundo, centro, max(2, raio))

        if detalhe_em is not None and 0 <= detalhe_em < world.P:
            i = detalhe_em
            vivo = bool(world.alive[i]) and not world.finished[i]
            centro = (int(pos[i, 0]), int(pos[i, 1]))
            cor = COR_ROBO if vivo else COR_ROBO_MORTO
            pygame.draw.circle(self.surf, cor, centro, raio)
            pygame.draw.circle(self.surf, (12, 40, 40), centro, raio, 2)
            frente = cam.to_screen(
                world.pos[i] + np.array([np.cos(world.theta[i]), np.sin(world.theta[i])])
                * world.cfg.robot.radius * 1.7)
            pygame.draw.line(self.surf, (255, 255, 255), centro, frente, 2)
            self._sensores(world, cam, detalhe_em)

    def _sensores(self, world, cam, i):
        """Mostra o cone de cada HC-SR04 até a distância que ele leu."""
        s = world.sensors
        origens = s.origins(world.pos[i:i + 1], world.theta[i:i + 1])[0]   # (n, 2)
        angulos = s.cone_angles(world.theta[i:i + 1])[0]                   # (n, k)
        leituras = s.last[i]
        alcance = world.cfg.sensor.max_range

        for j in range(s.n):
            d = float(leituras[j])
            perto = d < alcance * 0.999
            cor = COR_CONE if perto else (55, 70, 90)

            pontos = [cam.to_screen(origens[j])]
            for ang in angulos[j]:
                ponta = origens[j] + np.array([np.cos(ang), np.sin(ang)]) * d
                pontos.append(cam.to_screen(ponta))
            if len(pontos) >= 3:
                pygame.draw.polygon(self.surf, cor, pontos, 1)

            centro_ang = float(np.mean(angulos[j]))
            ponta = origens[j] + np.array([np.cos(centro_ang), np.sin(centro_ang)]) * d
            pygame.draw.line(self.surf, COR_RAIO if perto else (48, 60, 78),
                             cam.to_screen(origens[j]), cam.to_screen(ponta), 1)

    # ------------------------------------------------------------------ #
    def hud(self, world, extra_linhas=(), i=0):
        """Painel com as leituras, a idade de cada uma e o estado dos motores.

        A idade importa tanto quanto a distância: com o rodízio do HC-SR04 a rede
        decide sempre com dado velho, e é aqui que dá para ver o quanto.
        """
        cfg = world.cfg
        s = world.sensors
        linhas = []   # (texto, cor, fonte, fração_da_barra ou None)

        linhas.append(("SENSORES  (distância / idade da leitura)", COR_TEXTO, self.font, None))
        graus = np.rad2deg(s.mount)
        for j in range(s.n):
            d = float(s.last[i, j])
            idade = float(s.age[i, j])
            sem_eco = d >= cfg.sensor.max_range * 0.999
            cor = COR_APAGADO if sem_eco else (COR_ALERTA if d < 0.35 else COR_TEXTO)
            rotulo = "sem eco" if sem_eco else f"{d:5.2f} m"
            linhas.append((f"{graus[j]:+6.0f}°  {rotulo}  {idade*1000:4.0f} ms",
                           cor, self.font_peq, float(np.clip(d / cfg.sensor.max_range, 0, 1))))

        vl, vr = world.drive.wheel[i]
        linhas.append((None, None, None, None))
        linhas.append((f"MOTORES   esq {vl:+.2f}  dir {vr:+.2f} m/s", COR_TEXTO, self.font_peq, None))
        linhas.append((f"VELOC.    {world.v[i]:+.2f} m/s   giro "
                       f"{np.rad2deg(world.omega[i]):+6.1f} °/s", COR_TEXTO, self.font_peq, None))
        linhas.append((f"PERCURSO  {world.distance[i]:.2f} m   checkpoint "
                       f"{world.checkpoint[i]}/{len(world.track.checkpoints)}   "
                       f"batidas {world.bumps[i]}", COR_TEXTO, self.font_peq, None))
        if extra_linhas:
            linhas.append((None, None, None, None))
            linhas += [(t, COR_APAGADO, self.font_peq, None) for t in extra_linhas]

        x0, y0, larg_barra = 14, 12, 90
        largura = max(
            (f.size(t)[0] + (larg_barra + 10 if b is not None else 0))
            for t, _, f, b in linhas if t is not None)
        altura = sum(6 if t is None else f.get_linesize() + 2 for t, _, f, _ in linhas)

        painel = pygame.Surface((largura + 24, altura + 20), pygame.SRCALPHA)
        painel.fill((10, 13, 22, 214))
        pygame.draw.rect(painel, (46, 54, 74), painel.get_rect(), 1, border_radius=6)
        self.surf.blit(painel, (x0 - 12, y0 - 10))

        y = y0
        for texto, cor, fonte, barra in linhas:
            if texto is None:
                y += 6
                continue
            self.surf.blit(fonte.render(texto, True, cor), (x0, y))
            if barra is not None:
                bx = x0 + largura - larg_barra
                by = y + fonte.get_linesize() // 2 - 3
                pygame.draw.rect(self.surf, (34, 40, 56), (bx, by, larg_barra, 7), border_radius=3)
                if barra > 0:
                    pygame.draw.rect(self.surf, cor,
                                     (bx, by, max(2, int(larg_barra * barra)), 7), border_radius=3)
            y += fonte.get_linesize() + 2

    def aviso(self, texto, cor=COR_OBJETIVO):
        img = self.font.render(texto, True, cor)
        w, h = self.surf.get_size()
        caixa = img.get_rect(center=(w // 2, 40))
        fundo = caixa.inflate(24, 14)
        pygame.draw.rect(self.surf, (14, 18, 28), fundo, border_radius=8)
        pygame.draw.rect(self.surf, cor, fundo, 1, border_radius=8)
        self.surf.blit(img, caixa)
