import pygame
from visualization.core import Vector2, Rect
from sim.robot import RobotModel

class PygameRenderer:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))

    def draw_world(self, robots: list[RobotModel], obstacles: list[Rect]):
        self.screen.fill((0,0,0))
        for o in obstacles:
            pygame.draw.rect(self.screen, (255,255,255), pygame.Rect(o.x,o.y,o.w,o.h))
        for r in robots:
            center = (int(r.position.x), int(r.position.y))
            pygame.draw.circle(self.screen, (255,0,0), center, r.radius_px)
            # draw heading line…
        pygame.display.flip()

def run_visual(world, fps=60):
    clock = pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running = False
        world.step(1/fps)
        PygameRenderer.draw_world(world.robots, world.obstacles)
        clock.tick(fps)
    pygame.quit()
