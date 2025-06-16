import pygame, sys
import numpy as np
from math import cos, sin, tanh

# -- Simulation constants --
WORLD_W, WORLD_H = 4.0, 4.0    # meters
SCREEN_W, SCREEN_H = 600, 600  # pixels
SCALE = SCREEN_W / WORLD_W     # px per meter
DT = 0.05                      # sec per sim step
ROBOT_RADIUS = 0.1            # meters
SENSOR_ANGLES = [np.pi/4, 0, -np.pi/4]
SENSOR_RANGE = 1.5            # meters

# -- Neural net from before, shortened --
class SimpleNN:
    def __init__(self, weights=None):
        self.i, self.h, self.o = 3, 3, 2
        size = self.i*self.h + self.h*self.o + self.h + self.o
        self.w = np.random.uniform(-1,1,size) if weights is None else np.array(weights)
        self.decode()

    def decode(self):
        idx=0
        self.w1 = self.w[idx:idx+self.i*self.h].reshape(self.i,self.h); idx+=self.i*self.h
        self.w2 = self.w[idx:idx+self.h*self.o].reshape(self.h,self.o); idx+=self.h*self.o
        self.b1 = self.w[idx:idx+self.h]; idx+=self.h
        self.b2 = self.w[idx:idx+self.o]

    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)  # [throttle, steer]

# -- Robot with IR rays & simple kinematics --
class Robot:
    def __init__(self, genome=None):
        self.nn = SimpleNN(genome)
        self.reset()

    def reset(self):
        self.x, self.y, self.theta = WORLD_W/2, WORLD_H/2, 0.0
        self.alive, self.time = True, 0.0

    def sense(self, obstacles):
        dists = []
        for a in SENSOR_ANGLES:
            ray = self.theta + a
            for d in np.linspace(0, SENSOR_RANGE, 100):
                px = self.x + d*cos(ray)
                py = self.y + d*sin(ray)
                # check walls
                if not (0<=px<=WORLD_W and 0<=py<=WORLD_H):
                    dists.append(d); break
                # check boxes
                for ox,oy,ow,oh in obstacles:
                    if ox<=px<=ox+ow and oy<=py<=oy+oh:
                        dists.append(d); break
                else:
                    continue
                break
            else:
                dists.append(SENSOR_RANGE)
        return np.array(dists)/SENSOR_RANGE

    def step(self, obstacles):
        if not self.alive: return
        s = self.sense(obstacles)
        thrott, steer = self.nn.forward(s)
        v = thrott * 0.5
        w = steer * 1.0
        self.x += v*cos(self.theta)*DT
        self.y += v*sin(self.theta)*DT
        self.theta += w*DT
        self.time += DT
        # collision?
        if not (ROBOT_RADIUS<=self.x<=WORLD_W-ROBOT_RADIUS and ROBOT_RADIUS<=self.y<=WORLD_H-ROBOT_RADIUS):
            self.alive = False

    def draw(self, surf):
        # robot body
        cx = int(self.x * SCALE); cy = int(self.y * SCALE)
        pygame.draw.circle(surf, (0,200,0), (cx,cy), int(ROBOT_RADIUS*SCALE))
        # heading line
        hx = cx + int(cos(self.theta)*ROBOT_RADIUS*SCALE*2)
        hy = cy + int(sin(self.theta)*ROBOT_RADIUS*SCALE*2)
        pygame.draw.line(surf, (255,0,0), (cx,cy), (hx,hy), 2)
        # sensor rays
        for a in SENSOR_ANGLES:
            ray = self.theta + a
            dx = cos(ray)*SENSOR_RANGE*SCALE
            dy = sin(ray)*SENSOR_RANGE*SCALE
            pygame.draw.line(surf, (100,100,255), (cx,cy), (cx+dx, cy+dy), 1)

# -- Main Pygame loop --
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    # define your map here (x, y, width, height) in world coords
    obstacles = [
        (1.0, 1.0, 0.5, 2.0),
        (2.5, 0.5, 0.3, 3.0),
    ]

    robot = Robot()
    running, paused = True, True

    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            elif e.type==pygame.KEYDOWN:
                if e.key==pygame.K_SPACE: paused = not paused
                if e.key==pygame.K_r:
                    robot.reset()

        if not paused:
            robot.step(obstacles)

        # draw
        screen.fill((30,30,30))
        # walls
        pygame.draw.rect(screen,(200,200,200),(0,0,SCREEN_W,SCREEN_H),4)
        # obstacles
        for ox,oy,ow,oh in obstacles:
            rect = pygame.Rect(ox*SCALE, oy*SCALE, ow*SCALE, oh*SCALE)
            pygame.draw.rect(screen,(180,60,60), rect)
        # robot
        robot.draw(screen)

        pygame.display.flip()
        clock.tick(1/DT)

    pygame.quit()
    sys.exit()

if __name__=='__main__':
    main()
