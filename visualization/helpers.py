import numpy as np
import pygame
import jit_sim.robots.tt_2wheel as robot

def show_debug_info(screen,
                    sensor_readings:    np.ndarray,   # 3×
                    robot_state:        np.ndarray,   # [x, y, hd_deg]
                    controller_outputs: np.ndarray,   # 2×  (-1..1)
                    fitness:            np.float32):

    font = pygame.font.SysFont("consolas", 12)

    # ---- fixed-width formatting ------------------------------------
    sens_text = "Sensors: " + ", ".join(f"{s:6.4f}" for s in sensor_readings)
    fit_text = f"Fitness: {fitness:10.2f}"
    ctrl_text = "Ctrl: "    + ", ".join(f"{o:+7.4f}" for o in controller_outputs)
    pwm_text  = "Vel: "     + ", ".join(f"{(i * robot.MODELED_MAX_VELOCITY):+7.4f} m/s"  for i in controller_outputs)   # -255..255
    state_text= "Pos/Hdg: " + ", ".join(f"{v:7.4f}"   for v in robot_state)    # x y hd

    line1 = f"{sens_text} | {fit_text}"
    line2 = f"{ctrl_text} | {pwm_text} | {state_text}"

    # ---- erase old text area (optional but avoids ghosting) ---------
    bg_rect = pygame.Rect(8, 8, 900, 32)      # big enough for both lines
    pygame.draw.rect(screen, (0, 0, 0), bg_rect)

    # ---- render & blit ---------------------------------------------
    screen.blit(font.render(line1, True, (255, 255, 255)), (10, 10))
    screen.blit(font.render(line2, True, (255, 255,   0)), (10, 24))

