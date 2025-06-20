import numpy as np
import pygame

def show_debug_info(screen,
                    sensor_readings:    np.ndarray,   # 3×
                    robot_state:        np.ndarray,   # [x, y, hd_deg]
                    controller_outputs: np.ndarray,   # 2×  (-1..1)
                    robot_inputs:       np.ndarray):  # 2×  (PWM L,R)

    font = pygame.font.SysFont("consolas", 12)

    # ---- fixed-width formatting ------------------------------------
    sens_text = "Sensors: " + ", ".join(f"{s:6.1f}" for s in sensor_readings)
    ctrl_text = "Ctrl: "    + ", ".join(f"{o:+7.3f}" for o in controller_outputs)
    pwm_text  = "PWM: "     + ", ".join(f"{i:+4.0f}"  for i in robot_inputs)   # -255..255
    state_text= "Pos/Hdg: " + ", ".join(f"{v:7.1f}"   for v in robot_state)    # x y hd

    line1 = sens_text
    line2 = f"{ctrl_text} | {pwm_text} | {state_text}"

    # ---- erase old text area (optional but avoids ghosting) ---------
    bg_rect = pygame.Rect(8, 8, 900, 32)      # big enough for both lines
    pygame.draw.rect(screen, (0, 0, 0), bg_rect)

    # ---- render & blit ---------------------------------------------
    screen.blit(font.render(line1, True, (255, 255, 255)), (10, 10))
    screen.blit(font.render(line2, True, (255, 255,   0)), (10, 24))

