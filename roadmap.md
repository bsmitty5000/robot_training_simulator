Below is a **“ladder” roadmap** you can climb one rung at a time.
Each rung is:

* **Concrete** : a fully specified network you can evolve in your Python sim.
* **Transferable** : small enough to hand-translate (or auto-generate) into C for an Arduino Pro Mini (ATmega328P, 2 kB RAM).
* **More capable** than the previous one while re-using as much code and tooling as possible.

---

## 0 · Foundation (what you already have)

| Piece        | Spec                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------ |
| **Sensors**  | 3 forward-facing IR rangers (left-front, centre-front, right-front) → distances ∈ \[0 .. max]    |
| **Net**      | **3-4-2** (tanh hidden, linear outputs)                                                          |
| **Trainer**  | Genetic Algorithm (GA) with tournament + elitism                                                 |
| **Fitness**  | Max forward distance before collision                                                            |
| **Embedded** | Hard-code 22 weights + 6 biases as `float` constants → compute in \~50 µs per loop on 16 MHz AVR |

*Goal*: prove E2E flow works (sim → flash → drive).

---

## 1 · Hidden-size sweep (same topology, better capacity)

| Upgrade                   | Why                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------- |
| **3-8-2** then **3-12-2** | Doubles/triples parameters; often enough to learn smoother turns without adding layers. |
| **Mutation σ decay**      | Start σ=0.3, linearly drop to 0.05 by gen 75 to fine-tune.                              |

*Transfer*: still ≤ 30 floats—fits Pro Mini RAM.

---

## 2 · Second hidden layer (depth)

| Net                     | Params                         | Flash/RAM fit?                |
| ----------------------- | ------------------------------ | ----------------------------- |
| **3-8-4-2** (tanh,tanh) | 3×8+8 + 8×4+4 + 4×2+2 = **86** | OK in flash; RAM 86×4 = 344 B |

Benefits:

* Learns gentle S-curves; hidden₂ acts as a rudimentary “steering stabiliser.”
* Pure feed-forward math: still trivial in C.

---

## 3 · Recurrent memory (temporal context)

| Net                                             | Notes                                   |
| ----------------------------------------------- | --------------------------------------- |
| **Elman RNN**: 3 + 4 context → 6 hidden → 2 out | Copy hidden state to context each step. |
| **25–35 params** if you keep hidden small.      |                                         |

Why:

* Lets car “commit” to a turn instead of oscillating.
* Easy AVR port: one extra array to store last hidden activations.

GA training tip: set episode length > 3 s so memory helps.

---

## 4 · CMA-ES instead of vanilla GA (fewer evals)

*Population 64, generations 150* often beats GA-256×150.
Python: `pip install cma`, treat net parameters as one 𝑛-vector.
Same compiled AVR code—only trainer changes.

---

## 5 · “Rich” input vector (state fusion—no new sensors)

Add cheap numerical data already on board:

| New inputs                   | How to get them       |
| ---------------------------- | --------------------- |
| Wheel PWM last step (L,R)    | store last outputs    |
| Angular velocity estimate    | (PWM\_R − PWM\_L) × k |
| Norm. battery voltage (opt.) | ADC                   |

\=> **6 inputs**.  Network **6-12-6-2** (\~100 params).
Advantages:

* Robot can learn momentum compensation and avoid over-steer.
* Still fits Pro Mini (400 B weights).

---

## 6 · NEAT for topology search

When you’re unsure of ideal architecture:

* Use `neat-python` (works with your pure-Python sim / JIT kernels).
* Start with minimal 3 → 2 direct links.
* NEAT evolves neurons & connections; you freeze final genome and auto-serialise to C arrays.

Harder to port (variable graph), but code-gen scripts exist.

---

## 7 · Policy-gradient fine-tune (PPO “last mile”)

Once GA/CMA finds a solid policy:

1. Switch to PyTorch PPO with very small learning rate.
2. Initialise weights from best-GA nets.
3. Train 5–10 k steps to squeeze extra 5–10 % performance.

Finally export weights back to C.
(Requires more Python infra but **no** MCU change.)

---

## 8 · TinyML camera upgrade (stretch goal, not AVR-friendly)

* Add OV7670 or ESP32-Cam streaming grayscale to PC for training.
* CNN encoder →  latent → 2-motor QC output.
* Deploy on Teensy 4.0 / ESP32-S3 with TFLite Micro.

This is beyond Pro Mini RAM but keeps your Python sim as the sandbox.

---

### Recommended climb order

1. **Baseline 3-4-2** (already done).
2. Hidden-size sweep (3-8-2).
3. Add 2nd hidden layer (3-8-4-2).
4. Swap trainer to CMA-ES (same nets).
5. Add context memory (Elman).
6. Add wheel/PWM features → 6-12-6-2.
7. Try NEAT if architecture still feels cramped.
8. PPO fine-tune best genome.
9. (Future) switch hardware to a bigger MCU and explore vision.

---

### Tooling checklist at each rung

| Task           | Python side                                                            | Embedded side                       |
| -------------- | ---------------------------------------------------------------------- | ----------------------------------- |
| Export weights | `np.save`, or generate `const float W[] = {...};`                      | Copy to `.h` file                   |
| Activation fns | Keep **tanh** or **ReLU** (needs `float`)                              | One `fast_tanh()` LUT or `avr-libm` |
| Loop timing    | 20 Hz in sim **==** 20 Hz `millis()` loop                              | Guard with watchdog                 |
| Unit test      | Feed sample sensor triplet through Python & AVR code → compare outputs | Tiny serial diff tool               |

---

### Typical performance milestones

| Rung | Success rate (maze w/ 8 boxes)    |
| ---- | --------------------------------- |
| 0-1  | 40–60 % runs finish course        |
| 2    | 70–80 %                           |
| 3    | > 90 % (memory helps)             |
| 4-6  | Near-perfect, smoother trajectory |
| 7    | Polishes corner-cases             |

Climb until the real robot reaches the reliability you need, then stop—the next rungs are only if you crave tougher maps or want to add sensors.

Good luck, and feel free to circle back after each step!
