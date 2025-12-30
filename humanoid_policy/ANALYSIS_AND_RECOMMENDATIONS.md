# Training Analysis & Recommendations

## Run Comparison

### Run 1 (2025-12-30_01-31-53) - 1000 Episoden, 1024 envs:
- Episode 100: **1662.9**
- Episode 500: **1684.6** (Peak)
- Episode 1000: **1637.0** (stabil)

### Run 2 (2025-12-30_16-23-21) - 5000 Episoden, 2048 envs:
- Episode 100: **1671.8** (besser Start)
- Episode 1400: **1680.9** (Peak)
- Episode 5000: **1332.7** (stark abgefallen - Overfitting!)

## Problem: Overfitting in Run 2

**Symptom:** Reward fällt nach Peak stark ab (1680 → 1332)
**Ursache:** Zu viele Episoden ohne Early Stopping / Learning Rate Decay

---

## ✅ UMSETZUNGS-LISTE: Änderungen für Balance-Fokus

### 1. **PPO Config (rl_games_ppo_cfg.yaml) - Anymal Settings übernehmen**

**NN Layers (Anymal Standard):**
- ✅ `units: [256, 128, 64]` → `[512, 256, 128]` (größeres Netz)

**Hyperparameter (Anymal Standard):**
- ✅ `horizon_length: 64` → `24` (Anymal: 24, stabiler)
- ✅ `minibatch_size: 32768` → `16384` (Anymal: 16384, stabiler)
- ✅ `learning_rate: 3e-4` → **BLEIBT** (Anymal: 3e-4)
- ✅ `e_clip: 0.2` → **BLEIBT** (Anymal: 0.2)
- ✅ `mini_epochs: 5` → **BLEIBT** (Anymal: 5)
- ✅ `max_epochs: 5000` → `2000` (verhindert Overfitting)
- ✅ `save_best_after: 100` → `50` (früheres Best-Model Saving)

**Datei:** `humanoid_policy/source/humanoid_policy/humanoid_policy/tasks/direct/humanoid_policy/agents/rl_games_ppo_cfg.yaml`

---

### 2. **Rewards - NUR Balance, KEINE Bewegung**

**Velocity Reward entfernen (bereits deaktiviert):**
- ✅ `rew_scale_forward_vel = 0.0` → **BLEIBT** (bereits deaktiviert)

**Penalties reduzieren (weniger aggressiv für Balance):**
- ✅ `rew_scale_joint_vel: -0.001` → `-0.0005` (weniger aggressiv)
- ✅ `rew_scale_base_vel: -0.01` → `-0.005` (erlaubt kleine Balance-Bewegungen)

**Positive Rewards (bleiben):**
- ✅ `rew_scale_alive = 2.0` → **BLEIBT**
- ✅ `rew_scale_upright = 1.0` → **BLEIBT**
- ✅ `rew_scale_foot_contact = 0.5` → **BLEIBT**

**Negative Rewards (bleiben):**
- ✅ `rew_scale_terminated = -5.0` → **BLEIBT**
- ✅ `rew_scale_action = -0.0001` → **BLEIBT**
- ✅ `rew_scale_action_rate = -0.001` → **BLEIBT**
- ✅ `rew_scale_base_ang_vel = 0.0` → **BLEIBT**
- ✅ `rew_scale_joint_limit = -0.1` → **BLEIBT**

**Datei:** `humanoid_policy/source/humanoid_policy/humanoid_policy/tasks/direct/humanoid_policy/humanoid_policy_env_cfg.py`

---

### 3. **Episode Length - Schnelleres Lernen**

- ✅ `episode_length_s: 10.0` → `5.0` (schnelleres Lernen, mehr Resets)

**Datei:** `humanoid_policy/source/humanoid_policy/humanoid_policy/tasks/direct/humanoid_policy/humanoid_policy_env_cfg.py`

---

### 4. **Training Strategy**

**Environment Anzahl:**
- ✅ Training mit `--num_envs=1024` (Run 1 war stabiler als Run 2)

**Episoden:**
- ✅ Training mit `max_epochs=2000` (verhindert Overfitting)

---

## 📋 Zusammenfassung der Änderungen

### PPO Config (YAML):
1. `units: [512, 256, 128]` (von [256, 128, 64])
2. `horizon_length: 24` (von 64)
3. `minibatch_size: 16384` (von 32768)
4. `max_epochs: 2000` (von 5000)
5. `save_best_after: 50` (von 100)

### Env Config (Python):
1. `rew_scale_joint_vel: -0.0005` (von -0.001)
2. `rew_scale_base_vel: -0.005` (von -0.01)
3. `episode_length_s: 5.0` (von 10.0)
4. `rew_scale_forward_vel: 0.0` (bleibt - bereits deaktiviert)

---

## 🎯 Ziel: Reine Balance-Policy

**Fokus:**
- ✅ Roboter soll **NUR** aufrecht stehen bleiben
- ✅ **KEINE** Vorwärtsbewegung
- ✅ **KEINE** seitliche Bewegung
- ✅ Minimale Energie (kleine Penalties)
- ✅ Smooth Actions (action_rate penalty bleibt)

**Rewards fördern:**
- Stehen bleiben (alive + upright)
- Füße am Boden (foot_contact)
- Kein Umfallen (termination penalty)

**Rewards bestrafen:**
- Zu viel Bewegung (base_vel penalty reduziert)
- Zu schnelle Joints (joint_vel penalty reduziert)
- Hohe Torques (action penalty bleibt)
- Jerky Actions (action_rate penalty bleibt)

---

## Isaac Lab Referenzen

**Anymal PPO Config (Quelle):**
- `units: [512, 256, 128]`
- `horizon_length: 24`
- `minibatch_size: 16384`
- `learning_rate: 3e-4`
- `e_clip: 0.2`
- `mini_epochs: 5`

**GitHub Links:**
- Humanoid: `https://github.com/isaac-sim/IsaacLab/tree/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/humanoid`
- Quadruped: `https://github.com/isaac-sim/IsaacLab/tree/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/quadruped`

