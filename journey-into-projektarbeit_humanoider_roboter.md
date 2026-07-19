# Journey Into projektarbeit_humanoider_roboter

*A technical history reconstructed from claude-mem's persistent memory — 1,274 observations across 160 sessions, June 2–16, 2026.*

This is the story of a robotics machine-learning project that set out to teach a humanoid robot to stack blocks and, in the process, learned a much harder lesson about the gap between simulation and reality. The project fine-tunes NVIDIA's **GR00T N1.6** Vision-Language-Action model on the **Unitree G1 humanoid with DEX3 hands**, runs everything in Docker/Apptainer across vast.ai and the KISSKI HPC cluster, and ultimately confronts the question that haunts every sim-trained policy: *what happens when the simulator doesn't look like the world the model was trained on?* The memory record captures fourteen days of dense, often nocturnal work — training runs, simulation debugging, camera calibration marathons, and a steady accretion of documentation and academic write-up.

---

## 1. Project Genesis

The earliest observations, all timestamped **Jun 2, 2026 4:49–4:50 PM** (#1–#9), are not feature work but *cartography*. The very first session was a `/learn-codebase` pass that mapped the entire repository before any code was touched: the project structure (#1), the GR00T N1.6 fine-tuning architecture and workflow (#2), the git-submodule configuration (#3), and then the load-bearing infrastructure — the training Dockerfile with its CUDA base and pinned commit (#4), the entrypoint's "W&B always-offline" pattern (#5), the fine-tuning command and its training parameters (#6).

Crucially, the founding technical decisions were already in place and were *discovered* rather than invented in this period. Observation #7 captured the single most important operational constraint that recurs throughout the entire history: the **KISSKI SLURM job's critical bind-mount pattern for the GR00T fork**. Observation #8 documented the **vast.ai simulation Dockerfile that crams Isaac Lab + GR00T into one container**, and #9 surfaced the first landmine — the **Isaac Lab joint-ordering gotcha**, which would echo across two weeks of grasping bugs.

The problem being solved was clear from the architecture: take a 3.3-billion-parameter foundation model (#621 later confirms GR00T-N1.6-3B from NVIDIA's HuggingFace hub), fine-tune it on a single block-stacking dataset, and prove it works by running the resulting policy *closed-loop* inside a physics simulator. The dataset (#706–#708) was the Unitree G1 DEX3 BlockStacking set: 301 episodes, 28-DOF state/action, four cameras at 640×480/30fps. The action representation — a detail that would cause grief — was mixed: **arms RELATIVE, hands ABSOLUTE** (#248, #378, #837).

What is striking about the genesis period is how much of Jun 2 was *not* about the robot at all. Sessions S3 through S16 are a long detour into tooling: diagnosing claude-mem's own MCP server registration, chasing a `chroma-mcp` Windows spawn failure (#43–#70, an MSYS→cmd.exe bridge bug that was never fully fixed, only worked around by pre-warming the uvx cache, #55–#57), and building a markitdown auto-conversion hook (#77–#92). The robot project proper only resumed when a real signal appeared: **a training loss plateau observed at step 3200** (#114), which kicked off the W&B integration.

---

## 2. Architectural Evolution

The architecture the project *started* with was already two-headed, and the history is largely the story of those two heads diverging and then being reconciled.

**The training image vs. the simulation image.** From day one there were two Dockerfiles: `Training/Dockerfile` (pure GR00T fine-tuning, no Isaac Sim) and `Simulation/Dockerfile.vastai` (the combined Isaac Sim + GR00T container, #8, #101). The reason is hardware: training wants A100/H100 raw FLOPs, but Isaac Sim's camera rendering *requires RT-cores* — a constraint that became the project's hardest infrastructural wall (see §6). This split is why the RL work much later (#3646) could not simply run on KISSKI's A100s.

**The two-process ZMQ simulation architecture.** The closed-loop eval is not one program but two communicating over ZeroMQ (#102, #241, #376, #611, #2726, #4190). A GR00T policy server holds the model; an Isaac Lab client (`client.py`) builds observations, splits robot state into modality keys, sends them to the server, and applies the returned actions. Observation #513 captured a subtle but vital piece: **the GR00T server converts relative actions to absolute before returning them to the sim**. This decoupling let the team swap embodiments (DEX3 vs. a stock G1 gripper baseline) without touching the model server.

**Single-GPU → multi-GPU.** On **Jun 3 (#225)** a deliberate decision was logged to migrate KISSKI training from 1×A100 to 4×A100. The launcher gained dynamic `torchrun`/`python` selection (#229), the SLURM job was rewritten for 4 GPUs (#230), and hyperparameters were rescaled (#231). A latent bug surfaced immediately: the entrypoint never exported `LEARNING_RATE`, so the chain silently relied on Apptainer's `--env` forwarding (#232–#233) — a fragility that was patched but is emblematic of the env-var-everything design.

**Docker vs. Apptainer.** The same image runs under Docker locally/on vast.ai and under Apptainer on KISSKI. The detection logic in `run_finetuning.sh` checks for both `.dockerenv` and `$APPTAINER_NAME`. The bind-mount-the-scripts pattern (#7) means KISSKI repo changes take effect without an image rebuild — a deliberate dev-velocity choice that does not exist for vast.ai, where every script change demands a 50GB image rebuild (#618).

**The baseline pipeline pivot (Jun 4).** A significant architectural fork appeared in sessions S130–S140: a *parallel* `g1_gripper_sim` pipeline (#532, #534) to run the un-finetuned GR00T-N1.6-3B with a stock G1 + Dex1 binary gripper. This required confirming the base model uses the `UNITREE_G1`/`GR1` tag (#512, #539), building a whole new articulation config, env, client, and eval runner (#549, #552, #555, #560), and finally wiring it in as `SIM_MODE=baseline` (#564) so it lived alongside DEX3 without breaking it. This was the seed of the much later RoboCasa reference-eval work.

**The RL turn (Jun 12–16).** The final architectural layer was FPO reinforcement-learning fine-tuning. `rl_finetune.py` (#3633) trains only the action head, with a shaped dense reward added to the env (#3624–#3625) and a batched GPU observation API for rollouts (#3626). It got its own entrypoint (#3639) and SLURM template (#3645). The architecture honestly admitted its own incompatibility: **RL needs an RT-core GPU, so it cannot run on KISSKI** (#3646), and the BC entrypoint was given a *fail-fast guard* that errors with a pointer to the RL path (#3642) rather than silently doing the wrong thing.

---

## 3. Key Breakthroughs

The breakthroughs in this project were rarely "it works now" — more often they were "we finally understand *why* it doesn't."

**The domain-gap diagnosis (Jun 3–4).** The single most important intellectual breakthrough was reframing failure as expected behavior. The closed-loop policy froze in sim despite a beautifully converged training loss (loss → ~0.008–0.013, #247). After video analysis (#255–#257 measured motion-difference energy and found a flat profile), the root cause crystallized: **a frozen vision encoder trained on real images cannot interpret Isaac-Sim renderings** (#292, #598). This was later *quantified* — the left wrist camera showed a SEVERE cosine distance of 0.43, mean 0.26 (#645, #698, #2736) — and finally *vindicated by literature*: on **Jun 5 (#841)** a web-research agent surfaced the SIMPLER paper confirming that **visual gap causes near-zero success and that 0% is the expected result, not a bug**, and #842 noted NVIDIA's own eval uses open-loop action MSE, not closed-loop sim. The 0% result went from embarrassment to publishable finding.

**The black-hands USD fix (Jun 3–4).** A concrete attack on the domain gap: recolor the robot's hands black to match the dataset. This breakthrough was actually a *saga* (see §6) — the naive instance-root material binding silently failed to render (#367), and the real fix required de-instancing and direct mesh-level binding (#368, #371, #373).

**The camera-misalignment epiphany (Jun 4, #166 of S166).** Mid-calibration, a visual inspection revealed that `cam_left_wrist` "points at the wrong side of the joint — *not a real domain gap but a positioning error*." This split one problem into two: a genuine sim-to-real visual gap, and a fixable camera-pose bug. The fix landed in #701.

**The grasp-physics chain (Jun 4).** Over one evening the team root-caused why the robot grasped then dropped the cube: distal finger joints failed to close (#440), so finger actuators were strengthened (stiffness 20→60, effort 5→20 N·m, #441), cube friction was raised (#442, #505), and — the deepest fix — a **Dex3 finger-joint sign-convention inversion between dataset and USD** was discovered and corrected (#485, #495, #502). The payoff was logged precisely: **grasp confirmed at 2.8 cm lift after five targeted fixes** (#602).

**The flash-attn-in-Isaac-Sim fix (Jun 15, #3860).** The RL work required GR00T to run inside Isaac Sim's bundled Python 3.11, but `pip install flash-attn` exited code 1. The breakthrough was diagnostic: Isaac Sim's Python is 3.11 *with cxx11abi=True*, and the prebuilt wheels GR00T expected didn't match — yet **cp311 wheels DO exist on GitHub** (#3855). The fix installed flash-attn via a **direct GitHub wheel URL with pinned dependency versions** (#3860), validated by a real import test (#3865, #3872).

---

## 4. Work Patterns

The development rhythm is unmistakably that of a solo researcher working in intense, late-night bursts, with claude-mem as a continuity layer across dozens of short sessions.

The **daily cadence** is extreme front-loading: Jun 2 (182 obs / 13 sessions), Jun 3 (213 / 32), Jun 4 (**394 obs / 60 sessions** — the peak), Jun 5 (280 / 34). That is 1,069 observations and 139 sessions in four days. Then a gap until Jun 10 (17 / 1), Jun 12 (93 / 9), Jun 15 (59 / 7), and Jun 16 (36 / 4). The first four days were the build-and-debug sprint; the later days are punctuated, deliberate feature additions (RL infrastructure on Jun 12, vast.ai launch prep on Jun 15, RoboCasa scaffold on Jun 16).

Three distinct **modes** recur:

- **Tight debugging cycles** — especially the camera-calibration loop of Jun 4–5, which ran *twelve documented iterations* (#839 calls it "12-Iteration Overlay Process"). Each iteration was render → overlay against dataset reference → eyeball → adjust `g1_dex3_cfg.py` → repeat (#711–#760). These sessions are short and numerous, which inflates the Jun 4 session count.
- **Feature sprints** — the multi-GPU migration (#225–#235), the baseline pipeline (S130–S140), the `TUNE_VISUAL` vision-encoder path (#766–#807), and the RL infrastructure (#3613–#3692). Each follows the same shape: research the code, implement across all deployment targets, then a documentation-consistency sweep.
- **Documentation-as-first-class-work** — an unusually large fraction of effort. Whole sessions (S150, S155, S229, S230) are pure docs review and restructuring. The **Jun 5 docs restructure** (#944–#966) moved files into `ergebnisse/` and `weiterfuehrend/` sections, then verified all 295 links. And the **LaTeX thesis** (#872 onward) consumed most of Jun 5 evening — replacing placeholder chapters with real content, fixing fonts (IBMPlexMath → Fira Math → Libertinus), and reconstructing a 80-point training-loss figure from W&B HTML data (#1023).

A telltale meta-pattern: the work environment itself was frequently under repair. W&B MCP setup failed and was re-fixed at least five times (S22, S42, S108, S113–S121, S134–S141), eventually requiring a hand-built `wandb_gql` shim package (#469–#470, #575).

---

## 5. Technical Debt

The project accumulated and repaid debt visibly, often within the same week.

**Inactive train/test split.** The largest piece of debt: a train/test split was *implemented in code but never activated*. It was first flagged on Jun 4 (#636: "Train-Test Split Implemented but NOT Active — 60 Test Episodes Being Used in Training") and re-confirmed repeatedly (#2728, #3614, #3616) — both completed training runs used **all 301 episodes with no held-out test set** (#2728). This was finally paid down on **Jun 12 (#3622)** with runtime patching of `meta/info.json` gated behind `TRAIN_TEST_SPLIT=1`, and the docs were corrected to match (#3655). The thesis honestly disclosed the caveat (#1044).

**Stale defaults and docstrings.** A recurring class of debt. The episode-length docstring said "1200 steps = 40 Sekunden" long after the code changed (#674–#675); the `MAX_STEPS` default was stale at 30000 in the Dockerfile (#3649–#3650) and scattered through the FINETUNING_GUIDE (#3676–#3678); CLAUDE.md still pointed `/scratch` after the storage moved to VAST project storage (#662–#663). Each was caught by a consistency-review session and fixed.

**The `umgebungsanalyse.md` audit (Jun 4).** Session S149 produced a deliberate debt ledger — a "comprehensive development environment audit" (#614) that documented "Four Concrete Bugs and Three Structural Issues" (#672). Notably, one of its "bugs" turned out to be phantom: the episode-length bug "was based on a stale docstring" (#650, #673) — the code was already correct. This is debt-tracking honest enough to retract its own findings.

**Checkpoint-resume not implemented.** Documented as a known gap (#656) rather than silently broken.

**The stray-XML-tag debt.** The docs restructure left tool-artifact `</content>` tags in five files (#964), cleaned up in #966 — debt created and repaid inside one session.

---

## 6. Challenges and Debugging Sagas

**Saga 1 — GPU / RT-core compatibility.** The longest-running constraint, spanning the whole history. Isaac Sim 4.x dropped Turing support, so the KISSKI Quadro RTX 5000 is too old (#253, #652), while the A100/H100 partitions lack RT-cores entirely (#653). The blunt conclusion (#653): **KISSKI has no single GPU satisfying both Isaac Sim's requirements**. This forced sim eval and RL onto vast.ai with explicit GPU guards (Ampere+ with RT-cores: L40, RTX 4090, A6000). On Jun 15 it drove a real procurement exercise — querying the vast.ai market and finding **only 3 L40S instances available** that met the RL requirements (#3791).

**Saga 2 — the black-hands USD binding (Jun 3–4).** A four-act tragedy of USD composition. Act 1: recolor script written, "succeeds" with 16 bindings (#330). Act 2: discovered the **instance-root binding does NOT render — hands stayed white** (#367). Act 3: de-instance + direct mesh binding (#368, #371). Act 4: a deeper bug — **the black material loses to a pre-existing `material_white` on the mesh due to USD layer precedence** (#372), finally resolved with a `strongerThanDescendants` binding (#373, #375). The fix then had to be threaded through *every* deployment target's venv, because `pxr`/`usd-core` wasn't installed in the right Python (#384–#388, an isolated `/opt/usdtool` venv on KISSKI).

**Saga 3 — camera calibration / domain gap (Jun 4–5).** Twelve overlay iterations. The deepest single bug was a **wrist-camera roll error** — wrong `world_up` in a rotated link frame (#751) — fixed by setting both wrist cams to `world_up=(0,1,0)` (#753), then fine-tuned and partially reverted across iterations 10–12 (#754–#760). The FOV bounced 69°→90°→75° (#713, #720). This saga produced the project's *most expensive* memories by discovery cost (see §8) because each iteration loaded full 640×480 overlay panels for visual inspection.

**Saga 4 — flash-attn / venv / torch conflicts in Isaac Sim Python (Jun 15).** Detailed in §3. Beyond flash-attn itself, getting GR00T to import inside Isaac Sim's Python required mapping **7 missing runtime deps** (`lmdb` top-level, `av` lazy, `tyro` blocking all model imports via the config init chain, #3903–#3922) and installing them *without* breaking Isaac Sim's pinned versions — a delicate dependency-resolution puzzle ending in a validated import chain (#3921) and a pushed image (#3928).

**Saga 5 — the W&B MCP server.** Less glamorous but most persistent: the `wandb-mcp-server` doesn't exist on PyPI (#454), must be installed from GitHub as a uv tool (#459), and then crashes on a missing `wandb_gql` module that isn't a real package (#461, #465). The eventual fix was a hand-written shim re-exporting `gql` (#468–#470). It broke and was re-fixed across at least four separate sessions.

---

## 7. Memory and Continuity

The persistent memory system was, in this project, less a "recall engine" and more a **continuity substrate** for an extraordinarily fragmented workflow — 160 sessions in 14 days, many lasting only minutes. Explicit recall queries were rare (the database shows essentially one observation whose narrative references prior-session memory). The value came overwhelmingly from *passive* context injection.

The clearest evidence is in how observations cite each other across time. Observation **#653 explicitly cites #253** — "Memory Observation 253 Confirms KISSKI Has No GPU" — a day after #253 was written, letting a documentation-review session reach a hardware conclusion without re-deriving it. The domain-gap finding (#292, Jun 3) is referenced as settled fact through Jun 16 (#4154, #4157), anchoring the entire RL-as-path-forward decision (#603, #2727) without re-litigating the diagnosis each session.

The fragmented camera-calibration work (twelve iterations across two days, S173–S188) would have been nearly impossible to coordinate without memory: each iteration's exact config values and overlay verdicts were captured (#713–#760), so the next short session could resume mid-calibration rather than re-establishing where things stood. Similarly, the repeated "verify the sim setup" requests (S93–S100, all phrased nearly identically — *"Überprüfe ob bei der aktuellen Sim alles passt"*) leaned on accumulated state about what had already been fixed.

The project also used memory *reflexively* — many of the earliest sessions were spent diagnosing claude-mem itself (S2–S37), and the final observation in the record (#4254, Jun 16) is a worker health check, bookending the history with meta-maintenance.

---

## 8. Token Economics & Memory ROI

### Headline figures

| Metric | Value |
|---|---|
| Total discovery tokens | **13,579,861** |
| Distinct memory sessions | **160** |
| Observations (total) | **1,274** |
| Observations with discovery cost | 1,274 |
| Avg discovery tokens / obs | **10,659** |
| Avg "read" size / obs (title+subtitle+narrative+facts)/4 | **362 tokens** |
| Total read tokens invested | **460,835** |

### Top 5 most expensive observations (highest-value memories)

| ID | Discovery tokens | Title |
|---|---|---|
| #1053 | 320,036 | Visual Domain Gap Confirmed Between Real and Sim Camera Views |
| #719 | 309,998 | Visuelle Inspektion aller vier Kamera-Overlay-Panels durchgeführt |
| #725 | 306,701 | All Four Camera Overlay Panels Visually Inspected for New Run |
| #391 | 305,218 | Open-Loop Replay Frame-by-Frame Visual Analysis Underway for Arm-Twisting Diagnosis |
| #731 | 298,609 | All Four Camera Overlay Panels Visually Inspected for Run 202606002 |

Every one of the five priciest memories is an **image-analysis observation** — loading full camera frames or overlay panels into context. This is the literal cost of the camera-calibration and domain-gap sagas (§6), and it explains why Jun 4 dominates the discovery-token budget. These are also the *highest-value* memories: each encodes a visual diagnosis (the domain gap, the arm-twisting motion pattern, the overlay alignment) that would otherwise require re-loading and re-inspecting large images from scratch every time the question recurred.

### Daily breakdown

| Day | Observations | Discovery tokens | Sessions |
|---|---|---|---|
| 2026-06-02 | 182 | 1,002,850 | 13 |
| 2026-06-03 | 213 | 2,560,335 | 32 |
| 2026-06-04 | **394** | **5,887,823** | **60** |
| 2026-06-05 | 280 | 2,863,994 | 34 |
| 2026-06-10 | 17 | 179,060 | 1 |
| 2026-06-12 | 93 | 455,540 | 9 |
| 2026-06-15 | 59 | 195,754 | 7 |
| 2026-06-16 | 36 | 434,505 | 4 |

### Weekly rollup

| ISO week | Observations | Discovery tokens | Sessions |
|---|---|---|---|
| Week of Jun 2–8 (build & debug sprint) | 1,069 | 12,315,002 | 139 |
| Week of Jun 9–16 (RL + RoboCasa + thesis) | 205 | 1,264,859 | 21 |

The first week consumed **91%** of all discovery tokens — the heavy image-analysis and camera-calibration work front-loaded almost the entire budget.

### ROI computation (showing arithmetic)

Using the prescribed model:

- **Passive recall savings** = sessions × (value of a 50-observation context window) × 0.30
  - Value of a 50-obs window ≈ 50 × avg_read = 50 × 362 = **18,100 tokens**
  - = 160 × 18,100 × 0.30 = **868,800 tokens**
- **Explicit recall savings** ≈ 10,000 tokens/query × 1 explicit-recall observation = **10,000 tokens**
- **Total savings** ≈ 868,800 + 10,000 = **878,800 tokens**
- **Net ROI** = total_savings / total_read_tokens_invested = 878,800 / 460,835 = **≈ 1.9×**

In words: for every token spent storing and reading memory, the system returned roughly **1.9 tokens** of avoided re-discovery — and that conservative figure counts only passive injection at a 30% hit rate plus a single explicit recall. The qualitative continuity value (resuming twelve-iteration calibration loops across 60 micro-sessions in a single day) is not captured by the token arithmetic and is almost certainly larger.

---

## 9. Timeline Statistics

- **Date range:** 2026-06-02 14:49 → 2026-06-16 12:52 (GMT) — **14 calendar days**, 8 active days.
- **Total observations:** 1,274 across **160 sessions**.
- **Most active day:** Jun 4 (394 observations, 60 sessions) — the grasp-physics + baseline-pipeline + domain-gap day.
- **Quietest active day:** Jun 10 (17 observations, 1 session) — a focused PD-controller comparison.

### Observation type breakdown

| Type | Count | Share |
|---|---|---|
| discovery (○) | 831 | 65% |
| change (✓) | 197 | 15% |
| feature (◆) | 127 | 10% |
| bugfix (●) | 93 | 7% |
| decision (⚖) | 22 | 2% |
| refactor (↻) | 4 | <1% |

The profile is heavily **discovery-dominated** (two-thirds), consistent with a project where understanding the existing GR00T fork, the Isaac Lab internals, and the failure modes mattered more than writing net-new code. The 197 `change` and 127 `feature` observations reflect the steady stream of documentation edits, config tweaks, and the incremental build-out of multi-GPU, vision-encoder, baseline, and RL paths. Only 4 refactors — this was an *additive* project, layering capability onto a fork rather than reshaping it.

**Longest / densest sessions** by observation count include the LaTeX thesis sprint (S233–S243 on Jun 5, dozens of observations on figures and captions) and the Jun 12 RL-infrastructure session S448 (#3613–#3653, ~40 observations building train/test split, augmentation, shaped reward, the FPO trainer, and the RL entrypoint in one pass).

---

## 10. Lessons and Meta-Observations

A new developer joining this project would absorb a handful of hard-won truths:

1. **0% can be the correct answer.** The project's central scientific lesson is that a model with near-perfect training loss can score 0% in closed-loop sim *and that this is expected*, because the frozen vision encoder never saw simulator renderings (#841, #842). Recognizing this — rather than chasing it as a bug — is what turned a failed eval into a thesis result. The corollary, learned the hard way on Jun 5 (#1029, #2725), is that **naively fine-tuning the vision encoder on real-only data makes closed-loop *worse*** — it collapsed the policy. The domain gap is not solved by unfreezing; it is the subject of the project's future RL work.

2. **Hardware constraints are architectural constraints.** The RT-core requirement (§6, Saga 1) shaped everything: two Docker images, the vast.ai-vs-KISSKI split, and a fail-fast guard so RL never silently runs on the wrong GPU (#3642, #3646). Decide where each workload can physically run *before* designing the pipeline.

3. **Convention mismatches hide in plain sight.** The single most consequential code bug was a finger-joint **sign inversion between the dataset and the USD asset** (#485) — invisible in training, fatal in sim. Mixed action representations (relative arms, absolute hands) are a second such trap. When a model "works in replay but freezes closed-loop," suspect a frame/convention mismatch before suspecting the model.

4. **Simulation fidelity is a calibration discipline, not a one-time setup.** Twelve camera-overlay iterations (§6, Saga 3) say it plainly: matching sim cameras to a real dataset is iterative visual labor. The overlay-against-reference methodology (`overlay_camera_check.py`) was the workhorse, and it was worth documenting (#717, §16 of the notes).

5. **Documentation is part of the build.** An unusually large share of effort went into keeping `docs/`, `CLAUDE.md`, and the LaTeX thesis synchronized with reality — and it repeatedly caught real debt (stale defaults, inactive splits, broken links). The recurring "consistency sweep" after every feature is a habit worth copying.

6. **Test the environment, not just the model.** The final arc (Jun 15–16) pivots to a humbler goal: a **RoboCasa GR-1 zero-shot reference eval** (#4161, #4194) to *validate the dev environment itself* — because the base model can't zero-shot block-stacking (#4157), but it *can* hit ~47.6% on RoboCasa GR-1 tabletop (#4164). When your own task gives 0%, find a task with a known-good baseline to prove your pipeline isn't the problem. That instinct — distrust the harness before celebrating or mourning the result — is the maturity the project arrived at by its final entries.

The throughline of all fourteen days is intellectual honesty: every dead end (vision-encoder fine-tuning, the phantom episode-length bug, the wrong checkpoint uploaded at step 110000 instead of 175000 at #476) is recorded as plainly as every success. That candor, preserved in 1,274 observations, is what makes this memory record genuinely useful to whoever picks up the work next.
