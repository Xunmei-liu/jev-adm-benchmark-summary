# Jev-style decision models: inference-time benchmark results

The strongest completed result is **OpenWebRL-4B-SFT with our web-tuned Kev 4B selector: 108/300 Online Mind2Web tasks succeeded (36.00%), compared with 89/300 (29.67%) for SFT alone**. This is an observed gain of **6.33 percentage points, or 19 additional successful tasks**. The later rank-32 selector did not improve on rank 16 online.

## 1. Online action selection: complete 300-task evaluations

All rows use the same frozen set of 300 task IDs. OpenWebRL-4B-SFT generates actions; in selector rows it samples five complete next-action responses **at every visited state**, and the selector chooses one for execution. The actor is not fine-tuned in these experiments.

| Method | Candidates per state | Successes / all tasks | Overall success | Success excluding aborted and confirmed blocked tasks |
|---|---:|---:|---:|---:|
| OpenWebRL-4B-SFT alone | 1 | 89/300 | 29.67% | 89/258 = 34.50% |
| SFT + released OmniJev v1.1 4B selector | 5 | 83/300 | 27.67% | 83/256 = 32.42% |
| SFT + Piotr Scalar BT selector | 5 | 98/300 | 32.67% | 98/238 = 41.18% |
| **SFT + tuned Kev 4B, rank 16, step 1,748** | **5** | **108/300** | **36.00%** | **108/236 = 45.76%** |
| SFT + tuned Kev 4B, rank 32, step 1,748 | 5 | 96/300 | 32.00% | 96/229 = 41.92% |

**Evaluation settings:** candidate sampling temperature 0.7 and top-p 0.8; maximum 30 browser steps; one current screenshot for action selection; GPT-4.1 outcome judge using action history and up to three screenshots. Selector runs do not use GPT as a text-generation helper. This is the OpenWebRL evaluation harness, not the official Online Mind2Web WebJudge implementation.

**Interpretation:** the rank-16 Kev run is 6.33 points above SFT alone, 3.33 points above Scalar BT, and 8.33 points above the released OmniJev selector. Rank 32 is 4.00 points below rank 16, despite its higher offline development accuracy. No statistical significance or repeated-run robustness claim is made. These are live-web runs on matched task IDs, not identical-state or identical-candidate replays; websites, sampled candidates, trajectories, and run conditions can differ.

The all-300 rate should be the headline comparison. The filtered rate removes aborted tasks and separately confirmed blocked pages, leaving a different task subset for each method. Ordinary failures, malformed actions, and step-limit failures remain failures. The filtered rates therefore do not share a matched denominator.

This demonstrates a useful **best-of-five inference-time selection result**. It does not establish a scaling curve over candidate counts: a controlled K=1/2/5/10 sweep has not been reported here. Nor does it isolate the effect of selector quality from the extra actor sampling used by all best-of-five systems.

## 2. What “Jev-style ADM” means in these experiments

- **Tuned Kev:** initialized from released Kev 4B, using a Qwen3.5-4B-Base multimodal backbone, language-side LoRA, and a pointer decision head. It scores five supplied action/thought candidates in one forward pass. It does not generate an action or explanation token by token.
- **Training data:** teacher-labeled action-selection examples derived from OpenWebRL SFT demonstration trajectories; 27,956 training decisions and 3,994 development decisions, with separate 3,994 calibration and 3,993 test decisions. The loss is five-way cross-entropy against the teacher-selected candidate index.
- **Rank-16 configuration:** LoRA rank/alpha 16/32, dropout 0.05, learning rate 5e-5, global batch 16, AdamW, 5% warmup and cosine decay. The selected checkpoint is step 1,748. The visual components and original base weights are frozen.
- **Released OmniJev selector:** a separate, untuned OmniJev v1.1 4B checkpoint using its structured choice interface. It is not our fine-tuned Kev checkpoint. The adapter preserves full candidate text and executes the five-way argmax; native abstention is logged separately.

The Kev split holds out decision inputs, not whole tasks or states: 3,975 of 3,993 test rows share a state with training. Its 61.66% test accuracy measures teacher-index agreement on that split, not unseen-task success. This offline accuracy must not be substituted for the online success rate above.

### Paper-ready Kev training description

We adapt the released Kev-4B model for screenshot-conditioned web-action selection using 27,956 training decisions derived from OpenWebRL SFT demonstration trajectories. Each example pairs a trajectory state (screenshot, task context, and action history) with five candidate action–thought pairs sampled from OpenWebRL-4B-SFT and a GPT-5.5-selected target. We minimize five-way cross-entropy, updating only the language-side LoRA adapters and pointer head while freezing the original multimodal backbone. We use LoRA rank 16, alpha 32, and dropout 0.05, with AdamW, a peak learning rate of 5e-5, weight decay 0.01, and a global batch size of 16. Training follows a maximum three-epoch schedule with 5% warmup and cosine decay, using development NLL for early stopping and checkpoint selection. The selected checkpoint is at step 1,748, approximately one epoch. Our decision-level split reserves 3,994 examples for development, 3,994 for calibration, and 3,993 for testing; states and tasks may overlap across splits.

The original collector describes states extracted from SFT trajectories, repeated five-candidate draws at temperature 0.7, and GPT-5.5 pre-execution selections ([collection protocol](https://github.com/piotr-teterwak/action-reward-models#stage-1--data-generation-data_generation)). The released labels are reused; no new lookahead evidence is collected for this Kev run.

| Additional reproducibility setting | Value |
|---|---|
| Trainable parameters | 33,776,128, including LoRA and pointer head |
| Precision | BF16 backbone, FP32 adapter/head |
| Screenshot maximum area | 262,144 pixels |
| Gradient clipping | Global norm 1.0 |
| Dev evaluation | Every 500 updates and at epoch boundaries |
| Early stopping | Patience 3; minimum NLL improvement 0.002 to reset patience |
| Selected step / actual stopping step | 1,748 / 3,000 |
| Seed | 20261013 |
| Candidate-order augmentation / public replay | Neither in the selected run |

Temperature calibration is fitted only after checkpoint selection, on 3,994 separate calibration decisions: T = 0.91563. It preserves the selected candidate. Test accuracy is 61.66%, calibrated NLL 0.97872, Brier 0.51083, and ECE 2.09%; these are teacher-agreement metrics, not task-success probabilities.

## 3. OmniJev as actor: a separate diagnostic

Here OmniJev v1.1 4B chooses operations and targets directly, using its official browser grid and seven operation choices. GPT-4.1 supplies text for an already selected field or a final answer; it does not choose the operation. A later harness adds explicit Enter and bounded waiting.

| Actor experiment | Recorded successes | Scope |
|---|---:|---|
| Official seven-operation template, two five-task pilots | 0/10 | Small diagnostic task set |
| Seven operations plus Enter and waiting | 0/50 | Fixed subset of the same 300-task pool |
| Targeted rerun after placeholder and iframe repairs | 0/2 | Previously tested tasks; checks repairs only |

These task sets overlap and must not be pooled. In the 50-task run, 49 tasks reached the step limit and one aborted. OmniJev never selected `done` across 1,470 actions. The harness treated step-limit episodes as rule failures instead of independently judging every final screenshot.

Two confirmed execution defects affected the original run: a placeholder check rejected valid field interactions, and iframe hit-testing failed to resolve native dropdowns. The repaired two-task rerun verified that those operations could execute, but navigation/termination and helper mistakes remained. **There is no completed full 50-task rerun with all these repairs**, so 0/50 is a result of that model-plus-harness configuration, not an isolated measurement of model capability.

The separate `goal_met` score was not used to stop these runs. A retrospective inspection suggested two tasks might have been answerable at the first score above 0.5, but this is not a measured 4% success rate. The other crossings included incomplete tasks.

All of these actor results concern **OmniJev v1.1 4B**, not OneJev 1.6. Comparing tuned Kev selection with untuned OmniJev acting changes the model, adaptation, and execution interface; it is not a clean same-checkpoint actor-versus-selector ablation.

## 4. Reverse-KL actor training is not part of the positive inference-time result

Reverse-KL updates OpenWebRL-4B-SFT itself, whereas the selector evaluations above keep that actor fixed. Its held-out teacher top-one agreement improved from roughly 37% to 46%, but the trained step-560 model showed generation-format regressions.

In the retained corrected Hugging Face decoding control on six fixed inputs, the original actor produced 6/6 parseable responses and step 560 produced 3/6; neither hit the generation-length cap. This supports a checkpoint-associated regression beyond the initial serving path, but does not establish its training cause or estimate a benchmark-wide error rate. Do not report offline agreement as an online task-success improvement or claim that reverse KL necessarily causes failure.

## Suggested wording for the paper or discussion

> We evaluated Jev-style structured decision models as inference-time selectors on 300 Online Mind2Web tasks. With the OpenWebRL-4B-SFT actor held fixed, our web-tuned Kev 4B selector selected among five candidate actions per state and achieved 36.0% overall task success, compared with 29.7% for the single-sample actor, 32.7% with Scalar BT, and 27.7% with the released OmniJev selector. These live-web results support task-adapted decision models for action selection, while leaving robustness across runs and candidate budgets to future evaluation. Direct OmniJev acting had 0/50 recorded successes in a separate diagnostic, but confirmed execution defects and termination behavior prevent attributing that result entirely to the model.

## Evidence

Local paths are relative to `/mmfs1/gscratch/krishna/xunmei/`:

- Four complete 300-task aggregates and denominators: `omnijev_selector_pilot_20261020/reports/comparison_with_n1_verified_20261021.json`.
- Rank-32 final aggregate: `kev_rank32_online_20261029/reports/final_comparison.csv`.
- Selector settings, training setup, and exposure audit: `kev_vision_main_20261005/docs/KEV_TUNING_OMNIJEV_RESULTS_20261021.md`.
- Actor configuration and results: `kev_vision_main_20261005/docs/OMNIJEV_ACTOR_WEEKLY_20261029.md`.
- Original actor audit: `omnijev_actor_enter_wait_20261027/reports/manual_review_391777/audit_summary.json`.
- Repaired actor rerun: `omnijev_actor_fieldfix_20261028/reports/review_393906/manual_review.json`.
- Reverse-KL fixed-input control: `kev_actor_online_20261030/reports/vlm_debug_20261105/hf_corrected/summary.json`.
