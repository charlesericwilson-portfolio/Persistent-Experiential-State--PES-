# Persistent Experiential State (PES)

> **Status: Ongoing Experimental Research — Early Prototype**

Persistent Experiential State (PES) is an ongoing attempt to determine whether completed agent experiences can produce persistent, history-dependent changes in future model behavior without conventional retraining and without retrieving the original experiences into context.

This repository documents the experiment as it develops, including unsuccessful approaches, intermediate results, architectural changes, and working prototypes.

The central hypothesis is:

> **Episodic memory remembers what happened. Experiential state represents how what happened changed future behavior.**

PES combines existing techniques—including semantic embeddings, attention, persistent state, and low-rank model modulation—in an experimental architecture intended to test whether an agent can gradually adapt from the consequences of its own completed workflows.

**PES has not yet been demonstrated to produce persistent behavioral adaptation.**

The purpose of this repository is to find out whether it can.

---

## Research Goal

The long-term goal is an agent that can become better adapted to the environment in which it operates through accumulated experience.

Instead of periodically collecting new training data and retraining or fine-tuning the model, PES investigates whether completed experiences can update a compact persistent state during inference.

Conceptually:

```text
experience
    ↓
experience representation e
    ↓
persistent state update
    ↓
p(n+1)
    ↓
low-rank model modulation
    ↓
future behavior
```

A successful implementation would allow two initially identical agents to gradually behave differently because they accumulated different experiences.

The eventual experimental test is therefore:

```text
Clone A ── experiences X ──→ persistent state A
Clone B ── no X ──────────→ persistent state B

                    ↓

             same novel task Y

                    ↓

       compare resulting behavior
```

The original experience X must **not** be retrieved into the model's context during this test.

If the agents systematically behave differently in ways predicted by their different histories, PES has produced a measurable experiential effect.

---

# v-1: Event-Based Experience Representation

The first prototype treats a completed agent trajectory as an ordered sequence of events:

```text
USER
  ↓
ASSISTANT
  ↓
TOOL
  ↓
ASSISTANT
  ↓
TOOL
  ↓
...
  ↓
FINAL ASSISTANT
```

Every event is embedded independently using a role-aware semantic embedding model.

For an experience containing `T` events:

```text
H = [h1, h2, ... hT]

H ∈ R^(T × 1024)
```

Each event embedding has 1,024 dimensions.

Relative positional information is then added so the encoder can distinguish where events occurred within the completed trajectory.

The sequence is processed in both directions.

### Forward attention

Each event can attend to itself and events that occurred before it.

```text
past ──────────→ event
```

This represents:

> What does this event mean in relation to what led to it?

### Backward attention

Each event can attend to itself and events that occurred after it.

```text
event ─────────→ future
```

This represents:

> What does this event mean in relation to what happened because of or after it?

The directional representations are currently combined and pooled into a normalized 1,024-dimensional experience vector:

```text
e ∈ R^1024
```

---

# Preliminary v-1 Results

The initial experiment processed **200 complete agent experiences** containing **3,735 individual messages/events**.

All 200 trajectories were successfully converted into fixed-size experience vectors.

Initial cosine-similarity testing produced strong clustering between repeated workflow families.

Examples included approximately:

```text
0.9925  action-item extraction ↔ action-item extraction
0.9921  git backup/release     ↔ git backup/release
0.9918  web scraping           ↔ web scraping
0.9914  ffmpeg workflow        ↔ ffmpeg workflow
```

Some structurally different workflows were substantially farther apart.

For example:

```text
~0.69–0.72

reasoning without tools
        ↕
several tool-heavy workflows
```

These results suggest that the v-1 representation contains coherent information and strongly preserves workflow semantics.

They also exposed an important limitation.

## v-1 Limitation

Two experiences involving the same type of workflow can remain extremely similar even when the actual sequence of consequences and adaptations differs.

For PES, knowing:

> "This was an ffmpeg workflow."

is less important than knowing:

> "This action failed in this way, which caused the agent to change its next action, and that change succeeded."

The current representation appears likely to be overly influenced by **what the workflow was about**.

The next version will explicitly represent **what happened between events**.

---

# Next Step: The Beads and the String

The v-1 encoder represents the **beads** well.

```text
●     ●     ●     ●     ●
```

Each bead is an event:

```text
user intent
assistant decision
tool action
tool result
next decision
final outcome
```

But an experience is not only a collection of events.

It also contains the **string connecting them**:

```text
● ───→ ● ───→ ● ───→ ● ───→ ●
```

For PES, those transitions may contain the most important information.

The next prototype will therefore preserve both.

## 1. Action → Result

Combine an assistant tool action with the resulting tool output.

```text
CALL:
ffmpeg -i input.mp4 output.mp4

RESULT:
codec error
```

This represents:

> **I did X, and the environment responded with Y.**

---

## 2. Result → Next Action

Combine a tool result with the next action selected by the assistant.

```text
RESULT:
codec error

NEXT ACTION:
ffprobe input.mp4
```

This represents:

> **Y happened, so I did Z next.**

This transition is particularly important because it captures behavioral adaptation inside an experience.

Compare:

```text
error
   ↓
repeat same action
```

with:

```text
error
   ↓
inspect problem
   ↓
change strategy
```

The workflow topic may be identical while the experiential structure is very different.

---

## 3. Intent → Outcome

The initial user intent will also be connected to the terminal outcome.

```text
USER INTENT
      ↓
FINAL OUTCOME
```

This provides a global anchor describing what the experience was attempting to accomplish and how it ultimately ended.

---

# Proposed v-2 Representation

The next representation will therefore contain both event information and transition information.

Conceptually:

```text
                    EXPERIENCE

      BEADS                           STRING

 individual events              event relationships

 user                            action → result
 assistant                       result → next action
 tool                            intent → outcome
 assistant
 tool
   │                                   │
   └──────────────┬────────────────────┘
                  ↓
          experience encoder
                  ↓
                  e
```

The objective is **not** to discard individual event representations.

Instead, transition representations should prevent generic workflow semantics from overwhelming the consequences and adaptations that distinguish one experience from another.

---

# Next Experiment

The immediate experiment will compare trajectories belonging to the **same workflow family**.

For example:

```text
Experience A

ffmpeg command
      ↓
success
      ↓
complete
```

versus:

```text
Experience B

ffmpeg command
      ↓
codec failure
      ↓
inspect media
      ↓
change codec arguments
      ↓
success
      ↓
complete
```

The workflow topic is intentionally held approximately constant.

What changes is the experience.

The test will compare the similarity produced by:

```text
v-1 event-dominant encoder
```

against:

```text
v-2 event + transition encoder
```

If v-2 separates experiences with meaningfully different consequence/adaptation structures while keeping genuinely similar trajectories close together, it will provide evidence that the representation is capturing more than workflow topic.

---

# Future Persistent State

Once experience representation has been validated, PES will introduce persistent state.

A preliminary formulation is:

```text
Δp_n = Gθ(e_n, p_n)

p_(n+1) = λp_n + ηΔp_n
```

where:

* `e_n` = completed experience representation
* `p_n` = current persistent experiential state
* `Gθ` = state-update mechanism
* `η` = experiential learning strength
* `λ` = persistence / forgetting factor

The state may eventually modulate a frozen model through a low-rank mechanism such as:

```text
ΔW = B diag(p) A

W' = W + ΔW
```

This portion of PES has **not yet been experimentally validated**.

---

# What Would Count as Success?

PES does not necessarily require a large behavioral change from a single experience.

A weaker but directionally consistent signal may accumulate across repeated experiences.

For example, if an ideal mechanism required five reinforcing experiences to establish a useful behavioral bias while PES required ten, the system would be less sample-efficient but could still satisfy the underlying objective.

The critical distinction is:

```text
weak signal + consistent direction
                ≠
strong signal + unstable direction
```

Success requires persistent changes to be:

* history-dependent,
* directionally stable,
* behaviorally useful,
* capable of adapting as experience changes,
* and observable without retrieving the original experience into context.

---

# Current Status

Completed:

```text
[x] Complete trajectory ingestion
[x] Role-aware event embeddings
[x] 1024-dimensional event representation
[x] Relative positional encoding
[x] Forward masked attention
[x] Backward masked attention
[x] Fixed-size experience vector
[x] 200-experience prototype run
[x] Cross-experience similarity analysis
[x] Identification of workflow-semantic dominance
```

Next:

```text
[ ] Action → result embeddings
[ ] Result → next-action embeddings
[ ] Intent → outcome embedding
[ ] Beads + string experience representation
[ ] Same-workflow controlled comparison
[ ] Order/reversal sensitivity test
[ ] Failure/recovery sensitivity test
```

Later:

```text
[ ] Persistent state p
[ ] State update mechanism
[ ] Stability / forgetting experiments
[ ] Low-rank model modulation
[ ] Clone A / Clone B behavioral evaluation
[ ] Retrieval-independent adaptation test
```

---

# Research Notes

This repository intentionally preserves intermediate and unsuccessful approaches.

v-1 is not being removed because v-2 changes the architecture. The reason for the change is itself an experimental result:

> **v-1 demonstrated strong workflow-semantic clustering, but PES requires sensitivity to the relationships between decisions, consequences, adaptations, and outcomes.**

The project will continue to document those changes as the hypothesis is tested.

---

## Preliminary Report

The root of this repository includes:

`PES_vMinus1_Preliminary_Report.docx`

This document records the methodology, architecture, observations, limitations, and research status at the v-1 checkpoint.

---

## License

Apache License 2.0

---

**Persistent Experiential State is experimental research. No claim is currently made that PES produces useful persistent behavioral adaptation.**
This is being done with the help of AI Grok SpacexAI and ChatGPT Open-AI
