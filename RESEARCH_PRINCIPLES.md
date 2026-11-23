# Jordan's ML research principles

Distilled on 2026-10-10 from every session since 2026-09-18 (three reviews: the user's ideas, the assistant's proposals and
their outcomes, the user's directions). Each principle has dated quotes of the user. "Reading" lines are the assistant's
interpretation, not the user's words. Dates are UTC.

## A. What to build

1. **The model teaches itself.** A frozen copy of the model's own weights is the judge or teacher.
   - 09-25 "pixels are entirely learned through frozen encoder judgement"
   - 10-10 "gradient should flow through yet not update at the judging level"
2. **Never pull a representation with more information toward one with less.** Detach the richer side.
   - 09-27 "principle of not pulling more-information representations towards less information representations"
   - 10-08 "teacher with stop grad gets lower t"
3. **Only subtractive transformations.** Mask, crop, noise, patch drop.
   - 10-02 "only apply subtractive/erasure transformations ... to not open the door to the infinitude of additive transformations"
   - 09-28 "I don't want the network to learn reflections."
4. **Init and parameter-free changes before new modules, losses or schedules.**
   - 09-25 "a careful-init only change rather than a new skip path projection"
   - 09-30 "i don't want to add another rmsnorm, just init the lin projections small"
   - 10-10 "Prioritise changes that do note increase param count"
   - 10-11 one final un-embed "instead of training 2 extra unembeds"
5. **No crude remedies.** 09-25 "clipping grads/weight updates is crude and dangerous"
6. **Assume the network will cheat. Close the leak before trusting a gain.**
   - 09-28 "Needs to prevent secret code sharing between the encoder and decoder"
   - 09-29 "if that opens the door to high frequency cheat codes"
7. **Resolution is a free axis.** The weights do not depend on the patch count.
   - 10-03 "train first at lower resolutions with fewer patches and then increase resolution"
   - 10-11 "1 2 4 8 16 32 64 128. one doubling per layer readouts"
8. **Stability must hold across scale and learning rate.** Fragility is a design flaw.
   - 09-25 "25M feels like balancing on a tightrope"; "truly stabilizes across higher LR and beta2"
9. **Carry a mechanism that works into the next project.**
   - 10-09 "I wonder if any of them could apply to the unified pixel diffusion context?"
   - Reading: this transfer gave the best FID so far (114.19 against 135.62).

## B. How to test

10. **Diagnose the failure before the fix. Measure a suspected problem before a remedy.**
    - 09-25 "First, investigate the models that failed ... Identifying the failure mode more clearly will help designing a stabilizing architecture change"
    - 10-11 "start first by just measuring. don't institute any gradient balancing as we don't know if it'll be an issue or not."
    - But sized to the decision: 09-25 "mmm we have enough of a diagnostic"
11. **Every result prompts reasoning and a new idea. No repeats, no sweeps around a flat result.**
    - 10-11 "Each run that fails/improves should prompt an investigation ... figure out what is limiting the quality and think about NEW things to try"
    - Record: 87 loop tests of weights, shares and decays gave about 5 passes. The large gains came from new mechanisms.
12. **Name the follow-up for pass and for fail in advance. Combine only changes that win alone.**
    - 09-27 "Combine 2 and 3 if they're both successful."
    - 10-10 "if this experiment fails, add change a next, if it succeds, try change b on top"
13. **One change per run, against one existing control, at equal compute.**
    - 09-27 "only change the new z shortcut residual"
    - 10-10 "don't run the control multiple times."
    - 10-11 "exactly like a control we already have so we don't need to run a bunch more controls"
    - 10-10 "2 is enough" (seeds)
14. **Prove the treatment is applied. Results equal to the baseline are a bug.**
    - 10-05 "are you sure they're freezing differently?"
    - 10-06 "DON'T tell me your test wasn't a test...again..."
    - 10-08 "make sure new lr actually stuck"
    - Do not conclude from an early step: 10-11 "ofc grad will be aligned in first steps when everything is still way off"
15. **Test a suspected bug in the foundation first. Retire and label the tainted runs.**
    - 09-25 "hard to trust the insights now..."
    - 10-09 "the loss weighting potentially being flipped is a much bigger issue than the noise schedule"
16. **Judge by the right measure, not by training loss.**
    - 09-28 "the clean across runs is misleading due to different training schemes"
    - 10-07 "generative pixel flow models can have identical losses and different generative properties"
    - Reading: scoring FID from EMA weights is the assistant's rule, not a quote of the user.
17. **Velocity: small model, short boxes, branch from the best checkpoint. Then confirm the stack from scratch and at scale.**
    - 09-27 "drop model size down to 11M to accelerate testing velocity"
    - 10-10 "20 minute testing cycles. No runs from scratch."
    - 10-10 "the best recipe but starting from scratch, running out to 200 k steps"; "New long run is new checkpoint baseline"
18. **Stack small wins.** 10-10 "Add small wins."
    - Reading: the fixed pass margin of 1.5 FID dropped consistent smaller gains (weights 4 and 6). It may be too strict.
19. **Let a slow but interesting run finish.** The user's judgment beat the assistant's kill gates twice.
    - 09-29 "continue L1 linskip even though it failed your limit." (it became the best arm)
    - 10-08 "It's just learning slower than normal."
    - But end a flat one: 10-05 "plateaued pretty hard so please stop it and begin the next test"
20. **Step speed is a design constraint.** Nothing that breaks the compiled step or changes patch counts.
    - 09-28 "that'd make every crop have a different compiled forward and crush speed"
    - 10-05 "yes this is too slow. something is wrong."

## C. How to work with the user

21. **GPUs: never take one the user uses, never leave a free one idle.**
    - 10-11 "no touching 3090 till I say."; "Keep GPU busy don't stop at 2:10 am."
22. **Short, ordered replies and names.**
    - 10-11 "I didn't read your whole wall of text. keep conversation more concise"
    - 10-09 "the run descriptions are getting too long and unstructured"

## Changes of mind (for context)

- Warm starts: 09-18 "no warm start"; from 09-26 continuation tests; 10-10 "No runs from scratch" for the loop.
  Reading: continuation for screening, from scratch for confirmation or when a change alters early training.
- Seeds: 10-10 "go ahead with more seeds", then 22 minutes later "no more extra seeds 2 is enough".

## Untested ideas of the user (found in the record)

- 09-27 `mse(z.detach(), mz)` (X5): postponed, no result.
- 09-29 view pull from the full-image reconstruction to the raw image `z.detach()`.
- 10-11 `grad = 0.5 x my_readout + 0.5 x downstream`: waits for the gradient measure.
- 09-30 open question: "We never did figure why scaling doesn't improve results".
