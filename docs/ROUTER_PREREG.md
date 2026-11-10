# Router: pass bars and results (10 Nov 2026)

The router turns a heard sentence into one of LIBERO-Goal's 10 commands, or refuses it. Code: `voice/router.py` (frozen), scored by `voice/score_router.py`. The pass bars below were written before any sentence was seen. This published copy has edited paths and wording; the bars, fingerprints and times are unchanged.

## Pass bars (written 10 Nov 2026, before any sentence was seen)

**Question.** Can a local router turn natural wording into the right one of LIBERO-Goal's 10 commands, refuse everything else, and usually skip the language model?

**Data.** 130 rephrased commands (13 per command) and 45 requests the robot can't do. They were machine-generated, in a separate language-model session that never saw the router. Labels come from how each sentence was written, so nobody hand-labels anything.

**Split, fixed seed 20261110** (`voice/split_sentences.py`).
- Dev: 3 per command + 15 out-of-scope. Used to tune.
- Test: 10 per command + 30 out-of-scope. Scored once, after the router and its thresholds are frozen.

**Baseline.** The language model (Qwen3.5-4B) answering every sentence, with no fast path.

**Pass bars on the test set.**

| Measure | Bar |
|---|---|
| Right command, of the 100 rephrased commands | at least 90% |
| Wrong command (the worst failure, the robot does the wrong thing) | at most 3% |
| Correct refusal, of the 30 out-of-scope sentences | at least 80% |
| Rephrased commands answered without the language model | at least 50% |
| Median time per sentence | lower than the baseline |

**Rules.**
- Nobody reads the test sentences before scoring. The router writes them nowhere and prints only totals until it is frozen.
- The test set is scored once. If a bar fails, it is reported as failed.
- Any fix after that is checked on freshly written sentences, not on this test set.
- These results are about typed text. Whisper's errors on a real voice are checked separately ([RESULTS_VOICE.md](RESULTS_VOICE.md)).

**Test-set fingerprint (sha256 of `data/test.json`, recorded before the router was tuned):** `e34ec295559317d28ffcd912a21ed4793735bd8e4f2610c073e462cf4bba2ea3`

**Amendment before test scoring (10 Nov 2026).** On the dev sentences, a meaning-match level using embeddings answered only 23% of rephrased commands safely, so it was replaced by word rules. The router now has two levels: word rules, then the language model. The bars above are unchanged. Frozen `router.py` sha256: `cc50ba1ec46fb175349cabced4bfc6f4c73eba616fb660b91a65df4e526e2c74`

## The sentence sets are held back

The sentence files (`data/test.json`, `data/dev.json` and the unsplit `data/sentences_raw.json`) are not in this repository, and neither are the per-sentence scoring rows. Only the summaries are: `voice/results/router/{test,dev}_{router,llm}.json`. The sets are held back so that future router versions can be tested on sentences they have never seen. The fingerprint above shows which test file was scored. To score a router on your own sentences, write a file in the same format (`in_scope` items with `id`, `text`, `command_id`; `out_of_scope` items with `id`, `text`) as `voice/data/<split>.json` and run `python score_router.py <split> router`.

## Results (test set, typed sentences)

The test set was scored once, with `router.py` frozen.

| Measure | Bar | Router (word rules, then language model) | Language model alone |
|---|---|---|---|
| Right command, of 100 rephrased commands | at least 90% | **100%** | 94% |
| Wrong command | at most 3% | **0%** | 1% |
| Correct refusal, of 30 out-of-scope sentences | at least 80% | **96.7%** (29 of 30) | 100% |
| Rephrased commands answered without the language model | at least 50% | **88%** | 0% |
| Median time per sentence | below the baseline | **0.011 ms** | 103.9 ms |
| 99th-percentile time (p99) per sentence | none | 113 ms | 113 ms |

Peak GPU memory: 9.3 GB for the language model, 10.9 GB with Whisper loaded.

**Caveats.**
- One generator wrote all the sentences. The synonyms in the word rules overlap with examples given to that generator, so 100% is probably optimistic.
- The router made one wrong call: an out-of-scope stove request was routed to "turn on the stove" when it should have refused. It is reported, not fixed, because this test set can't be reused for a fix.
- The first design, a meaning-match level using embeddings, was dropped on the dev sentences because it answered only 23% safely.
- Before the voice runs, Whisper was checked only on a computer-generated voice: median 62 ms on 1.4 to 2.3 s clips, and it misheard 2 of 5 lines. The real-voice results are in [RESULTS_VOICE.md](RESULTS_VOICE.md).
