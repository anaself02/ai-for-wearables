# Architecture: sense - think - act

```mermaid
flowchart LR
  subgraph SENSE
    W["WESAD wrist BVP, 64 Hz<br/>15 subjects (data/WESAD, git-ignored)"]
    F["Fitbit heart rate<br/>Google Takeout / Health API (private)"]
  end
  subgraph FEATURES["feature extraction (scripts/, models/features.py)"]
    HR["beat-level HR: band-pass, peaks,<br/>IBI median filter (5), 40-180 bpm"]
    WIN["60 s windows, 30 s step<br/>5 HR features (+ 35 other channels)"]
    R2["R2: beats averaged in 3 s bins<br/>(eval/resolution.py)"]
  end
  subgraph THINK
    FT[("results/features.csv<br/>1015 windows, 15 subjects")]
    EV["eval/: nested LOSO, SVM / RF / MLP,<br/>seeds 0-4, bootstrap 6322"]
    INT["eval/interaction: model x resolution"]
    ERR["eval/errors: exploratory error analysis"]
    SVM["models/svm_hr_only.joblib<br/>(default hyperparameters)"]
  end
  subgraph ACT
    DEMO["demo/run_demo.py:<br/>LOSO tables + one scored Fitbit day"]
    DAY["daily summary (10 allowlisted fields)<br/>act/schema.py"]
    TPL["arm A: deterministic template"]
    LLM["arm B: Gemini message<br/>(act/llm.py, key in .env)"]
    RUB["rubric: numbers, medical terms,<br/>caution, length (act/rubric)"]
  end
  W --> HR --> WIN --> FT
  FT --> EV --> INT
  EV --> ERR
  HR --> R2 --> INT
  FT --> SVM
  F --> DEMO
  SVM --> DEMO
  SVM --> DAY
  F --> DAY
  DAY --> TPL --> RUB
  DAY --> LLM --> RUB
  PRE["PRE_DECLARED.md<br/>(append-only protocol)"] -.-> EV
  PRE -.-> INT
  PRE -.-> RUB
```

- **Sense:** WESAD gives labelled lab data (baseline, stress, amusement); a Fitbit day is unlabelled except for
  a few self-labelled sessions held in the separate dashboard project.
- **Think:** only heart-rate features go into the models, so what is evaluated is what a consumer wrist wearable
  can provide. Labels from personal sessions only grade predictions; they never train or tune anything.
- **Act:** the daily summary is turned into a short message by a template or by an LLM; both are scored with the
  same rubric. Only ten allowlisted, de-identified fields leave the machine, and only synthetic inputs have
  been sent to the LLM so far.
