# Results: SVM, random forest and MLP under one protocol

Protocol: `PRE_DECLARED.md` Amendment 4 (frozen before any nested result). Leave-one-subject-out, 15 subjects, inner subject-grouped tuning, 5 seeds averaged per subject first. Mean and SD are over the 15 subjects; CIs are 95% percentile bootstrap over subjects (10,000 resamples). Pairwise CIs are **not adjusted for multiplicity**.

## Primary: binary, stress vs non-stress

### Macro-F1 per subject (headline)

| model | mean | SD over subjects | 95% bootstrap CI of the mean |
|---|---|---|---|
| majority | 0.413 | 0.004 | [0.411, 0.415] |
| threshold | 0.689 | 0.189 | [0.594, 0.778] |
| svm | 0.731 | 0.183 | [0.639, 0.818] |
| rf | 0.747 | 0.160 | [0.669, 0.826] |
| mlp | 0.760 | 0.157 | [0.679, 0.833] |

### Balanced accuracy per subject

| model | mean | SD over subjects | 95% bootstrap CI of the mean |
|---|---|---|---|
| majority | 0.500 | 0.000 | [0.500, 0.500] |
| threshold | 0.710 | 0.142 | [0.641, 0.779] |
| svm | 0.759 | 0.152 | [0.685, 0.832] |
| rf | 0.753 | 0.151 | [0.681, 0.827] |
| mlp | 0.786 | 0.132 | [0.719, 0.849] |

### AUROC per subject

| model | mean | SD over subjects | 95% bootstrap CI of the mean |
|---|---|---|---|
| majority | 0.500 | 0.000 | [0.500, 0.500] |
| threshold | 0.801 | 0.186 | [0.710, 0.890] |
| svm | 0.869 | 0.134 | [0.798, 0.928] |
| rf | 0.877 | 0.136 | [0.806, 0.937] |
| mlp | 0.898 | 0.126 | [0.830, 0.951] |

### Pairwise differences in macro-F1

| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |
|---|---|---|---|
| mlp - svm | +0.030 | [+0.005, +0.057] | yes |
| mlp - rf | +0.013 | [-0.033, +0.061] | no |
| rf - svm | +0.017 | [-0.023, +0.065] | no |
| mlp - majority | +0.347 | [+0.267, +0.420] | yes |
| mlp - threshold | +0.071 | [+0.008, +0.134] | yes |
| rf - majority | +0.334 | [+0.257, +0.413] | yes |
| rf - threshold | +0.058 | [-0.029, +0.138] | no |
| svm - majority | +0.318 | [+0.227, +0.405] | yes |
| svm - threshold | +0.041 | [-0.032, +0.109] | no |

### Pairwise differences in balanced accuracy

| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |
|---|---|---|---|
| mlp - svm | +0.027 | [+0.007, +0.048] | yes |
| mlp - rf | +0.033 | [-0.004, +0.073] | no |
| rf - svm | -0.007 | [-0.038, +0.027] | no |
| mlp - majority | +0.286 | [+0.219, +0.349] | yes |
| mlp - threshold | +0.076 | [+0.023, +0.130] | yes |
| rf - majority | +0.253 | [+0.181, +0.327] | yes |
| rf - threshold | +0.043 | [-0.025, +0.102] | no |
| svm - majority | +0.259 | [+0.185, +0.332] | yes |
| svm - threshold | +0.050 | [-0.011, +0.109] | no |

### Pairwise differences in AUROC

| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |
|---|---|---|---|
| mlp - svm | +0.029 | [+0.011, +0.051] | yes |
| mlp - rf | +0.021 | [-0.002, +0.052] | no |
| rf - svm | +0.008 | [-0.008, +0.022] | no |
| mlp - majority | +0.398 | [+0.330, +0.451] | yes |
| mlp - threshold | +0.097 | [+0.025, +0.171] | yes |
| rf - majority | +0.377 | [+0.306, +0.437] | yes |
| rf - threshold | +0.076 | [-0.011, +0.159] | no |
| svm - majority | +0.369 | [+0.298, +0.428] | yes |
| svm - threshold | +0.068 | [-0.006, +0.143] | no |

### Stability across the 5 seeds (mean macro-F1 over subjects, per seed)

| model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range |
|---|---|---|---|---|---|---|
| majority | 0.413 | 0.413 | 0.413 | 0.413 | 0.413 | 0.000 |
| threshold | 0.689 | 0.689 | 0.689 | 0.689 | 0.689 | 0.000 |
| svm | 0.724 | 0.719 | 0.727 | 0.746 | 0.738 | 0.027 |
| rf | 0.750 | 0.745 | 0.742 | 0.746 | 0.753 | 0.011 |
| mlp | 0.765 | 0.762 | 0.752 | 0.753 | 0.769 | 0.017 |

### Mean per-class recall (over subjects)

| model | non-stress | stress |
|---|---|---|
| majority | 1.000 | 0.000 |
| threshold | 0.810 | 0.610 |
| svm | 0.768 | 0.751 |
| rf | 0.852 | 0.653 |
| mlp | 0.786 | 0.786 |

### Pooled out-of-fold macro-F1 (the Table I quantity)

| model | pooled macro-F1, mean over 5 seeds | range over seeds | Table I (3-class, HR only, default hyperparameters) |
|---|---|---|---|
| majority | 0.413 | 0.413 to 0.413 | - |
| threshold | 0.706 | 0.706 to 0.706 | - |
| svm | 0.738 | 0.727 to 0.750 | - |
| rf | 0.754 | 0.749 to 0.762 | - |
| mlp | 0.763 | 0.755 to 0.771 | - |

### Chosen hyperparameters

**svm** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__C": 10, "clf__gamma": 0.1}` | 41 |
| `{"clf__C": 1, "clf__gamma": 0.01}` | 11 |
| `{"clf__C": 10, "clf__gamma": "scale"}` | 9 |
| `{"clf__C": 10, "clf__gamma": 0.01}` | 5 |
| `{"clf__C": 100, "clf__gamma": 0.1}` | 4 |
| `{"clf__C": 100, "clf__gamma": 0.01}` | 2 |
| `{"clf__C": 1, "clf__gamma": 0.1}` | 1 |
| `{"clf__C": 1, "clf__gamma": "scale"}` | 1 |
| `{"clf__C": 100, "clf__gamma": "scale"}` | 1 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__C": 10, "clf__gamma": 0.1}` | 2 of 5 |
| S3 | `{"clf__C": 10, "clf__gamma": 0.1}` | 3 of 5 |
| S4 | `{"clf__C": 10, "clf__gamma": 0.1}` | 4 of 5 |
| S5 | `{"clf__C": 10, "clf__gamma": 0.1}` | 3 of 5 |
| S6 | `{"clf__C": 1, "clf__gamma": 0.01}` | 3 of 5 |
| S7 | `{"clf__C": 10, "clf__gamma": 0.1}` | 5 of 5 |
| S8 | `{"clf__C": 10, "clf__gamma": 0.1}` | 4 of 5 |
| S9 | `{"clf__C": 1, "clf__gamma": 0.01}` | 2 of 5 |
| S10 | `{"clf__C": 10, "clf__gamma": 0.01}` | 4 of 5 |
| S11 | `{"clf__C": 10, "clf__gamma": 0.1}` | 4 of 5 |
| S13 | `{"clf__C": 10, "clf__gamma": "scale"}` | 5 of 5 |
| S14 | `{"clf__C": 10, "clf__gamma": 0.1}` | 3 of 5 |
| S15 | `{"clf__C": 10, "clf__gamma": 0.1}` | 2 of 5 |
| S16 | `{"clf__C": 10, "clf__gamma": 0.1}` | 4 of 5 |
| S17 | `{"clf__C": 10, "clf__gamma": 0.1}` | 3 of 5 |

**rf** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 31 |
| `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 30 |
| `{"clf__max_depth": 20, "clf__min_samples_leaf": 1}` | 9 |
| `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 3 |
| `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 2 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 3 of 5 |
| S3 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 2 of 5 |
| S4 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S5 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 4 of 5 |
| S6 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 4 of 5 |
| S7 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S8 | `{"clf__max_depth": 20, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S9 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 5 of 5 |
| S10 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 5 of 5 |
| S11 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 3 of 5 |
| S13 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 3 of 5 |
| S14 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 3 of 5 |
| S15 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S16 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 3 of 5 |
| S17 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 3 of 5 |

**mlp** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 36 |
| `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 29 |
| `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 6 |
| `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 4 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 2 of 5 |
| S3 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 2 of 5 |
| S4 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 4 of 5 |
| S5 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 2 of 5 |
| S6 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S7 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S8 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S9 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 2 of 5 |
| S10 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S11 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 5 of 5 |
| S13 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 4 of 5 |
| S14 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 4 of 5 |
| S15 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 5 of 5 |
| S16 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S17 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |

## Secondary: 3-class (baseline / stress / amusement), WESAD only

### Macro-F1 per subject (headline)

| model | mean | SD over subjects | 95% bootstrap CI of the mean |
|---|---|---|---|
| majority | 0.236 | 0.003 | [0.235, 0.238] |
| threshold | 0.440 | 0.130 | [0.373, 0.501] |
| svm | 0.467 | 0.132 | [0.406, 0.533] |
| rf | 0.482 | 0.158 | [0.410, 0.564] |
| mlp | 0.496 | 0.112 | [0.444, 0.552] |

### Balanced accuracy per subject

| model | mean | SD over subjects | 95% bootstrap CI of the mean |
|---|---|---|---|
| majority | 0.333 | 0.000 | [0.333, 0.333] |
| threshold | 0.488 | 0.105 | [0.436, 0.539] |
| svm | 0.493 | 0.116 | [0.440, 0.552] |
| rf | 0.510 | 0.142 | [0.446, 0.584] |
| mlp | 0.520 | 0.104 | [0.472, 0.573] |

### Pairwise differences in macro-F1

| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |
|---|---|---|---|
| mlp - svm | +0.029 | [+0.000, +0.057] | yes |
| mlp - rf | +0.014 | [-0.026, +0.053] | no |
| rf - svm | +0.015 | [-0.016, +0.046] | no |
| mlp - majority | +0.260 | [+0.208, +0.315] | yes |
| mlp - threshold | +0.057 | [+0.008, +0.109] | yes |
| rf - majority | +0.246 | [+0.174, +0.328] | yes |
| rf - threshold | +0.042 | [-0.020, +0.105] | no |
| svm - majority | +0.231 | [+0.169, +0.296] | yes |
| svm - threshold | +0.028 | [-0.017, +0.074] | no |

### Pairwise differences in balanced accuracy

| comparison (first minus second) | mean difference | 95% CI | CI excludes 0 |
|---|---|---|---|
| mlp - svm | +0.027 | [+0.002, +0.051] | yes |
| mlp - rf | +0.010 | [-0.028, +0.048] | no |
| rf - svm | +0.018 | [-0.013, +0.046] | no |
| mlp - majority | +0.187 | [+0.139, +0.240] | yes |
| mlp - threshold | +0.032 | [-0.009, +0.077] | no |
| rf - majority | +0.177 | [+0.113, +0.251] | yes |
| rf - threshold | +0.023 | [-0.035, +0.077] | no |
| svm - majority | +0.160 | [+0.107, +0.218] | yes |
| svm - threshold | +0.005 | [-0.033, +0.044] | no |

### Stability across the 5 seeds (mean macro-F1 over subjects, per seed)

| model | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | range |
|---|---|---|---|---|---|---|
| majority | 0.236 | 0.236 | 0.236 | 0.236 | 0.236 | 0.000 |
| threshold | 0.440 | 0.440 | 0.440 | 0.440 | 0.440 | 0.000 |
| svm | 0.458 | 0.481 | 0.473 | 0.481 | 0.443 | 0.037 |
| rf | 0.476 | 0.492 | 0.473 | 0.485 | 0.484 | 0.020 |
| mlp | 0.501 | 0.488 | 0.506 | 0.491 | 0.496 | 0.018 |

### Mean per-class recall (over subjects)

| model | baseline | stress | amusement |
|---|---|---|---|
| majority | 1.000 | 0.000 | 0.000 |
| threshold | 0.816 | 0.647 | 0.000 |
| svm | 0.551 | 0.708 | 0.220 |
| rf | 0.636 | 0.709 | 0.186 |
| mlp | 0.518 | 0.734 | 0.308 |

### Pooled out-of-fold macro-F1 (the Table I quantity)

| model | pooled macro-F1, mean over 5 seeds | range over seeds | Table I (3-class, HR only, default hyperparameters) |
|---|---|---|---|
| majority | 0.236 | 0.236 to 0.236 | - |
| threshold | 0.449 | 0.449 to 0.449 | - |
| svm | 0.483 | 0.463 to 0.498 | 0.499 |
| rf | 0.502 | 0.496 to 0.510 | 0.498 |
| mlp | 0.502 | 0.495 to 0.509 | - |

### Chosen hyperparameters

**svm** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__C": 100, "clf__gamma": 0.1}` | 16 |
| `{"clf__C": 10, "clf__gamma": "scale"}` | 13 |
| `{"clf__C": 1, "clf__gamma": 0.01}` | 12 |
| `{"clf__C": 0.1, "clf__gamma": "scale"}` | 9 |
| `{"clf__C": 10, "clf__gamma": 0.1}` | 6 |
| `{"clf__C": 1, "clf__gamma": 0.1}` | 5 |
| `{"clf__C": 0.1, "clf__gamma": 0.1}` | 4 |
| `{"clf__C": 10, "clf__gamma": 0.01}` | 4 |
| `{"clf__C": 1, "clf__gamma": "scale"}` | 4 |
| `{"clf__C": 100, "clf__gamma": 0.01}` | 1 |
| `{"clf__C": 0.1, "clf__gamma": 0.01}` | 1 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__C": 0.1, "clf__gamma": "scale"}` | 2 of 5 |
| S3 | `{"clf__C": 10, "clf__gamma": "scale"}` | 2 of 5 |
| S4 | `{"clf__C": 1, "clf__gamma": 0.1}` | 2 of 5 |
| S5 | `{"clf__C": 10, "clf__gamma": 0.1}` | 2 of 5 |
| S6 | `{"clf__C": 1, "clf__gamma": 0.01}` | 4 of 5 |
| S7 | `{"clf__C": 1, "clf__gamma": 0.01}` | 4 of 5 |
| S8 | `{"clf__C": 100, "clf__gamma": 0.1}` | 2 of 5 |
| S9 | `{"clf__C": 10, "clf__gamma": "scale"}` | 3 of 5 |
| S10 | `{"clf__C": 10, "clf__gamma": 0.01}` | 3 of 5 |
| S11 | `{"clf__C": 100, "clf__gamma": 0.1}` | 2 of 5 |
| S13 | `{"clf__C": 100, "clf__gamma": 0.01}` | 1 of 5 |
| S14 | `{"clf__C": 100, "clf__gamma": 0.1}` | 2 of 5 |
| S15 | `{"clf__C": 0.1, "clf__gamma": "scale"}` | 2 of 5 |
| S16 | `{"clf__C": 100, "clf__gamma": 0.1}` | 3 of 5 |
| S17 | `{"clf__C": 10, "clf__gamma": "scale"}` | 5 of 5 |

**rf** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 27 |
| `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 26 |
| `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 9 |
| `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 7 |
| `{"clf__max_depth": 20, "clf__min_samples_leaf": 1}` | 5 |
| `{"clf__max_depth": 20, "clf__min_samples_leaf": 5}` | 1 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S3 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 3 of 5 |
| S4 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 2 of 5 |
| S5 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 4 of 5 |
| S6 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 1}` | 1 of 5 |
| S7 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 3 of 5 |
| S8 | `{"clf__max_depth": null, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S9 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 3 of 5 |
| S10 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 3 of 5 |
| S11 | `{"clf__max_depth": 20, "clf__min_samples_leaf": 1}` | 2 of 5 |
| S13 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 3 of 5 |
| S14 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 5 of 5 |
| S15 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 5 of 5 |
| S16 | `{"clf__max_depth": 10, "clf__min_samples_leaf": 5}` | 2 of 5 |
| S17 | `{"clf__max_depth": null, "clf__min_samples_leaf": 5}` | 4 of 5 |

**mlp** - chosen setting over 75 outer fits (15 subjects x 5 seeds):

| setting | times chosen |
|---|---|
| `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 27 |
| `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 25 |
| `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 12 |
| `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [32]}` | 11 |

Per subject (the setting chosen most often over the seeds; in brackets how many of the seeds chose it):

| subject | modal choice | seeds agreeing |
|---|---|---|
| S2 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S3 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 2 of 5 |
| S4 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S5 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S6 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 3 of 5 |
| S7 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S8 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S9 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 2 of 5 |
| S10 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [32]}` | 2 of 5 |
| S11 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S13 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 2 of 5 |
| S14 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S15 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 3 of 5 |
| S16 | `{"clf__alpha": 0.0001, "clf__hidden_layer_sizes": [64, 32]}` | 2 of 5 |
| S17 | `{"clf__alpha": 0.01, "clf__hidden_layer_sizes": [64, 32]}` | 5 of 5 |

## Failed or unstable runs

150 record(s) in `results/failures.csv`:

| task | model | kind | records |
|---|---|---|---|
| binary | mlp | convergence warnings | 75 |
| three | mlp | convergence warnings | 75 |

MLP convergence (max_iter=300, no early stopping): inner fits that ran to max_iter, summed per task:

| task | inner fits not converged | of | outer refits not converged | of outer fits |
|---|---|---|---|---|
| binary | 2100 | 2100 | 75 | 75 |
| three | 2100 | 2100 | 75 | 75 |

## Interaction: model x heart-rate resolution (binary, Amendment 5)

Protocol: `PRE_DECLARED.md` Amendment 5 (frozen before any interaction result). R1s = beat-level HR, R2 = the same beats averaged into 3.0 s bins (a controlled change of time resolution, not a device model), R1stoR2 = trained on R1s and tested on R2. Same 1012 windows in every condition; per-subject macro-F1, 5 seeds averaged first; 95% paired bootstrap CIs over the 15 subjects, **not adjusted for multiplicity**. Source: `results/resolution/*.csv`.

### Macro-F1 per subject by condition (mean [95% CI])

| model | R1s | R2 | R1stoR2 |
|---|---|---|---|
| majority | 0.412 [0.410, 0.414] | 0.412 [0.410, 0.414] | 0.412 [0.410, 0.414] |
| threshold | 0.694 [0.598, 0.783] | 0.681 [0.590, 0.769] | 0.659 [0.577, 0.742] |
| svm | 0.729 [0.639, 0.814] | 0.749 [0.659, 0.832] | 0.712 [0.626, 0.800] |
| rf | 0.755 [0.679, 0.832] | 0.724 [0.643, 0.804] | 0.661 [0.581, 0.745] |
| mlp | 0.752 [0.672, 0.825] | 0.761 [0.669, 0.842] | 0.737 [0.659, 0.814] |

### Interaction contrasts (the pre-declared test; difference of differences)

| contrast | difference at R1s | difference at R2 | contrast (95% CI) | CI excludes 0 |
|---|---|---|---|---|
| (mlp-svm at R1s) - (mlp-svm at R2) | +0.023 | +0.011 | +0.012 [-0.010, +0.034] | no |
| (mlp-rf at R1s) - (mlp-rf at R2) | -0.003 | +0.037 | -0.041 [-0.073, -0.009] | yes |
| (rf-svm at R1s) - (rf-svm at R2) | +0.026 | -0.026 | +0.052 [+0.029, +0.073] | yes |

### Resolution effect (R2 minus R1s) and transfer (R1stoR2 minus R1s / R2)

| contrast | mean difference [95% CI] | CI excludes 0 |
|---|---|---|
| majority at R2 - majority at R1s | +0.000 [+0.000, +0.000] | no |
| majority at R1stoR2 - majority at R1s | +0.000 [+0.000, +0.000] | no |
| majority at R1stoR2 - majority at R2 | +0.000 [+0.000, +0.000] | no |
| threshold at R2 - threshold at R1s | -0.013 [-0.032, +0.004] | no |
| threshold at R1stoR2 - threshold at R1s | -0.035 [-0.072, +0.001] | no |
| threshold at R1stoR2 - threshold at R2 | -0.022 [-0.057, +0.010] | no |
| svm at R2 - svm at R1s | +0.021 [+0.001, +0.044] | yes |
| svm at R1stoR2 - svm at R1s | -0.017 [-0.057, +0.022] | no |
| svm at R1stoR2 - svm at R2 | -0.038 [-0.086, +0.009] | no |
| rf at R2 - rf at R1s | -0.031 [-0.046, -0.016] | yes |
| rf at R1stoR2 - rf at R1s | -0.094 [-0.130, -0.059] | yes |
| rf at R1stoR2 - rf at R2 | -0.063 [-0.108, -0.017] | yes |
| mlp at R2 - mlp at R1s | +0.009 [-0.023, +0.043] | no |
| mlp at R1stoR2 - mlp at R1s | -0.014 [-0.049, +0.017] | no |
| mlp at R1stoR2 - mlp at R2 | -0.024 [-0.069, +0.023] | no |

### Order of the three models by mean macro-F1 (descriptive; best first)

- R1s: rf > mlp > svm
- R2: mlp > svm > rf
- R1stoR2: mlp > svm > rf

MLP convergence (max_iter=300): 4196 of 4200 inner fits ran to max_iter and 150 of 150 outer refits did not converge (`results/resolution/failures.csv`).

Limits (stated in Amendment 5): 15 subjects, so the interaction test has low power; binning does not reproduce a device's smoothing; R1stoR2 also carries a shift in the feature distributions.

