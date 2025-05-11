# Reports

## Experiment 1

### (A) standard tokenizer, train: score

```
16.4 K    Trainable params
1.2 B     Non-trainable params
1.2 B     Total params
4,943.323 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```
GPU mem consumption: 8.231 GB

Training: `exp1_a.csv`

```
test/f1        0.06593702733516693
test/loss      0.39342889189720154
test/precision 0.24583333730697632
test/recall    0.03958333283662796
```

### (B) syntax-aware tokenizer, train: score, embeddings

```
262 M     Trainable params
973 M     Non-trainable params
1.2 B     Total params
4,943.479 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```

GPU mem consumption: 37.466 GB

Training: `exp1_b.csv`

```
test/f1        0.2899688482284546
test/loss      0.34830784797668457
test/precision 0.49507609009742737
test/recall    0.2187500149011612
```

## Experiment 2

### (A) standard tokenizer, train: score, layer 15

```
60.8 M    Trainable params
1.2 B     Non-trainable params
1.2 B     Total params
4,943.323 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```
GPU mem consumption: 10.274 GB

Training: `exp2_a.csv`

```
test/f1        0.469761461019516
test/loss      0.5562769174575806
test/precision 0.5502395629882812
test/recall    0.4208333194255829
```

### (B) syntax-aware tokenizer, train: score, embeddings, layer 15

```
323 M     Trainable params
912 M     Non-trainable params
1.2 B     Total params
4,943.479 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```

GPU mem consumption: 38.833 GB

Training: `exp2_b.csv`

```
test/f1        0.34084242582321167
test/loss      0.5937626957893372
test/precision 0.5053246021270752
test/recall    0.26249998807907104
```

## Experiment 3

### (A) standard tokenizer, train: score, layers 12–15

```
243 M     Trainable params
992 M     Non-trainable params
1.2 B     Total params
4,943.323 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```

GPU mem consumption: 19.879 GB

Training: `exp3_a.csv`

```
test/f1        0.7180450558662415
test/loss      0.39477473497390747
test/precision 0.7653037309646606
test/recall    0.706250011920929
```

### (B) syntax-aware tokenizer, train: score, embeddings, layers 12–15

```
506 M     Trainable params
729 M     Non-trainable params
1.2 B     Total params
4,943.479 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```

GPU mem consumption: 43.614 GB

Training: `exp3_b.csv`

```
test/f1        0.31342220306396484
test/loss      0.5332949161529541
test/precision 0.4106231927871704
test/recall    0.27916666865348816
```

## Ablation Study

### (A) standard tokenizer, train: score, embeddings

Training: `abl_a.csv`

```
test/f1        0.14745356142520905
test/loss      0.4418472647666931
test/precision 0.3871813118457794
test/recall    0.09583334624767303
```

### (B) syntax-aware tokenizer, train: score

```
16.4 K    Trainable params
1.2 B     Non-trainable params
1.2 B     Total params
4,943.479 Total estimated model params size (MB)
18        Modules in train mode
215       Modules in eval mode
```

Training: `abl_b.csv`

```
test/f1        0.1125616654753685
test/loss      0.41154807806015015
test/precision 0.3099411129951477
test/recall    0.0729166716337204
```
