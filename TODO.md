# TODO

## Reference

- [x] Pinned XFeat and LighterGlue pipeline
- [x] Reproduce published MegaDepth-1500 AUC
- [x] Measure pruning on vs off
- [x] Measure edge-row fix vs upstream

## Export

- [x] Spike: static XFeat graph matches reference
- [x] Spike: padded, fused LighterGlue graph
- [x] `xfeat-export` CLI (height, width, K, batch)

## Shared core

- [x] `xfeat-core` wheel
- [x] `xfeat-bench` harness
- [x] Contract test: export, load, compare

## Backends

- [x] ORT on WSL2 CPU
- [x] TensorRT on WSL2 GPU (FP32)
- [ ] TensorRT FP16
- [ ] Offline throughput mode
- [ ] Raspberry Pi 5
- [ ] M2 Mac
- [ ] Jetson Orin NX

## Evaluation

- [x] Custom image-pair loader
- [x] MegaDepth-1500 scoring of backend predictions
- [ ] HPatches scoring
- [ ] Ablations: precision, resolution, K

## Targets

- [ ] K 2048 or more at low latency
- [ ] K 4096 or more offline
- [ ] Fused LighterGlue on every backend
