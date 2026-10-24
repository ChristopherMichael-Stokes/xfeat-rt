# TODO

## Reference

- [x] Pinned XFeat and LighterGlue pipeline
- [x] Reproduce published MegaDepth-1500 AUC
- [x] Measure pruning on vs off
- [x] Measure edge-row fix vs upstream

## Export

- [ ] Spike: static XFeat graph matches reference
- [ ] Spike: padded, fused LighterGlue graph
- [ ] `xfeat-export` CLI (height, width, K, batch)

## Shared core

- [ ] `xfeat-core` wheel
- [ ] `xfeat-bench` harness
- [ ] Contract test: export, load, compare

## Backends

- [ ] ORT on WSL2 CPU
- [ ] TensorRT on WSL2 GPU
- [ ] Offline throughput mode
- [ ] Raspberry Pi 5
- [ ] M2 Mac
- [ ] Jetson Orin NX

## Evaluation

- [ ] Custom image-pair loader
- [ ] MegaDepth-1500 and HPatches scoring
- [ ] Ablations: precision, resolution, K

## Targets

- [ ] K 2048 or more at low latency
- [ ] K 4096 or more offline
- [ ] Fused LighterGlue on every backend
