# DPA1-L0-M PT-EXPT Kokkos water-MD reproduction

This repository contains a minimal public reproduction bundle for the related DeepMD-kit issue:

https://github.com/deepmodeling/deepmd-kit/issues/5864

## Scope

The same 192-atom, 64-water periodic system was tested with:

1. The released pretrained DPA1-L0-M ordinary PT-EXPT model.
2. A 2M-step fine-tuned DPA1-L0-M H/O ordinary PT-EXPT model.
3. The same fine-tuned model after PT-EXPT compression using five evenly spaced frames from each of 21 training systems.

The fine-tuned ordinary and compressed models fail before the first useful MD step in both NVE and NVT. The released pretrained ordinary model reproduces the earlier trajectory blow-up followed by a Kokkos CUDA illegal-address abort during longer NVE.

The pure-water data source is the public Zenodo dataset:

https://zenodo.org/records/14780363

## Repository layout

- `models/`: ordinary pretrained and fine-tuned PT-EXPT models, plus the sampled-compressed fine-tuned model.
- `data/`: wrapped periodic 64-water LAMMPS data file.
- `inputs/`: portable LAMMPS inputs. Model, data, and output paths are supplied with `-var`.
- `scripts/`: generic SLURM launchers. Set `MODEL_PATH`, `DATA_PATH`, and `OUTPUT_PREFIX` rather than editing private paths into the scripts.
- `outputs/`: sanitized thermo/error excerpts from the tested jobs.
- `SHA256SUMS`: hashes for the uploaded artifacts.

## Environment

- NVIDIA GeForce RTX 3090, 24 GiB
- NVIDIA driver 550.90.07
- CUDA runtime 12.6
- LAMMPS 4 Jul 2026
- Kokkos 5.1.99
- CUDA + LAMMPS + Kokkos DeePMD-kit build

## Common input settings

- 192 atoms = 64 H2O molecules
- Periodic orthogonal box: 12.42 x 12.42 x 12.42 Angstrom
- `pair_coeff * * H O`
- `neighbor 1.0 bin`
- `neigh_modify every 1 delay 0 check yes`
- Timestep: 0.0005 ps (0.5 fs)
- Initial velocities: 300 K
- LAMMPS and XYZ output every 100 steps

## Reproduction

From the repository root, use the ordinary fine-tuned model:

```bash
export MODEL_PATH="$PWD/models/finetuned_ordinary.pt2"
export DATA_PATH="$PWD/data/water64_wrapped.data"
export OUTPUT_PREFIX="$PWD/outputs/local_finetuned_nve"
sbatch --export=ALL,MODEL_PATH,DATA_PATH,OUTPUT_PREFIX scripts/slurm_nve.sh
```

For the released pretrained model, replace `MODEL_PATH` with:

```bash
export MODEL_PATH="$PWD/models/pretrained_ordinary.pt2"
```

For the sampled-compressed fine-tuned model, use:

```bash
export MODEL_PATH="$PWD/models/finetuned_compressed_sampled5.pt2"
```

For NVT, use `scripts/slurm_nvt.sh` and set `OUTPUT_PREFIX` to a different output prefix.

The scripts assume a local module named `DeepMD-kit/dpa1-master-cu126-kk`; replace the module line with the corresponding local DeePMD-kit/LAMMPS+Kokkos environment when needed.

## Results

The ordinary and sampled-compressed fine-tuned models both print only the initial thermo record at step 0 and then abort while building the first dynamic Kokkos neighbor list. No integration step completes.

The released pretrained ordinary model can complete a short 100-step smoke test, but the longer NVE trajectory becomes numerically unstable before aborting with the same Kokkos CUDA illegal-address error.

The key error is:

```text
cudaDeviceSynchronize() error( cudaErrorIllegalAddress): an illegal memory access was encountered
```

The relevant stack reaches:

```text
Kokkos::Impl::ExecSpaceManager::static_fence
LAMMPS_NS::NBinKokkos<Kokkos::Cuda>::bin_atoms
LAMMPS_NS::NeighborKokkos::build_kokkos
LAMMPS_NS::VerletKokkos::run
```

The uploaded models are provided for exact artifact-level reproduction. The report does not claim that the model is physically valid for pure water; it reports the reproducible failure mode and the opaque Kokkos error.
