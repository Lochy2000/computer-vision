# Dataset licence register

This file is an engineering register, not legal advice. Before using any data to
train a distributable model, preserve the exact licence and download terms that
were displayed on the acquisition date. A licence on a code repository does not
necessarily license data hosted elsewhere, and a dataset mirror cannot always
prove the rights to its source images.

| Dataset/source | Modality | Displayed terms | v3 status |
|---|---|---|---|
| UB-SOD / Figshare 33733090 | RGB, UAV + bird | CC BY 4.0 on the formal Figshare record | **Preferred first candidate**; annotation and split archives verified locally, images pending |
| Roboflow Anti-UAV (`gia-bao-nguyen-k74rm`) | RGB | CC BY 4.0 displayed by host | Candidate; verify image provenance |
| Roboflow Anti-drone (`drone-detection-fn0bd`) | RGB | CC BY 4.0 displayed by host | Candidate; verify image provenance |
| DroneDetect / IEEE DataPort | RGB | Claimed CC BY 4.0; record could not be independently fetched | Verify before use |
| 4th Anti-UAV Challenge / Zenodo 15103888 | Multi-track | Zenodo licence field is blank; files restricted | Do not use commercially without written terms |
| Original Anti-UAV / Anti-UAV410 | RGB/thermal | Research/non-commercial restrictions reported | Benchmark only; exclude from commercial training |
| Bhattacharya et al. Drone Detection Dataset | Thermal | GitHub repository is GPL-3.0; external dataset grant is unclear | Verify dataset rights separately |
| LRDDv3 | RGB/thermal | CDLA Permissive 2.0 stated, subject to access/export requirements | Strong candidate after access approval |

## Source links

- https://doi.org/10.6084/m9.figshare.33733090
- https://github.com/DMCcodefinder/UB-SOD-benchmark
- https://universe.roboflow.com/gia-bao-nguyen-k74rm/anti-uav-s8wri-9plnw
- https://universe.roboflow.com/drone-detection-fn0bd/anti-drone/dataset/2
- https://ieee-dataport.org/documents/dronedetect-benchmark-uav-dataset-deep-learning-based-drone-detection
- https://zenodo.org/records/15103888
- https://anti-uav.github.io/dataset/
- https://github.com/purbaditya/Drone-Detection-Dataset
- https://research.coe.drexel.edu/ece/imaple/lrddv3/

## First-dataset decision: UB-SOD

UB-SOD is the preferred first experiment because it directly labels both UAVs
and birds, provides fixed train/validation/test splits, and is dominated by
small objects rather than close product photographs. Its formal repository
reports 7,118 real RGB images from about 32 airport environments and uses CC BY
4.0. It is not a complete production corpus: it lacks thermal data and still
needs a visual audit for missed boxes, near-duplicates and domain bias.

Acquisition audit performed 2026-10-07:

- Annotation ZIP SHA-256 matched the publisher checksum.
- Split/metadata ZIP SHA-256 matched the publisher checksum.
- 5,765 train, 641 validation and 712 test entries; no filename overlap.
- 7,118 YOLO label files with 10,047 UAV and 7,995 bird instances.
- No unexpected class IDs were found in the annotation archive.
- The 2,198,608,426-byte image archive has not yet been downloaded.

See `manifests/datasets.json` for immutable URLs and checksums. Dataset files
remain under ignored `data/`; attribution and the licence must accompany any
redistribution or derived release.

