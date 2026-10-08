# UB-SOD scene-v1 split

This is a derived split definition; it does not modify or redistribute UB-SOD
images or labels. Every visually connected component is assigned wholly to one
partition.

## Counts

| Split | Images | UAV boxes | Bird boxes | Visual components |
|---|---:|---:|---:|---:|
| Train | 5,367 | 7,899 | 6,347 | 1,262 |
| Validation | 876 | 1,062 | 820 | 73 |
| Test | 875 | 1,086 | 828 | 67 |

## Grouping rule

Candidate pairs are generated with a 64-bit difference hash. Images are joined
when either of these resized-grayscale rules is satisfied:

- dHash distance <=18, correlation >=0.88 and normalized MAE <=0.12; or
- dHash distance <=22, correlation >=0.85 and normalized MAE <=0.04.

Connected components are indivisible. A deterministic greedy allocator balances
image counts, UAV boxes, Bird boxes and component counts toward an 80/10/10
target. Large components make the achieved ratio approximately 75/12/12.

## Integrity hashes

- `train.txt`: `78a1c9fe94229617b61fcd2a6082c41f639d2974eaeb9fa086c29a476688954d`
- `val.txt`: `bd7ef42705b146e2592e3b8fc7cc906e25e5446fd3c62a36ffa488472664a6de`
- `test.txt`: `74e3b8763d3da4578b001e26e7b4d85f5095636056f8ea3c46580dda955e0a07`
- `split_manifest.json`: `f13185c90db108831ded3ab50fe0fa5f4b986aaa6f423ec5d5653debab5b49de`

## Limitation

The release does not provide authoritative recording/session or airport IDs.
This split removes the observed visual leakage under the documented rules, but
cannot prove that every different viewpoint from the same airport is isolated.
Final claims still require independent external video tests.
