# Benchmark Results — temporaldata Hot Path Optimizations

## Baseline (before any optimizations)


| Benchmark                       | Iters   | Mean (µs) |
| ------------------------------- | ------- | --------- |
| ArrayDict.keys() x100k          | 100,000 | 0.622     |
| Interval.**and** (single)       | 1,000   | 577.393   |
| Interval.**and** (multi)        | 200     | 781.266   |
| Interval.**or**                 | 200     | 4,317.873 |
| Interval.difference             | 200     | 4,157.842 |
| Data.slice() end-to-end         | 500     | 177.922   |
| LazyInterval .start/.end access | 2,000   | 545.691   |


## After Phase 1: Cache `ArrayDict.keys()`


| Benchmark                       | Iters   | Mean (µs) | Speedup |
| ------------------------------- | ------- | --------- | ------- |
| ArrayDict.keys() x100k          | 100,000 | 0.126     | 4.94x   |
| Interval.**and** (single)       | 1,000   | 545.288   | 1.06x   |
| Interval.**and** (multi)        | 200     | 743.250   | 1.05x   |
| Interval.**or**                 | 200     | 4,249.148 | 1.02x   |
| Interval.difference             | 200     | 4,184.210 | 0.99x   |
| Data.slice() end-to-end         | 500     | 181.881   | 0.98x   |
| LazyInterval .start/.end access | 2,000   | 532.968   | 1.02x   |


