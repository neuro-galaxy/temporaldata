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


## After Phase 2: Vectorize `__and__`, `__or__`, `difference`


| Benchmark                       | Iters   | Mean (µs) | vs Baseline | vs Phase 1 |
| ------------------------------- | ------- | --------- | ----------- | ---------- |
| ArrayDict.keys() x100k          | 100,000 | 0.123     | 5.06x       | 1.02x      |
| Interval.**and** (single)       | 1,000   | 26.494    | 21.79x      | 20.37x     |
| Interval.**and** (multi)        | 200     | 620.331   | 1.26x       | 1.20x      |
| Interval.**or**                 | 200     | 43.614    | 99.01x      | 97.45x     |
| Interval.difference             | 200     | 3,841.421 | 1.08x       | 1.06x      |
| Data.slice() end-to-end         | 500     | 195.049   | 0.91x       | 0.93x      |
| LazyInterval .start/.end access | 2,000   | 531.261   | 1.03x       | 1.00x      |


## After Phase 3: Replace `all_loaded` O(n) scan with `_n_lazy` O(1) counter


| Benchmark                       | Iters   | Mean (µs) | vs Baseline | vs Phase 2 |
| ------------------------------- | ------- | --------- | ----------- | ---------- |
| ArrayDict.keys() x100k          | 100,000 | 0.122     | 5.10x       | 1.01x      |
| Interval.**and** (single)       | 1,000   | 26.400    | 21.87x      | 1.00x      |
| Interval.**and** (multi)        | 200     | 626.479   | 1.25x       | 0.99x      |
| Interval.**or**                 | 200     | 43.664    | 98.88x      | 1.00x      |
| Interval.difference             | 200     | 3,837.573 | 1.08x       | 1.00x      |
| Data.slice() end-to-end         | 500     | 193.029   | 0.92x       | 1.01x      |
| LazyInterval .start/.end access | 2,000   | 527.686   | 1.03x       | 1.01x      |

**Note:** Phase 3 replaces the O(n) `all(isinstance(...) for key in self.keys())` check that ran on every attribute access with an O(1) integer counter comparison. The benchmark shows minimal change because the test `LazyInterval` has only 2 public keys, making the O(n) scan already trivial. The optimization prevents degradation as the number of attributes grows.
