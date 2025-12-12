# Test Coverage Analysis for temporaldata

**Generated with pytest-cov**

```
TOTAL: 1385 statements, 204 missing, 85% coverage
```

---

## Coverage Summary by Module

| Module | Statements | Missing | Coverage | Key Gaps |
|--------|------------|---------|----------|----------|
| `__init__.py` | 7 | 0 | **100%** | ✅ Complete |
| `arraydict.py` | 185 | 31 | **83%** | `__repr__`, `from_hdf5` error, `from_dataframe` warnings |
| `concat.py` | 24 | 4 | **83%** | Error paths for mismatched keys/types |
| `data.py` | 216 | 40 | **81%** | `select_by_interval`, `__repr__`, `_check_for_data_leakage` |
| `interval.py` | 484 | 48 | **90%** | `register_timekey`, `from_dataframe`, lazy edge cases |
| `irregular_ts.py` | 297 | 45 | **85%** | `from_dataframe`, `add_split_mask`, domain setter |
| `regular_ts.py` | 150 | 18 | **88%** | `timekeys`, `select_by_mask`, lazy `to_hdf5` |
| `utils.py` | 22 | 18 | **18%** | `_size_repr` (only used internally via `__repr__`) |

---

## Part 1: Missing Coverage (by Module)

### 1. ArrayDict (`arraydict.py`) — 83% Coverage

**Uncovered Lines:** 75, 94-102, 139, 169, 181, 199, 202, 216, 253-254, 286-302, 350, 360, 368-369, 410, 444, 447

| Line(s) | Code | Issue |
|---------|------|-------|
| 75 | `raise ValueError(...)` | 0-dimensional array error path |
| 94-102 | `__repr__` | String representation never tested |
| 139 | mask length mismatch | Error path in `select_by_mask` |
| 169 | `raise ValueError(...)` | Column override in `from_dataframe` |
| 181, 199, 202, 216 | `from_dataframe` | ndarray shape mismatch warnings, object column handling |
| 253-254 | `to_hdf5` | UnicodeEncodeError path |
| 286-302 | `from_hdf5` | Entire method on base `ArrayDict` class (only Lazy variant tested) |
| 350, 360, 368-369, 410 | `LazyArrayDict` | `_maybe_first_dim` edge cases, mask operations |
| 444, 447 | `LazyArrayDict` | `from_dataframe` and `to_hdf5` NotImplementedError |

### 2. Interval (`interval.py`) — 90% Coverage

**Uncovered Lines:** 113, 117-120, 223, 322, 350-356, 440, 481-501, 570-573, 629-639, 673-694, 739-745, 843, 865, 931-989

| Line(s) | Code | Issue |
|---------|------|-------|
| 113 | `timekeys()` | Method return value never tested |
| 117-120 | `register_timekey()` | Entire method untested |
| 223 | `slice` with unsorted | Auto-sort path in slice |
| 322 | `coalesce` unsorted | Sort path in coalesce |
| 350-356 | `difference` | Validation errors for non-disjoint/unsorted |
| 440 | `add_split_mask` | Duplicate mask name error |
| 481-501 | `add_split_mask` / `allow_split_mask_overlap` | Overlap detection logic |
| 570-573 | `from_dataframe` | Missing column errors |
| 629-639 | `to_hdf5` | UnicodeEncodeError path |
| 673-694 | `from_hdf5` | `allow_split_mask_overlap` restoration |
| 688-694 | `__and__` | Validation errors for non-disjoint/unsorted |
| 739-745 | `__or__` | Validation errors for non-disjoint/unsorted |
| 843, 865 | `LazyInterval` | `_maybe_first_dim` edge cases |
| 931-989 | `LazyInterval.slice` | Various edge cases with unresolved slices |

### 3. IrregularTimeSeries (`irregular_ts.py`) — 85% Coverage

**Uncovered Lines:** 106-114, 127-140, 167, 175, 271-280, 305-308, 339-357, 403-417, 446-471, 513, 539, 630-693

| Line(s) | Code | Issue |
|---------|------|-------|
| 106, 111, 114 | Domain validation | Non-Interval domain and unsorted domain errors |
| 127-129 | `domain` setter | Property setter validation |
| 137-140 | `register_timekey()` | Entire method untested |
| 167, 175 | `_maybe_start/_maybe_end` | Unsorted data paths |
| 271-280 | `add_split_mask` | Duplicate mask and mask creation logic |
| 305-308 | `from_dataframe` | Missing timestamp column error |
| 339-340, 347-357 | `to_hdf5` | Unicode handling and sorting warning |
| 403, 417 | `from_hdf5` | Type mismatch error |
| 446-471 | `LazyIrregularTimeSeries` | `_maybe_first_dim`, `load()` methods |
| 513, 539 | Lazy operations | Unicode conversion and slice handling |
| 630-633, 659, 675-693 | Lazy slice | Edge cases in `_resolve_timestamps_after_slice` |

### 4. RegularTimeSeries (`regular_ts.py`) — 88% Coverage

**Uncovered Lines:** 66, 87, 90, 157, 241-251, 270, 323-331, 364

| Line(s) | Code | Issue |
|---------|------|-------|
| 66 | `domain_start` validation | Error when `domain_start` is not a number |
| 87 | `timekeys()` | Returns undefined `_timekeys` — **POTENTIAL BUG** |
| 90 | `select_by_mask()` | `NotImplementedError` path |
| 157 | `add_split_mask` | Overlap assertion |
| 241-251 | `from_hdf5` | Entire non-lazy `from_hdf5` method |
| 270 | `LazyRegularTimeSeries._maybe_first_dim` | Edge case with sliced domain |
| 323-331 | `LazyRegularTimeSeries.slice` | Reset origin handling |
| 364 | `LazyRegularTimeSeries.to_hdf5` | `NotImplementedError` path |

### 5. Data (`data.py`) — 81% Coverage

**Uncovered Lines:** 134, 137, 164, 178, 183, 255-276, 279-294, 432-448, 457, 519, 558

| Line(s) | Code | Issue |
|---------|------|-------|
| 134, 137 | Nested Data domain check | Setting attribute with domain on domainless parent |
| 164 | `__setattr__` | Error when adding time-based attr without domain |
| 178, 183 | `start`/`end` properties | When domain is None |
| 255-276 | `select_by_interval()` | **Entire method untested** |
| 279-290 | `__repr__` | String representation |
| 294 | `to_dict()` | Method never called |
| 432-448 | `_check_for_data_leakage()` | **Entire method untested** |
| 457 | `__contains__` | Only tested via `in` operator implicitly |
| 519 | `__deepcopy__` | h5py.Dataset branch |
| 558 | `serialize()` | Type match in `serialize_fn_map` |

### 6. concat (`concat.py`) — 83% Coverage

**Uncovered Lines:** 49, 62, 68, 85

| Line(s) | Code | Issue |
|---------|------|-------|
| 49 | Type mismatch error | `ValueError` when objects have different types |
| 62 | Key mismatch error | `ValueError` when keys don't match |
| 68 | Timekey mismatch error | `ValueError` when timekeys don't match |
| 85 | Unsupported type | `NotImplementedError` for non-IrregularTimeSeries |

### 7. utils (`utils.py`) — 18% Coverage

**Uncovered Lines:** 7-29

| Line(s) | Code | Issue |
|---------|------|-------|
| 7-29 | `_size_repr()` | **Entire function untested** (used by `__repr__` methods which are also untested) |

---

## Part 2: Weak Tests (Tests That Exist but Are Insufficient)

### 🔴 Critical Weaknesses

#### `test_regular_ts.py` — Most Problematic

| Test | Weakness | Impact |
|------|----------|--------|
| **No basic construction test** | Only tests slicing; no tests for basic `RegularTimeSeries` construction, properties, or attribute access | Could miss basic functionality issues |
| **No input validation tests** | What happens with `sampling_rate=0`, negative sampling rate, or mismatched domain? | Invalid objects could be created |
| **Uses random data** | `np.random.random()` makes failures non-reproducible | Debugging flaky tests is hard |
| **No empty timeseries test** | What happens with 0 samples? | Edge case untested |
| **No multi-attribute test** | Only tests with 1-2 attributes | Attribute handling bugs could be missed |

```python
# Example: test_regulartimeseries only tests slicing, nothing else
def test_regulartimeseries(test_filepath):
    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"  # random!
    )
    # Only slicing is tested, not:
    # - data.sampling_rate
    # - data.domain properties
    # - data.timestamps correctness
    # - attribute access
```

| Test | Weakness |
|------|----------|
| `test_regular_to_irregular_timeseries` | Only checks timestamps and one attribute; **domain transfer is not verified** |

```python
# Weak: doesn't verify domain is preserved
def test_regular_to_irregular_timeseries():
    a = RegularTimeSeries(lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto")
    b = a.to_irregular()
    assert np.allclose(b.timestamps, np.arange(0, 10, 0.1))
    assert np.allclose(b.lfp, a.lfp)
    # MISSING: assert b.domain == a.domain
    # MISSING: assert b.is_sorted()
```

---

#### `test_concat.py` — Only 1 Test!

| Weakness | Impact |
|----------|--------|
| Only ONE test function | No edge cases tested |
| No `sort=False` test | Default sorting is tested, but disabling it is not |
| No domain verification | Merged domain is not checked |
| No empty input test | `concat([])` behavior unknown |
| No single-element test | `concat([ts1])` behavior unknown |

```python
# The ENTIRE test file:
def test_irregular_timeseries_concat():
    # ... creates two IrregularTimeSeries ...
    data = concat([data1, data2])
    assert len(data) == len(data1) + len(data2)  # Only checks length
    assert np.all(data.unit_index == ...)
    assert np.all(data.timestamps == ...)
    # MISSING: assert data.domain covers both original domains
    # MISSING: test with sort=False
    # MISSING: test with overlapping timestamps
```

---

#### `test_data.py`

| Test | Weakness |
|------|----------|
| Line 72 | **Useless assertion**: `assert ["session_id", ...]` always passes (non-empty list is truthy) |
| `test_data` | Domain is updated after slicing but **never verified** |
| `test_data` | Only checks `spikes`, `lfp`, `trials` after slice; other attributes not verified |
| `test_data_copy` | Mutates original test data mid-test, making later assertions dependent on earlier mutations |

```python
# Line 72 - This assertion does NOTHING:
assert ["session_id", "spikes", "lfp", "units", "trials", "drifting_gratings_imgs"]
# ^ This is always True because a non-empty list is truthy

# Should be:
assert data.keys() == ["session_id", "spikes", "lfp", "units", "trials", "drifting_gratings_imgs"]
```

---

### 🟡 Moderate Weaknesses

#### `test_interval.py`

| Test | Weakness |
|------|----------|
| `test_interval_select_by_mask` | Doesn't verify `_sorted` state is preserved |
| `test_interval_select_by_mask` | Doesn't verify `_timekeys` are preserved |
| `test_interval_slice` | No test for slicing that results in empty interval |
| `test_interval_slice` | No test for partial overlap (slice window partially overlaps interval) |
| `test_split` | No test with `sizes` that don't sum correctly (error path) |

```python
# select_by_mask doesn't verify important state:
def test_interval_select_by_mask():
    data = Interval(...)
    data = data.select_by_mask(mask)
    assert len(data) == 3
    # MISSING: assert data._sorted == original._sorted
    # MISSING: assert data._timekeys == original._timekeys
```

---

#### `test_irregular_ts.py`

| Test | Weakness |
|------|----------|
| `test_irregular_timeseries_slice` | Domain is checked but `domain.is_sorted()` and `domain.is_disjoint()` are not verified |
| `test_irregular_timeseries_sortedness` | Tests sorting but not the warning that's supposed to be logged |
| No test for empty slice result | What happens when slice returns 0 timestamps? |

---

#### `test_arraydict.py`

| Test | Weakness |
|------|----------|
| `test_array_dict_select_by_mask` | Doesn't verify returned object is independent (modifying it shouldn't affect original) |
| `test_array_dict_from_dataframe` | Uses hardcoded expected dtypes but they may change with pandas versions |

---

#### `test_add_split_mask.py`

| Test | Weakness |
|------|----------|
| Only tests `RegularTimeSeries` | No tests for `IrregularTimeSeries.add_split_mask` |
| Only tests `RegularTimeSeries` | No tests for `Interval.add_split_mask` |
| No overlap test | What happens when split intervals overlap? |
| No gap test | What happens when there's a gap between splits (some points belong to no split)? |

```python
# Only RegularTimeSeries is tested:
def test_regulartimeseries():
    data = RegularTimeSeries(...)
    data.add_split_mask("train", train_domain)
    # ...

# MISSING:
# def test_irregular_timeseries_add_split_mask(): ...
# def test_interval_add_split_mask(): ...
```

---

### 🟢 Minor Weaknesses

| Test File | Weakness |
|-----------|----------|
| `test_h5_io.py` | Tests save/load but doesn't verify ALL attributes survive round-trip |
| `test_materialize.py` | Checks types but doesn't verify actual data values are correct |
| `test_serialize.py` | Only tests with Enum; other custom types untested |

---

## Part 3: Critical Issues Summary

### Bug Found
**`RegularTimeSeries.timekeys()` (Line 87)** references `self._timekeys` which is **never defined** in the class. Calling this method will raise `AttributeError`.

### Completely Untested Public Methods

| Method | Class |
|--------|-------|
| `select_by_interval()` | `Data` |
| `_check_for_data_leakage()` | `Data` |
| `register_timekey()` | `Interval`, `IrregularTimeSeries` |
| `from_dataframe()` | `Interval`, `IrregularTimeSeries` |
| `to_dict()` | `Data` |
| `__repr__` | All classes |

### Error Paths Never Exercised

All `ValueError` and `NotImplementedError` raises in:
- `concat()` — all 4 error paths
- `from_dataframe()` — column validation
- `add_split_mask()` — duplicate mask detection

---

## Part 4: Recommended Test Additions

### Priority 1: Fix the Bug + Critical Missing Tests

```python
# test_regular_ts.py - Add:
def test_regular_ts_timekeys_raises():
    """RegularTimeSeries.timekeys() references undefined _timekeys"""
    ts = RegularTimeSeries(lfp=np.zeros((100, 3)), sampling_rate=10, domain="auto")
    with pytest.raises(AttributeError):
        ts.timekeys()

def test_regular_ts_basic_construction():
    """Basic RegularTimeSeries construction and properties"""
    ts = RegularTimeSeries(
        lfp=np.zeros((100, 3)),
        sampling_rate=10.0,
        domain="auto"
    )
    assert ts.sampling_rate == 10.0
    assert len(ts) == 100
    assert ts.domain.start[0] == 0.0
    assert np.allclose(ts.timestamps, np.arange(0, 10, 0.1))

def test_regular_ts_select_by_mask_not_implemented():
    ts = RegularTimeSeries(lfp=np.zeros((10, 3)), sampling_rate=10, domain="auto")
    with pytest.raises(NotImplementedError):
        ts.select_by_mask(np.array([True] * 10))
```

### Priority 2: Data Methods

```python
# test_data.py - Add:
def test_data_select_by_interval():
    """Data.select_by_interval() is completely untested"""
    data = Data(
        spikes=IrregularTimeSeries(...),
        domain=Interval(0., 10.)
    )
    interval = Interval(np.array([1., 5.]), np.array([3., 7.]))
    result = data.select_by_interval(interval)
    # Verify spikes are filtered
    # Verify domain is updated

def test_data_check_for_data_leakage():
    """Data._check_for_data_leakage() is completely untested"""
    ...

# Fix the useless assertion on line 72
def test_data():
    ...
    # CHANGE FROM:
    # assert ["session_id", "spikes", ...]  # Always True!
    # TO:
    assert data.keys() == ["session_id", "spikes", ...]
```

### Priority 3: concat Error Paths

```python
# test_concat.py - Add:
def test_concat_type_mismatch():
    ts1 = IrregularTimeSeries(timestamps=np.array([0.]), domain="auto")
    ts2 = RegularTimeSeries(lfp=np.zeros((10, 1)), sampling_rate=10, domain="auto")
    with pytest.raises(ValueError, match="same type"):
        concat([ts1, ts2])

def test_concat_key_mismatch():
    ts1 = IrregularTimeSeries(timestamps=np.array([0.]), a=np.array([1]), domain="auto")
    ts2 = IrregularTimeSeries(timestamps=np.array([1.]), b=np.array([2]), domain="auto")
    with pytest.raises(ValueError, match="same keys"):
        concat([ts1, ts2])

def test_concat_unsupported_type():
    i1 = Interval(0., 1.)
    i2 = Interval(1., 2.)
    with pytest.raises(NotImplementedError):
        concat([i1, i2])

def test_concat_sort_false():
    ts1 = IrregularTimeSeries(timestamps=np.array([2., 3.]), domain="auto")
    ts2 = IrregularTimeSeries(timestamps=np.array([0., 1.]), domain="auto")
    result = concat([ts1, ts2], sort=False)
    assert np.array_equal(result.timestamps, np.array([2., 3., 0., 1.]))
```

### Priority 4: from_dataframe Methods

```python
# test_interval.py - Add:
def test_interval_from_dataframe():
    df = pd.DataFrame({
        'start': [0., 1., 2.],
        'end': [1., 2., 3.],
        'label': ['a', 'b', 'c']
    })
    interval = Interval.from_dataframe(df)
    assert len(interval) == 3
    assert np.array_equal(interval.label, np.array(['a', 'b', 'c']))

def test_interval_from_dataframe_missing_start():
    df = pd.DataFrame({'end': [1., 2., 3.]})
    with pytest.raises(AssertionError, match="start"):
        Interval.from_dataframe(df)
```

---

## Summary

| Category | Count |
|----------|-------|
| Total statements | 1385 |
| Covered | 1181 (85%) |
| Missing | 204 (15%) |
| Untested public methods | 6+ |
| Untested error paths | 15+ |
| Weak tests identified | 20+ |
| Potential bugs found | 1 |

**Key Takeaways:**
1. **85% coverage is misleading** — many tests are too weak to catch real bugs
2. **`RegularTimeSeries` is the weakest tested class** — no basic construction tests, uses random data
3. **`concat` has only 1 test** with no edge cases or error paths
4. **Line 72 in `test_data.py` is a no-op assertion** that should be fixed
5. **All `__repr__` methods are untested** (along with `utils._size_repr`)
