# macOS/Windows Multiprocessing Guide for Parallel LHC

## Important Note for macOS and Windows Users

If you're writing **scripts** (not interactive code) that use the parallel LHC optimizer, you **MUST** protect your code with `if __name__ == '__main__':`.

### Why?

macOS and Windows use **'spawn'** instead of **'fork'** for multiprocessing. This means the Python interpreter re-imports your script in each worker process. Without the guard, this creates an infinite loop of process spawning.

---

## ✅ CORRECT Usage (Scripts)

```python
#!/usr/bin/env python
"""My kriging script."""

import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan


def main():
    """Main function."""
    # Your code here
    sp = samplingplan(k=2)
    X = sp.optimallhc(100)  # Uses parallel optimization

    y = np.array([my_function(x) for x in X])
    k = kriging(X, y)
    k.train()

    print(f"Done! Trained with {X.shape[0]} points")


if __name__ == '__main__':
    # CRITICAL on macOS/Windows!
    main()
```

---

## ❌ INCORRECT Usage (Will Crash on macOS/Windows)

```python
#!/usr/bin/env python
"""My kriging script - WILL CRASH on macOS!"""

import numpy as np
from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan

# This code runs immediately when script is imported
sp = samplingplan(k=2)
X = sp.optimallhc(100)  # CRASH! Infinite process spawning

y = np.array([my_function(x) for x in X])
k = kriging(X, y)
k.train()
```

**Error you'll see:**
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

---

## 🟢 Interactive Use (No Guard Needed)

If you're using pyKriging **interactively** (Jupyter, IPython, Python REPL), you don't need the guard:

```python
# In Jupyter/IPython - this is fine!
from pyKriging.samplingplan import samplingplan

sp = samplingplan(k=2)
X = sp.optimallhc(100)  # Works fine interactively
```

---

## When Do You Need the Guard?

| Scenario | Need Guard? | Why |
|----------|-------------|-----|
| **Script (.py file)** | ✅ YES | Prevents re-execution in workers |
| **Jupyter Notebook** | ❌ NO | Jupyter handles this automatically |
| **IPython/REPL** | ❌ NO | Interactive shells are safe |
| **Imported module** | ❌ NO | Only scripts need it |
| **Linux only** | ⚠️ MAYBE | Works without, but better with |

---

## Platform-Specific Behavior

| Platform | Multiprocessing Method | Guard Required? |
|----------|----------------------|-----------------|
| **Linux** | fork | No (but recommended) |
| **macOS** | spawn | **YES** |
| **Windows** | spawn | **YES** |

---

## Quick Fix Template

If you have existing scripts that crash, wrap them like this:

```python
# Original script
import numpy as np
from pyKriging.samplingplan import samplingplan

sp = samplingplan(k=2)
X = sp.optimallhc(100)
# ... rest of code ...

# Fixed script
import numpy as np
from pyKriging.samplingplan import samplingplan


def main():
    sp = samplingplan(k=2)
    X = sp.optimallhc(100)
    # ... rest of code ...


if __name__ == '__main__':
    main()
```

---

## Why This Matters for Parallel LHC

The parallel LHC optimizer (new default for n ≥ 50) uses `multiprocessing.Pool`:

```python
# Inside samplingplan.py
with Pool(processes=n_workers) as pool:
    X_list = pool.map(worker_func, q)
```

On macOS/Windows:
1. Pool spawns new Python processes
2. Each process re-imports your script
3. Without guard: script runs again → spawns more processes → infinite loop!
4. With guard: script is imported but main() not called → works correctly

---

## Troubleshooting

### Error: "RuntimeError: An attempt has been made to start a new process..."

**Solution:**
```python
if __name__ == '__main__':
    # Your code here
```

### Error Still Happens?

Check that you're not calling code **outside** the guard:

```python
# BAD - this runs during import!
X = sp.optimallhc(100)

if __name__ == '__main__':
    print("Done")  # Too late, already crashed above

# GOOD - everything inside guard
if __name__ == '__main__':
    X = sp.optimallhc(100)
    print("Done")
```

### Works on Linux, Crashes on macOS?

Linux uses 'fork', which doesn't need the guard. But add it anyway for cross-platform compatibility:

```python
# Works everywhere
if __name__ == '__main__':
    main()
```

---

## Alternative: Force Serial Mode

If you can't modify your script, disable parallelization:

```python
# Force serial mode (no multiprocessing)
X = sp.optimallhc(100, n_jobs=1)  # No guard needed
```

But you lose the 5x speedup!

---

## Examples in This Repo

All scripts in this repo follow the pattern:

- ✅ `benchmark_metal.py` - Has guard (correct)
- ✅ `test_auto_parallel_lhc.py` - Has guard (correct)
- ✅ `benchmark_lhc_parallel.py` - Has guard (correct)

Use these as templates for your scripts.

---

## Summary

**For macOS/Windows users writing scripts:**

1. ✅ **Wrap your code in a `main()` function**
2. ✅ **Add `if __name__ == '__main__': main()`**
3. ✅ **Never call parallel code at module level**

**Interactive users (Jupyter, IPython):**
- ✅ **No changes needed!** Just use it normally.

**Linux users:**
- ⚠️ **Recommended but not required**

This ensures your code works everywhere and takes advantage of the 5x parallel LHC speedup!
