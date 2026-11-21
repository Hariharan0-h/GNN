# vendor/

`pymetis-2025.2.2-cp312-cp312-win_amd64.whl` is a local Windows build of
[pymetis](https://pypi.org/project/pymetis/). PyPI only ships source for Windows, and that source
does not compile with MSVC out of the box.

To build it, the pymetis 2025.2.2 sdist was patched in two places.

1. `src/metis/GKlib/gk_arch.h`. The `rint` and `INFINITY` macros clash with the modern UCRT
   `math.h`. They are now defined only for old MSVC:
   ```c
   #if defined(__MSC__) && defined(_MSC_VER) && (_MSC_VER < 1800)
   ```
2. `meson.build`, the `c_args` of `metis_internal_static`, which were extended to
   ```
   '-D_GNU_SOURCE', '-DUSE_GKREGEX', '-DWIN32', '-D__thread=__declspec(thread)', '-D_CRT_SECURE_NO_DEPRECATE'
   ```

The wheel was then built with MSVC 2019 using `pip wheel <patched sdist> --no-deps`.

Install it into the project venv with:

```
.venv\Scripts\python.exe -m pip install --no-deps vendor\pymetis-2025.2.2-cp312-cp312-win_amd64.whl
```

The build was checked against a brute-force edge-cut count; see `tests/test_partitioning.py`.
On Linux or macOS, install `pymetis` from PyPI as usual.
