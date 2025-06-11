---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
import os
from pathlib import Path
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter("ignore", )

import sunpy
from sunraster.instr.spice import read_spice_l2_fits
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.modeling.models as m
from astropy.modeling.fitting import parallel_fit_dask, TRFLSQFitter
import numpy as np
from ndcube import NDCollection, NDCube
from dask import delayed
import dask.bag as db
```

```{code-cell} ipython3
%matplotlib widget
```

```{code-cell} ipython3
base_path = Path("/data/astropy-fitting-talk/")
all_files = sorted(list(base_path.glob("*.fits")))
```

```{code-cell} ipython3
first_file = read_spice_l2_fits(all_files[0:1])
```

```{code-cell} ipython3
data_unit = first_file["Mg IX 706 - Peak"].unit
```

```{code-cell} ipython3
initial_models = {
    'Mg IX 706 - Peak': (
        m.Const1D(amplitude=0 * data_unit) +
        m.Gaussian1D(amplitude=0.35 * data_unit, mean=70.6 * u.nm, stddev=0.03 * u.nm)
    ),
    'N IV 765 - Peak': (
        m.Const1D(amplitude=0.1 * data_unit) +
        m.Gaussian1D(amplitude=1 * data_unit, mean=76.51 * u.nm, stddev=0.05 * u.nm)
    ),
    'Ne VIII 770 - Peak': (
        m.Const1D(amplitude=0.1 * data_unit) +
        m.Gaussian1D(amplitude=4 * data_unit, mean=77.04 * u.nm, stddev=0.03 * u.nm)
    ),
    'Ly-gamma-CIII group (Merged)': (
        m.Const1D(amplitude=0.1 * data_unit) +
        m.Gaussian1D(amplitude=25 * data_unit, mean=97.7 * u.nm, stddev=0.03 * u.nm)
    ),
    'Ly Beta 1025 (Merged)': (
        m.Const1D(amplitude=0.1 * data_unit) +
        m.Gaussian1D(amplitude=19 * data_unit, mean=102.56 * u.nm, stddev=0.05 * u.nm)
    ),
    'O VI 1032 - Peak': (
        m.Const1D(amplitude=0.1 * data_unit) +
        m.Gaussian1D(amplitude=15 * data_unit, mean=103.18 * u.nm, stddev=0.04 * u.nm)
    ),
}
```

```{code-cell} ipython3
def fit_single_spice_file(filename, initial_models):
    keys = tuple(initial_models.keys())
    spice = read_spice_l2_fits([filename])[0, 120:-120, :]
    assert tuple(spice.keys()) == keys

    # Remove NDMeta because of https://github.com/sunpy/ndcube/issues/847
    new_items = []
    for key, cube in spice.items():
        new_items.append((key, NDCube(cube, meta=dict(cube.meta))))
    spice = NDCollection(new_items, aligned_axes=tuple(spice.aligned_axes.values()))
    
    spice_model_fits = {
        key: parallel_fit_dask(
            data=cube,
            model=initial_models[key],
            fitter=TRFLSQFitter(),
            fitting_axes=0,
            fitter_kwargs={"filter_non_finite": True}, # Filter out non-finite values,
            scheduler="default",
        )
        for key, cube in spice.items()
    }

    return filename, spice_model_fits
    
```

```{code-cell} ipython3
spice, spice_model_fits = fit_single_spice_file(all_files[1], initial_models)
```

```{code-cell} ipython3
from dask.distributed import LocalCluster
cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=1)
client = cluster.get_client()
client
```

```{code-cell} ipython3
bag_o_files = db.from_sequence(all_files[:10])
```

```{code-cell} ipython3
fit_bag = dask.bag.map(fit_single_spice_file, bag_o_files, initial_models)
```

```{code-cell} ipython3
fit_bag.compute()
```

```{code-cell} ipython3
spice_model_fits = {
    key: delayed(parallel_fit_dask)(
        data=cube,
        model=average_fits[key],
        fitter=TRFLSQFitter(),
        fitting_axes=0,
        fitter_kwargs={"filter_non_finite": True}, # Filter out non-finite values,
        scheduler="default",
    )
    for key, cube in spice.items()
}
```

```{code-cell} ipython3
[m.compute() for m in spice_model_fits.values()]
```

```{code-cell} ipython3
def plot_spice_fit(spice, spice_model_fits):
    fig = plt.figure(figsize=(11, 6))
    for i, (key, cube) in enumerate(spectral_means.items()):
        ax = fig.add_subplot(3, 3, i + 1, projection=cube)
        model_fit = spice_model_fits[key]

        g1_peak_shift = model_fit.mean_1.quantity.to(u.km/u.s, equivalencies=u.doppler_optical(u.Quantity(average_fits[key].mean_1)))
        g1_max = np.nanpercentile(np.abs(g1_peak_shift.value), 90)
        mean_1 = ax.imshow(g1_peak_shift.value, cmap="coolwarm", vmin=-g1_max, vmax=g1_max)
        #fig.colorbar(mean_1, ax=axs[1], extend="both", label=f"Velocity from Doppler shift [{g1_peak_shift.unit:latex}]", shrink=0.8)

        #i1_max = np.nanpercentile(np.abs(model_fit.amplitude_1.value), 95)
        #peak = ax.imshow(model_fit.amplitude_1.value, cmap="plasma", vmin=-i1_max, vmax=i1_max)

        ax.set_title(key, pad=40)
        ax.set_aspect(cube.meta["CDELT2"] / cube.meta["CDELT1"])
        ax.coords[0].set_ticklabel(exclude_overlapping=True)
        ax.coords[0].set_axislabel("Helioprojective Longitude")
        ax.coords[1].set_axislabel("Helioprojective Latitude")
        ax.coords[2].set_ticklabel(exclude_overlapping=True)
    
    fig.tight_layout()
```

```{code-cell} ipython3
plot_spice_fit(spice, spice_model_fits)
```
