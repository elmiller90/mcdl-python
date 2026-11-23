import json
from pathlib import Path

ROOT = Path("results/dictionary_learning/paired")

rows = []

for path in sorted(ROOT.glob("snr*_seed100.json"), reverse=True):
    d = json.loads(path.read_text())

    for key in ("non_temporal", "temporal"):
        r = d[key]
        rows.append({
            "snr": d["requested_snr_db"],
            "model": r["name"],
            "clean": r["clean_error"],
            "disc": r["discrepancy"],
            "xerr": r["coefficient_error"],
            "cos_g": r["atom_cosine_0"],
            "cos_r": r["atom_cosine_1"],
            "nz": r["significant_count"],
            "conv": r["admm_converged"],
        })

rows.sort(key=lambda r: (-r["snr"], r["model"] != "non-temporal"))

print(
    "SNR  model          cleanerr   disc    Xerr      "
    "cosG      cosR      nz  conv"
)

for r in rows:
    print(
        f"{r['snr']:3.0f}  {r['model']:12s} "
        f"{r['clean']:9.3e} {r['disc']:6.3f} "
        f"{r['xerr']:9.3e} {r['cos_g']:9.6f} "
        f"{r['cos_r']:9.6f} {r['nz']:4d} "
        f"{str(r['conv']):5s}"
    )