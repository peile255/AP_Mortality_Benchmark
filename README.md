# AP Mortality Benchmark

Research-only retrospective secondary analysis of mortality prediction in critically ill patients with acute pancreatitis. Not a validated clinical decision tool.

## Reproduction

Python 3.13 and pinned dependencies in `requirements.txt`. Obtain the source workbook separately (see `data/README.md`); it is not redistributed.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest test_pipeline -v
python run_all.py --smoke --out smoke_test
python run_all.py --bootstrap 1000 --jobs 1 --out final_run
python compare_results.py final_run
python run_v3.py --repeats 30 --outer 500 --oob 1000 --jobs 1 --v2 final_run --out v3_final
python code/make_publication_figures.py
```

`results/` contains previously generated aggregate tables; the figure script uses these tables. Fresh analysis outputs are written to `final_run/` and `v3_final/`. See `documentation/V2.md` and `documentation/V3.md` for estimands, limitations and cluster execution. The SLURM script is an example requiring local module and partition settings. Do not run compute-intensive tasks on a login node.

The analyses are exploratory, conditional on a released imputed dataset, and lack independent external validation. The repeated-CV percentile ranges describe split variability; outer-bootstrap percentile intervals are exploratory, not confirmatory tests. No patient-level data or predictions are included.
