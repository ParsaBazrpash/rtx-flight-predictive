# RTX Flight Predictive Analytics — DFW Dataset Pipeline

This repository contains the dataset pipeline for the RTX-sponsored Predictive Analytics Application.

## Current data scope

- **Airport:** DFW-origin flights only
- **Years:** 2016-2025
- **Flight source:** U.S. DOT/BTS Reporting Carrier On-Time Performance
- **Weather source:** NOAA/NCEI Global Hourly
- **NOAA station:** 72259003927 (Dallas/Fort Worth International Airport)
- **Weather match:** nearest observation to scheduled departure, within 75 minutes

## Build the dataset

Install dependencies:

```bash
pip install -r data_pipeline/requirements.txt
```

Run:

```bash
python data_pipeline/build_dfw_rtx_dataset.py \
  --start-year 2016 \
  --end-year 2025 \
  --origin DFW \
  --station 72259003927 \
  --output-dir rtx_dfw_data
```

## Outputs

The final directory contains:

- `rtx_dfw_flights_weather_2016_2025.parquet` — complete joined flight + weather data
- `rtx_dfw_flights_weather_2016_2025_model_input.parquet` — ML-safe predictors + project targets
- `dfw_weather_2016_2025.parquet` — cleaned DFW weather observations
- `data_dictionary.csv`
- `dataset_summary.json`
- `safe_feature_columns.txt`
- `target_columns.txt`

The four prediction targets are:

- departure delay >= 15 minutes
- departure delay duration
- cancellation
- diversion

Post-event fields are prefixed with `outcome_` and must not be used as pre-departure model inputs because they would introduce data leakage.

## GitHub Actions

The workflow in `.github/workflows/build-dfw-dataset.yml` builds the complete 2016-2025 dataset and uploads the finished files as downloadable GitHub Actions artifacts.

## Official sources

- BTS: https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=&gnoyr_VQ=FGJ
- BTS pre-zipped files: https://transtats.bts.gov/PREZIP/
- NOAA Global Hourly: https://www.ncei.noaa.gov/access/search/datasets/global-hourly/
