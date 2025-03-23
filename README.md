# eeg_study
Synthetic data for a somatosensory evoked potential EEG experiment

```{shell}
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To generate the raw data:

```{shell}
python3 create_dataset.py
```

To go from raw data to processed data:

```{shell}
python3 process_data.py
```

To go from processed data to the summary table `summary_table.csv`:

```{shell}
python3 generate_tables.py
```


