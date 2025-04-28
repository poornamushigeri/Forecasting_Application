import pandas as pd

def preprocess_data(df, date_col=None, target_col=None, external_cols=None, resample_freq=None):
    df = df.copy()

    # --- Auto-detect Date column if not provided ---
    if date_col is None:
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        if date_col is None:
            raise ValueError("No date column found automatically. Please select manually.")

    # --- Auto-detect Target column if not provided ---
    if target_col is None:
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric column found for target.")
        target_col = numeric_cols[0]

    # --- Parse datetime safely ---
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # --- Drop rows where date or target is NaN ---
    df = df.dropna(subset=[date_col, target_col])

    # --- Select important columns ---
    cols = [date_col, target_col] + (external_cols if external_cols else [])
    df = df[cols]

    # --- Sort by date ---
    df = df.sort_values(date_col)

    # --- Rename columns ---
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})

    # --- Check for duplicate timestamps ---
    if df.duplicated(subset=['ds']).any():
        df = df.drop_duplicates(subset=['ds'])
        print("⚠️ Warning: Duplicate timestamps found and removed.")

    # --- Force datetime again for safety ---
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

    # --- Handle resampling ---
    if resample_freq:
        df = df.set_index('ds')

        # Resample (mean aggregation) — allows missing dates to be filled
        df = df.resample(resample_freq).mean()

        # Fill missing 'y' values (created during resample) via interpolation
        df['y'] = df['y'].interpolate(method='linear')

        # Fill missing regressors (if any) with forward fill (best for time series)
        for col in df.columns:
            if col != 'y':
                df[col] = df[col].fillna(method='ffill')

        df = df.reset_index()
    else:
        # Try to infer frequency
        try:
            inferred_freq = pd.infer_freq(df['ds'])
            if inferred_freq:
                print(f"✅ Detected Frequency: {inferred_freq}")
            else:
                print("⚠️ Warning: Could not infer frequency automatically.")
        except Exception as e:
            print(f"⚠️ Frequency detection failed: {e}")

    # --- Final check: Fill any remaining NaNs safely ---
    df = df.fillna(method='ffill')  # Forward fill
    df = df.fillna(method='bfill')  # Backward fill if needed

    return df
