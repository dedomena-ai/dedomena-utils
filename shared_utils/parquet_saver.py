import os
import tempfile
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

class ParquetSaver:
    def __init__(self, inspector, file_base_name: str, user_name: str, bucket_storage):
        self.inspector = inspector
        self.file_base_name = file_base_name
        self.user_name = user_name
        self.bucket_storage = bucket_storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_name = os.path.splitext(file_base_name)[0]

    def save(self):
        fmt = getattr(self.inspector, "fmt", None)
        if fmt == "csv":
            return self._save_csv()
        elif fmt == "excel":
            return self._save_excel()
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def _save_csv(self):
        df = getattr(self.inspector, "df", None)
        if df is None or df.empty:
            return

        parquet_name = f"{self.base_name}.parquet"
        blob_path = f"{self.user_name}/{parquet_name}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_parquet_file:
            tmp_path = tmp_parquet_file.name

        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path)

        blob = self.bucket_storage.blob(blob_path)
        blob.upload_from_filename(tmp_path)

        return blob_path

    def _save_excel(self):
        
        def fix_numeric_columns(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        paths = []
        if not hasattr(self.inspector, "dataframes") or not hasattr(self.inspector, "sheet_names"):
            return paths

        for df, sheet in zip(self.inspector.dataframes, self.inspector.sheet_names):
            if df is None or df.empty:
                continue
            clean_sheet = sheet.replace(" ", "_").replace("/", "_")[:30]
            parquet_name = f"{self.base_name}_{clean_sheet}.parquet"
            blob_path = f"{self.user_name}/{parquet_name}"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_parquet_file:
                tmp_path = tmp_parquet_file.name
                
            df_clean = fix_numeric_columns(df)
            table = pa.Table.from_pandas(df_clean)
            pq.write_table(table, tmp_path)

            blob = self.bucket_storage.blob(blob_path)
            blob.upload_from_filename(tmp_path)

            paths.append(blob_path)
        return paths