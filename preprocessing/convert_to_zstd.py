import os
import sys
import glob
import time
import duckdb
import pyarrow.parquet as pq

def check_and_convert_zstd(data_dir: str):
    print("=" * 80)
    print(" PARQUET DATASET ZSTD COMPRESSION CHECK & CONVERSION")
    print(f" Target Directory: {data_dir}")
    print("=" * 80)

    parquet_files = sorted(glob.glob(os.path.join(data_dir, '**', '*.parquet'), recursive=True))
    
    target_files = []
    for f in parquet_files:
        if 'model=' in f:
            continue
        target_files.append(f)

    print(f"\n[Step 1] Found {len(target_files)} main Parquet dataset files.\n")
    
    non_zstd_files = []
    zstd_files = []

    for f in target_files:
        with open(f, 'rb') as fp:
            pf = pq.ParquetFile(fp)
            codec = pf.metadata.row_group(0).column(0).compression
        rel_path = os.path.relpath(f, data_dir)
        if codec.upper() != 'ZSTD':
            non_zstd_files.append((f, codec))
            print(f"  - [NON-ZSTD] {rel_path:<60} | Current Codec: {codec}")
        else:
            zstd_files.append(f)
            print(f"  - [  ZSTD  ] {rel_path:<60} | Current Codec: {codec}")

    print("\n" + "-" * 80)
    print(f" Summary: {len(zstd_files)} files already ZSTD | {len(non_zstd_files)} files need conversion to ZSTD")
    print("-" * 80)

    if not non_zstd_files:
        print("\nAll dataset Parquet files are already using ZSTD compression! No conversion needed.")
        return

    total_orig_bytes = 0
    total_new_bytes = 0

    for idx, (fpath, current_codec) in enumerate(non_zstd_files, 1):
        rel_path = os.path.relpath(fpath, data_dir)
        orig_size_bytes = os.path.getsize(fpath)
        orig_size_mb = orig_size_bytes / (1024 * 1024)
        total_orig_bytes += orig_size_bytes

        print(f"\n[{idx}/{len(non_zstd_files)}] Converting '{rel_path}' ({current_codec} -> ZSTD)...")
        t0 = time.time()

        fpath_clean = fpath.replace("\\", "/")
        tmp_fpath = fpath + ".tmp_zstd"
        tmp_fpath_clean = tmp_fpath.replace("\\", "/")

        with open(fpath, 'rb') as fp:
            ref_schema = pq.read_schema(fp)
        expected_cols = ref_schema.names

        # Open connection per file
        con = duckdb.connect(database=':memory:')
        con.execute("SET memory_limit='8GB';")
        con.execute("SET threads=4;")

        # 1. Inspect source statistics
        source_row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{fpath_clean}')").fetchone()[0]
        null_queries = [f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "null_{col}"' for col in expected_cols]
        source_nulls = con.execute(f"SELECT {', '.join(null_queries)} FROM read_parquet('{fpath_clean}')").fetchdf().iloc[0].to_dict()

        # 2. Re-encode to ZSTD
        if os.path.exists(tmp_fpath):
            os.remove(tmp_fpath)

        con.execute(f"COPY (SELECT * FROM read_parquet('{fpath_clean}')) TO '{tmp_fpath_clean}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        # Close DuckDB connection so read locks on fpath are released!
        con.close()

        # 3. Post-conversion strict verification
        con_verify = duckdb.connect(database=':memory:')
        new_row_count = con_verify.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp_fpath_clean}')").fetchone()[0]
        with open(tmp_fpath, 'rb') as fp:
            merged_schema = pq.read_schema(fp)
        new_cols = merged_schema.names
        new_nulls = con_verify.execute(f"SELECT {', '.join(null_queries)} FROM read_parquet('{tmp_fpath_clean}')").fetchdf().iloc[0].to_dict()
        con_verify.close()

        mismatches = []
        if source_row_count != new_row_count:
            mismatches.append(f"Row count mismatch: {source_row_count:,} vs {new_row_count:,}")
        if expected_cols != new_cols:
            mismatches.append("Column schema mismatch")
        
        for col in expected_cols:
            k = f"null_{col}"
            if source_nulls[k] != new_nulls[k]:
                mismatches.append(f"Null count mismatch for '{col}': {source_nulls[k]} vs {new_nulls[k]}")

        if mismatches:
            if os.path.exists(tmp_fpath):
                os.remove(tmp_fpath)
            raise ValueError(f"Integrity check failed for {rel_path}: {mismatches}")

        # 4. Replace original file atomically with retries
        replaced = False
        for attempt in range(10):
            try:
                os.replace(tmp_fpath, fpath)
                replaced = True
                break
            except Exception as e:
                time.sleep(0.5)

        if not replaced:
            os.replace(tmp_fpath, fpath)

        new_size_bytes = os.path.getsize(fpath)
        new_size_mb = new_size_bytes / (1024 * 1024)
        total_new_bytes += new_size_bytes
        saved_mb = orig_size_mb - new_size_mb
        ratio = (1 - (new_size_bytes / orig_size_bytes)) * 100

        print(f"  [OK] Done in {time.time() - t0:.2f}s | Size: {orig_size_mb:.2f} MB -> {new_size_mb:.2f} MB (Saved {saved_mb:.2f} MB / {ratio:.1f}%)")

    total_saved_mb = (total_orig_bytes - total_new_bytes) / (1024 * 1024)
    total_saved_gb = total_saved_mb / 1024
    total_ratio = (1 - (total_new_bytes / total_orig_bytes)) * 100

    print("\n" + "=" * 80)
    print(" ZSTD CONVERSION AND INTEGRITY VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"  - Total Original Size  : {total_orig_bytes / (1024**2):,.2f} MB ({total_orig_bytes / (1024**3):,.2f} GB)")
    print(f"  - Total ZSTD Size      : {total_new_bytes / (1024**2):,.2f} MB ({total_new_bytes / (1024**3):,.2f} GB)")
    print(f"  - Total Space Saved    : {total_saved_mb:,.2f} MB ({total_saved_gb:,.2f} GB) - {total_ratio:.1f}% reduction")
    print("=" * 80)

if __name__ == "__main__":
    data_directory = r"C:\Workspace\projects\26_2_COIN\data"
    check_and_convert_zstd(data_directory)
