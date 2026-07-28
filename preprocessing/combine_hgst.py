import os
import sys
import glob
import time
import duckdb
import pyarrow.parquet as pq

def main():
    raw_dir = r"C:\Workspace\projects\26_2_COIN\data\raw"
    source_folder = os.path.join(raw_dir, "model=HGST_20HUH721212ALN604")
    output_file = os.path.join(raw_dir, "HGST_20HUH721212ALN604.parquet")
    
    print("=" * 80)
    print(" HGST_20HUH721212ALN604 RAW PARQUET MERGE & INTEGRITY VALIDATION")
    print(f" - Source Folder : {source_folder}")
    print(f" - Output File   : {output_file}")
    print(f" - Compression   : ZSTD (Zstandard)")
    print("=" * 80)

    # 1. Source Files Inventory
    pattern = os.path.join(source_folder, "**", "*.parquet")
    source_files = sorted(glob.glob(pattern, recursive=True))
    n_files = len(source_files)
    print(f"\n[Step 1] Scanning source directory...")
    print(f"  - Found {n_files:,} daily source Parquet files.")
    if n_files == 0:
        raise FileNotFoundError(f"No parquet files found in {source_folder}")

    t0 = time.time()
    con = duckdb.connect(database=':memory:')
    con.execute("SET memory_limit='8GB';")
    con.execute("SET threads=4;")

    # 2. Inspect Source Data & Pre-Merge Aggregates
    print("\n[Step 2] Inspecting source schema and collecting pre-merge statistics...")
    source_glob_path = os.path.join(source_folder, "*.parquet").replace("\\", "/")
    
    sample_file = source_files[0]
    ref_schema = pq.read_schema(sample_file)
    raw_cols = ref_schema.names
    expected_cols = ["date"] + raw_cols
    
    print(f"  - Raw file column count  : {len(raw_cols)}")
    print(f"  - Combined column count  : {len(expected_cols)} (Added 'date' as 1st column)")
    
    # Base SELECT query for source reading with date parsing
    col_select_str = "CAST(regexp_extract(filename, '(\\d{4}-\\d{2}-\\d{2})') AS DATE) AS date, " + ", ".join([f'"{c}"' for c in raw_cols])
    
    source_stats_query = f"""
        WITH source_df AS (
            SELECT {col_select_str}
            FROM read_parquet('{source_glob_path}', filename=true)
        )
        SELECT 
            COUNT(*) AS total_rows,
            COUNT(DISTINCT serial_number) AS unique_serials,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(CAST(failure AS BIGINT)) AS total_failures
        FROM source_df
    """
    source_stats = con.execute(source_stats_query).fetchdf().iloc[0].to_dict()
    
    print(f"  - [Source Stats] Total Rows       : {source_stats['total_rows']:,}")
    print(f"  - [Source Stats] Unique Serials   : {source_stats['unique_serials']:,}")
    print(f"  - [Source Stats] Date Range       : {source_stats['min_date']} ~ {source_stats['max_date']}")
    print(f"  - [Source Stats] Total Failures   : {int(source_stats['total_failures']):,}")

    # Calculate pre-merge null count for each column
    col_null_queries = [f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "null_{col}"' for col in expected_cols]
    source_null_query = f"""
        WITH source_df AS (
            SELECT {col_select_str}
            FROM read_parquet('{source_glob_path}', filename=true)
        )
        SELECT {', '.join(col_null_queries)}
        FROM source_df
    """
    print("  - Calculating per-column null distributions across source files...")
    source_nulls = con.execute(source_null_query).fetchdf().iloc[0].to_dict()

    # 3. Combine & Export with ZSTD Compression
    print(f"\n[Step 3] Merging {n_files:,} daily files into single Parquet file (ZSTD Compression)...")
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception as e:
            print(f"  - Warning: Could not remove existing target file: {e}")
        
    out_file_clean = output_file.replace("\\", "/")
    merge_query = f"""
        COPY (
            SELECT {col_select_str}
            FROM read_parquet('{source_glob_path}', filename=true)
        ) TO '{out_file_clean}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    t_merge_start = time.time()
    con.execute(merge_query)
    merge_duration = time.time() - t_merge_start
    
    out_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    out_size_gb = out_size_mb / 1024
    print(f"  - Merged Parquet file successfully created!")
    print(f"  - Saved File Path : {output_file}")
    print(f"  - Saved File Size : {out_size_mb:.2f} MB ({out_size_gb:.2f} GB)")
    print(f"  - Merge Duration  : {merge_duration:.2f} seconds")

    # 4. Strict Integrity Validation (Post-Merge Verification)
    print("\n[Step 4] Running Strict Integrity Verification on Merged File...")
    
    merged_stats_query = f"""
        SELECT 
            COUNT(*) AS total_rows,
            COUNT(DISTINCT serial_number) AS unique_serials,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(CAST(failure AS BIGINT)) AS total_failures
        FROM read_parquet('{out_file_clean}')
    """
    merged_stats = con.execute(merged_stats_query).fetchdf().iloc[0].to_dict()
    
    merged_null_query = f"""
        SELECT {', '.join(col_null_queries)}
        FROM read_parquet('{out_file_clean}')
    """
    merged_nulls = con.execute(merged_null_query).fetchdf().iloc[0].to_dict()

    # Schema Validation
    merged_schema = pq.read_schema(output_file)
    merged_cols = merged_schema.names

    mismatches = []
    
    if len(expected_cols) != len(merged_cols):
        mismatches.append(f"Column count mismatch: Expected={len(expected_cols)}, Merged={len(merged_cols)}")
    if expected_cols != merged_cols:
        mismatches.append("Column order/names mismatch!")
        
    if source_stats['total_rows'] != merged_stats['total_rows']:
        mismatches.append(f"Row count mismatch: Source={source_stats['total_rows']:,}, Merged={merged_stats['total_rows']:,}")
        
    if source_stats['unique_serials'] != merged_stats['unique_serials']:
        mismatches.append(f"Unique serials mismatch: Source={source_stats['unique_serials']:,}, Merged={merged_stats['unique_serials']:,}")

    if str(source_stats['min_date']) != str(merged_stats['min_date']) or str(source_stats['max_date']) != str(merged_stats['max_date']):
        mismatches.append(f"Date range mismatch: Source=({source_stats['min_date']}, {source_stats['max_date']}), Merged=({merged_stats['min_date']}, {merged_stats['max_date']})")

    if source_stats['total_failures'] != merged_stats['total_failures']:
        mismatches.append(f"Total failures mismatch: Source={source_stats['total_failures']:,}, Merged={merged_stats['total_failures']:,}")

    # Check null count equality for every single column
    null_mismatches = []
    for col in expected_cols:
        key = f"null_{col}"
        if source_nulls[key] != merged_nulls[key]:
            null_mismatches.append(f"Column '{col}' Null count mismatch: Source={source_nulls[key]}, Merged={merged_nulls[key]}")

    if null_mismatches:
        mismatches.extend(null_mismatches)

    # Date-wise row count distribution validation
    date_dist_mismatch_query = f"""
        WITH s_dist AS (
            SELECT CAST(regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})') AS DATE) AS date, COUNT(*) AS cnt 
            FROM read_parquet('{source_glob_path}', filename=true) GROUP BY 1
        ),
        m_dist AS (
            SELECT date, COUNT(*) AS cnt FROM read_parquet('{out_file_clean}') GROUP BY 1
        )
        SELECT s_dist.date, s_dist.cnt AS src_cnt, m_dist.cnt AS mrg_cnt
        FROM s_dist
        FULL OUTER JOIN m_dist ON s_dist.date = m_dist.date
        WHERE s_dist.cnt != m_dist.cnt OR s_dist.cnt IS NULL OR m_dist.cnt IS NULL
    """
    dist_mismatches = con.execute(date_dist_mismatch_query).fetchall()
    if dist_mismatches:
        mismatches.append(f"Date-wise row count distribution mismatch in {len(dist_mismatches)} dates!")

    print("=" * 80)
    print(" INTEGRITY VALIDATION RESULT SUMMARY")
    print("=" * 80)
    print(f"  [OK] Total Rows              : {merged_stats['total_rows']:,} (Match: {source_stats['total_rows'] == merged_stats['total_rows']})")
    print(f"  [OK] Total Columns           : {len(merged_cols)} (Match: {expected_cols == merged_cols})")
    print(f"  [OK] Unique Serial Numbers   : {merged_stats['unique_serials']:,} (Match: {source_stats['unique_serials'] == merged_stats['unique_serials']})")
    print(f"  [OK] Date Range              : {merged_stats['min_date']} ~ {merged_stats['max_date']} (Match: True)")
    print(f"  [OK] Total Failure Count     : {int(merged_stats['total_failures']):,} (Match: {source_stats['total_failures'] == merged_stats['total_failures']})")
    print(f"  [OK] Column-wise Null Check  : All {len(expected_cols)} columns matched (Mismatches: {len(null_mismatches)})")
    print(f"  [OK] Date-wise Distribution  : All {n_files:,} dates matched perfectly (Mismatches: {len(dist_mismatches)})")
    print(f"  [OK] Output Compression Codec: ZSTD ({out_size_mb:.2f} MB / {out_size_gb:.2f} GB)")
    print("=" * 80)

    if mismatches:
        print("\n INTEGRITY VALIDATION FAILED!")
        for err in mismatches:
            print(f"  - {err}")
        raise ValueError("Data integrity verification failed!")
    else:
        print("\n PERFECT INTEGRITY VERIFIED! ZERO DATA LOSS OR OMISSION DETECTED.")
        print(f" Total Script Execution Time: {time.time() - t0:.2f} seconds")
        print("=" * 80)

    con.close()

if __name__ == "__main__":
    main()
