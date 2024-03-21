import argparse
import psycopg2
import psycopg2.extras
import numpy as np
import json
import random
import string
import time
from datetime import datetime, date, timedelta
from multiprocessing import Process, Value
import threading
from collections import namedtuple

start_date = date(2023, 3, 20)

# Shared counters for update operations
total_operations = Value('i', 0)  # Total number of operations completed by all workers
total_duration = Value('d', 0.0)  # Total duration of all operations in seconds

# Shared counters for query operations
total_queries = Value('i', 0)
total_query_time = Value('d', 0.0)

def parse_arguments():

    # Argument parsing for script configuration
    parser = argparse.ArgumentParser(description="PostgreSQL Performance and Vacuum Optimization Experiment.")
    parser.add_argument("--db-name", type=str, required=True, help="The name of the database.")
    parser.add_argument("--db-user", type=str, required=True, help="The username for the database.")
    parser.add_argument("--db-password", type=str, required=True, help="The password for the database.")
    parser.add_argument("--db-host", type=str, default="localhost", help="The host of the database.")
    parser.add_argument("--initial-rows", type=int, default=10000, help="Total number of rows to initially insert.")
    parser.add_argument("--updated-percentage", type=int, default=20, help="Percentage of initially inserted rows to update.")
    parser.add_argument("--updates-per-cycle", type=int, default=100, help="Number of rows to update in each update cycle.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for update operations.")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the experiment in seconds.")
    parser.add_argument("--disable-autovacuum", action='store_true', help="Disable autovacuum for manual vacuum control.")
    parser.add_argument("--manualvacuum-enable", action='store_true', help="Enable manual vacuum, disable to avoid manual VACUUM calls.")
    parser.add_argument("--manualvacuum-interval", type=int, default=5, help="Interval between each manual vacuum call in seconds.")
    parser.add_argument("--extra-columns", type=int, default=0, help="Number of extra columns to add to the table.")
    parser.add_argument("--num-indexes", type=int, default=0, help="Number of indexes to add to the table.")
    parser.add_argument("--num-partitions", type=int, default=0, help="Number of table partitions.")

    args = parser.parse_args()
    return args

def is_partitioned(args):
    return args.num_partitions > 1

def get_num_partitions(args):
    return args.num_partitions if is_partitioned(args) else 0

def get_table_name(args):
    return f"test_data_{args.initial_rows}_c{args.extra_columns}_i{args.num_indexes}_p{args.num_partitions}"  # Include the desired size in the table name

def create_stored_procedures(args, cur):
    table_name = get_table_name(args)
    partitions = get_num_partitions(args)
    column_usages = ''.join([f", extra{i}" for i in range(1, args.extra_columns + 1)])
    column_data_def = ''.join([f"extra_data{i} text;\n" for i in range(1, args.extra_columns + 1)])
    column_data = ''.join([f"extra_data{i} := md5(random()::text);\n" for i in range(1, args.extra_columns + 1)])
    column_data_usages = ''.join([f", extra_data{i}" for i in range(1, args.extra_columns + 1)])
    BULK_INSERT_PROCEDURE = f"""
    CREATE OR REPLACE PROCEDURE bulk_insert_data(total_rows int, start_date date)
    LANGUAGE plpgsql
    AS $$
    DECLARE
        batch_size int := 5000;
        i int := 0;
        data_text text;
        {column_data_def}
    BEGIN
        LOOP
            EXIT WHEN i >= total_rows;
            data_text := md5(random()::text);
            {column_data}
            INSERT INTO {table_name} (data, event_date {column_usages})
            SELECT
                data_text,
                start_date + (random() * {partitions-1})::int {column_data_usages}
            FROM generate_series(1, LEAST(batch_size, total_rows - i));
            i := i + batch_size;
        END LOOP;
    END;
    $$;
    """

    BULK_UPDATE_PROCEDURE = f"""
    CREATE OR REPLACE PROCEDURE bulk_update_data(ids int[])
    LANGUAGE plpgsql
    AS $$
    DECLARE
        id_param int;
    BEGIN
        FOREACH id_param IN ARRAY ids LOOP
            UPDATE {table_name} SET data = md5(random()::text) WHERE id = id_param;
        END LOOP;
    END;
    $$;
    """

    print(f"{datetime.now()} - Creating stored procedures...")
    cur.execute(BULK_INSERT_PROCEDURE)
    cur.execute(BULK_UPDATE_PROCEDURE)
    print(f"{datetime.now()} - BULK_INSERT_PROCEDURE: {BULK_INSERT_PROCEDURE}")
    print(f"{datetime.now()} - BULK_UPDATE_PROCEDURE: {BULK_UPDATE_PROCEDURE}")
    print(f"{datetime.now()} - Stored procedures created.")

def create_non_partitioned_table(args, cur):
    table_name = get_table_name(args)
    print(f"{datetime.now()} - Creating non-partitioned table {table_name} with {args.extra_columns} extra columns...")
    
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")

    column_definitions = ''.join([f"extra{i} TEXT, " for i in range(1, args.extra_columns + 1)])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            data TEXT NOT NULL,
            {column_definitions}
            event_date DATE NOT NULL
        );
    """)

def create_partitioned_table(args, cur):
    table_name = get_table_name(args)
    partitions = get_num_partitions(args)
    print(f"{datetime.now()} - Creating partitioned table {table_name} with daily partitions and {args.extra_columns} extra columns...")
    
    cur.execute(f"DROP TABLE IF EXISTS {table_name};")

    column_definitions = ''.join([f"extra{i} TEXT, " for i in range(1, args.extra_columns + 1)])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL,
            data TEXT NOT NULL,
            {column_definitions}
            event_date DATE NOT NULL,
            PRIMARY KEY (id, event_date)
        ) PARTITION BY RANGE (event_date);
    """)

    # Create daily partitions
    for i in range(partitions):
        partition_date = start_date + timedelta(days=i)
        partition_name = f"{table_name}_{partition_date.strftime('%Y%m%d')}"
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
            FOR VALUES FROM (%s) TO (%s);
        """, (partition_date, partition_date + timedelta(days=1)))

def has_n_partitions(args, cur):
    """
    Check if the specified table has the expected number of partitions.
    
    :param cur: The database cursor.
    :param table_name: The name of the table to check.
    :param expected_partition_count: The expected number of partitions.
    :return: True if the table has the expected number of partitions, False otherwise.
    """
    table_name = get_table_name(args)
    partitions = get_num_partitions(args)
    cur.execute("""
        SELECT COUNT(*)
        FROM pg_inherits
        JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
        JOIN pg_class child ON pg_inherits.inhrelid = child.oid
        WHERE parent.relname = %s;
    """, (table_name,))
    partition_count = cur.fetchone()[0]
    return partition_count == partitions

def initialize_table_with_mixed_states(args):
    table_name = get_table_name(args)
    """
    Create a table with a mix of inserted and updated rows to simulate different database states.
    """
    print(f"{datetime.now()} - Checking table status for {table_name}...")

    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
    conn.autocommit = True
    cur = conn.cursor()

    skip_insert = False

    # Check if the specific-sized table already exists
    cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = %s AND table_schema = 'public'", (table_name,))
    if cur.fetchone()[0] == 1:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        if cur.fetchone()[0] == args.initial_rows and has_n_partitions(args, cur):
            # Table exists, skip initialization
            print(f"{datetime.now()} - Table {table_name} exists with the desired number of rows. Skipping truncation and initial population.")
            skip_insert = True
            print(f"{datetime.now()} - Vacuuming table {table_name}.")
            cur.execute(f"VACUUM FULL {table_name}")

    if skip_insert:
        create_stored_procedures(args, cur)
    else:
        # Table doesn't exist, create it and insert data
        if is_partitioned(args):
            create_partitioned_table(args, cur)
        else:
            create_non_partitioned_table(args, cur)

        create_indexes(args, cur)
        create_stored_procedures(args, cur)

        print(f"{datetime.now()} - Populating table {table_name} with {args.initial_rows} rows and {args.extra_columns} extra columns via bulk insert...")
        # Use CALL for procedures instead of callproc
        cur.execute("CALL bulk_insert_data(%s, %s);", (args.initial_rows, start_date))

    # Adjust autovacuum setting based on command-line flag
    adjust_autovacuum_setting(args)

    print(f"{datetime.now()} - Table {table_name} applying initial updates.")
    # Update specified percentage of rows
    update_count = int(args.initial_rows * (args.updated_percentage / 100.0))
    apply_batch_updates(cur, 0, update_count, 1, args.initial_rows)

    print(f"{datetime.now()} - Table {table_name} preparation completed.")

    cur.close()
    conn.close()

def create_indexes(args, cur):
    table_name = get_table_name(args)
    indexes = args.num_indexes
    print(f"{datetime.now()} - Creating {indexes} indexes on {table_name}...")
    for i in range(1, min(indexes, args.extra_columns) + 1):
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_extra{i}_idx ON {table_name} (extra{i});")

def apply_batch_updates(cur, worker_id, update_count, start_id, end_id):
    """
    Apply a batch of updates to the table using a stored procedure.
    Zipfian distribution is used to select the rows for updates.
    """

    zipfian_a = 1.2  # Zipf distribution parameter
    # Generate a list of IDs to update based on Zipfian distribution within this worker's range
    update_ids = np.random.zipf(a=zipfian_a, size=update_count)
    update_ids = np.mod(update_ids, (end_id - start_id)) + start_id
    update_ids = update_ids.tolist()

    # print(f"{datetime.now()} - Running bulk update stored procedure...")
    # Use CALL for procedures instead of callproc
    cur.execute(f"CALL bulk_update_data(ARRAY{update_ids}::int[]);")

def continuous_update(args, worker_id, end_time):
    """
    Continuously update rows in the table using a Zipfian distribution for row selection.
    """

    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    while True:
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
        conn.autocommit = True
        cur = conn.cursor()

        try:
            while datetime.now() < end_time:
                start_time = datetime.now()

                total_rows = args.initial_rows
                rows_per_worker = total_rows // args.num_workers
                start_id = worker_id * rows_per_worker + 1
                end_id = start_id + rows_per_worker
                if worker_id == args.num_workers - 1:  # Ensure the last worker covers the remaining rows
                    end_id = total_rows

                apply_batch_updates(cur, worker_id, args.updates_per_cycle, start_id, end_id)

                duration = (datetime.now() - start_time).total_seconds()
                with total_operations.get_lock():
                    total_operations.value += args.updates_per_cycle
                    total_duration.value += duration
                # print(f"{datetime.now()} - Worker {worker_id}: Batch update completed.")
            break
        except Exception as e:
            print(f"Error in worker {worker_id}: {e}")
        finally:
            cur.close()
            conn.close()

def execute_queries(args, end_time):
    table_name = get_table_name(args)

    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    while True:
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
        cur = conn.cursor()
        try:
            while datetime.now() < end_time:
                start_time = time.time()
                cur.execute(f"SELECT * FROM {table_name}")
                _ = cur.fetchall()
                duration = time.time() - start_time
                
                with total_queries.get_lock(), total_query_time.get_lock():
                    total_queries.value += 1
                    total_query_time.value += duration
            break

        except Exception as e:
            print(f"Error in execute_queries: {e}")
        finally:
            cur.close()
            conn.close()

def monitor_performance(args, end_time):
    """
    Periodically log the average latency, cumulative throughput, and the percentage of live and dead tuples.
    """
    table_name = get_table_name(args)

    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
    conn.autocommit = True
    cur = conn.cursor()
    
    last_autovacuum_timestamp = None
    autovacuum_count = 0

    while datetime.now() < end_time:
        time.sleep(1)
        with total_operations.get_lock(), total_duration.get_lock(), total_queries.get_lock(), total_query_time.get_lock():
            live_percentage = -1.0
            dead_percentage = -1.0
            average_update_latency = -1.0
            average_query_latency = -1.0
            
            # Fetch live and dead tuple counts
            cur.execute("""
                SELECT n_live_tup, n_dead_tup
                FROM pg_stat_user_tables
                WHERE relname = %s;
            """, (table_name,))
            result = cur.fetchone()
            if result:
                n_live_tup, n_dead_tup = result
                total_tup = n_live_tup + n_dead_tup
                if total_tup > 0:
                    live_percentage = (n_live_tup / total_tup) * 100
                    dead_percentage = (n_dead_tup / total_tup) * 100
            
            # Print operation and query metrics as before
            if total_operations.value > 0:
                average_update_latency = total_duration.value / total_operations.value
            if total_queries.value > 0:
                average_query_latency = total_query_time.value / total_queries.value

            cur.execute("""
                SELECT last_autovacuum
                FROM pg_stat_user_tables
                WHERE relname = %s;
            """, (table_name,))
            result = cur.fetchone()

            if result and result[0] is not None:  # Ensure there is a timestamp to compare
                current_timestamp = result[0]
                if last_autovacuum_timestamp is None or current_timestamp > last_autovacuum_timestamp:
                    autovacuum_count += 1
                    last_autovacuum_timestamp = current_timestamp
            
            print(f"{datetime.now()} - AutoVacCount: {autovacuum_count}, Live Tuples: {live_percentage:.2f}%, Dead Tuples: {dead_percentage:.2f}%, Query Avg Latency: {average_query_latency:.4f}s/query, Query Throughput: {total_queries.value} queries, Update Avg Latency: {average_update_latency:.4f}s/op, Update Throughput: {total_operations.value} ops")
    
    cur.close()
    conn.close()

def vacuum_worker(args, end_time):
    table_name = get_table_name(args)

    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
    conn.autocommit = True  # VACUUM cannot run in a transaction block
    cur = conn.cursor()

    vacuum_count = 0  # Keep track of how many times VACUUM has been executed

    while datetime.now() < end_time:
        vacuum_start_time = time.time()
        
        # Execute the VACUUM command (consider using VACUUM (ANALYZE, VERBOSE) for more detailed output)
        cur.execute(f"VACUUM {table_name};")
        
        vacuum_duration = time.time() - vacuum_start_time
        vacuum_count += 1
        
        print(f"{datetime.now()} - VACUUM completed on {table_name}. Duration: {vacuum_duration:.2f}s. Total vacuums: {vacuum_count}")
        
        time.sleep(args.vacuum_interval)  # Wait for a second before the next VACUUM

    cur.close()
    conn.close()

def adjust_autovacuum_setting(args):
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    HOST = args.db_host

    conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST)
    conn.autocommit = True
    cur = conn.cursor()

    table_name = get_table_name(args)

    autovacuum_setting = 'false' if args.disable_autovacuum else 'true'
    print(f"{datetime.now()} - Setting autovacuum: {autovacuum_setting} for table {table_name}.")

    if is_partitioned(args):
        for i in range(get_num_partitions(args)):
            partition_date = start_date + timedelta(days=i)
            partition_name = f"{table_name}_{partition_date.strftime('%Y%m%d')}"

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {partition_name} PARTITION OF {table_name}
                FOR VALUES FROM (%s) TO (%s);
            """, (partition_date, partition_date + timedelta(days=1)))

            # Set autovacuum_enabled storage parameter for each partition
            cur.execute(f"""
                ALTER TABLE {partition_name} SET (autovacuum_enabled = {autovacuum_setting}, toast.autovacuum_enabled = {autovacuum_setting});
            """)
    else:
        cur.execute(f"ALTER TABLE {table_name} SET (autovacuum_enabled = {autovacuum_setting}, toast.autovacuum_enabled = {autovacuum_setting});")

    cur.close()
    conn.close()


def main(args, barrier):
    initialize_table_with_mixed_states(args)

    if barrier:
        barrier.wait()

    end_time = datetime.now() + timedelta(seconds=args.duration)
    
    processes = [Process(target=continuous_update, args=(args, i, end_time)) for i in range(args.num_workers)]
    for p in processes:
        p.start()
    
    query_process = Process(target=execute_queries, args=(args, end_time,))
    query_process.start()

    monitoring_thread = threading.Thread(target=monitor_performance, args=(args, end_time,), daemon=True)
    monitoring_thread.start()

    if args.manualvacuum_enable:
        vacuum_process = Process(target=vacuum_worker, args=(args, end_time,))
        vacuum_process.start()

    for p in processes:
        p.join()
    query_process.join()
    monitoring_thread.join()
    if args.manualvacuum_enable:
        vacuum_process.join()

def run_with_default_settings(barrier, env_info):
    ManualInput = namedtuple(
        "ManualInput",
        [
            "db_name",
            "db_host",
            "db_user",
            "db_password",
            "initial_rows",
            "updated_percentage",
            "updates_per_cycle",
            "num_workers",
            "duration",
            "disable_autovacuum",
            "manualvacuum_enable",
            "manualvacuum_interval",
            "extra_columns",
            "num_indexes",
            "num_partitions",
        ],
    )
    args = ManualInput(
        db_name=env_info["db_name"],
        db_host=env_info["db_host"],
        db_user=env_info["db_user"],
        db_password=env_info["db_pwd"],
        initial_rows=500_000,
        updated_percentage=5,
        updates_per_cycle=10_000,
        num_workers=50,
        duration=120,
        disable_autovacuum=False,
        manualvacuum_enable=False,
        manualvacuum_interval=1,
        extra_columns=0,
        num_indexes=0,
        num_partitions=0,
    )
    main(args, barrier)

if __name__ == "__main__":
    args = parse_arguments()
    main(args, None)
