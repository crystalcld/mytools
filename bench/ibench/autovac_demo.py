import os
import sys
from threading import Thread

def do_run(cmd):
    print("Running command: %s" % cmd)
    os.system(cmd)

if __name__ == '__main__':
    cmd_base = ('python3 iibench.py --setup --dbms=postgres --db_user=postgres --max_rows=100000000 '
                '--secs_per_report=10 --query_threads=3 --delete_per_insert --max_seconds=120 --initial_size=100000 '
                '--inserts_per_second=0 --initial_autovac_delay=5 --db_name=postgres --rows_per_commit=10000')

    cmd_nopid = cmd_base + ' --db_host='+sys.argv[1]+' --db_password='+sys.argv[2]+' --tag="no_pid"'
    cmd_pid = cmd_base + ' --db_host='+sys.argv[3]+' --db_password='+sys.argv[4]+' --tag="pid" --enable_pid --control_autovac'

    t1 = Thread(target = do_run, args = (cmd_nopid, ))
    t2 = Thread(target = do_run, args = (cmd_pid, ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    os.system("cat no_pid_dataQuery_thread_#* | sort -nr > no_pid_latencies.txt")
    os.system("cat pid_dataQuery_thread_#* | sort -nr > pid_latencies.txt")
    os.system('echo "plot \'no_pid_latencies.txt\'\nreplot \'pid_latencies.txt\'\npause -1" > gnuplot_script.txt ')
    os.system('gnuplot gnuplot_script.txt')
