e=$1
nr=$2
nrt=$3

for d in 16jun17 8may18 ; do
 cd ~/b/myrocks.$d; bash ini.sh; sleep 60
 cd ~/git/mytools/bench/ibench
 bash iq.sh $e "" ~/b/myrocks.$d/bin/mysql /data/m/my/data nvme0n 1 1 no no no 0 no $nr no
 mkdir my.$nrt.rx.$d.$e ; mv l scan q100* my.$nrt.rx.$d.$e
 sleep 60; ~/b/myrocks.$d/bin/mysqladmin -uroot -ppw -h127.0.0.1 shutdown; sleep 60
done
