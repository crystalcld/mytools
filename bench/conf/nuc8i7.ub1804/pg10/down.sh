
bdir=/data/m/pg

echo "stop and remove files"
bin/pg_ctl -D $bdir stop
sleep 5
rm -rf $bdir/*
rm -f logfile; touch logfile

