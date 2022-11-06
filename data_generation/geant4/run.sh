NSIM=1000
NTHREADS=12
NPERTHREAD=$(expr $NSIM/$NTHREADS)
for i in $(seq 1 $NTHREADS)
do  
    echo $i
    { for j in $(seq 1 $NTHREADS)
    do
        t=$(time ./mu run.mac out${i}_${j} > /dev/null 2>&1)
        echo $i $j
    done }&
done