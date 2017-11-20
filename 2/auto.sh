#!/bin/bash

counter=1
while [ $counter -le 3 ]
do
a=$(( $counter * 5 ))
python3 letor.py $a .001 .1 1000000
((counter++))
echo -e '\n'
done
