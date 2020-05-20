#!/bin/bash
yr=1999
while [ $yr -le 2018 ]
do
  cd $yr
  ncecat *nc4 -O 3B42_Daily.$yr.7.nc4
  cd ..
  yr=$(( $yr + 1 ))
done
