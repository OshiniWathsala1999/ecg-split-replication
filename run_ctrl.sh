#!/bin/bash
for b in weight oversample; do for s in 0 1 2 3 4; do
  f=results/rebalance/ctrl_B_${b}_${s}.json; [ -f $f ] && continue
  python3 rebalance_control.py --protocol B --balance $b --seed $s | tail -1
done; done
for b in weight oversample; do for s in 0 1 2; do
  f=results/rebalance/ctrl_A_${b}_${s}.json; [ -f $f ] && continue
  python3 rebalance_control.py --protocol A --balance $b --seed $s | tail -1
done; done
echo ALLDONE
