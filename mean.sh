awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
