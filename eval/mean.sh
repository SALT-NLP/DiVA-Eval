awk -F "\[SEP_DIAL\]" '{ sum += $3 } END { if (NR > 0) print sum / NR }'
