import csv, subprocess;

def perf_stat_counts(filename):
    src = open(filename)
    # The following are names for the observed output fields with pert stat -x, -r
    # These are described in the man page for pert-stat, but in a slightly 
    # different order.
    reader = csv.DictReader(src, ['count', 'units', 'event', 'variance', 'runtime', 'pct', 'aggregate', 'aggregate units'])
    result_dict = {}
    for row in reader:
        if row['event'] == 'cycles':
            result_dict['cycles'] = row['count']
        elif row['event'] == 'instructions':
            result_dict['instructions'] = row['count']
        elif row['event'] == 'branches':
            result_dict['branches'] = row['count']
        elif row['event'] == 'branch-misses':
            result_dict['branch-misses'] = row['count']
    return result_dict

def run_with_perf_stat(program_under_test, args):
    # First run the program without measurement to ensure
    # that generated kernels are compiled to the object cache.
    subprocess.run([program_under_test] + args, encoding="utf-8")
    # Now do a perf stat run with a repeat count of 5 and generating csv output
    outf = open("/tmp/commandout", "w")
    statsf = open("/tmp/perfout", "w")
    perf_args = ["perf", "stat", "-x,", "-r5", program_under_test] + args
    subprocess.run(perf_args, encoding="utf-8", stdout=outf, stderr=statsf)
    outf.close()
    statsf.close()
    return perf_stat_counts("/tmp/perfout")

