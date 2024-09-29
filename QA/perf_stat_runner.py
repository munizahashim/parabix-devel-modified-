import csv, subprocess;

def perf_stat_counts(filename, perf_counters):
    src = open(filename)
    # The following are names for the observed output fields with pert stat -x, -r
    # These are described in the man page for pert-stat, but in a slightly 
    # different order.
    reader = csv.DictReader(src, ['count', 'units', 'event', 'variance', 'runtime', 'pct', 'aggregate', 'aggregate units'])
    result_dict = {}
    for row in reader:
        if row['event'] in perf_counters:
            result_dict[row['event']] = row['count']
    return result_dict

def run_with_perf_stat(program_under_test, args, perf_counters):
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
    return perf_stat_counts("/tmp/perfout", perf_counters)

class PerformanceTester:
    def __init__(self, program_under_test):
        self.PUT = program_under_test
        self.keyword_list = []
        self.positional_parameter_list = []
        self.performance_counters = ['instructions', 'cycles', 'branches', 'branch-misses']
        self.parameter_map = {}

    def addKeywordParameter(self, keyword, choices):
        self.keyword_list.append(keyword)
        self.parameter_map[keyword] = choices

    def addPositionalParameter(self, name, choices):
        self.positional_parameter_list.append(name)
        self.parameter_map[name] = choices

    def run_1_test(self, run_parameter_map):
        keyword_parms = [kw + "=" + run_parameter_map[kw] for kw in self.keyword_list]
        positional_parms = [run_parameter_map[p] for p in self.positional_parameter_list]
        result_map = run_with_perf_stat(self.PUT, keyword_parms + positional_parms, self.performance_counters)
        for kw in self.keyword_list:
            result_map[kw] = run_parameter_map[kw]
        for kw in self.positional_parameter_list:
            result_map[kw] = run_parameter_map[kw]
        self.writer.writerow(result_map)

    def run_tests(self, report_file):
        csv_sink = open(report_file, 'w')
        self.writer = csv.DictWriter(csv_sink, self.keyword_list + self.positional_parameter_list + self.performance_counters)
        self.writer.writeheader()
        run_parameter_map = {}
        self.run_tests_with_parms(run_parameter_map, self.keyword_list, self.positional_parameter_list)
        csv_sink.close()

    def run_tests_with_parms(self, run_parameter_map, remaining_keys, remaining_parms):
        if remaining_keys == [] and remaining_parms == []:
            self.run_1_test(run_parameter_map)
        elif remaining_keys == []:
            parm = remaining_parms[0]
            remain = remaining_parms[1:]
            for v in self.parameter_map[parm]:
                extended_map = run_parameter_map
                run_parameter_map[parm] = v
                self.run_tests_with_parms(extended_map, [], remain)
        else:
            kw = remaining_keys[0]
            remain = remaining_keys[1:]
            for v in self.parameter_map[kw]:
                extended_map = run_parameter_map
                extended_map[kw] = v
                self.run_tests_with_parms(extended_map, remain, remaining_parms)
