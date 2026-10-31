"""Paired log-ratio summaries; assumptions are part of the returned result."""
import itertools
import math
import random
import statistics

def summarize(logs,observed):
    assert len(logs)==len(observed)==8 and sorted(observed)==[-1]*4+[1]*4
    mean=statistics.mean(logs)
    null=[statistics.mean(d*s for d,s in zip(logs,signs))for signs in itertools.product((-1,1),repeat=8)]
    p=sum(abs(x)>=abs(mean)-1e-12 for x in null)/len(null)
    balanced=[]
    for first_candidate in itertools.combinations(range(8),4):
        selected=set(first_candidate)
        balanced.append(statistics.mean(d*o*(1 if i in selected else-1)for i,(d,o)in enumerate(zip(logs,observed))))
    balanced_p=sum(abs(x)>=abs(mean)-1e-12 for x in balanced)/len(balanced)
    rng=random.Random(2026103101)
    boot=sorted(statistics.mean(rng.choices(logs,k=8))for _ in range(20000))
    ci=(boot[500],boot[19499])
    return dict(paired_log_ratios=logs,mean_log_ratio=mean,
        wave_latency_reduction_pct=100*(1-math.exp(-mean)),
        paired_geometric_throughput_increase_pct=100*math.expm1(mean),
        bootstrap95_latency_reduction_pct=[100*(1-math.exp(-v))for v in ci],
        bootstrap95_throughput_increase_pct=[100*math.expm1(v)for v in ci],
        predeclared_two_sided_signflip_p=p,
        signflip_assumption='Pairwise sign symmetry;256signpatterns, not the constrained assignment distribution.',
        balanced_assignment_sensitivity_p=balanced_p,balanced_assignments=70,
        positive_pairs=sum(x>0 for x in logs),negative_pairs=sum(x<0 for x in logs))
