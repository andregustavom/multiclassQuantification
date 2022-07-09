import pandas as pd
import numpy as np

# it is not working
def EMQ(tr_prev, posterior_probabilities, epsilon = 1e-4):
    Px = posterior_probabilities
    Ptr = np.copy(tr_prev)
    qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

    s, converged = 0, False
    qs_prev_ = None
    while not converged and s < 5:

        # E-step: ps is Ps(y|xi)
        ps_unnormalized = (qs / Ptr) * Px    
        ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

        # M-step:
        qs = ps.mean(axis=0)
        if qs_prev_ is not None and np.mean(abs(qs-qs_prev_)) < epsilon and s > 10:
            converged = True

        qs_prev_ = qs
        s += 1

    #if not converged:
    #    print('[warning] the method has reached the maximum number of iterations; it might have not converged')

    return qs[1]