"""Let X ~ Beta(1.8, 1). An unsuspecting researcher has 100 i.i.d samples of X,
and would like to conduct the following test at 10% significance - H0: E(X) =
2, H1: E(X) < 1.
What’s the probability that the researcher rejects the null hypothesis?
What does 10% in 10% significance level mean? Show your argument using a
simulation.
Explain, in your own words, what Type I and Type II errors are.
"""

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

def beta(alpha, beta, draws):
    return np.random.beta(1.8, 1, size=100)

#10% is the maximum proportion of sample that can lead to rejection of
#null hypothesis to conclude a strong evidence of rejection of null hypothesis
alpha = 0.1
a = beta(1.8, 1, 100)
t_test, pval = stats.ttest_1samp(simulation, 2)
if pval > alpha:
    print("accept H0")
else:
    print("reject H0")

power_one_sample = sm.stats.TTestPower()
result = power_one_sample.solve_power(nobs1=100, alpha=0.1, effect_size=0.8, alternative='smaller')
print("The probability of rejecting the null hypothesis is ", result)


#Type 1 error is when H0 is rejected when it should not be (H0 is true), it is related to the reliability of the test (alpha),
#Type 2 error is when H0 is not rejected when it should be (H0 is not true),
#it is related to the power of the test (beta)
