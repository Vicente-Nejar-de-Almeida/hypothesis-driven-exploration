import numpy as np
from scipy import stats
from typing import Union


def one_sample_hypothesis_test(data: Union[np.array, list], aggregation: str, null_value: float | str, alternative: str) -> float:
    """
    Statistical hypothesis testing.

    This function applies a one-sample statistical
    test to draw a conclusion about a population
    parameter or distribution.

    Parameters
    ----------
    data : array
        Sample observation.
    aggregation : {'mean', 'variance', 'distribution'}
        Aggregation function.
        > 'mean' : applies a one-sample t-test
        > 'variance' : applies a one-sample variance Chi-square test
        > 'distribution' : applies a one-sample Kolmogorov-Smirnov test
    null_value : float or {'uniform', 'norm', 'expon'}
        Null value of the hypothesis. Is str if aggregation is 'distribution', otherwise is float.
        > 'uniform': uniform distribution
        > 'norm': gaussian distribution
        > 'expon': exponential distribution
    alternative : {'less', 'greater', 'two-sided'}
        Defines the alternative hypothesis.
        > 'less': mean of sample is less than the null value
        > 'greater': mean of sample is greater than null value
        > 'two-sided': mean of sample is different than null value

    Returns
    -------
    float
        The p-value associated with the given alternative.
    """
    if aggregation == 'mean':
        return stats.ttest_1samp(data, popmean=null_value, alternative=alternative).pvalue
    
    elif aggregation == 'variance':
        n = len(data)
        test_statistic = (n - 1) * np.var(data) / null_value
        if alternative == 'less':
            return stats.chi2.cdf(test_statistic, n - 1)
        elif alternative == 'greater':
            return stats.chi2.sf(test_statistic, n - 1)
        elif alternative == 'two-sided':
            return 2 * min(stats.chi2.cdf(test_statistic, n - 1), stats.chi2.sf(test_statistic, n - 1))
        else:
            raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    
    elif aggregation == 'distribution':
        return stats.kstest(data, null_value, alternative=alternative).pvalue
    
    else:
        raise ValueError("aggregation must be 'mean', 'variance' or 'distribution'")


if __name__ == "__main__":
   
   sample = [0.4, 0.6, 0.7, 0.8, 0.9, 1]

   # Mean greater than 0.5
   print(one_sample_hypothesis_test(sample, 'mean', 0.5, 'greater'))

   # Variance greater than 0.1
   print(one_sample_hypothesis_test(sample, 'variance', 0.1, 'greater'))

   # Does not follow uniform distribution
   print(one_sample_hypothesis_test(sample, 'distribution', 'uniform', 'two-sided'))
