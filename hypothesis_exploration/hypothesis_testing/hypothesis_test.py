from hypothesis_exploration.hypothesis_testing.one_sample import one_sample_hypothesis_test


class HypothesisTest:
    # number of samples
    ONE_SAMPLE = 1

    # alternative to natural language
    ALTERNATIVE_TO_NL = {'less': 'less than', 'greater': 'greater than', 'two-sided': 'not equal to'}

    def __init__(self, aggregation: str, null_value: float | str, alternative: int, n_sample: int):
        self.aggregation = aggregation
        self.null_value = null_value
        self.alternative = alternative
        self.n_sample = n_sample

    def test(self, *args) -> float:
        if self.n_sample == self.ONE_SAMPLE:
            p_value = one_sample_hypothesis_test(
                *args,
                aggregation=self.aggregation,
                null_value=self.null_value,
                alternative=self.alternative,
            )
            return p_value
    
    def __str__(self):
        if self.n_sample == self.ONE_SAMPLE:
            if self.aggregation == 'distribution':
                return f'Does not follow {self.null_value} distribution'
            else:
                return f'{self.aggregation.capitalize()} is {self.ALTERNATIVE_TO_NL[self.alternative]} {self.null_value}'
        else:
            return ''
