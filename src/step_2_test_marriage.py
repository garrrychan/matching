import unittest
import pickle
import pandas as pd

from pandas.testing import assert_frame_equal

from step_1_daa_many_to_many import Suited, Suitor, pipeline, stableMarriage, verifyStable


class TestClass(unittest.TestCase):
    """ Class for running unitests
    for stableMarriage algorithm.
    """

    @classmethod
    def setUpClass(self):
        '''
        A class method called before tests in an individual class are run.
        '''
        self.ground_truth_marriage4 = pd.read_csv(DIR+'test_marriage4.csv')

    # def test_marriage1(self):
    #     '''Base case example'''
    #     suitors = [Suitor(0, [0, 1], 1), Suitor(1, [1, 0], 1)]
    #     # Suited(id, preferences, capacity)
    #     suiteds = [Suited(0, [0, 1], 1), Suited(1, [1, 0], 1)]
    #     # marriage where keys are ids and values are the matchings
    #     marriage = stableMarriage(suitors.copy(), suiteds.copy())
    #     ground_truth = {0: [0], 1: [1]}
    #     self.assertEqual(ground_truth, marriage)
    #     self.assertEqual(True, verifyStable(suitors, suiteds, marriage))
    #
    # def test_marriage2(self):
    #     '''All Suitors prefer Suited 0 as first choice'''
    #     suitors = [Suitor(0, [0], 1), Suitor(1, [0], 1),
    #                Suitor(2, [0], 1), Suitor(3, [0], 1)]
    #     suiteds = [Suited(0, [0, 1, 2, 3], 4)]
    #     marriage = stableMarriage(suitors.copy(), suiteds.copy())
    #     ground_truth = {0: [0, 1, 2, 3]}
    #     self.assertEqual(ground_truth, marriage)
    #     self.assertEqual(True, verifyStable(suitors, suiteds, marriage))
    #
    # def test_marriage3(self):
    #     '''Scenario where Suited 2 does not get top picks because
    #        Suited 1 is more prefered.
    #        Suited have capacity > 1'''
    #     suitors = [Suitor(0, [0, 1], 1), Suitor(1, [0, 1], 1),
    #                Suitor(2, [0, 1, 2, 3], 1), Suitor(3, [0, 1, 2, 3], 1)]
    #     suiteds = [Suited(0, [0, 1, 2, 3], 2), Suited(1, [3, 2, 1, 0], 2)]
    #     marriage = stableMarriage(suitors.copy(), suiteds.copy())
    #     ground_truth = {0: [0, 1], 1: [2, 3]}
    #     self.assertEqual(ground_truth, marriage)
    #     self.assertEqual(True, verifyStable(suitors, suiteds, marriage))

    def test_marriage4(self):
        """
        A toy example of Friends in the 1-to-1 scenario.

        Rachel: id 0		Ross: id 0
        Phoebe: id 1		Chandler: id 1
        Monica: id 2		Joey: id 2
        """
        df = pipeline(DIR+'friends_toy_ex_suitors_1_to_1.csv',
                      DIR+'friends_toy_ex_suited_1_to_1.csv', run_report=False)
        assert_frame_equal(self.ground_truth_marriage4, df)


if __name__ == '__main__':
    DIR = '/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/'
    unittest.main()

# fix doesn't like the data objects
# pickle.dump(marriage, open('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/test_marriage3.pkl', 'wb'))
# dill.dump(marriage, open('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/test_marriage1.pkl', 'wb'))
# test an example where the capacity of the suitor is > number of preflist, because it always picks it's most pref, it's sorted
# Scenario where suitor 2 and 3's prefList does not include 2,3 ... therefore no matching
# with open('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/test_marriage3.pkl', 'rb') as file:
#     ground_truth = pickle.load(file)
