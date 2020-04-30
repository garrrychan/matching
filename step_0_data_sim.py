from scipy.stats import geom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
plt.style.use('fivethirtyeight')
np.set_printoptions(suppress=True)


def simulate_mentor_data(n_mentors: int, n_ventures: int, capacity_m: int, capacity_v: int, proportion_of_uniform_dist: float, pmf_proba_m: float) -> pd.DataFrame:
    '''
    Return a dataframe of simulated mentor data for each mentor, with mentor capacity and mentor rankings.

    n_mentor: number of mentors
    n_ventures: number of ventures
    capacity_m: capacity of mentors
    capacity_v: capacity of ventures
    proportion_of_uniform_dist: a float between (0,1) representing percentage of sample that is drawn from a uniform distribution
    pmf_proba_m: a float between (0,1) representing probability of success for the pmf of a geometric distribution

    '''
    mentor_col = np.arange(0, n_mentors).reshape(-1, 1)
    capacity_m_col = np.full(n_mentors, capacity_m).reshape(-1, 1)
    arr_m = np.empty((n_mentors, n_ventures))

    # First 60% of mentors will select from a uniform distribution of ventures
    split_index = int(arr_m.shape[0]*proportion_of_uniform_dist)
    # np.random.seed(42)
    arr_m[:split_index] = [np.random.choice(
        np.arange(n_ventures), n_ventures, replace=False) for i in range(0, split_index)]

    # Next 40% of mentors will select from a geometric distribution of ventures
    proba = geom.pmf(np.arange(n_ventures), pmf_proba_m)

    # Ensure probability sums to 1. Each venture has non-zero probability to be selected.
    proba[0] += 1 - np.sum(proba)
    proba = sorted(proba, reverse=True)
    arr_m[split_index:] = [np.random.choice(
        a=np.arange(n_ventures), size=n_ventures, p=proba, replace=False) for i in range(0, arr_m.shape[0] - split_index)]
    return pd.DataFrame(np.concatenate([mentor_col, capacity_m_col, arr_m], axis=1))


def simulate_venture_data(n_mentors: int, n_ventures: int, capacity_m: int, capacity_v: int, proportion_of_uniform_dist: float, pmf_proba_v: float) -> pd.DataFrame:
    '''
    Return a dataframe of simulated venture data for each venture, with venture capacity and venture rankings.

    n_mentor: number of mentors
    n_ventures: number of ventures
    capacity_m: capacity of mentors
    capacity_v: capacity of ventures
    proportion_of_uniform_dist: a float between (0,1) representing percentage of sample that is drawn from a uniform distribution
    pmf_proba_v: a float between (0,1) representing probability of success for the pmf of a geometric distribution

    '''
    venture_col = np.arange(0, n_ventures).reshape(-1, 1)
    capacity_v_col = np.full(n_ventures, capacity_v).reshape(-1, 1)
    arr_v = np.empty((n_ventures, n_mentors))

    # 60% of mentors will select from a uniform distribution of ventures
    split_index = int(arr_v.shape[0]*proportion_of_uniform_dist)
    # np.random.seed(42)
    arr_v[:split_index] = [np.random.choice(
        np.arange(n_mentors), n_mentors, replace=False) for i in range(0, split_index)]

    # 40% of mentors will select from a geometric distribution of ventures
    proba = geom.pmf(np.arange(n_mentors), pmf_proba_v)

    # Ensure probability sums to 1. Each venture has non-zero probability to be selected.
    proba[0] += 1 - np.sum(proba)
    proba = sorted(proba, reverse=True)
    arr_v[split_index:] = [np.random.choice(
        a=np.arange(n_mentors), size=n_mentors, p=proba, replace=False) for i in range(0, arr_v.shape[0] - split_index)]
    return pd.DataFrame(np.concatenate([venture_col, capacity_v_col, arr_v], axis=1))


if __name__ == '__main__':
    n_mentors = 50
    n_ventures = 14
    capacity_m = 5
    capacity_v = 5
    proportion_of_uniform_dist = 0.6
    pmf_proba_m = 1/3
    pmf_proba_v = 1/2
    simulate_mentor_data(n_mentors, n_ventures, capacity_m, capacity_v, proportion_of_uniform_dist, pmf_proba_m).to_csv(
        '/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/mentor_ranking_sim_room_1.csv', index=False, header=False)
    simulate_venture_data(n_mentors, n_ventures, capacity_m, capacity_v, proportion_of_uniform_dist, pmf_proba_v).to_csv(
        '/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/venture_ranking_sim_room_1.csv', index=False, header=False)

# # plot for slide deck
# from collections import Counter
#
# plt.hist(np.where(arr_m==0)[1],bins=14);
# plt.title('Venture 0 (most preferred)')
# plt.xlabel('Rank');
# plt.ylabel('Count');
# plt.savefig('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/fig1.png', bbox_inches = "tight");
#
# plt.hist(np.where(arr_m==13)[1]);
# plt.title('Venture 13 (least preferred)')
# plt.xlabel('Rank');
# plt.ylabel('Count');
# plt.savefig('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/fig2.png', bbox_inches = "tight");
#
# plt.hist(np.where(arr_v==0)[1], bins=50);
# plt.title('Mentor 0 (most preferred)')
# plt.xlabel('Rank');
# plt.ylabel('Count');
# plt.savefig('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/fig3.png', bbox_inches = "tight");
#
# plt.hist(np.where(arr_v==49)[1], bins=50);
# plt.title('Mentor 13 (least preferred)')
# plt.xlabel('Rank');
# plt.ylabel('Count');
# plt.savefig('/Users/gc/cdl/other_projects/matching/DAA1920/ss_match_sim/fig4.png', bbox_inches = "tight");

# # eda
# s_ranks = mentorFile.loc[:,2:]
# v_ranks = ventureFile.loc[:,1:]
# # there are 12 sites
# s_ranks.shape[0]
# # there are 1777 ventures
# v_ranks.shape[0]
# # only 443/1777 are ranked
# # we want all of the ventures/mentors to be matched
#
# # most sites have a capacity between 27-31
# # a few sites have 13
# # one site took 54
# plt.hist(mentorFile[1]);
# plt.title('Capacity per Sites');
#
# # most sites rank 30-31
# # some sites rank 36-41
# # only two sites ranked 50
# # follow similar patterns for mentors ranking sites (some will rank a lot, but most will only rank a lesser number)
# plt.hist(s_ranks.count(axis=1))
# plt.title('Number of Rankings per Site');
#
# # most ventures rank all the Sites, 12
# # or only rank 1-2
# plt.hist(v_ranks.count(axis=1), bins=range(1,13));
# plt.title('Number of Rankings per Venture');
# plt.xticks(range(1,13));
#
# pd.DataFrame(v_ranks.count(axis=1), columns=['count']).reset_index().groupby('count').count()
#
# # very few ventures were rated across sites
# # do we expect this behaviour for ss?
# # probably, they don't know enough about the ventures so the ventures typically only show up once per mentor
# # some overlap
# arr = np.ndarray.flatten(s_ranks.values)
# arr_ = arr[~np.isnan(arr)]
# plt.hist(arr_, bins=1777);
# plt.title('Number of sites who ranked venture j');
#
# {k:v for k,v in Counter(arr_).items() if v>1}
#
# arr2 = np.ndarray.flatten(v_ranks.values)
# arr2_ = arr2[~np.isnan(arr2)]
# plt.hist(arr2_, bins=range(1,13));
# plt.title('Number of ventures who ranked site k')
# plt.xticks(range(1,13));
##
# df = pd.DataFrame({k:[v] for k,v in Counter(arr2_).items()}).T
# df.columns = ['count']
# df.sort_index(inplace=True)
# df.sort_values(by='count', ascending=False)
# plt.bar(df.reset_index().values[:,0], df.reset_index().values[:,1])
#
# list(df.sort_values(by='count', ascending=False).index)
# # site 2,3,4,5 are most popular
# # site 7 and 10 are the least popular
#
# So... sample from this distribution?
# Ventures who rank 'mentors' should be similar
# 4/12 - 33% are very popular
# 7/12 - 58% are moderately popular
# 1/12 - 8% are least popular
