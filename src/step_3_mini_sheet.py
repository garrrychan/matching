from scipy.spatial import distance
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import random
import copy
# random.seed(42)
np.set_printoptions(suppress=True)
# import itertools


def mini_sheet(matchesFile: str, n_suiteds: int, m_suitors: int) -> np.array:
    '''
    Returns a schedule in the format of a minisheet for small group meetings.

    Each row or column may still have conflicts to be resolved.
    '''
    matches = pd.read_csv(matchesFile).values
    results = np.zeros((n_suiteds, m_suitors))
    for suited in range(0, n_suiteds):
        suitors = matches[:, suited]
        n_meetings_for_venture = len(suitors)
        # a venture cannot meet multiple mentors at the same time
        # random scheduling is better than 1, 2, ..., n_meetings_for_venture
        schedule = list(range(1, n_meetings_for_venture + 1))
        random.shuffle(schedule)
        results[suited, suitors] = schedule

        # replace 0 with nan for google sheet
        results[results == 0] = np.nan
    return results


def is_valid(arr: np.array) -> np.array:
    '''
    Check if an 1D array is a valid solution (column wise, or row wise).
    This means there the row contains only unique values.

    Return a boolean array such that each index is True
    if it is a unique value, otherwise False.

    Inspired by Google Sheets
    =if(OR(countif(D$5:D$20, D5)>1, countif($D5:$BA5, D5)>1), TRUE, FALSE)

    '''
    vals, counts = np.unique(arr, return_counts=True)
    idx_val_repeated = np.where(counts > 1)[0]
    vals_repeated = vals[idx_val_repeated]
    # no duplicates
    if len(vals_repeated) < 1:
        return np.tile(True, arr.shape)

    else:
        bool_array = np.logical_not(
            np.any([arr == val for val in vals_repeated], axis=0))
        return bool_array


def check_valid(arr: np.array) -> np.array:
    '''
    Ensure resulting schedule is valid.

    Return a boolean_array n x m of valid indices, such that each
    index is True if the it is unique in the row and col. Otherwise,
    the index is False.
    '''
    check_rows = np.apply_along_axis(is_valid, axis=0, arr=arr)  # check_rows

    if arr.ndim == 1:  # 1D
        return check_rows

    if arr.ndim == 2:  # 2D
        check_cols = np.apply_along_axis(is_valid, axis=1, arr=arr)
        return np.logical_and(check_cols, check_rows)


def schedule_optimiser(arr: np.array):
    '''
    Return an optimised schedule with as few errors as possible.
    '''
    errors = np.where(check_valid(arr[:, :]) == False)
    n_errors = len(errors[0])
    print(f'There are {n_errors} errors before optimising.')
    d_errors = dict(enumerate(zip(errors[0], errors[1])))
    if len(d_errors) == 0:
        print(
            f'There are {n_errors} errors before optimising. Therefore, there is no need to optimise.')
        return arr

    round = 0
    # cap 10 rounds to fix
    while len(errors[0]) > 0 and (round < 10):
        round += 1
        num_errors_current = len(errors[0])
        d_errors = dict(enumerate(zip(errors[0], errors[1])))
        df_errors = pd.DataFrame(d_errors).T
        df_errors.rename(columns={0: 'row', 1: 'column'}, inplace=True)
        # start with row with the least number of errors
        row_order = df_errors.groupby(
            'row').count().sort_values(by='column').index
        # len(list(itertools.permutations([1,2,3,4,5])))
        # in theory, if all of the timeslots were conflicts, there would be 120 permutations to go through
        # therefore, cap it at 120 otherwise.
        for row in row_order:
            # print('\n')
            iteration = 0
            # print(f'Fixing row: {row}')
            # print('\n')
            # bool_arr of non na
            not_na = ~np.isnan(arr[row, :])
            # note, you must verify ALL the columns of the error row, not just the error columns.
            # otherwise, you introduce potentially new errors.
            # i.e. is_valid on the row, and its respective columns to ensure solution is valid for all columns

            # while 1) there are errors, and 2) if iteration < 20
            # continue to shuffle and resolve conflicts
            # possible_loc is a list of (row, column) of the meetings
            possible_locs = list(
                zip([row]*np.sum(not_na), np.where(not_na)[0]))
            while not(np.all([np.all(is_valid(arr[:, loc[1]])) for loc in possible_locs])) and (iteration < 120):
                num_valid_before = np.sum(
                    [np.all(is_valid(arr[:, loc[1]])) for loc in possible_locs])

                iteration += 1
                # iterate on each error for that row
                for error in [v for k, v in d_errors.items() if v[0] == row]:
                    # get timeslots (i.e. non na)
                    timeslots = arr[row, :][~np.isnan(arr[row, :])]

                    # randomly shuffle timeslots
                    np.random.shuffle(timeslots)

                    arr_ = copy.copy(arr)
                    arr_[row, :][~np.isnan(arr_[row, :])] = timeslots

                    num_valid_after = np.sum(
                        [np.all(is_valid(arr_[:, loc[1]])) for loc in possible_locs])

                    # reassign only if this condition, so you do not add errors
                    if num_valid_after >= num_valid_before:
                        arr[row, :][~np.isnan(arr[row, :])] = timeslots
        # update errors
        errors = np.where(check_valid(arr[:, :]) == False)

        # print(f'iteration: {iteration}')
        # print(f'row: {row} done.')

    errors_ = errors
    n_errors_ = len(errors_[0])
    print(f'There are {n_errors_} errors after optimising.')
    print('\n')
    return arr


def simulate_runs(og_schedule, k_runs):
    '''
    Return number of runs before a perfect schedule,
    after simulating up to random k_runs (no random seed)

    Time complexity is poor, k_runs of 10 is recommended.
    '''
    # for k_run scenarios
    is_perfect = False
    while k_runs > 0 and not(is_perfect):
        print(f'--------------- run #{k_runs} ---------------')
        # try schedule optimiser up to 10 times to return a perfect schedule
        for i in range(0, 10):
            print(f'iteration {i+1}')
            optimised_schedule = schedule_optimiser(copy.deepcopy(og_schedule))
            is_perfect = np.all(check_valid(optimised_schedule))
            if is_perfect:
                break
            else:
                continue
        k_runs -= 1

    print(f'Number of tries for a perfect schedule: {i+1}')
    return i+1


def pair_mentors():
    '''
    Pair Mentors who are similar to each other
    '''
    # 0 is identical, 1 is farthest
    # distance.jaccard([1,1,1], [2,2,2])
    # distance.jaccard([1,1,1], [1,1,0])
    # post DAA
    # jaccard_similarity
    # pair mentors up with no matches with mentors they're similar to
    pass


def main(matchesFile, simulate=None):
    print(f'simulate data? {simulate}')
    n_suiteds = 14
    m_suitors = 50
    og_schedule = mini_sheet(matchesFile, n_suiteds, m_suitors)

    if simulate:
        return simulate_runs(og_schedule, 10)

    errors = np.where(check_valid(og_schedule[:, :]) == False)
    d_errors = dict(enumerate(zip(errors[0], errors[1])))
    print(f'old_errors: {d_errors}, count: {len(d_errors)}')
    arr = copy.deepcopy(og_schedule)
    schedule_optimiser(arr)
    new_errors = dict(enumerate(tuple(
        zip(np.where(check_valid(arr) == 0)[0], np.where(check_valid(arr) == 0)[1]))))
    print(f'new_errors: {new_errors}, count: {len(new_errors)}')
    # double check entire schedule
    print(f'perfect schedule? {np.all(check_valid(arr))}')
    # if np.all(check_valid(arr)):
    pd.DataFrame(og_schedule).to_csv(
        f'./ss_match_sim/sim_100/schedules/original_schedule_test.csv')
    pd.DataFrame(arr).to_csv(
        f'./ss_match_sim/sim_100/schedules/optimised_schedule_test.csv')


if __name__ == '__main__':
    # main('./ss_match_sim/ss_matches_room_0.csv', True) # test 1
    main('./ss_match_sim/sim_100/matches/ss_matches_room_0.csv')
