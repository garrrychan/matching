from step_0_data_sim import simulate_mentor_data, simulate_venture_data
from step_1_daa_many_to_many import pipeline
from step_3_mini_sheet import main

import numpy as np
# /Users/gc/cdl/other_projects/matching/DAA1920


def run(simulate_data_bool, n):
    '''
    simulate_data_bool: True if simulate mentor and ventor data
    n: number of runs
    Prints a list of the number of times it took to obtain a perfect schedule
    '''
    # step_0
    n_mentors = 50
    n_ventures = 14
    capacity_m = 5
    capacity_v = 5
    proportion_of_uniform_dist = 0.6
    pmf_proba_m = 1/3
    pmf_proba_v = 1/2
    results = []
    # step 0
    if simulate_data_bool:
        for i in range(0, n):
            simulate_mentor_data(n_mentors, n_ventures, capacity_m, capacity_v, proportion_of_uniform_dist, pmf_proba_m).to_csv(
                f'./ss_match_sim/sim_100/data/mentor_ranking_sim_room_{i}.csv', index=False, header=False)
            simulate_venture_data(n_mentors, n_ventures, capacity_m, capacity_v, proportion_of_uniform_dist, pmf_proba_v).to_csv(
                f'./ss_match_sim/sim_100/data/venture_ranking_sim_room_{i}.csv', index=False, header=False)

    # step_1
    for i in range(0, n):
        ventureFile = f'./ss_match_sim/sim_100/data/venture_ranking_sim_room_{i}.csv'
        mentorFile = f'./ss_match_sim/sim_100/data/mentor_ranking_sim_room_{i}.csv'
        suitedFile = ventureFile
        suitorFile = mentorFile
        # run pipeline and save match file to csv
        result_df = pipeline(suitorFile, suitedFile,
                             run_report=False, matchFile=None)
        result_df.to_csv(
            f'./ss_match_sim/sim_100/matches/ss_matches_room_{i}.csv', index=False)

    # step_3
    results = []
    for i in range(0, n):
        print(f'--------------- matchfile #{i} ---------------')
        results.append(
            main(f'./ss_match_sim/sim_100/matches/ss_matches_room_{i}.csv', True))

    print(results)


if __name__ == '__main__':
    run(False, 100)
