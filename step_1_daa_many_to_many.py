import pandas as pd
import numpy as np
from typing import Dict, List
from collections import Counter
import collections
import csv
import itertools
import os.path
import random
# for reproducable results, remove when re-running for different results
# random.seed(42)

np.set_printoptions(suppress=True)


class Suitor(object):
    '''
    Class for the suitor.
    '''

    def __init__(self, id, prefList, capacity):
        self.prefList = prefList
        self.capacity = capacity  # capacity of the Suitor, a constraint
        self.held = set()
        self.proposals = 0
        self.id = id

    def preference(self):
        # number of proposals is also the index of the next available option as it increments
        # returns the next most prefered suited (size 1)
        return self.prefList[self.proposals]

    def __repr__(self):
        return repr(self.id)


class Suited(object):
    '''
    Class for the suited.
    '''

    def __init__(self, id, prefList, capacity):
        self.prefList = prefList
        self.capacity = capacity  # the capacity of the Suited, a constraint
        self.held = set()
        self.id = id

    def reject(self):
        '''
        Trim the self.held set down to its capacity, returning the list of rejected suitors
        '''
        if len(self.held) < self.capacity:
            # do not reject if held < capacity
            return set()
        else:
            # sort the list of held suitors, by suited's preference
            # this is to keep the suited's best preferred suitors (at each iteration)
            sortedSuitors = sorted(
                list(self.held), key=lambda suitor: self.prefList.index(suitor.id))
            # held is the number of suitors up to the suited's capacity
            self.held = set(sortedSuitors[:self.capacity])

            # reject all of the other suitors not held and return the rejected suitors
            return set(sortedSuitors[self.capacity:])

    def __repr__(self):
        return repr(self.id)


def stableMarriage(suitors: List, suiteds: List):
    '''
    Implement the stableMarriage algorithm, a.k.a. Deferred Acceptance

    A stable marriage is a polygamous (many to many) marriage between suitors and suiteds.
    One suited can match to many suitors, and vice versa.

    Return the matches from the suiteds' perspective.

    e.g. Each SS room: 14 unique suitors, 50 unique suiteds

    suitors: List of Suitors
    suiteds: List of Suiteds
    '''
    unassigned = suitors
    # randomizes to ensure fairness of order
    random.shuffle(suitors)
    counter = 0
    round_num = 1
    for suitor in suitors:
        if len(suitor.prefList) == 0:
            # print("\n\nSuitor", suitor.id, "has submitted no rankings and is removed from the matching process.")
            unassigned.remove(suitor)

    for suited in suiteds:
        if len(suited.prefList) == 0:
            # print("\n\nSuited", suited.id, "has submitted no rankings and is removed from the matching process.")
            suiteds.remove(suited)

    # all_removed = []
    # while loop until there are no more unassigned suitors
    while unassigned:
        print(f'Unassigned remaining: {len(unassigned)}')
        print(f'Round: {round_num}')
        round_num += 1
        for suitor in unassigned:
            counter += 1
            # print("\n\n-------------Matching number", counter, "for suitor", suitor.id,"-------------\n\n")
            # print(f'This suitor {suitor} has proposed {suitor.proposals} times so far, and the suitor wants to be next matched with suited {suitor.preference()}')
            # Before adding to held:
            # 1) suitor.held < suitor.capacity
            # 2) suitor.proposals < len(prefList) to (strictly less to prevent index errors)
            if (len(suitor.held) < suitor.capacity) and (suitor.proposals < len(suitor.prefList)):
                if suitor.id in suiteds[suitor.preference()].prefList:
                    suiteds[suitor.preference()].held.add(suitor)
                    suitor.held.add(suiteds[suitor.preference()])
                    # print("\nSuitor", suitor.id, "is paired with", suitor.preference())
                # else:
                    # print("\nSuitor", suitor.id,"is NOT in the preference list of suited", suitor.preference(),"and is queued to be matched with another suited")
            # else:
                # the suitor cannot be paired with anymore suited
                # print(f"This suitor {suitor} has hit its capacity or has proposed to all of its preferred suiteds.")

        for suitor in unassigned:
            # incrementing all unassigned suitors' number of proposals
            # they will try their next preferred
            suitor.proposals += 1

        for suited in suiteds:
            # reject existing suitor(s) if a more preferred suitor comes along and fills the capacity
            removed_suitors = suited.reject()

            for removed_suitor in removed_suitors:
                # remove suited from suitors held since they were rejected,
                # allowing them to be potentially paired with another suited
                # print(f'suited removed: {suited}')
                removed_suitor.held.remove(suited)

        # remove from unassigned if:
        #  1) suitor has proposed to all of its preferred suited or
        #  2) if suitor is at capacity
        removed = [suitor for suitor in unassigned if (suitor.proposals >= len(
            suitor.prefList) or (len(suitor.held) >= suitor.capacity))]

        # update unassigned to smaller set
        unassigned = set(unassigned)-set(removed)

        # if len(removed) > 0:
        # print("\nThe following suitors are removed in this round", removed, f'Count: {len(removed)}')

        # all_removed.extend(removed)
        # print("\nList of suitors available to be matched after this round is", unassigned, f'Count: {len(unassigned)}')
        # randomize unassigned at each iteration
        unassigned = list(unassigned)
        random.shuffle(unassigned)

    # ordered from least to biggest # of suitor id; only for readability
    return dict([(suited, list(sorted(suited.held, key=lambda x: x.id))) for suited in suiteds])


def setup(suitorFile: str, suitedFile: str):
    '''
    Load files and return a ([Suitors], [Suiteds])

    Order matters. Be careful of the file order you input!

    suitorFile: csv of suitor file, with capacity and ranking
    suitedFile: csv of suited file, with capacity and ranking
    '''

    # note to self, potentially refactor with column headings in csv,
    # so we can just pull the column names directly
    with open(suitorFile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        suitors = list(reader)
    for i in range(len(suitors)):
        while '' in suitors[i]:
            suitors[i].remove('')
    for i in range(len(suitors)):
        suitors[i] = list(map(lambda x: int(float(x)), suitors[i]))

    with open(suitedFile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        suiteds = list(reader)

    for i in range(len(suiteds)):
        while '' in suiteds[i]:
            suiteds[i].remove('')
    for i in range(len(suiteds)):
        suiteds[i] = [int(float(suited)) for suited in suiteds[i]]

    list_of_suitors = []
    list_of_suiteds = []

    # for each suitor, append the suitor's ID, preference and capacity
    for i in range(len(suitors)):
        list_of_suitors.append(
            Suitor(suitors[i][0], suitors[i][2:], suitors[i][1]))
        # capacity is position 1
        print("Suitor", suitors[i][0], "meets", suitors[i][1], "suiteds")

    # For each suited, append the suited's ID, preference and capacity
    for j in range(len(suiteds)):
        list_of_suiteds.append(
            Suited(suiteds[j][0], suiteds[j][2:], suiteds[j][1]))
        # capacity is position 1
        print("Suited", suiteds[j][0], "accepts", suiteds[j][1], "suitors")
    return list_of_suitors, list_of_suiteds


def rank(suitor, list_of_suited, mode, suitors, suiteds):
    '''
    Utility function for report function.

    Returns a list of ranks for where_the_suitor_ranked_its_matched_suited
    or where_the_matched_suited_ranked_this_suitor for suitor_rank and suited_rank
    respectively.
    '''
    rank_list = []
    for suited in list_of_suited:
        if mode == 'suitor_rank':
            # rank of where the suitor ranked its matched suited
            rank_list.append(suitors[suitor].prefList.index(suited)+1)
        elif mode == 'suited_rank':
            # rank of where the suited ranked its matched suitor
            rank_list.append(suiteds[suited].prefList.index(suitor)+1)
        else:
            raise ValueError(
                'Invalid mode. Please choose "suitor_rank" or "suited_rank"')
    return rank_list


def report(suitors, suiteds, matchFile):
    '''
    Create a report of matches for each suited with explanation of results.
    '''

    matches = pd.read_csv(matchFile)

    for suited in range(len(suiteds)):
        print("Generating Report for Suited", suited)
        suited_preferences = suiteds[suited].prefList

        result = pd.DataFrame(
            np.empty((len(suited_preferences), 5)), dtype=int)

        # suited_matched_to
        # can be multiple suiteds in the many to many case
        # make it a list to show commas, np array uses spaces by default

        suited_matched_to = pd.Series(suited_preferences).apply(
            lambda suitor: repr(np.where(matches.values == suitor)[1]))

        # drop suited with no matches
        suited_matched_to_ = suited_matched_to[suited_matched_to.str.len() > 0]

        # where_the_suitor_ranked_you
        where_the_suitor_ranked_you = pd.Series(suited_preferences).apply(
            lambda suitor: (suitors[suitor].prefList.index(suited))+1)

        # where_the_suitor_ranked_its_matched_suited
        df = pd.concat([pd.Series(suited_preferences), suited_matched_to_],
                       axis=1).dropna().rename(columns={0: 'suitor', 1: 'suited'})

        # this is a list of rankings instead of a single ranking in the many to many case
        where_the_suitor_ranked_its_matched_suited = df.apply(lambda my_df: rank(
            my_df.suitor, my_df.suited, 'suitor_rank', suitors, suiteds), axis=1)

        # where_the_matched_suited_ranked_this_suitor
        # this is a list of rankings instead of a single ranking in the many to many case
        where_the_matched_suited_ranked_this_suitor = df.apply(lambda my_df: rank(
            my_df.suitor, my_df.suited, 'suited_rank', suitors, suiteds), axis=1)

        result = pd.DataFrame({'Suitor rankings by you': suited_preferences,
                               'Suited matched to': suited_matched_to_,
                               'Where the suitor ranked you': where_the_suitor_ranked_you,
                               'Where the suitor ranked its matched suited': where_the_suitor_ranked_its_matched_suited,
                               'Where the matched suited ranked this suitor': where_the_matched_suited_ranked_this_suitor})

        result['Where the suitor ranked you'].fillna(
            "Suitor did not rank you", inplace=True)
        result.fillna(
            "Suited rejected suitor - not in top pref list", inplace=True)
        result.to_csv(
            f'./ss_match_sim/report_for_suited_{suited}.csv')


def pipeline(suitorFile, suitedFile, run_report=True, matchFile=None):
    '''
    Run the pipeline to create a Matching and
    return the matching csv and associated reports
    '''
    suitors, suiteds = setup(suitorFile, suitedFile)

    if run_report:
        # report in the perspective for the suited
        report(suitors, suiteds, matchFile)
    else:
        matches = stableMarriage(suitors.copy(), suiteds.copy())
        # use Series since different row lengths (each suited takes in a variable number of suitors) as if blank, default is NaN
        df = pd.DataFrame({k: pd.Series([x.id for x in v])
                           for k, v in matches.items()})
        df.columns = [f'Suited {suited}' for suited in suiteds]
        print("\n-------------------------Matching is Complete-------------------------\n")
        # most suited have 5 suitors
        print(pd.DataFrame({k: [v] for k, v in Counter([len(v) for k, v in matches.items()]).items()}).T.reset_index(
        ).sort_values(by='index').rename(columns={'index': 'matched_meetings_for_suiteds', 0: 'count'}))
        print('Verifying stability...')
        print(verifyStable(suitors, suiteds, matches))
        return df


def verifyStable(suitors, suiteds, marriage):
    '''
    verifyStable: [Suitor], [Suited], {Suited -> [Suitor]} -> bool
    Check that the assignment of suitors to suited is a stable marriage.
    '''
    def precedes(L, item1, item2): return L.index(item1) < L.index(item2)
    # find the partner(s) of suitor
    def partner(suitor): return filter(
        lambda s: suitor in marriage[s], suiteds)

    def suitorPrefers(suitor, suited):
        '''
        Return True if the suitor prefers the suited over its existing matches.
        Otherwise, returns False.
        '''
        return any(map(lambda x: precedes(suitor.prefList, suited.id, x.id), list(partner(suitor))))

    def suitedPrefers(suited, suitor):
        '''
        Return True if the suited prefers the suitor over its existing matches
        '''
        # map this function to each of the suited's prefList
        # check if there is any suitor that the suited prefers over than its existing paired suitor
        # if there is any True in the iterator return True

        # e.g. Suited 0 is married to [25, 33, 104, 237, 281]
        # (Suitor = 93, Suited = 0)
        # check if Suitor 93 is preferred over Suitor 25 (True!)
        # check if Suitor 93 is preferred over Suitor 3 (False) ...
        # -> [True, False, False, False, False]
        # > True
        return any(map(lambda x: precedes(suited.prefList, suitor.id, x.id), marriage[suited]))

    # iterate over the cartesian products of each suitor, suited set
    counter = 0
    for (suitor, suited) in itertools.product(suitors, suiteds):

        counter += 1
        if counter % (len(suiteds)*len(suitors)/10) == 0:
            print(f'{counter/(len(suiteds)*len(suitors))*100}% checks complete')

        # stability condition
        if (suitor not in marriage[suited]) and suitorPrefers(suitor, suited) and suitedPrefers(suited, suitor):
            print(
                f'Suitor {suitor} and Suited {suited} is not a stable pairing')
            print(f'suitorPrefers: {suitorPrefers(suitor, suited)}')
            print(f'suitedPrefers: {suitedPrefers(suited, suitor)}')
            return False

    return True


if __name__ == "__main__":
    ventureFile = './ss_match_sim/venture_ranking_sim_room_1.csv'
    mentorFile = './ss_match_sim/mentor_ranking_sim_room_1.csv'
    # matchFile = './ss_match_sim/ss_matches_room_0.csv'
    suitedFile = ventureFile
    suitorFile = mentorFile
    # run pipeline
    result_df = pipeline(suitorFile, suitedFile,
                         run_report=False, matchFile=None)
    result_df.to_csv("./ss_match_sim/ss_matches_room_0.csv", index=False)

    # run report
    # pipeline(suitorFile, suitedFile, run_report=True, matchFile=matchFile)

    # friends toy example
    # A = './matching/DA1920/ss_match_sim/friends_toy_ex_suited_1_to_1.csv'
    # B = './matching/DA1920/ss_match_sim/friends_toy_ex_suitors_1_to_1.csv'
    # suitedFile = A
    # suitorFile = B
    # result_df = pipeline(suitorFile, suitedFile, run_report=False)
    # result_df.to_csv("./matching/DA1920/ss_match_sim/test_marriage4.csv", index=False)


# code to check average ranking of mentor13
# np.mean(np.where(pd.read_csv(ventureFile, header=None).iloc[:,2:]==13)[1]+1)
