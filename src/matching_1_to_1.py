import copy  # deepcopy constructs a new compound object, recursively, inserts copies into it
import random


class Person:
    # constructor to initialize the attributes of Person class
    def __init__(self, name, preferences):
        self.name = name
        self.partner = None
        self.preferences = preferences

    # return object representation
    def __repr__(self):
        if self.partner:
            return f'{self.name} ⚭ {self.partner}'
        else:
            return f'{self.name} ⌀'


class Alpha(Person):
    def __init__(self, name, preferences):
        # super() refers to parent class, and inherits methods
        super().__init__(name, preferences)
        # prefered person not asked yet
        # recursively copy
        self.not_asked = copy.deepcopy(preferences)

    def ask(self):
        # drop the first element which is the next preferred person
        return self.not_asked.pop(0)

    # for check_stability function
    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )


class Beta(Person):
    def __init__(self, name, preferences):
        super().__init__(name, preferences)
        # this person does not ask

    def accept(self, suitor):
        return self.partner is None or(
            # check that the suitor is strictly preferred to the existing partner
            self.preferences.index(suitor) <
            self.preferences.index(self.partner)
        )


def setup(preferences, proposing, accepting):
    """
    Initialize the set up and return a dictionary of alphas and betas.

    No one is matched at the beginning.
    """
    # modify the variable in a local context
    global alphas
    global betas

    alphas = {}
    # loop over the preferences proposing
    for key, value in preferences.get(proposing).items():
        alphas[key] = Alpha(key, value)

    betas = {}
    for key, value in preferences.get(accepting).items():
        betas[key] = Beta(key, value)


def run_da(preferences, proposing, accepting):
    """
    Run the deferred acceptance algo and print the match results.

    1) Each unengaged man propose to the woman he prefers most
    2) Each woman says "maybe" to the suitor she most prefers and "no" to all other suitors
    3) Continue while there are still unengaged men
    """
    # Friends came out in 1994
    random.seed(1994)
    setup(preferences, proposing, accepting)
    print("Proposing: ", alphas)
    print("Accepting: ", betas)
    print()
    # all alphas are unmatched at the beginning
    unmatched = list(alphas.keys())

    while unmatched:
        # randomly select one of the alphas to choose next
        alpha = alphas[random.choice(unmatched)]
        # alpha ask his first choice
        beta = betas[alpha.ask()]
        print(f'{alpha.name} asks {beta.name}')
        # if beta accepts alpha's proposal
        if beta.accept(alpha.name):
            print(f'{beta.name} accepts')
            # # if beta has a partner
            if beta.partner:
                # this existing alpha partner is now an ex
                ex = alphas[beta.partner]
                print(f'{beta.name} dumps {ex.name}')
                # this alpha person has no partner now :(
                ex.partner = None
                # add this alpha person back to the list of unmatched
                unmatched.append(ex.name)
            unmatched.remove(alpha.name)
            # log the match
            alpha.partner = beta.name
            beta.partner = alpha.name
        else:
            print(f'{beta.name} rejects')
            # move on to the next unmatched male
    print()
    print("Everyone is matched. This is a stable matching")
    print(alphas)
    print(betas)


def print_pairings(people):
    for p in people.values():
        if p.partner:
            print(
                f'{p.name} is paired with {p.partner} ({p.preferences.index(p.partner) + 1})')
        else:
            print(f'{p.name} is not paired')


def check_not_top_matches(matches):
    '''Generate a list of people who do not have their top matches'''
    not_top_matches = []
    for person in matches.keys():
        if matches[person].partner != matches[person].preferences[0]:
            not_top_matches.append(person)
    return not_top_matches


def check_stability(proposing, accepting, list_of_not_top_matches):
    for i in list_of_not_top_matches:
        more_preferred = proposing[i].preferences[:proposing[i].preferences.index(
            proposing[i].partner)]
        # check to see if it's reciprocated
        for j in more_preferred:
            # print reason why the female rejects
            if accepting[j].accept(proposing[i].name) == False:
                print(
                    f'{proposing[i].name} prefers {accepting[j].name} more, but {accepting[j].name} prefers {accepting[j].partner}.')
            else:
                print("This matching is NOT stable!")
                break
    print("Therefore, this matching is stable.")


def run_all(preferences, proposing, accepting):
    run_da(preferences, proposing, accepting)
    print()
    print_pairings(alphas)
    print()
    print_pairings(betas)
    print()
    check_stability(alphas, betas, check_not_top_matches(alphas))
    print()
    check_stability(betas, alphas, check_not_top_matches(betas))


# preferences
people = {
    "Men": {
        "Ross": ["Rachel", "Phoebe", "Monica"],
        "Chandler": ["Rachel", "Monica", "Phoebe"],
        "Joey": ["Phoebe", "Rachel", "Monica"]
    },
    "Women": {
        "Rachel": ["Joey", "Ross", "Chandler"],
        "Phoebe": ["Ross", "Chandler", "Joey"],
        "Monica": ["Joey", "Chandler", "Ross"]
    }
}


if __name__ == '__main__':
    run_all(people, "Men", "Women")
    # try running it if women propose first, and compare the results!
    # run_all(people, "Women", "Men")
