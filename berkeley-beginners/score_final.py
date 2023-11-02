import csv
import sys

from pathlib import Path
from typing import List, Dict, Tuple


def multi_dance(
    results: List, tables: Dict, all_marks: List
) -> Tuple[Dict, Dict, Dict]:
    '''
    Score a multi dance. 
    :param results: The ordered list of couple placements for each dance
    :param tables: The 'tabulation table' for each dance. 
    :param all_marks: A list of dicts mapping each couple to their marks for 
      each dance.
    :return: We return
      - a dict of final results mapping each couple to their place
      - a dict of tiebreak rules mapping each couple to the tiebreak rule that 
        was used if one was used to break their tie.
      - a dict mapping each couple to their placements in each dance
    '''
    dance_results = {}
    dances = [key for key in tables.keys()]
    # for each couple in the event
    for couple in results[0]:
        res = []
        for dance in dances:
            # We need to store both their official place and the point value of
            # that place (for example place: 1 and point: 1.5 in the case of
            # a tie for 1st place) because these values are used differently
            # at different points of multidance scoring
            point_value = float(tables[dance][couple][-1])
            place = float(tables[dance][couple][-2])
            res.append((place, point_value))
        dance_results[couple] = res
    # This will be the "Final Summary" table for the multidance
    final_summary = []
    # We sum the placements for each couple across all dances and add this to 
    # the final summary
    for couple, results in dance_results.items():
        places = [b for a, b in results]
        dance_results[couple].append(sum(places))
        final_summary.append((couple, results))
    # Now it's time to begin placing couples
    currently_awarding = 1
    # This will track the placement of each couple
    final_results = {}
    # RULE 9 - lowest total placements wins
    # We sort the couples from lowest to highest placement total
    sorted_summary = sorted(final_summary, key=lambda result: result[1][-1])
    # We go through every couple and if their total is unique from everyone
    # else's, we can award them a place
    i = 0
    while i < len(sorted_summary):
        couple, results = sorted_summary[i]
        # if this is the last place to award we can't look ahead anymore
        if currently_awarding == len(sorted_summary) or \
            i == len(sorted_summary) - 1:
            final_results[couple] = currently_awarding
            currently_awarding += 1
            i += 1
        else: 
            # look at the next couple
            j = 1
            next_couple, next_results = sorted_summary[i + j]
            # if this is a tie, continue down the list until we find the next
            # couple that is not tied
            if results[-1] == next_results[-1]:
                currently_awarding += 1
                while results[-1] == next_results[-1]:
                    j += 1
                    if i + j == len(sorted_summary):
                        break
                    next_couple, next_results = sorted_summary[i + j]
                    currently_awarding += 1
                i += j
            # if it's not a tie we can award the place
            else:
                final_results[couple] = currently_awarding
                currently_awarding += 1
                i += 1
    # It's now time to break the ties
    awarding = 1
    i = 0
    # This will track which rule we used to break each tie
    tiebreak_rules = {}
    # for all of the couples
    while i < len(sorted_summary):
        couple_results = sorted_summary[i]
        couple, results = couple_results
        # RULE 10 
        # if the couple hasn't been assigned a place yet, they are tied
        if couple not in final_results.keys():
            # we need to find the couples they are tied with
            tied_couples = [couple_results]
            j = 1
            next_couple = sorted_summary[i+j]
            while next_couple[0] not in final_results.keys() \
                and next_couple[1][-1] == results[-1]:
                tied_couples.append(next_couple)
                j += 1
                if i + j == len(sorted_summary):
                    break
                next_couple = sorted_summary[i+j]
            i += j
            # as long as we haven't assgined a place to all couples in this tie
            while tied_couples != []:
                # if this is the last couple we can assign the place
                if len(tied_couples) == 1:
                    # we add the couple's placement to the final results
                    final_results[tied_couples[0][0]] = awarding
                    # we note that rule 10 was used to break this tie
                    tiebreak_rules[tied_couples[0][0]] = "R10"
                    # we remove this couple from the list of tied couples
                    tied_couples.remove(tied_couples[0])
                    awarding += 1
                    break
                # for each couple we count the number of places equal to or 
                # lower than the place we are trying to award
                #  the results are returned in sorted order
                current_place_totals = _get_place_counts(tied_couples, awarding)
                # This is the highest number of relevant marks
                top = list(current_place_totals.values())[0]
                # We collect all the couples who achieved this number
                couples_with_most_marks = [
                    couple for couple in tied_couples if \
                        current_place_totals[couple[0]] == top
                ]
                # If only one couple has the highest number, they win the place
                if len(couples_with_most_marks) == 1:
                    final_results[couples_with_most_marks[0][0]] = awarding
                    tiebreak_rules[couples_with_most_marks[0][0]] = "R10"
                    tied_couples.remove(couples_with_most_marks[0])
                    awarding += 1
                # If more than one couple has the highest num, we are still tied
                else:
                    sum_totals = {}
                    # We now add the relevant places together to get a total
                    for couple, marks in couples_with_most_marks:
                        sum_totals[couple] = _get_sum(marks, awarding)
                    sorted_totals = {k: v for k, v in sorted(
                        sum_totals.items(), key=lambda item: item[1]
                    )}
                    # We identify all of the couples with the lowest total
                    lowest = list(sorted_totals.values())[0]
                    couples_with_lowest_totals = [
                        couple for couple in couples_with_most_marks if \
                            sorted_totals[couple[0]] == lowest
                    ]
                    # If only one couple has the lowest total, the tie is broken
                    if len(couples_with_lowest_totals) == 1:
                        final_results[
                            couples_with_lowest_totals[0][0]
                        ] = awarding
                        tiebreak_rules[couples_with_lowest_totals[0][0]] = "R10"
                        tied_couples.remove(couples_with_lowest_totals[0])
                        awarding += 1
                    # Otherwise we have to go to Rule 11
                    else:
                        # We smush together all marks for each couple across all 
                        # dances
                        smushed_marks = {}
                        for couple, _ in couples_with_most_marks:
                            smushed_marks[couple] = []
                        for dance_results in all_marks:
                            for couple, _ in couples_with_most_marks:
                                smushed_marks[couple].extend(
                                    dance_results[couple]
                                )
                        # RULE 11 - Treat all marks from every dance like a 
                        # giant single dance
                        _, t = place_couples(
                            smushed_marks, awarding, awarding + len(
                                couples_with_most_marks
                            )
                        )
                        # We sort the results from first to last
                        t = list(t.items())
                        t = sorted(t, key=lambda result: int(result[1][-1]))
                        # These are all the places that were awarded
                        places = [p[1][-1] for p in t]
                        # If every couple was awarded the same place, this is 
                        #   truly a tie that we cannot break
                        if all([p == places[0] for p in places]):
                            # all couples win the place being awarded
                            winning_couples = [c for c in tied_couples]
                            # we award them all the same place
                            for c in winning_couples:
                                tied_couples.remove(c)
                                tiebreak_rules[c[0]] = "R11"
                                final_results[c[0]] = awarding
                            # but we make sure to increment the next place to 
                            #   award correctly
                            awarding += len(places)
                        # If more than two couples were tied and sent to R11,
                        #   we only care about the winner, any remaining couples
                        #   will go back to the tiebreak rules under R10
                        elif len(smushed_marks) > 2:
                            # will count the number of couples who are able
                            #   to be placed after R11
                            placed = 0
                            # for each tied couple
                            for couple, r11_res in t.items():
                                p = r11_res[-1]
                                # if that couple is the winner under R11, then 
                                #   award them the place
                                if p == "1":
                                    final_results[couple] = awarding
                                    tiebreak_rules[couple] = "R11"
                                    couple_to_remove = [
                                        c for c in tied_couples \
                                            if c[0] == couple
                                    ][0]
                                    tied_couples.remove(couple_to_remove)
                                    placed += 1
                            awarding += placed
                        # If only two couples were tied, and the tie was broken
                        #  with R11, then we can award them their places
                        else:
                            winner, other = t
                            final_results[winner[0]] = awarding
                            tiebreak_rules[winner[0]] = "R11"
                            winner_to_remove = [
                                w for w in tied_couples if w[0] == winner[0]
                            ][0]
                            tied_couples.remove(winner_to_remove)
                            awarding += 1
                            final_results[other[0]] = awarding
                            tiebreak_rules[other[0]] = "R11"
                            other_to_remove = [
                                o for o in tied_couples if o[0] == other[0]
                            ][0]
                            tied_couples.remove(other_to_remove)
                            awarding += 1 
        # if this couple is not tied then just move on
        else:
            i += 1
            awarding += 1 
    for couple in final_results.keys():
        if couple not in tiebreak_rules.keys():
            tiebreak_rules[couple] = "--"
    final_dict = dict(final_summary)
    return final_results, tiebreak_rules, final_dict


def single_dance(marks: Dict) -> Tuple[List, Dict]:
    '''
    Score a single dance. 
    :param marks: A dict mapping couple (number) to a list of their marks
    :return: the sorted_placements which is a list of couple, placement tuples
    sorted by placement AND a dict that shows how the placements were
    calculated. The dict will be printed as part of the results. 
    '''
    number_of_couples = len(marks.keys())
    # We pass in the marks, the place we are currently awarding, and the total 
    #   number of places to award (this is not always the same as the number
    #   of couples, because this function is sometimes called with a subset of 
    #   couples in order to break ties). 
    results, tabulation_table = place_couples(marks, 1, len(marks.keys()))
    # We now have the results of the single dance, we need to calculate the 
    # point values in the case of ties. (e.g. if two couples tie for first in
    # some cases that should count as a 1 but in others it counts as 1.5 when
    # when scoring multi dances.)
    i = 0
    while i < number_of_couples:
        j = i + 1
        relevant_place = int(tabulation_table[results[i]][-1])
        total = int(tabulation_table[results[i]][-1])
        while j < number_of_couples:
            next_place = int(tabulation_table[results[j]][-1])
            if relevant_place == next_place:
                total += j + 1
            else: 
                break
            j += 1
        number_tied = j - i
        tied_value = total / number_tied
        for k in range(i, j):
            tabulation_table[results[k]].append(tied_value)
        i = j
    # a list of couple, place tuples
    placements = [(k, v[-2]) for k, v in tabulation_table.items()]
    sorted_placements = sorted(placements, key=lambda result: float(result[1]))
    return sorted_placements, tabulation_table


def place_couples(
    marks: Dict[str, List[int]], current_mark: int, places_to_award: int
) -> Tuple[List, Dict]:
    """
    Takes in a set of marks and outputs an ordered list of couples from first 
    to last along with a dictionary showing the tabulation of results. 
    :param marks: a Dict mapping each couple number to a list of all marks they 
        were given
    :param current_mark: This specifies the mark that we are currently 
        evaluating, it may be different from the place we are currently awarding
    :param places_to_award: This specifies how many places we are awarding 
        in this dance overall. This may be different than the number of couples
        passed into this function, because we may only pass in a subset of the 
        couples in the case of a tiebreak.
    :return: an ordered list of couples from first to last and a table showing
        the tabulation of results.
    """
    # An ordered list of couples - first through last 
    results = []
    # All couples that have not yet been placed
    unranked_couples = list(marks.keys())
    # Table that will display the calculations used to place couples
    tabulation_table = {}
    # Create a row for each couple in the tabulation table
    for couple in unranked_couples:
        tabulation_table[couple] = []
    # We get the num of judges by looking at the number of marks for any couple
    number_of_judges = len(list(marks.values())[0])
    majority = ( number_of_judges // 2) + 1 
    currently_awarding = 1
    for i in range(current_mark, places_to_award + 1):
        # Collect the marks less than or equal to the place being looked at
        relevant_marks = _get_relevant_marks(marks, unranked_couples, i)
        # Counts and couples with a majority for the place being looked at
        majority_couples_counts, majority_couples = _get_majority_couples(
            relevant_marks, majority
        )
        # Update the tabulation table for this place
        for k, v in relevant_marks.items():
            # If a couple has no relevant marks, mark empty
            if len(v) == 0:
                tabulation_table[k].append("--")
            # Otherwise document the number of relevant marks the couple has
            else: 
                tabulation_table[k].append(str(len(v)))
        # RULE 5 - single couple with majority
        if len(majority_couples) == 1: 
            results.append(majority_couples[0]) # Add couple to results
            unranked_couples.remove(majority_couples[0]) # remove from unranked
            # fill out the rest of their tabulation table with empties
            tabulation_table[majority_couples[0]].extend(
                ["--"] * (places_to_award - i)
            )  
            # Add the placement to the end of the tabulation table 
            tabulation_table[majority_couples[0]].append(
                str(currently_awarding)
            )
            currently_awarding += 1
        # RULE 8 - no couple has a majority
        elif len(majority_couples) == 0:
            continue    # We just move on to the next set of marks
        # RULE 6 - more than one couple has a majority
        else:
            # for as long as we have tied couples
            while len(majority_couples) > 0:
                # identify the max majority
                max_majority = max([m for c, m in majority_couples_counts])
                couples_with_max = [
                    c for c, m in majority_couples_counts if m == max_majority
                ]
                # If just one couple has the max we can break the tie
                if len(couples_with_max) == 1:
                    results.append(couples_with_max[0])
                    majority_couples.remove(couples_with_max[0])
                    majority_couples_counts.remove(
                        (couples_with_max[0], max_majority)
                    )
                    unranked_couples.remove(couples_with_max[0])
                    tabulation_table[couples_with_max[0]].extend(
                        ["--"] * (places_to_award - i)
                    )
                    tabulation_table[couples_with_max[0]].append(
                        str(currently_awarding)
                    )
                    currently_awarding += 1
                # RULE 7 - more than one couple have the max majority
                else:
                    # We calculate the sum of the relevant marks
                    totals_for_couples_with_max = [
                        (c, sum(relevant_marks[c])) for c in couples_with_max
                    ]
                    totals = [t for c, t in totals_for_couples_with_max]
                    # for as long as we have tied couples
                    while len(couples_with_max) > 0:
                        # we identify the minimum total
                        min_total = min(totals)
                        couples_with_min = [
                            c for c, t in totals_for_couples_with_max \
                            if t == min_total
                        ]
                        # if just one couple has the min we can break the tie
                        if len(couples_with_min) == 1:
                            min_couple = couples_with_min[0]
                            results.append(min_couple)
                            couples_with_max.remove(min_couple)
                            majority_couples.remove(min_couple)
                            unranked_couples.remove(min_couple)
                            totals.remove(min_total)
                            tabulation_table[min_couple][-1] = \
                            tabulation_table[min_couple][-1] + f"({min_total})"
                            tabulation_table[min_couple].extend(
                                ["--"] * (places_to_award - i)
                            )
                            tabulation_table[min_couple].append(
                                str(currently_awarding)
                            )
                            currently_awarding += 1
                        # if there are still tied couples
                        else:
                            tied = {}
                            for couple in couples_with_min:
                                tabulation_table[couple][-1] = \
                            tabulation_table[couple][-1] + f"({min_total})"
                                tied[couple] = marks[couple]
                            # if no more places to award, this is truly a tie
                            if i == places_to_award:
                                for couple in couples_with_min:
                                    results.append(couple)
                                    tabulation_table[couple].append(
                                        str(currently_awarding)
                                    )
                                return results, tabulation_table
                            else:
                                # continue with the next mark, 
                                #   but for only the tied couples
                                #   This is where we recursively call this 
                                #   method on a subset, and why we need 
                                #   the places_to_award tracker, so the 
                                #   recursion eventually stops.
                                tiebreak_res, tiebreak_table = place_couples(
                                    tied, i + 1, places_to_award
                                )
                                for couple in tiebreak_res:
                                    results.append(couple)
                                    tabulation_table[couple].extend(
                                        tiebreak_table[couple]
                                    )
                                    tabulation_table[couple][-1] = int(
                                        tiebreak_table[couple][-1]
                                    ) + currently_awarding - 1
                                    unranked_couples.remove(couple)
                                    couples_with_max.remove(couple)
                                    majority_couples.remove(couple)
                                currently_awarding += len(couples_with_min)
                                if currently_awarding > i:
                                    i = currently_awarding                   
    return results, tabulation_table


def _get_sum(marks: List[Tuple[int, float]], place: int) -> float:
    matches = [real for _, real in marks[:-1] if real <= place]
    return sum(matches)


def _get_place_counts(
    couples: List[Tuple[str, List[Tuple[int, float]]]], place: int
) -> Dict[str, int]:
    place_totals = {}
    for couple, results in couples:
        marks = [rounded for rounded, _ in results[:-1] if rounded <= place]
        place_totals[couple] = len(marks)
    return {k: v for k, v in sorted(
        place_totals.items(), key=lambda item: item[1], reverse=True
    )}


def _get_relevant_marks(
    marks: Dict[str, int], couples: List[str], place: int
) -> Dict[str, List[int]]:
    relevant_marks = {}
    for couple in couples:
        relevant_marks[couple] = [
            mark for mark in marks[couple] if mark <= place
        ]
    return relevant_marks


def _get_majority_couples(
    relevant_marks: Dict[str, int], majority: int
) -> Tuple[List[Tuple[str, int]], List[str]]:
    majority_couples_counts = [
        (c, len(rms)) for c, rms in relevant_marks.items() if 
        len(rms) >= majority
    ] 
    majority_couples = [c for c, _ in majority_couples_counts]
    return majority_couples_counts, majority_couples


def write_singledance_results(
    output_file: Path, sorted_placements: List, tabulation_table: Dict, 
    original_couple_order: List
):
    '''
    Write the results to an output file or print them to stdout.
    :param output_file: the output file to write results too, may be sys.stdout
    :param sorted_placements: a list of couples and their placements
    :param tabulation_table: the table showing how the dance was scored
    :param original_couple_order: list of couples in the order they were in the 
      input file so that we can output results in the same order. 
    '''
    if output_file == sys.stdout:
        o = sys.stdout
    else:
        o = open(output_file, "w")
    # Write the title row
    o.write(
        "\t".join(["Couple", "1"] + [
            f"1-{i}" for i in range(2, len(original_couple_order) + 1)
        ] + ["result", "average"]) + "\n"
    )
    # Write results for each couple
    for couple in original_couple_order:
        string_marks = [str(m) for m in tabulation_table[couple]]
        o.write("\t".join([couple] + string_marks) + "\n")
    o.write("\n\n")
    # Also write a simplified lists of results to read during awards
    o.write("Results to read\n")
    o.write("Couple" + "\t" + "Place\n")
    for couple, place in sorted_placements:
        o.write(couple + "\t" + str(place) + "\n")
    if o != sys.stdout:
        o.close()


def write_multidace_results(
    output_file: Path, final_results: Dict, tables: Dict, tiebreak_rules: Dict, 
    final_dict: Dict, original_couple_order: List
):
    '''
    Write the results of the multidance to an output file or print them.
    :param output_file: the output file to write results to, may be sys.stdout
    :param final_results: a dict of final overall results for the multidance
    :param tables: a dict showing how all results were calculated for the 
      individual dances.
    :param tiebreak_rules: a dict showing which (if any) tiebreak rules were 
      used for each couple/placement.
    :param final_dict: a dict mapping each couple to their placements in each 
      dance. 
    :param original_couple_order: list of couples in the order they were in the 
      input file so that we can output results in the same order. 
    '''
    sorted_results = {k: v for k, v in sorted(
        final_results.items(), key=lambda item: item[1]
    )}
    if output_file == sys.stdout:
        o = sys.stdout
    else:
        o = open(output_file, "w")
    # Write the title row
    for dance, marks in tables.items():
        o.write(dance + "\n")
        o.write("\t".join(["Couple", "1"] + [f"1-{i}" for i in range(
            2, len(original_couple_order) + 1
        )] + ["result", "average"]) + "\n")
        # Write results for each couple
        for couple in original_couple_order:
            string_marks = [str(m) for m in marks[couple]]
            o.write("\t".join([couple] + string_marks) + "\n")
        o.write("\n\n")
    # Write final summary
    o.write("\t".join(
        ["Couple"] + list(tables.keys()) + ["Total"] + ["Final Place"]
    ) + "\n")
    for couple in original_couple_order: 
        placements = final_dict[couple]
        raw_placements = [str(r) for p, r in placements[:-1]]
        final_place = f'{final_results[couple]} ({tiebreak_rules[couple]})'\
            if tiebreak_rules[couple] != "--" else final_results[couple]
        o.write("\t".join([couple] +  raw_placements + [
            str(placements[-1]), str(final_place)
        ]) + "\n")
    # Write sorted placements for easy reading during awards      
    o.write("\n\n")
    o.write("Results to Read\n")
    for couple, place in sorted_results.items():
        o.write(couple + "\t" + str(place) + "\n")
    if o != sys.stdout:
        o.close()


def read_marks_file(filename: Path) -> Tuple[List, Dict]:
    '''
    Reads an input file of marks for a single dance. Checks rules 2-4 that all 
    judges have marked each couple exactly once, and given them a unique place.
    Also ensures there is an odd number of judges.
    :param filename: The input file of results. It is assumed that this .tsv
    file contains one column called "Couples" and any additional number of 
    uniquely titled columns.
    :return: A list of couple numbers, ordered in the order they were listed in 
    the input file. We keep this so that we can print the results in the same 
    order. AND the marks for each couple in a Dict mapping couples to marks
    '''
    rows = [row for row in csv.DictReader(open(filename), delimiter="\t")]
    original_couple_order = [row["Couple"] for row in rows]
    number_of_couples = len(original_couple_order)
    marks = {}
    # get the marks for each couple
    for row in rows:
        marks[row["Couple"]] = [
            int(mark) for title, mark in row.items() if title != "Couple"
        ]
    # Rules 2-4
     # # RULES 2-4 - All judges mark every couple once
    number_of_judges = 0
    for key in rows[0].keys():
        # Get all marks for each judge and do checks
        if key != "Couple":
            number_of_judges += 1
            judges_marks = [int(row[key]) for row in rows]
            try:
                assert len(judges_marks) == len(set(judges_marks))
            except AssertionError:
                print(f"Problem with {filename}")
                print(f"Judge: {key} did not give unique marks")
                print(f"The marks were: {judges_marks}")
                sys.exit()
            try:
                assert sorted(judges_marks) == \
                    list(range(1, number_of_couples + 1))
            except AssertionError:
                print(f"Problem with {filename}")
                print(
                    f"{key} did not rank couples 1 through {number_of_couples}"
                )
                print(f"The marks were: {judges_marks}")
                sys.exit()
    try:
        assert number_of_judges % 2 == 1 # number of judges must be odd
    except AssertionError:
        print(f"Problem with {filename}")
        print("The number of judges must be odd")
    return original_couple_order, marks


def main(args):
    if args.output_dir is None:
        output_dir = args.dance_results[0].parent # Just use input dir
    else:
        output_dir = args.output_dir
    # We assume {level}-{dance}.tsv format for the input files
    event_names = [filename.stem for filename in args.dance_results]
    level = event_names[0].split("-")[0]
    levels_and_dances = [event.split("-") for event in event_names]
    # Check to make sure all input files are for the same level
    assert all(lev==level for lev, _ in levels_and_dances) 
    dances = [dance for _, dance in levels_and_dances]
    all_marks_all_couples = [
        read_marks_file(filename) for filename in args.dance_results
    ]
    all_marks = [marks for _, marks in all_marks_all_couples]
    
    
    # We save this for the purpose of printing the results aligned with input
    original_couple_order = all_marks_all_couples[0][0]
    # If this is a MULTIDANCE event
    if len(dances) > 1:
        results = []
        tables = {}
        # Mark each dance as a single dance and store the results
        for i, marks in enumerate(all_marks):
            sorted_placements, tt = single_dance(marks)
            result = [couple for couple, _ in sorted_placements]
            results.append(result)
            tables[dances[i]] = tt
        # Score the multidance event, we return info for printing the results
        final_results, tiebreak_rules, final_dict = multi_dance(
            results, tables, all_marks
        )
        if args.print:
            output_file = sys.stdout
            print(f"\n{level} - {''.join([dance[0] for dance in dances])}\n")
        else:
            output_file = output_dir.joinpath(
                f"{level}-{''.join([dance[0] for dance in dances])}-RESULTS.tsv"
            )
        write_multidace_results(
            output_file, final_results, tables, tiebreak_rules, final_dict, 
            original_couple_order
        )
    # If this is a SINGLE dance event
    else:
        sorted_placements, tt = single_dance(
            all_marks[0]
        )
        if args.print:
            output_file = sys.stdout
            print(f"\n{level} - {dances[0]}\n")
        else:
            output_file = output_dir.joinpath(
                f"{level}-{dances[0]}-RESULTS.tsv"
            )
        write_singledance_results(
            output_file, sorted_placements, tt, original_couple_order
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "dance_results", nargs="+", type=Path, 
        help="The results for each dance. Usually between 1 to 5 files. If only\
            one file is input it will be scored as a single dance. If more \
            than one file is input it will be scored as a multidance.  *NOTE: \
            The filenames are expected in {LEVEL}-{DANCE}.tsv format. The file \
            itslef should have one column titled 'Couple' and any additional \
            odd number of columns full of marks. The columns MUST have unique \
            names."
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, 
        help="The directory to write the results files to"
    )
    parser.add_argument(
        "-p", "--print", action="store_true", 
        help="Print the results to stdout instead of writing them to file(s)"
    )
    args = parser.parse_args()
    main(args)