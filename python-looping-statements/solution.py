def countdown_with_skip(start: int) -> list:
    """
    Using a while loop, count down from `start` to 1 (inclusive),
    appending each number to a list, but skip appending any
    number that is exactly 3 (use continue for this).
    Do not use break here.
    Return the resulting list.
    """
    list_counts = []
    while start >= 1 :
        if start == 3:
            start -= 1  
            continue 
        list_counts.append(start)
        start -= 1  
    return list_counts


def find_first_negative(numbers: list) -> int | None:
    """
    Using a while loop with an index variable, find and return
    the first negative number in `numbers`. Use break once found.
    If no negative number exists, use the loop's else clause to
    return None after the loop completes normally.
    """
    index = 0
    while index < len(numbers):
        if numbers[index] < 0:
            return numbers[index]
        index+=1
    return None


def sum_with_index(numbers: list) -> dict:
    """
    Using a for loop with enumerate(), build and return a
    dictionary mapping each index to the running sum of all
    numbers up to and including that index.

    Example: sum_with_index([10, 20, 30])
    -> {0: 10, 1: 30, 2: 60}
    """

    dict = {}
    sum = 0
    for index,value in enumerate(numbers):
        sum += value
        dict[index] = sum

    return dict


def manual_iteration_trace(items: list) -> list:
   
    iterator = iter(items)
    return_list = []
    while True:
        try:
            item = next(iterator)
            return_list.append(item)
        except StopIteration:
            break
    return return_list
