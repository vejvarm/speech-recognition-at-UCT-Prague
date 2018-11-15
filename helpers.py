# helper functions for this project
import random


def check_equal(iterator):
    """check if all elements in an iterator are equal

    :return True if all elements in an iterator are equal (==)
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def random_shuffle(iter1, iter2, seed=0):
    """shuffle two iterators the same way

    :return tuple of the two lists shuffled the same way
    """
    list_zip = list(zip(iter1, iter2))
    random.seed(seed)
    random.shuffle(list_zip)

    return zip(*list_zip)
