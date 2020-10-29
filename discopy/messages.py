# -*- coding: utf-8 -*-

"""
discopy error messages.
"""

from typing import Any

NUMPY_THRESHOLD = 16
IGNORE_WARNINGS = [
    "No GPU/TPU found, falling back to CPU.",
    "Casting complex values to real discards the imaginary part"]


def empty_name(got: Any) -> str:
    """ Empty name error. """
    return "Expected non-empty name, got {}.".format(repr(got))


def type_err(expected: type, got: Any) -> str:
    """ Type error. """
    return "Expected {}.{}, got {} of type {} instead.".format(
        expected.__module__, expected.__name__,
        repr(got), type(got).__name__)


def does_not_compose(left: Any, right: Any) -> str:
    """ Composition error. """
    return "{} does not compose with {}.".format(left, right)


def is_not_connected(diagram: Any) -> str:
    """ Disconnected error. """
    return "{} is not connected.".format(str(diagram))


def boxes_and_offsets_must_have_same_len() -> str:
    """ Disconnected diagram error. """
    return "Boxes and offsets must have the same length."


def no_winding_number_for_complex_types() -> str:
    """ No winding number for complex types. """
    return "Only atomic types have a winding number."

def are_not_adjoints(left, right) -> str:
    """ Adjunction error. """
    return "{} and {} are not adjoints.".format(left, right)


def wrong_adjunction(left, right, cup) -> str:
    """ Wrong adjunction error. """
    return "There is no {0}({2}, {3}) in a rigid category. "\
           "Maybe you meant {1}({2}, {3})?".format(
               "Cup" if cup else "Cap", "Cap" if cup else "Cup", left, right)


def cup_vs_cups(left, right) -> str:
    """ Simple type error. """
    return "Cup can only witness adjunctions between simple types. "\
           "Use Diagram.cups({}, {}) instead.".format(left, right)


def cap_vs_caps(left, right) -> str:
    """ Simple type error. """
    return cup_vs_cups(left, right).replace('up', 'ap')


def swap_vs_swaps(left, right) -> str:
    """ Simple type error. """
    return cup_vs_cups(left, right).replace("adjunctions", "symmetry")\
        .replace("Cup", "Swap").replace("cups", "swap")


def cannot_add(left, right) -> str:
    """ Addition error. """
    return "Cannot add {} and {}.".format(left, right)


def expected_pregroup() -> str:
    """ pregroup.draw error. """
    return "Expected a pregroup diagram of shape `word @ ... @ word >> cups`,"\
           " use diagram.draw() instead."


def expected_input_length(function, values) -> str:
    """ Unexpected input length error. """
    return "Expected input of length {}, got {} instead.".format(
        len(function.dom), len(values))
