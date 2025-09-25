# Global level helper function to determine the numerical value of a class
def class_encoding():
    classes = {
        "cross": 1,
        "hook": 2,
        "jab": 3,
        "uppercut": 4,
        "boxer_1": 5,
        "boxer_2": 6,
        "hook_left_body": 8,
        "hook_left_head": 9,
        "hook_right_body": 10,
        "hook_right_head": 11,
        "straight_left_body": 12,
        "straight_left_head": 13,
        "straight_right_body": 14,
        "straight_right_head": 15
    }

    return classes