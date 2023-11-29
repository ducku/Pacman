"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [Enter a description of what you did here.]
    """
    # Changed Noise to 0 so agent doesn't end up off the bridge,
    # giving it more confidence to cross

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    [Enter a description of what you did here.]
    """
    # decrease living reward to incentivize a fast patht to a closer exit

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -3

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    [Enter a description of what you did here.]
    """
    # decrease living reward for incentivizing a closer exit
    # decrease discount to incentivize a longer but safer path

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    [Enter a description of what you did here.]
    """
    # decrease noise to incentivize walking close to the cliff

    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    [Enter a description of what you did here.]
    """
    # default parameters

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    [Enter a description of what you did here.]
    """
    # increase living reward to avoid exiting and dying

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 100

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    """
    # No combinations of epsilon and learning rate can find an optimal policy before 50 iterations

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
