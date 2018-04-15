# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        if self.iterations == 0:
            return

        newVals = util.Counter() # A Counter is a dict with default 0

        for state in self.mdp.getStates():
            totalMax = -1*sys.maxint
            for action in self.mdp.getPossibleActions(state):
                totalTrans = 0
                for transitions in self.mdp.getTransitionStatesAndProbs(state, action):
                    nextState, prob = transitions
                    reward = self.mdp.getReward(state, action, nextState)
                    # print state, nextState, reward, prob, self.discount
                    val = self.getValue(nextState)
                    if  self.mdp.isTerminal(state):
                        val = 0
                    totalTrans += prob * (reward + self.discount * val)
                    # print state, nextState, totalTrans, prob, reward, val * self.discount

                # print "final: ", state, totalTrans
                if totalTrans > totalMax:
                    totalMax = totalTrans
            if totalMax == -1*sys.maxint:
                totalMax = 0
            newVals[state] = totalMax

        self.values = newVals

        self.iterations -= 1
        if self.iterations > 0:
            self.runValueIteration()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        val = 0

        for transitions in self.mdp.getTransitionStatesAndProbs(state, action):
            nextState, prob = transitions
            reward = self.mdp.getReward(state, action, nextState)
            if not self.mdp.isTerminal(nextState):
                val = self.values[nextState]
            qVal += prob * (reward + self.discount * val)
            val = 0
        return qVal



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # print self.values
        if self.mdp.isTerminal(state):
            return None

        actions = {}
        for action in self.mdp.getPossibleActions(state):
            totalTrans = 0
            for transitions in self.mdp.getTransitionStatesAndProbs(state, action):
                nextState, prob = transitions
                reward = self.getValue(nextState)
                totalTrans += reward * prob
            actions[action] = totalTrans

        bestAction = self.mdp.getPossibleActions(state)[0]
        maxVal = -1*sys.maxint
        for key, val in actions.items():
            if val > maxVal:
                maxVal = val
                bestAction = key

        return bestAction
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        if self.iterations == 0:
            return

        for i in range(0, self.iterations):


            index = 0
            if i < len(self.mdp.getStates()):
                index = i
            else:
                index = i % len(self.mdp.getStates())
                
            state = self.mdp.getStates()[index]

            if self.mdp.isTerminal(state):
                self.values[state] = 0
                continue

            totalMax = -1*sys.maxint
            for action in self.mdp.getPossibleActions(state):
                totalTrans = 0
                for transitions in self.mdp.getTransitionStatesAndProbs(state, action):
                    nextState, prob = transitions
                    reward = self.mdp.getReward(state, action, nextState)
                    val = self.getValue(nextState)
                    if self.mdp.isTerminal(state):
                        val = 0
                    totalTrans += prob * (reward + self.discount * val)

                if totalTrans > totalMax:
                    totalMax = totalTrans
            if totalMax == -1*sys.maxint:
                totalMax = 0

            self.values[state] = totalMax



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

