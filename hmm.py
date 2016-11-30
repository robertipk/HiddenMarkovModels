#! /usr/bin/env python
import sys
import re
# https://github.com/robertipk/HiddenMarkovModels
usage = "\nusage:   ./hmm.py weather data\nexample: ./hmm.py weather foggy-1000.txt (to test weather model on foggy-1000.txt)"

# This section defines states and probabilities used by the HMMs.

### Weather Model ###

# state map
weatherStateMap   = {'sunny' : 0, 'rainy' : 1, 'foggy' : 2}
weatherStateIndex = {0 : 'sunny', 1 : 'rainy', 2 : 'foggy'}

# observation map
weatherObsMap   = {'no' : 0, 'yes' : 1}
weatherObsIndex = {0 : 'no', 1 : 'yes'}

# prior probability on weather states
# P(sunny) = 0.5  P(rainy) = 0.25  P(foggy) = 0.25
weatherProb = [0.5, 0.25, 0.25]

# transition probabilities
#                    tomorrrow
#    today     sunny  rainy  foggy
#    sunny      0.8    0.05   0.15
#    rainy      0.2    0.6    0.2
#    foggy      0.2    0.3    0.5
weatherTProb = [ [0.8, 0.05, 0.15], [0.2, 0.6, 0.2], [0.2, 0.3, 0.5] ]

# conditional probabilities of evidence (observations) given weather
#                          sunny  rainy  foggy
# P(umbrella=no|weather)    0.9    0.2    0.7
# P(umbrella=yes|weather)   0.1    0.8    0.3
weatherEProb = [ [0.9, 0.2, 0.7], [0.1, 0.8, 0.3] ]

# Using the prior probabilities and state map, return:
#     P(state)
def getStatePriorProb(prob, stateMap, state):
   return prob[stateMap[state]]

# Using the transition probabilities and state map, return:
#     P(next state | current state)
def getNextStateProb(tprob, stateMap, current, future):
   return tprob[stateMap[current]][stateMap[future]]

# Using the observation probabilities, state map, and observation map, return:
#     P(observation | state)
def getObservationProb(eprob, stateMap, obsMap, state, obs):
   return eprob[obsMap[obs]][stateMap[state]]

# Normalize a probability distribution
def normalize(pdist):
   s = sum(pdist)
   for i in range(0,len(pdist)):
      pdist[i] = pdist[i] / s
   return pdist


# Filtering.
# Input:  The HMM (state and observation maps, and probabilities)
#         A list of T observations: E(0), E(1), ..., E(T-1)
#         (ie whether the umbrella was seen [yes, no, ...])
#
# Output: The posterior probability distribution over the most recent state
#         given all of the observations: P(X(T-1)|E(0), ..., E(T-1)).
def filter( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   length = len(prob)
   pdist = prob
   arr = [0]*length
   for i in range(0,len(observations)):
       for x in range(0,length):
           arr[x] = pdist[0]*getNextStateProb(tprob, stateMap, 'sunny', stateIndex[x]) + pdist[1]*getNextStateProb(tprob, stateMap, 'rainy', stateIndex[x]) + pdist[2]*getNextStateProb(tprob, stateMap, 'foggy', stateIndex[x])
       new_prob = [0]*length
       for x in range(0,length):
          new_prob[x] = arr[x]*getObservationProb(eprob, stateMap, obsMap, stateIndex[x], observations[i])
       pdist = normalize(new_prob)
   print("filter results: ", pdist)
   return pdist

# Prediction.
# Input:  The HMM (state and observation maps, and probabilities)
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Output: The posterior probability distribution over the next state
#         given all of the observations: P(X(T)|E(0), ..., E(T-1)).
def predict( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   filtered = filter( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations)
   predictions = [0]*len(prob)
   for i in range(0,len(prob)):
     predictions[i] = filtered[0]*getNextStateProb(tprob, stateMap, 'sunny', stateIndex[i]) + filtered[1]*getNextStateProb(tprob, stateMap, 'rainy', stateIndex[i]) + filtered[2]*getNextStateProb(tprob, stateMap, 'foggy', stateIndex[i])
   print("predictions: ", predictions)
   return predictions

# Smoothing.
# Input:  The HMM (state and observation maps, and probabilities)
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Ouptut: The posterior probability distribution over each state given all
#         of the observations: P(X(k)|E(0), ..., E(T-1) for 0 <= k <= T-1.
#
#         These distributions should be returned as a list of lists.
def smooth( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   num_states = len(prob)
   forward = []
   pdist = prob
   # forward algorithm
   for i in range(0,len(observations)):
     has_umbrella = observations[i]
     p_sun = pdist[0]*getNextStateProb(tprob, stateMap, 'sunny', 'sunny') + pdist[1]*getNextStateProb(tprob, stateMap, 'rainy', 'sunny') + pdist[2]*getNextStateProb(tprob, stateMap, 'foggy', 'sunny')
     p_rain = pdist[0]*getNextStateProb(tprob, stateMap, 'sunny', 'rainy') + pdist[1]*getNextStateProb(tprob, stateMap, 'rainy', 'rainy') + pdist[2]*getNextStateProb(tprob, stateMap, 'foggy', 'rainy')
     p_fog = pdist[0]*getNextStateProb(tprob, stateMap, 'sunny', 'foggy') + pdist[1]*getNextStateProb(tprob, stateMap, 'rainy', 'foggy') + pdist[2]*getNextStateProb(tprob, stateMap, 'foggy', 'foggy')

     new_prob = [0]*num_states
     new_prob[0] = p_sun*getObservationProb(eprob, stateMap, obsMap, 'sunny', has_umbrella)
     new_prob[1] = p_rain*getObservationProb(eprob, stateMap, obsMap, 'rainy', has_umbrella)
     new_prob[2] = p_fog*getObservationProb(eprob, stateMap, obsMap, 'foggy', has_umbrella)
     new_prob = normalize(new_prob)
     forward.append(new_prob)
     pdist = new_prob
   # print("printing forward")
   # for i in range(0,len(forward)):
   #     print(forward[i])
   # backwards algorithm
   backwards = [[1,1,1]]
   dp = [1,1,1]
   for i in range(0,len(forward)):
     has_umbrella = observations[len(observations)-1-i]
     smoothed_probs = [0]*num_states

     p_sun_prior = 0
     for j in range(0,num_states):
       p_sun_prior += getNextStateProb(tprob, stateMap, 'sunny', stateIndex[j])*getObservationProb(eprob, stateMap, obsMap, stateIndex[j], has_umbrella)
     smoothed_probs[0] = p_sun_prior*dp[0]

     p_rain_prior = 0
     for j in range(0,num_states):
       p_rain_prior += getNextStateProb(tprob, stateMap, 'rainy', stateIndex[j])*getObservationProb(eprob, stateMap, obsMap, stateIndex[j], has_umbrella)
     smoothed_probs[1] = p_rain_prior*dp[1]

     p_fog_prior = 0
     for j in range(0,num_states):
       p_fog_prior += getNextStateProb(tprob, stateMap, 'foggy', stateIndex[j])*getObservationProb(eprob, stateMap, obsMap, stateIndex[j], has_umbrella)
     smoothed_probs[2] = p_fog_prior*dp[2]

     dp = normalize(smoothed_probs)
     backwards.insert(0,dp)

   backwards.pop(0)
   # print("printing backwards")
   # for i in range(0,len(backwards)):
   #     print(backwards[i])
   # compute the smoothed probability values
   posterior = []
   for i in range(0,len(backwards)):
     smooth_probs = [0]*num_states
     for x in range(0,num_states):
       smooth_probs[x] = forward[i][x]*backwards[i][x]
     posterior.append(normalize(smooth_probs))
   print("printing smoothing: ")
   for i in range(0,len(posterior)):
       print(posterior[i])
   return posterior


# Viterbi algorithm.
# Input:  The HMM (state and observation maps, and probabilities)
#         A list of T observations: E(0), E(1), ..., E(T-1)
#
# Output: A list containing the most likely sequence of states.
#         (ie [sunny, foggy, rainy, sunny, ...])
def viterbi( \
   stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations):
   prev_dp = [[0 for y in range(len(observations))] for x in range(3)] # DP matrix for backtracing
   probs_dp = [[0 for y in range(len(observations))] for x in range(3)] # DP matrix for probabilities
   # fill in first column
   for i in range(0,3):
     probs_dp[i][0] = prob[i]*getObservationProb(eprob, stateMap, obsMap, stateIndex[i], observations[0])
   # fill in rest of probabilities and tables
   for i in range(1,len(observations)):
     has_umbrella = observations[i]
     for j in range(0,3):
         max_prob = 0
         prev_node = None
         for z in range(0,3):
             prob = probs_dp[z][i-1]*getNextStateProb(tprob, stateMap, stateIndex[z], stateIndex[j])*getObservationProb(eprob, stateMap, obsMap, stateIndex[j], has_umbrella)
             if prob > max_prob:
               max_prob = prob
               prev_node = z
         probs_dp[j][i] = max_prob
         prev_dp[j][i] = prev_node
   max_prob = 0
   prev_node = None
   # find max in last column
   for j in range(0,3):
     if probs_dp[j][len(observations)-1] > max_prob:
       max_prob = probs_dp[j][len(observations)-1]
       prev_node = prev_dp[j][len(observations)-1]
   probable_sequence = [stateIndex[prev_node]]
   # backtrace
   for i in range(len(observations)-1,0,-1):
       prev_node = prev_dp[prev_node][i]
       probable_sequence.insert(0,stateIndex[prev_node])
   print("viterbi: ", probable_sequence)
   return probable_sequence



# Functions for testing.
# You should not change any of these functions.
def loadData(filename):
   input = open(filename, 'r')
   input.readline()
   data = []
   for i in input.readlines():
      x = i.split()
      y = x[0].split(",")
      data.append(y)
   return data

def accuracy(a,b):
   total = float(max(len(a),len(b)))
   c = 0
   for i in range(min(len(a),len(b))):
      if a[i] == b[i]:
         c = c + 1
   return c/total

def test(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, data):
   observations = []
   classes = []
   for c,o in data:
      observations.append(o)
      classes.append(c)
   n_obs_short = 10
   obs_short = observations[0:n_obs_short]
   classes_short = classes[0:n_obs_short]
   print('Short observation sequence:')
   print('   '), obs_short
   for i in range(0,len(classes_short)):
     print(classes_short[i])
   # test filtering
   result_filter = filter( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print ('\nFiltering - distribution over most recent state:')
   for i in range(0,len(result_filter)):
     print(result_filter[i])
   print(stateIndex[0],stateIndex[1],stateIndex[2])

   for i in range(0,len(result_filter)):
      print ('   '), stateIndex[i], ('%1.3f') % result_filter[i],
   # test prediction
   result_predict = predict( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   # print ('\n\nPrediction - distribution over next state:')
   for i in range(0,len(result_filter)):
      print ('   '), stateIndex[i], ('%1.3f') % result_predict[i],
   # test smoothing
   result_smooth = smooth( \
      stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print ('\n\nSmoothing - distribution over state at each point in time:')
   for t in range(0,len(result_smooth)):
      result_t = result_smooth[t]
      print ('   '), ('time'), t,
      for i in range(0,len(result_t)):
         print ('   '), stateIndex[i], ('%1.3f') % result_t[i],
      print (' ')
   # test viterbi
   result_viterbi = viterbi(stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, obs_short)
   print ('\nViterbi - predicted state sequence:\n   '), result_viterbi
   print ('Viterbi - actual state sequence:\n   '), classes_short
   print ('The accuracy of your viterbi classifier on the short data set is'), \
      accuracy(classes_short, result_viterbi)
   # result_viterbi_full = viterbi( \
   #    stateMap, stateIndex, obsMap, obsIndex, prob, tprob, eprob, observations)
   # print ('The accuracy of your viterbi classifier on the entire data set is'), \
   #    accuracy(classes, result_viterbi_full)

if __name__ == '__main__':
   type = None
   filename = None
   if len(sys.argv) > 1:
      type = sys.argv[1]
   if len(sys.argv) > 2:
      filename = sys.argv[2]

   if filename:
      data = loadData(filename)
   else:
      print(usage)
      exit(0)

   if (type == 'weather'):
      test( \
         weatherStateMap, \
         weatherStateIndex, \
         weatherObsMap, \
         weatherObsIndex, \
         weatherProb, \
         weatherTProb, \
         weatherEProb, \
         data)
   else:
      print(usage)
      exit(0)
