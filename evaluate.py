import numpy as np

def evaluate(estimated, groundTruth, threshold = 0.07): #code taken from lab2
    """
    Usage:  [stats, isFP, isFN] = evaluate(estimated, groundTruth, threshold)
    with input parameters:
      estimated: a vector of estimated event times
      groundTruth:   a vector of the correct (ground truth) event times
      threshold: the allowed absolute error in detection times (default 50 ms)
    and output parameters:
      stats: a 5-element vector containing:
         number of correct detections
         number of false positives
         number of false negatives
         F-measure
         mean error of correct detections
      isFP: boolean vector indicating locations of false positives
      isNP: boolean vector indicating locations of false negatives
      """
    sz1 = len(estimated)
    sz2 = len(groundTruth)
    err1 = np.zeros(sz1)
    err2 = np.zeros(sz2)
    index1 = np.zeros(sz1, dtype=int)
    index2 = np.zeros(sz2, dtype=int)
    if sz2 > 0:
        for i in range(sz1):
            l = [abs(x - estimated[i]) for x in groundTruth]
            index1[i] = np.argmin(l)
            err1[i] = l[index1[i]]
        isFP = [x > threshold for x in err1]
    else:
        isFP = np.ones(sz1)
    if sz1 > 0:
        for i in range(sz2):
            l = [abs(x - groundTruth[i]) for x in estimated]
            index2[i] = np.argmin(l)
            err2[i] = l[index2[i]]
        isFN = [x > threshold for x in err2]
    else:
        isFN = np.ones(sz2)
    fp = sum(isFP)
    fn = sum(isFN)
    correct = sz2 - fn
    if sum([x <= threshold for x in err2]) > 0:
        meanError = (sum([(x <= threshold) * x for x in err2]) /
                   sum([x <= threshold for x in err2]))
    else:
        meanError = 0
    F = 2 * correct / (2 * correct + fp + fn)
    return [ [ correct, fp, fn, F, meanError ], isFP, isFN ]