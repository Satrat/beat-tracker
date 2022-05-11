# Beat Tracking with Dynamic Programming

Offline beat tracking algorithm for ballroom dance music that combines a LogFiltSpecFlux onset detection function with the global tempo estimation and beat tracking functions presented in the Ellis 2007 paper “Beat Tracking by Dynamic Programming. The algorithm takes a dance style label as input, and performance is tuned for the characteristics of each ballroom dance style.

The Ballroom dataset can be downloaded here: http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html

## Directory Layout
### beat_track.py
Beat tracker implementation

### evaluate.py
Evaluation function for calculating # of hits, false positive and false negative beats

### example_inference.ipynb
demonstration of running the beat tracking algorithm on a single input file

### graphs_and_stats.ipynb
Plots output at different stages of the algorithm and reports overall accuracy on the Ballroom dataset

## How to Run
```
from beat_track import beatTracker, Style

est_beats,_ = beatTracker(INPUT_FILE, danceStyle=STYLE)
```


