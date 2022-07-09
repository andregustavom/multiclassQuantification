
import pdb
import numpy as np




def build_histograms(self, y, y_scores, Y_cts, clf_type, nbins):
    Y = np.unique(y)
    if clf_type == 'prob':
        score_range = (0, 1)
    else:
        score_range = (np.min(y_scores), np.max(y_scores))
        self.CM = np.vstack([np.histogram(y_scores[np.where(y == l)[0]], bins= nbins, range= score_range)[0]
                             for l in Y]).T / Y_cts