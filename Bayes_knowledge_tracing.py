import numpy as np
import itertools
import sys


ALMOST_ONE = 0.999
ALMOST_ZERO = 0.001


class BKT:
    def __init__(self, step=0.1, bounded=True, best_k0=True):
        self.k0 = ALMOST_ZERO
        self.transit = ALMOST_ZERO
        self.guess = ALMOST_ZERO
        self.slip = ALMOST_ZERO
        self.forget = ALMOST_ZERO

        self.k0_limit = ALMOST_ONE
        self.transit_limit = ALMOST_ONE
        self.guess_limit = ALMOST_ONE
        self.slip_limit = ALMOST_ONE
        self.forget_limit = ALMOST_ONE

        self.step = step
        self.best_k0 = best_k0

        if bounded:
            self.k0_limit = 0.85
            self.transit_limit = 0.3
            self.guess_limit = 0.3
            self.slip_limit = 0.1

    def fit(self, X, y=None):

        if self.best_k0:
            self.k0 = self._find_best_k0(X)
            self.k0_limit = self.k0

        k0s = np.arange(self.k0, min(self.k0_limit + self.step, ALMOST_ONE), self.step)
        transits = np.arange(self.transit, min(self.transit_limit + self.step, ALMOST_ONE), self.step)
        guesses = np.arange(self.guess, min(self.guess_limit + self.step, ALMOST_ONE), self.step)
        slips = np.arange(self.slip, min(self.slip_limit + self.step, ALMOST_ONE), self.step)

        all_parameters = [k0s, transits, guesses, slips]

        parameter_pairs = list(itertools.product(*all_parameters))
        min_error = sys.float_info.max
        for (k, t, g, s) in (parameter_pairs):
            error = self.computer_error(X, k, t, g, s)
            if error < min_error:
                self.k0 = k
                self.transit = t
                self.guess = g
                self.slip = s
                min_error = error
        return self.k0, self.transit, self.guess, self.slip

    def computer_error(self, X, k, t, g, s):
        error = 0.0
        for seq in X:
            pred = k
            for res in seq:
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = p + (1 - p) * t
                next_pred = k * (1 - s) + (1 - k) * g
                error += (res - pred) ** 2
                pred = next_pred
        return error

    def _find_best_k0(self, X):
        res = 0.5
        kc_best = np.mean([seq[0] for seq in X])
        if kc_best > 0:
            res = kc_best
        return res

    def inter_predict(self, S, X, k, t, g, s, num_skills):
        all_skills_mastery = {}
        for j in S.keys():
            # j's exercise sequence
            skills = list(map(int, S[j]))
            # j's exercise response
            responses = list(map(int, X[j]))
            # Record the students' mastery of a certain knowledge point last time
            last_mastery = np.zeros(num_skills + 1)
            # Record the students' mastery of a certain knowledge point
            skills_mastery = np.zeros(len(skills))
            # Record whether a certain knowledge point has been done
            has_done = np.zeros(num_skills + 1, dtype=np.int32)

            if len(skills) <= 1:
                continue
            for i, skill_id in enumerate(skills):
                if has_done[skill_id] == 0:
                    has_done[skill_id] = 1
                    pL = k[skill_id]
                else:
                    pL = last_mastery[skill_id]
                skills_mastery[i] = pL

                # update skill mastery after observed their reponse
                if responses[i] == 1:
                    pL = pL * (1 - s[skill_id]) / (pL * (1 - s[skill_id]) + (1 - pL) * g[skill_id])
                else:
                    pL = pL * s[skill_id] / (pL * s[skill_id] + (1 - pL) * (1 - g[skill_id]))
                # Predict students' mastery status
                last_mastery[skill_id] = pL + (1 - pL) * t[skill_id]

            all_skills_mastery[j] = skills_mastery

        return all_skills_mastery


if __name__ == "__main__":
    pass