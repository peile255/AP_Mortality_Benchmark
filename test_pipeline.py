import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from analysis_pipeline import ROOT, GROUPS, UNCERTAIN, ID, TARGET, metrics, net, fit

class Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.d = pd.read_excel(ROOT / 'data/pone.0323330.s001.xlsx')

    def test_data(self):
        self.assertEqual(self.d.shape, (477, 72))
        self.assertEqual(self.d[TARGET].sum(), 70)
        self.assertTrue(self.d[ID].is_unique)

    def test_whitelist(self):
        for g, cols in GROUPS.items():
            self.assertTrue(set(cols) <= set(self.d.columns))
            self.assertFalse(any((c in [ID, TARGET, 'TBAR1', 'Age1', 'icu_mortality', 'los_icu', 'los_hospital', 'crrt', 'is_ventilator', 'vasopressin', 'ercp', 'ERCP', 'antibiotics'] or c.startswith('hosp_') for c in cols)))
            if g != 'F4':
                self.assertFalse(set(cols) & set(UNCERTAIN))

    def test_grouped_folds(self):
        d = self.d.iloc[np.random.default_rng(42).integers(len(self.d), size=len(self.d))]
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(d, d[TARGET], d[ID]):
            self.assertFalse(set(d.iloc[a][ID]) & set(d.iloc[b][ID]))

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        s = metrics(y, p)
        self.assertAlmostEqual(s[0], 1)
        self.assertAlmostEqual(s[2], 0.025)
        self.assertTrue(np.isfinite(net(y, p)).all())
if __name__ == '__main__':
    unittest.main()
