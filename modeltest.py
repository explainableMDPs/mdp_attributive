import unittest
from fixed_mdp import *
from solver import LinearEncoding, QuadraticProblem

class ModelTest(unittest.TestCase):
    """Unittest class, uses the fixed MDP in fixed_mdp.py
    """
    def test_loop_example(self):
        for encoding in [LinearEncoding, QuadraticProblem]:
            mdp = loop_mdp()
            qp = encoding(mdp, 'a', 'b', 'c', debug=True)
            r = qp.solve_lower_upper().df()
            self.assertTrue((r['lower_reachability_value'] == r['upper_reachability_value']).all())
            self.assertEqual(r['lower_reachability_value'].iloc[0], 1)
            self.assertEqual(r['lower_importance_value'].iloc[0], 0)
            self.assertEqual(r['upper_importance_value'].iloc[0], 1)
    
    def test_loop_example_half(self):
        for encoding in [LinearEncoding, QuadraticProblem]:
            mdp = loop_half_mdp()
            qp = encoding(mdp, 'a', 'b', 'pos', debug=True)
            r = qp.solve_lower_upper().df()
            self.assertTrue((r['lower_reachability_value'] == r['upper_reachability_value']).all())
            self.assertEqual(r['lower_reachability_value'].iloc[0], 0.5)
            self.assertTrue((r['lower_importance_value'] == r['upper_importance_value']).all())
            self.assertEqual(r['lower_importance_value'].iloc[0], 1)

    def test_paper_example(self):
        for encoding in [LinearEncoding, QuadraticProblem]:
            mdp = paper_example_mdp()
            qp = encoding(mdp, 'q0: start_customer', 'consultation_customer', 'positive', debug=True)
            r = qp.solve_lower_upper().df()
            self.assertTrue((r['lower_reachability_value'] == r['upper_reachability_value']).all())
            self.assertEqual(round(r['lower_reachability_value'].iloc[0],2), 0.98)
            self.assertEqual(round(r['lower_importance_value'].iloc[0], 3), 0.525)
            self.assertEqual(r['upper_importance_value'].iloc[0], 1)

    def test_grid_world(self):
        mdp = gridworld_mdp()
        for s in mdp.nodes():
            qp = QuadraticProblem(mdp, 's00', s, 's60', debug=True)
            r = qp.solve_lower_upper().df()
            self.assertTrue((r['lower_reachability_value'] == r['upper_reachability_value']).all(), r[['lower_reachability_value', 'upper_reachability_value']])
            self.assertEqual(r['lower_reachability_value'].iloc[0], 1)
            if s in ['s00', 's23', 's33', 's43', 's60']:                
                self.assertTrue((r['lower_importance_value'] == r['upper_importance_value']).all(), r[['lower_importance_value', 'upper_importance_value']])
                self.assertEqual(r['lower_importance_value'].iloc[0], 1)
            elif s in ['s30', 's31', 's32', 's34', 's35', 's36']:
                self.assertEqual(r['lower_importance_value'].iloc[0], 0)
                self.assertEqual(r['lower_importance_value'].iloc[0], 0)
            else:
                self.assertEqual(r['upper_importance_value'].iloc[0], 1)
                self.assertEqual(r['lower_importance_value'].iloc[0], 0)
            
if __name__ == '__main__':
    unittest.main()