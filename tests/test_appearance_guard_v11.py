"""Safety contract for the optional research-only appearance filter."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from appearance_guard_v11 import guard_obstacles

class AppearanceGuardContract(unittest.TestCase):
    def test_confident_obstacle_and_original_exclusions_survive(self):
        old=np.array([[True,True,False,False]])
        p=np.zeros((4,1,4));p[1]=[.8,.97,.8,.8]
        aux=np.zeros((2,1,4));aux[1]=[.01,.01,.99,.99]
        result,removed=guard_obstacles(old,p,aux)
        np.testing.assert_array_equal(result,[[False,True,False,False]])
        np.testing.assert_array_equal(removed,[[True,False,False,False]])
    def test_uncertain_appearance_is_retained_at_strict_boundary(self):
        p=np.zeros((4,1,3));p[1]=[.949,.95,.8];aux=np.zeros((2,1,3));aux[1]=[.1,.099,.099]
        result,_=guard_obstacles(np.ones((1,3),bool),p,aux)
        np.testing.assert_array_equal(result,[[True,True,False]])
    def test_misaligned_geometry_is_rejected(self):
        with self.assertRaises(ValueError):guard_obstacles(np.zeros((3,4),bool),np.zeros((4,3,4)),np.zeros((2,4,3)))

if __name__=='__main__':unittest.main()
