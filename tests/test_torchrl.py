# this file tests torchrl algorithms
from unittest import TestCase


class testSAC(TestCase):
    def test_sac_implementation(self):
        from roboticsgym.algorithms.torchrl.sac import main

        main()
