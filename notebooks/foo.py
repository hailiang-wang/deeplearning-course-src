#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /home/hai/Courses/ML26Q2/courses/04_classcodes/notebooks/foo.py
# Author: Hai Liang Wang
# Date: 2026-08-12:14:57:33
#
# ===============================================================================

"""
This script does not support python2.
"""
__copyright__ = "Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2026-08-12:14:57:33"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)

import env3
ENV = env3.load_env(dotenv_file=os.path.join(curdir, os.pardir, ".env"))

import log5
logger = log5.get_logger(log5.LN(__name__), output_mode=log5.OUTPUT_STDOUT)

##########################################################################
# Actual Works
##########################################################################



##########################################################################
# Testcases
##########################################################################
import unittest

# run testcase: python /home/hai/Courses/ML26Q2/courses/04_classcodes/notebooks/foo.py Test.testExample
class Test(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_001(self):
        print("test_001")

def test():
    '''
    Run tests, two ways available
    '''

    # run as a suite
    #suite = unittest.TestSuite()
    #suite.addTest(Test("test_001"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    # run as main, accept pass testcase name with argvs
    unittest.main()

def main():
    test()

if __name__ == '__main__':
    main()
