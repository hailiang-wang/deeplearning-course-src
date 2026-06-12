
import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(curdir, os.pardir))

from weatherforcast import weather_forecast

##########################################################################
# Testcases
##########################################################################
import unittest

# run testcase: python /d/Academia/Courses/ML26Q2/courses/04_classcodes/extended/02_py_unittest/weatherforcast.py Test.testExample


class Test(unittest.TestCase):
    '''

    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_001(self):
        '''
        Testcase
        '''
        print("test_001")

        l = 10
        result = weather_forecast(l)
        assert result == "晴朗", "天气预测失败"

    def test_002(self):
        '''
        Testcase
        '''
        print("test_002")
        l = 11
        result = weather_forecast(l)
        assert result == "晴朗", "天气预测失败"


def test():
    '''
    Run tests, two ways available
    '''

    # run as a suite
    suite = unittest.TestSuite()
    suite.addTest(Test("test_001"))
    suite.addTest(Test("test_002"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # run as main, accept pass testcase name with argvs
    # unittest.main()


def main():
    test()


if __name__ == '__main__':
    main()
