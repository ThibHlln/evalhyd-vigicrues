import unittest
import doctest
import evalhyd.vigicrues


if __name__ == '__main__':
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    test_suite.addTests(doctest.DocTestSuite(evalhyd.vigicrues.read.csv))
    test_suite.addTests(doctest.DocTestSuite(evalhyd.vigicrues.read.prv))
    test_suite.addTests(doctest.DocTestSuite(evalhyd.vigicrues.read.xml))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    if not result.wasSuccessful():
        exit(1)
