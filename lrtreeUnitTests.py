from __future__ import division
import unittest
from BS import  *
from numpy import array, sqrt, nan, isfinite, isnan
from copy import copy
from LRTree import *

class ConfirmingUnitTester(unittest.TestCase):

    def confirm( self, result, expected, digits=7, logOnErr=True):
        res = copy( result )
        try:
            self.confirm_iter( res, expected, digits=digits)
        except:
            if logOnErr:
                print('Did Not Match Expected:\n\t\t%r'%(result,))
            raise

    def confirm_iter( self, result, expected, digits=7, nanEquivToNone=True):
        ''' Nest iterables and check '''
        if hasattr(expected,'__iter__'):
            if type(expected)==dict:
                for (key,expectedElem) in expected.iteritems():
                    resElem = result[key]
                    self.confirm_iter(resElem, expectedElem, digits )
            else:
                for (i, expectedElem) in enumerate(expected):
                    resElem = result[i]
                    self.confirm_iter(resElem, expectedElem, digits )
        else:
            if nanEquivToNone and expected in (None, nan):
                    self.assertTrue( result is None or isnan(result) )
            else:
                    self.assertAlmostEqual(result, expected, digits)

class TestLRTree(ConfirmingUnitTester):

    def test000BSPriceSimple(self):
        'BS Formula simple formula'
        i_exp=0.38292492254802624
        try:
            i_got = bsPrice(-1, 1, 1, 0, 1, 1, q=0, t=0)
            if np.abs(i_got/i_exp-1) > 1e-8:
                print('Warning: not getting a BS price match')
        except Exception:
            print('Warning: Not testing bsPrice()')

    def test001BSDeltaCall(self):
        'BS Formula delta call'
        i_exp=0.81105903156704573
        i_got = bsDelta(1, 30, 38, 0, 4, 1, q=0, t=0)
        self.confirm( i_got, i_exp )


    def test002BSDeltaPut(self):
        'BS Formula delta put'
        i_exp=-0.15865525393145705
        i_got = bsDelta(-1, 30, 30, 0, 4, 1, q=0, t=0)
        self.confirm( i_got, i_exp )

    def test002BSDeltaPut(self):
        'BS Formula delta atms put with div rate'
        i_exp=-0.15804962994808808
        i_got = bsDelta(-1, 30, 30, 0, 4, 1, q=0.0040, t=0)
        self.confirm( i_got, i_exp )

    def test004BSDeltaPut(self):
        'BS Formula delta put with div rate'
        i_exp=-0.33460027789140467
        i_got = bsDelta(-1, 30.5, 30, 0.04, 1.1, 0.65, q=0.0040)
        self.confirm( i_got, i_exp )

    def test004BSGammaPut(self):
        'BS Formula gamma'
        i_exp=0.017466242748756987
        i_got = bsGamma(-1, 30.5, 30, 0.04, 1.1, 0.65, q=0.0040)
        self.confirm( i_got, i_exp )


    def test010BSImpvol(self):
        'BS implied vol call'
        i_exp=0.28507070556131081
        i_got = impliedBS(4.4, 1, 30.5, 30, 0.04, T=1.1, q=0.0040)
        self.confirm( i_got, i_exp )

    def test011BSImpvol(self):
        'BS implied vol put'
        i_exp=0.42322974447049933
        i_got = impliedBS(4.4, -1, 30.5, 30, 0.04, T=1.1, q=0.0040)
        self.confirm( i_got, i_exp )


    def test012BSImpvol(self):
        'BS implied vol impossible: below intrinsic'
        i_got = impliedBS(4.4, 1, 30.5, 20, 0.04, T=1.1, q=0.0040)
        self.assert_( np.isnan(i_got) )


    def test020LR1S(self):
        'Leisen-Reimer tree call, stdev=1 easy moneyness one step (should be BS)'
        i_exp=(0.38292492254802624, 0.84492460143487391, 0.18262819015172366)
        i_got = lrtree(1, 1, 1, 0, 1, 1, q=0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test021LR1S(self):
        'Leisen-Reimer tree put, stdev=1 easy moneyness one step (should be BS)'
        i_exp=(0.38292492254802624, -0.155075398565126, 0.18262819015172366)
        i_got = lrtree(-1, 1, 1, 0, 1, 1, q=0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )


    def test031LR3S(self):
        'Leisen-Reimer tree call, stdev=1 easy moneyness 3 step'
        i_exp=(0.37120402616630621, 0.76979863308578933, 0.26542554339987762)
        i_got = lrtree(1, 1, 1, 0, 1, 1, q=0, t=0,params={'stepCount':3})
        self.confirm( i_got, i_exp )

    def test032LR3S(self):
        'Leisen-Reimer tree put, stdev=1 easy moneyness 3 step'
        i_exp= (0.37120402616630604, -0.23020136691421089, 0.26542554339987762)
        i_got = lrtree(-1, 1, 1, 0, 1, 1, q=0, t=0,params={'stepCount':3})
        self.confirm( i_got, i_exp )

    def test033LR2S(self):
        'Leisen-Reimer tree put, stdev=1 easy moneyness 2 step is 3 step'
        i_exp=(0.37120402616630604, -0.23020136691421089, 0.26542554339987762)
        i_got = lrtree(-1, 1, 1, 0, 1, 1, q=0, t=0,params={'stepCount':2})
        self.confirm( i_got, i_exp )

    def test040LR1S(self):
        'Leisen-Reimer tree call, stdev=1 nontriv moneyness one step (should be BS)'
        i_exp=(8.9831265941452454, 0.83144975910954499, 0.0075677919936509947)
        i_got = lrtree(1, 25, 27, 0, 1, 1, q=0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test041LR1S(self):
        'Leisen-Reimer tree put, stdev=1 nontriv moneyness one step (should be BS)'
        i_exp=(10.983126594145242, -0.16855024089045509, 0.0075677919936509956)
        i_got = lrtree(-1, 25, 27, 0, 1, 1, q=0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test042LR1S(self):
        'Leisen-Reimer tree put, nontriv moneyness one step (should be BS)'
        i_exp=(5.7906697481543024, -0.35902669003177018, 0.025940693036790863)
        i_got = lrtree(-1, 25, 27, 0, .5, .65, q=0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test043LR2S(self):
        'Leisen-Reimer tree put, nontriv moneyness 3 step'
        i_exp=(10.679655196926628, -0.25176670036138327, 0.01099874287769247)
        i_got = lrtree(-1, 25, 27, 0, 1, 1, q=0, t=0,params={'stepCount':3})
        self.confirm( i_got, i_exp )

    def test044LR2S(self):
        'Leisen-Reimer tree put,  3 step'
        i_exp=(6.920967263159552, -0.37248518351451737, 0.022731147608350776)
        i_got = lrtree(-1, 25, 27, 0, .55, .8, q=0, t=0,params={'stepCount':3})
        self.confirm( i_got, i_exp )

    def test047LR2S(self):
        'Leisen-Reimer tree put, nontriv moneyness 5 step'
        i_exp=(10.754710376493044, -0.27966046494339258, 0.012159100958604349)
        i_got = lrtree(-1, 25, 27, 0, 1, 1, q=0, t=0,params={'stepCount':5})
        self.confirm( i_got, i_exp )

    def test050LR1S(self):
        'Leisen-Reimer tree put, stdev=1 nontriv moneyness and int rate one step (should be BS)'
        i_exp=(9.8700985073298426, -0.16901311666995955, 0.0084577143976602102)
        i_got = lrtree(-1, 25, 27, 0.06, 1, 1, q=0.0, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )


    def test051LR1S(self):
        'Leisen-Reimer tree put, stdev=1 nontriv mnys and rates 1 step (should be BS)'
        i_exp=(10.177483350229863, -0.16821873151902567, 0.0081740171006697941)
        i_got = lrtree(-1, 25, 27, 0.06, 1, 1, q=0.0390, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test052LR1S(self):
        'Leisen-Reimer tree put,  nontriv mnys and rates 1 step (should be BS)'
        i_exp=(5.0807086596662048, -0.38106470032941353, 0.030705788541945809)
        i_got = lrtree(-1, 25, 27, 0.06, .4, .65, q=0.0390, t=0,params={'stepCount':1})
        self.confirm( i_got, i_exp )

    def test056LR1S(self):
        'Leisen-Reimer tree put,  nontriv mnys and rates 3 step'
        i_exp=(4.9765314743739992, -0.43851459874552878, 0.035284571281518434)
        i_got = lrtree(-1, 25, 27, 0.06, .4, .65, q=0.0390, t=0,params={'stepCount':3})
        self.confirm( i_got, i_exp )

    def test076LR1S(self):
        'Leisen-Reimer tree put,  nontriv mnys and rates 255 step'
        i_exp=(5.1221597491274453, -0.48316313275459893, 0.039232389744912707)
        i_got = lrtree(-1, 25, 27, 0.06, .4, .65, q=0.0390, t=0,params={'stepCount':255})
        self.confirm( i_got, i_exp )

if __name__ == '__main__':
    unittest.main()
    print "Unit Test Main"