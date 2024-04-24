from operator import truediv
from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, ConversionObservation, Observation
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS': 0, 'CHOCOLATE': 0, 'STRAWBERRIES': 0, 'ROSES': 0, 'GIFT_BASKET': 0, 'COCONUT': 0, 'COCONUT_COUPON': 0}

def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:
    
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100, 'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES':60, 'GIFT_BASKET': 60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    volume_traded = copy.deepcopy(empty_dict)
    
    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    starfruit_cache = []
    starfruit_dim = 4    
    
# orchids
    buy_orchids = False
    sell_orchids = False
    clear_orchids = False
    orchidpnl = 0
    last_orchid = 0
    sunlight_value = 0
    humidity_value = 0
    steps = 0
    start_sunlight = 0
    last_sunlight = -1
    last_humidity = -1
    last_export = -1
    last_import = -1
    

# gift baskets
    std = 25    
    basket_std = 50 # 191.1808805 standard deviation

    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    
# coconuts and coconut coupons    

    # average coconut: 10040.633, 10067.36595, 9891.704
    # average coconut coupon: 656.97875, 668.4934, 579.66715
    rate = 0.0656
    years = 1
    Tstep = 250
    stdev_cc = 0.0240898021
    strike = 10000
    option_fv = 637.63
    buy_coconut = False
    sell_coconut = False
    buy_ccoupon = False
    sell_ccoupon = False
    coconut_dim = 4
    coconuts_cache = []

    
# calculates the next price of starfrruit
    def calc_next_price_starfruit(self):
        # bananas cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [0.39244518,  0.24708895,  0.14171926,  0.21181824]
        intercept = 34.95416609
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))
    

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
# Amethysts orders function
    def compute_orders_amethysts(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid-1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask+1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 20):
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 2), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(20, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -20):
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-20, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    
# perform regression on starfruit
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

# Compute conditional regression on orchids - Sunlight, humidity
    def compute_orders_orchids(self,order_depth, convobv, timestamp):    
        orders = {'ORCHIDS' : []}
        prods = ['ORCHIDS']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
       
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            osunlight = convobv[p].sunlight
            ohumidity = convobv[p].humidity
            oshipping = convobv[p].transportFees
            oexport = convobv[p].exportTariff
            oimport = convobv[p].importTariff
            southbid = convobv[p].bidPrice
            southask = convobv[p].askPrice
            
            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
            
            
        if self.last_sunlight != -1 and (osunlight - self.last_sunlight > 1) and (mid_price['ORCHIDS'] - self.last_orchid > 2.33):
            self.buy_orchids = True
        if self.last_export != -1 and ((oexport - self.last_export >= 1.5) or (oexport - self.last_export<= -1.5)):
            self.buy_orchids = True
        if self.last_export != -1 and (oexport - self.last_export == 1):
            self.sell_orchids = True
            
        # export tariff only changes by increments of 1
        # import tariff only by 0.2

        if self.buy_orchids and self.sell_orchids:
            self.buy_orchids = False
            self.sell_orchids = False
            
        if self.position['ORCHIDS'] > 0 and self.sell_orchids == False:
            self.buy_orchids = False
            self.sell_orchids = False
            self.clear_orchids = True
           
        # stop gap so we don't exceed position limit    
        if self.position['ORCHIDS'] == self.POSITION_LIMIT['ORCHIDS']:
            self.buy_orchids = False
        elif self.position['ORCHIDS'] == -self.POSITION_LIMIT['ORCHIDS']:
            self.sell_orchids = False
            
        self.sell_orchids = False
        self.buy_orchids = False
            
        if self.clear_orchids:
            vol = 10
            orders['ORCHIDS'].append(Order('ORCHIDS', worst_buy['ORCHIDS'], -vol))
        if self.buy_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS']  - self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', (best_sell['ORCHIDS']), vol))     
        if self.sell_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS'] + self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', (best_buy['ORCHIDS']), -vol))
                
        self.last_export = convobv['ORCHIDS'].exportTariff
        self.last_sunlight = convobv['ORCHIDS'].sunlight
        self.last_orchid = mid_price['ORCHIDS']

        return orders
    
# compute if we want to make a conversion or not
    def conversion_opp(self):
        conversions = [1]
        prods = ['ORCHIDS']
        
        return sum(conversions)
    
# compute orders for basket
    def compute_orders_basket(self, order_depth):

        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 388
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 388

        trade_at = self.basket_std*0.5
        close_at = self.basket_std*(-1000)

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        rose_pos = self.position['ROSES']
        rose_neg = self.position['ROSES']

        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol
                
        if int(round(self.person_position['Olivia']['ROSES'])) > 0:

            val_ord = self.POSITION_LIMIT['ROSES'] - rose_pos
            if val_ord > 0:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], val_ord))
        if int(round(self.person_position['Olivia']['ROSES'])) < 0:

            val_ord = -(self.POSITION_LIMIT['ROSES'] + rose_neg)
            if val_ord < 0:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], val_ord))

        return orders
    
    def calc_next_price_coconut(self):
        # coconut cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [0.0171,  -0.0126, -0.0088, -0.0436]
        intercept = 6.96667
        nxt_price = intercept
        for i, val in enumerate(self.coconuts_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))
                
# coupon orders
    
    def compute_orders_coupons(self, order_depth):
        orders = {'COCONUT': [], 'COCONUT_COUPON': []}
        prods = ['COCONUT', 'COCONUT_COUPON']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

# calculating coconut coupon, $637.63 is fair value  
        u = math.exp(self.stdev_cc*math.sqrt(self.years/self.Tstep))
        d = 1/u
        probup = (((math.exp((self.rate)*self.years/self.Tstep)) - d) / (u - d))
        discount_factor = self.rate/self.Tstep
        duration_of_time_step = (self.years/self.Tstep)
        payoffs = []
        for n in range(self.Tstep+1): 
            payoffs.append(max(0, (mid_price['COCONUT']*(u**((self.Tstep)-n))*(d**n) - self.strike)))   
        
        for x in reversed(range(1, self.Tstep+1)):
            discounting1 = []
            for i in range(0,x):
                discounting1.append((((probup)*payoffs[i]) + ((1-probup)*payoffs[i+1])) / (math.exp(discount_factor)))
                
            payoffs.clear()
            payoffs.extend(discounting1)
            
        calculated_ccoupon = discounting1[0]
        
# ************************************* End of calculations ************************************
        
        if best_sell['COCONUT_COUPON'] < calculated_ccoupon: # fair value is $637.63
            self.buy_ccoupon = True
        else:
            self.sell_ccoupon = True            

        if self.position['COCONUT_COUPON'] == self.POSITION_LIMIT['COCONUT_COUPON']:
            self.buy_ccoupon = False
        if self.position['COCONUT_COUPON'] == -self.POSITION_LIMIT['COCONUT_COUPON']:
            self.sell_ccoupon = False
            
        if self.buy_ccoupon:
            vol = self.position['COCONUT']
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_sell['COCONUT_COUPON'], vol))
        if self.sell_ccoupon:
            vol = self.position['COCONUT']
            orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', best_buy['COCONUT_COUPON'], -vol))
            
        # # EXAMPLE OF POSITION TAKING WITH COUNTERPARTIES           
        # if int(round(self.person_position['Olivia']['UKULELE'])) > 0:

        #     val_ord = self.POSITION_LIMIT['UKULELE']
        #     if val_ord > 0:
        #         orders['UKULELE'].append(Order('UKULELE', worst_sell['UKULELE'], val_ord))

        return orders

    
    # compute orders
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == "COCONUT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
    


 # RUN function, Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [],  'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES': [], 'GIFT_BASKET': [], 'COCONUT': [], 'COCONUT_COUPON': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')
            
        # assert abs(self.position.get('ROSES', 0)) <= self.POSITION_LIMIT['ROSES']

        timestamp = state.timestamp

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)

        _, bs_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.starfruit_cache.append((bs_starfruit+bb_starfruit)/2)

        INF = 1e9
    
        starfruit_lb = -INF
        starfruit_ub = INF

        if len(self.starfruit_cache) == self.starfruit_dim:
            starfruit_lb = self.calc_next_price_starfruit() - 1
            starfruit_ub = self.calc_next_price_starfruit() + 1

        amethysts_lb = 10000
        amethysts_ub = 10000
        
        # for coconuts
        if len(self.coconuts_cache) == self.coconut_dim:
            self.coconuts_cache.pop(0)
            
        _, bs_coconut = self.values_extract(collections.OrderedDict(sorted(state.order_depths['COCONUT'].sell_orders.items())))
        _, bb_coconut = self.values_extract(collections.OrderedDict(sorted(state.order_depths['COCONUT'].buy_orders.items(), reverse=True)), 1)
        
        self.coconuts_cache.append((bs_coconut+bb_coconut)/2)
        
        coconut_lb = -INF
        coconut_ub = INF
        
        if len(self.coconuts_cache) == self.coconut_dim:
            coconut_lb = self.calc_next_price_coconut() - 1
            coconut_ub = self.calc_next_price_coconut() + 1

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS' : amethysts_lb, 'STARFRUIT' : starfruit_lb, 'COCONUT': coconut_lb} # we want to buy at slightly below
        acc_ask = {'AMETHYSTS' : amethysts_ub, 'STARFRUIT' : starfruit_ub, 'COCONUT': coconut_ub} # we want to sell at slightly above

        self.steps += 1

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] = 1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity
         

        # orders for the different products
          
        orders = self.compute_orders_orchids(state.order_depths, state.observations.conversionObservations, state.timestamp)
        result['ORCHIDS'] += orders['ORCHIDS']
        orders = self.compute_orders_basket(state.order_depths)
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['ROSES'] += orders['ROSES']
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        orders = self.compute_orders_coupons(state.order_depths)
        result['COCONUT_COUPON'] += orders['COCONUT_COUPON']
        
        
        for product in ['AMETHYSTS', 'STARFRUIT', 'COCONUT']:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product])
            result[product] += orders

        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        totpnl = 0

        for product in state.order_depths.keys():
            settled_pnl = 0
            best_sell = min(state.order_depths[product].sell_orders.keys())
            best_buy = max(state.order_depths[product].buy_orders.keys())

            if self.position[product] < 0:
                settled_pnl += self.position[product] * best_buy
            else:
                settled_pnl += self.position[product] * best_sell
            totpnl += settled_pnl + self.cpnl[product]
            print(f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl+self.cpnl[product])/(self.volume_traded[product]+1e-20)}")

        print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        print("End transmission")
        
        		# string value holding trader state data required. 
				# it will be delivered as tradingstate.traderdata on next execution.
        traderdata = "ohcanada"

				# sample conversion request. check more details below. 
        
        conversions = self.conversion_opp()
        print(f"Total Conversions: {conversions}")

        return result, conversions, traderdata
               
        


