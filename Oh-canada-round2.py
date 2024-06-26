from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order, ConversionObservation, Observation
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS': 0}

def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:
    
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100}
    volume_traded = copy.deepcopy(empty_dict)
    
    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    starfruit_cache = []
    coconuts_cache = []
    starfruit_dim = 4
    buy_orchids = False
    sell_orchids = False
    close_orchids = False
    sunlight_value = 0
    humidity_value = 0
    steps = 0
    start_sunlight = 0
    last_sunlight = -1
    last_humidity = -1
    
    conversion_default = 0
    
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
                
        # average sunlight per hour is 2500 units
        # each day in orchid trading is equal to 12 hours on the island, assume orchid quality doesn't deteriorate overnight
        # data plot shows instantaneous rate of sunlight on any moment of the day
               
        
        if self.last_sunlight != -1 and ((osunlight - self.last_sunlight)/self.last_sunlight > 0):  
            self.buy_orchids = True
        elif self.last_sunlight != -1 and ((osunlight - self.last_sunlight)/self.last_sunlight < 0):
            self.sell_orchids = True
        else:
            self.sell_orchids = True
            
        if self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity > 0.01) and ((ohumidity < 80) and (ohumidity > 60)):
            self.buy_orchids = True
            self.sell_orchids = False
        elif self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity < -0.01) and ((ohumidity < 80) and (ohumidity > 60)):
            self.buy_orchids = False
            self.buy_orchids = True
        elif self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity > 0.01) and (ohumidity < 60):
            self.sell_orchids = False
            self.buy_orchids = True
        elif self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity > 0.01) and (ohumidity > 80):
            self.sell_orchids = True
            self.buy_orchids = False
        elif self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity < 0.01) and (ohumidity < 60):
            self.sell_orchids = True
            self.buy_orchids = False
        elif self.last_humidity != -1 and ((ohumidity - self.last_humidity)/self.last_humidity < -0.01) and (ohumidity > 80):
            self.sell_orchids = False
            self.buy_orchids = True
        else:
            self.sell_orchids = True
            
        
            
        if southbid > best_sell['ORCHIDS']:
            price_to_buy = southbid
        else:
            price_to_buy = best_sell['ORCHIDS']
            
        if southask < best_buy['ORCHIDS']:
            price_to_sell = southask
        else:
            price_to_sell = best_buy['ORCHIDS']
           
       
        
        # stopgap so we don't exceed position limit    
        if self.buy_orchids and self.position['ORCHIDS'] == self.POSITION_LIMIT['ORCHIDS']:
            self.buy_orchids = False
        if self.sell_orchids and self.position['ORCHIDS'] == -self.POSITION_LIMIT['ORCHIDS']:
            self.sell_orchids = False
        if self.close_orchids and self.position['ORCHIDS'] == 0:
            self.close_orchids = False 

        if self.buy_orchids:
            if timestamp > 9989*100 and self.position['ORCHIDS'] < 0:
                vol = self.position['ORCHIDS']
            else:
                vol = self.POSITION_LIMIT['ORCHIDS'] - self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', price_to_buy, vol))     
        if self.sell_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS'] + self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', price_to_sell , -vol))
        if self.close_orchids:
            vol = -self.position['ORCHIDS']
            if vol < 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', best_sell['ORCHIDS'], -vol))
            else:
                orders['ORCHIDS'].append(Order('ORCHIDS', best_buy['ORCHIDS'], vol))
                
        self.last_humidity = convobv['ORCHIDS'].humidity
        self.last_sunlight = convobv['ORCHIDS'].sunlight

        return orders
        



 # Compute orders as a whole   
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):

        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
    
        
 # compute if we want to make a conversion or not
    def conversion_opp(self, convobv, timestamp):
        conversions = [0]
        prods = ['ORCHIDS']
        
        if self.position['ORCHIDS'] <= 100 and self.position['ORCHIDS'] > 0:
            conversions.append(0)#self.position['ORCHIDS'])
        elif self.position['ORCHIDS'] >= -100 and self.position['ORCHIDS'] < 0:
            conversions.append(0)#(self.position['ORCHIDS']+1))
        elif timestamp == 998*100:
            conversions.append(abs(self.position['ORCHIDS']))
        
        # if timestamp == 999*100:
        #     conversions.append(self.position['ORCHIDS'])
        # else:
        #     conversions.append(0)
        # for product in prods:
        #     value = self.position[product]
                  
        #     if value < self.POSITION_LIMIT[product]:
        #         conversions.append(abs(value)/2)
        #     else:
        #         conversions.append(0)
        return sum(conversions)


 # RUN function, Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : []}

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

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

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS' : amethysts_lb, 'STARFRUIT' : starfruit_lb} # we want to buy at slightly below
        acc_ask = {'AMETHYSTS' : amethysts_ub, 'STARFRUIT' : starfruit_ub} # we want to sell at slightly above

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

        for product in ['AMETHYSTS', 'STARFRUIT']:
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
        
        conversions = self.conversion_opp(state.observations.conversionObservations, state.timestamp)
        print(f"Total Conversions: {conversions}")

        return result, conversions, traderdata
               
        

