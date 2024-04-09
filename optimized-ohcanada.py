from typing import Dict, List, Tuple
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, OrderDepth, TradingState, Order, ConversionObservation, Observation
from typing import Any
import jsonpickle



# empty dict
empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0}

def def_value():
    return copy.deepcopy(empty_dict)

class Trader:
    # the order position
    position = copy.deepcopy(empty_dict)
    
    # limit
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    
    # for STARFRUIT regression
    starfruit_cache = []
    starfruit_dim = 4
    steps = 0
    
    
    # calculate next price for STARFRUIT
    def calculate_next_price(self, starfruit_cache):
        # Ensure there are enough points for regression
        if len(starfruit_cache) >= self.starfruit_dim:
            # Use the last self.starfruit_dim number of prices for regression
            prices = starfruit_cache[-self.starfruit_dim:]
            # Perform linear regression to predict the next price
            next_price = np.poly1d(np.polyfit(range(self.starfruit_dim), prices, 1))(self.starfruit_dim)
            return next_price
        else:
            # If not enough data, return the average of available prices
            return np.mean(starfruit_cache) if starfruit_cache else 0

    # compute order for AMETHYSTS
    def compute_amethysts_order(self, product, order_depth, acc_bid, acc_ask):
        # store orders
        orders: list[Order] = []
        
        sell_dict = order_depth.sell_orders.items()
        buy_dict = order_depth.buy_orders.items()
        
        # get current position
        current_position = self.position[product]
        
        # ============================ ***for placing buy order*** ============================
        
        # Buy all orders on the market
        for ask, vol in sell_dict:
            # if the ask price is less than the acc_bid price
            # and price == our acc_bid and we are shorting the product 
            if ((ask < acc_bid) or ((ask == acc_bid) and (self.position[product] < 0)) 
                and (current_position < self.POSITION_LIMIT[product])):
                
                # buy to our limit
                buy_volume = min(-vol, self.POSITION_LIMIT[product] - current_position)
                # price = their ask price 
                price = ask
                
                # place order
                orders.append(Order(product, int(price), buy_volume))
                
                # add to position
                current_position += buy_volume
                
        # if we are currently short, we need to buy to cover
        if (current_position < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
            # buy to our limit, max amount we can buy is 10
            buy_volume = min(self.POSITION_LIMIT[product] - current_position, 10)
            
            # let us buy at price 1 lower than the acc_ask
            price = acc_ask - 1
            
            # place order
            orders.append(Order(product, int(price), buy_volume))
            
            # add to position
            current_position += buy_volume
            
        # if we are currently long (more than 10 shares), we want to buy more lol
        if (current_position < self.POSITION_LIMIT[product]) and (self.position[product] > 10):
            # we will buy max 10 shares
            buy_volume = min(self.POSITION_LIMIT[product] - current_position, 10)
            
            # we will buy at price 2 lower than the acc_ask
            price = acc_ask - 2
            
            # place order
            orders.append(Order(product, int(price), buy_volume))    
            
            # add to position
            current_position += buy_volume
            
        # let place the rest at the acc_ask, buy to our limit
        if (current_position < self.POSITION_LIMIT[product]):
            buy_volume = min(self.POSITION_LIMIT[product] - current_position, 10)
            price = acc_ask
            orders.append(Order(product, int(price), buy_volume))
            
            # add to current position
            current_position += buy_volume
            
        
        # ============================ ***for placing sell order*** ============================
        current_position = self.position[product]
        for bid, vol in buy_dict:
            # if the bid price is greater than the acc_ask price
            # and price == our acc_ask and we are long the product
            if ((bid > acc_ask) or ((bid == acc_ask) and (self.position[product] > 0)) 
                and (current_position > -self.POSITION_LIMIT[product])):
                
                # sell to our limit
                sell_volume = max(-vol, -self.POSITION_LIMIT[product] - current_position)
                price = bid
                
                # place order
                orders.append(Order(product, int(price), sell_volume))
                
                # add to position
                current_position += sell_volume
                
        # if we are currently long, we need to sell to cover
        if (current_position > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
            sell_volume = max(-self.POSITION_LIMIT[product] - current_position, -10)
            price = acc_bid + 1
            
            orders.append(Order(product, int(price), -sell_volume))
            
            # add to position
            current_position += sell_volume
            
        # if we are currently short, we want to sell more lol
        if (current_position > -self.POSITION_LIMIT[product]) and (self.position[product] < -10):
            sell_volume = max(-self.POSITION_LIMIT[product] - current_position, -10)
            price = acc_bid + 2
            orders.append(Order(product, int(price), -sell_volume))
            
            # add to position
            current_position += sell_volume
            
            
        # let place the rest at the acc_bid, sell to our limit
        if (current_position > -self.POSITION_LIMIT[product]):
            sell_volume = max(-self.POSITION_LIMIT[product] - current_position, -10)
            price = acc_bid
            orders.append(Order(product, int(price), -sell_volume))
            
            # add to current position
            current_position += sell_volume
            
        return orders
    
    # compute order for STARFRUIT
    # Since the STARFRUIT is a regression problem, we will use the last 4 prices to predict the next price
    def compute_starfruit_order(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []
        
        buy_dict = order_depth.buy_orders.items()
        sell_dict = order_depth.sell_orders.items()
        
        # get current position
        current_position = self.position[product]
        
        # ============================ ***for placing buy order*** ============================
        
        # Buy all orders on the market
        for ask, vol in sell_dict:
            # if the ask price is less than the acc_bid price
            # and price == our acc_bid and we are shorting the product 
            if ((ask <= acc_bid) or ((ask == acc_bid) and (self.position[product] < 0)) 
                and (current_position < self.POSITION_LIMIT[product])):
                
                # buy to our limit
                buy_volume = min(-vol, self.POSITION_LIMIT[product] - current_position)
                # price = their ask price 
                price = ask
                
                # place order
                orders.append(Order(product, int(price), buy_volume))
                
                # add to position
                current_position += buy_volume
                
        # if we still have space to buy, we will buy at the acc_ask + 1
        if (current_position < self.POSITION_LIMIT[product]):
            buy_volume = min(self.POSITION_LIMIT[product] - current_position, 10)
            price = acc_ask + 1
            orders.append(Order(product, int(price), buy_volume))
            
            # add to current position
            current_position += buy_volume
            
            
        # ============================ ***for placing sell order*** ============================
        current_position = self.position[product]
        # Sell all orders on the market
        for bid, vol in buy_dict:
            # if the bid price is greater than the acc_ask price
            # and price == our acc_ask and we are long the product
            if ((bid >= acc_ask) or ((bid == acc_ask) and (self.position[product] > 0)) 
                and (current_position > -self.POSITION_LIMIT[product])):
                
                # sell to our limit
                sell_volume = max(-vol, -self.POSITION_LIMIT[product] - current_position)
                price = bid
                
                # place order
                orders.append(Order(product, int(price), -sell_volume))
                
                # add to position
                current_position += sell_volume
                
        # if we still have space to sell, we will sell at the acc_bid - 1
        if (current_position > -self.POSITION_LIMIT[product]):
            sell_volume = -self.POSITION_LIMIT[product] - current_position
            price = acc_bid - 1
            orders.append(Order(product, int(price), -sell_volume))
            
            # add to current position
            current_position += sell_volume
        
        return orders
    
    # the main function to compute order
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # save result
        result = {'AMETHYSTS' : [], 'STARFRUIT' : []}

        # updating the position
        for key, val in state.position.items():
            self.position[key] = val
            
        # for the regression
        # Sorting buy orders in descending order to get the highest bid
        sorted_buy = sorted(state.order_depths['STARFRUIT'].buy_orders.items(), key=lambda x: x[0], reverse=True)
        best_bid = sorted_buy[0][0]  # Gets the price (key) of the first item, which is the highest bid

        # Sorting sell orders in ascending order to get the lowest ask
        sorted_sell = sorted(state.order_depths['STARFRUIT'].sell_orders.items(), key=lambda x: x[0])
        best_ask = sorted_sell[0][0]  # Gets the price (key) of the first item, which is the lowest ask
        # get the mid price
        mid_price = (best_bid + best_ask) / 2
        
        # get the cache from trader_data
        traderData_encoded = state.traderData
        if traderData_encoded:  # Ensure the string is not empty
            try:
                traderData = jsonpickle.decode(traderData_encoded)
            except  :
                # Handle the case where decoding fails
                print("Error decoding traderData: JSONDecodeError")
                traderData = {}
        else:
            # Initialize as empty or default if encoded data is empty
            traderData = {}

        if 'starfruit_cache' in traderData:
            self.starfruit_cache = traderData['starfruit_cache']
        else:
            self.starfruit_cache = []
        
        # update the starfruit cache, not ore than dim
        if len(self.starfruit_cache) < self.starfruit_dim:
            # add the mid price
            self.starfruit_cache.append(mid_price)
        else:
            self.starfruit_cache.pop(0)
            self.starfruit_cache.append(mid_price)
            
        # update the trader_data
        traderData['starfruit_cache'] = self.starfruit_cache    
        
        
        # for AMETHYSTS, the acc_bid and acc_ask are the same = 10000
        amethysts_acc_bid = 10000
        amethysts_acc_ask = 10000
        
        # for STARFRUIT, the acc_bid and acc_ask will be calculated with the regression
        starfruit_acc_bid = self.calculate_next_price(self.starfruit_cache)
        starfruit_acc_ask = self.calculate_next_price(self.starfruit_cache)
        
        # acc bid and ask dict
        acc_bid = {'AMETHYSTS' : amethysts_acc_bid, 'STARFRUIT' : starfruit_acc_bid}
        acc_ask = {'AMETHYSTS' : amethysts_acc_ask, 'STARFRUIT' : starfruit_acc_ask}
        
        
        for product in ['AMETHYSTS', 'STARFRUIT']:
            if product == 'AMETHYSTS':
                amethysts_orders = self.compute_amethysts_order('AMETHYSTS', state.order_depths['AMETHYSTS'], acc_bid['AMETHYSTS'], acc_ask['AMETHYSTS'])
                result['AMETHYSTS'] = amethysts_orders
            if product == 'STARFRUIT':
                starfruit_orders = self.compute_starfruit_order('STARFRUIT', state.order_depths['STARFRUIT'], acc_bid['STARFRUIT'], acc_ask['STARFRUIT'])
                result['STARFRUIT'] = starfruit_orders
        
        
        # convertion rate
        conversion_rate = 0 # current conversion rate
        
        # encode the trader data
        traderData_encoded = jsonpickle.encode(traderData)
        
        return result, conversion_rate, traderData_encoded