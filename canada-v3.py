from operator import truediv
from typing import Dict, List, Tuple
from datamodel import (
    OrderDepth,
    TradingState,
    Order,
    ConversionObservation,
    Observation
)
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {"AMETHYSTS": 0, "STARFRUIT": 0, "ORCHIDS": 0, "CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0, "GIFT_BASKET": 0, "COCONUT": 0, "COCONUT_COUPON": 0
}

PRICE_TERM = {"AMETHYSTS": 0, "STARFRUIT": 0, "ORCHIDS": 2, "GIFT_BASKET": 30, "ROSES": 30, "COCONUT": 30, "COCONUT_COUPON": 30}


def def_value():
    return copy.deepcopy(empty_dict)


INF = int(1e9)


class Trader:
    THRESHOLDS = {"AMETHYSTS": {"over": 0, "mid": 10}, "STARFRUIT": {"over": 0, "mid": 10}, "ORCHIDS": {"over": 20, "mid": 40}, "GIFT_BASKET": {"over": 0, "mid": 10}, "ROSES": {"over": 0, "mid": 10}, "COCONUT": {"over": 0, "mid": 10}, "COCONUT_COUPON": {"over": 0, "mid": 10}}
    
    market_taking: list[tuple[str, int, bool]] = []

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {
        "AMETHYSTS": 20,
        "STARFRUIT": 20,
        "ORCHIDS": 100,
        "CHOCOLATE": 250,
        "STRAWBERRIES": 350,
        "ROSES": 60,
        "GIFT_BASKET": 60,
        "COCONUT": 300,
        "COCONUT_COUPON": 600,
    }
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda: 0)
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
    basket_std = 200  # 191.1808805 standard deviation

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

        coef = [0.39244518, 0.24708895, 0.14171926, 0.21181824]
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
            if buy == 0:
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
        obuy = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if (
                (ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))
            ) and cpos < self.POSITION_LIMIT["AMETHYSTS"]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT["AMETHYSTS"] - cpos)
                cpos += order_for
                assert order_for >= 0
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(
            undercut_buy, acc_bid - 1
        )  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMIT["AMETHYSTS"]) and (self.position[product] < 0):
            num = min(20, self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT["AMETHYSTS"]) and (self.position[product] > 20):
            num = min(20, self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 2), num))
            cpos += num

        if cpos < self.POSITION_LIMIT["AMETHYSTS"]:
            num = min(20, self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if (
                (bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))
            ) and cpos > -self.POSITION_LIMIT["AMETHYSTS"]:
                order_for = max(-vol, -self.POSITION_LIMIT["AMETHYSTS"] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert order_for <= 0
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT["AMETHYSTS"]) and (self.position[product] > 0):
            num = max(-20, -self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT["AMETHYSTS"]) and (
            self.position[product] < -20
        ):
            num = max(-20, -self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT["AMETHYSTS"]:
            num = max(-20, -self.POSITION_LIMIT["AMETHYSTS"] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    # perform regression on starfruit
    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if (
                (ask <= acc_bid)
                or ((self.position[product] < 0) and (ask == acc_bid + 1))
            ) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert order_for >= 0
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(
            undercut_buy, acc_bid
        )  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if (
                (bid >= acc_ask)
                or ((self.position[product] > 0) and (bid + 1 == acc_ask))
            ) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert order_for <= 0
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    # Compute conditional regression on orchids - Sunlight, humidity

    def arbitrage(self, state: TradingState, product: str) -> tuple[list[Order], int]:
        # Calculate the adjusted buy and sell prices
        adjusted_prices = self.calculate_adjusted_prices(state, product)
        acceptable_buy_price = math.floor(adjusted_prices["buy_price"])
        acceptable_sell_price = math.ceil(adjusted_prices["sell_price"])

        # Generate buy and sell orders based on the adjusted prices and price aggression
        orders = self.generate_market_orders(
            state, acceptable_sell_price, acceptable_buy_price, product
        )

        # Handle conversions and update positions based on market activity
        conversions = self.update_positions_and_conversions(state, product)

        return orders

    def calculate_adjusted_prices(self, state: TradingState, product: str) -> dict:
        conversion_obs = state.observations.conversionObservations[product]
        adjusted_buy_price = (
            conversion_obs.bidPrice
            - conversion_obs.transportFees
            - conversion_obs.exportTariff
        )
        adjusted_sell_price = (
            conversion_obs.askPrice
            + conversion_obs.transportFees
            + conversion_obs.importTariff
        )
        return {"buy_price": adjusted_buy_price, "sell_price": adjusted_sell_price}

    def generate_market_orders(
        self, state: TradingState, sell_price: int, buy_price: int, product: str
    ) -> list[Order]:
        return self.get_orders(
            state, sell_price, buy_price, product, PRICE_TERM[product]
        )

    def update_positions_and_conversions(
        self, state: TradingState, product: str
    ) -> int:
        conversions = 0
        conversion_limit = state.position.get(product, 0)
        for index, (market_taking_product, market_taking_amount, seen) in enumerate(
            self.market_taking
        ):
            if conversion_limit == 0 or market_taking_product != product:
                continue
            conversions += self.adjust_market_taking(
                index, conversion_limit, market_taking_amount
            )
            conversion_limit += conversions
        return conversions

    def adjust_market_taking(
        self, index: int, conversion_limit: int, market_taking_amount: int
    ) -> int:
        if conversion_limit > 0 and market_taking_amount > 0:
            adjustment = -min(conversion_limit, market_taking_amount)
            self.market_taking[index] = (
                self.market_taking[index][0],
                market_taking_amount,
                True,
            )
            return adjustment
        elif conversion_limit < 0 and market_taking_amount < 0:
            adjustment = min(-conversion_limit, -market_taking_amount)
            self.market_taking[index] = (
                self.market_taking[index][0],
                market_taking_amount,
                True,
            )
            return adjustment
        return 0

    def get_orders(
        self,
        trading_state: TradingState,
        max_sell_price: int,
        min_buy_price: int,
        product_id: str,
        aggression_level: int,
    ) -> List[Order]:
        """
        Calculate and generate trading orders based on current market conditions, position limits,
        and defined thresholds for buying and selling.

        Args:
            trading_state (TradingState): Current trading market state including order depths.
            max_sell_price (int): Maximum price threshold for selling.
            min_buy_price (int): Minimum price threshold for buying.
            product_id (str): Identifier for the trading product.
            aggression_level (int): Level of price aggression to apply.

        Returns:
            List[Order]: A list of Order objects that represent the trading decisions (buy/sell).
        """

        # Retrieve the order book depths for the specified product
        order_depth = trading_state.order_depths[product_id]
        position_limit = self.POSITION_LIMIT[product_id]
        trading_orders = []

        # Sorting sell and buy order books by price
        sell_orders = sorted(list(order_depth.sell_orders.items()), key=lambda x: x[0])
        buy_orders = sorted(
            list(order_depth.buy_orders.items()), key=lambda x: x[0], reverse=True
        )

        # Getting the lowest and highest prices available in the order books
        lowest_sell_price = sell_orders[0][0]
        highest_buy_price = buy_orders[0][0]

        # Buy-side logic: Process buy orders based on the sell order book
        current_buy_position = trading_state.position.get(product_id, 0)

        for asking_price, volume in sell_orders:
            if position_limit - current_buy_position <= 0:
                break

            if asking_price < min_buy_price - aggression_level:
                buy_volume = min(-volume, position_limit - current_buy_position)
                current_buy_position += buy_volume
                assert buy_volume > 0
                trading_orders.append(Order(product_id, asking_price, buy_volume))
                self.market_taking.append((product_id, buy_volume, False))

            if (
                asking_price == min_buy_price - aggression_level
                and current_buy_position < 0
            ):
                buy_volume = min(-volume, -current_buy_position)
                current_buy_position += buy_volume
                assert buy_volume > 0
                trading_orders.append(Order(product_id, asking_price, buy_volume))
                self.market_taking.append((product_id, buy_volume, False))

        # Additional buy orders based on remaining capacity and thresholds
        if position_limit - current_buy_position > 0:
            self.handle_additional_buy_orders(
                trading_orders,
                product_id,
                current_buy_position,
                min_buy_price,
                aggression_level,
                highest_buy_price,
            )

        # Sell-side logic: Process sell orders based on the buy order book
        current_sell_position = trading_state.position.get(product_id, 0)

        for bid_price, volume in buy_orders:
            if -position_limit - current_sell_position >= 0:
                break

            if bid_price > max_sell_price + aggression_level:
                sell_volume = max(-volume, -position_limit - current_sell_position)
                current_sell_position += sell_volume
                assert sell_volume < 0
                trading_orders.append(Order(product_id, bid_price, sell_volume))
                self.market_taking.append((product_id, sell_volume, True))

            if (
                bid_price == max_sell_price + aggression_level
                and current_sell_position > 0
            ):
                sell_volume = max(-volume, -current_sell_position)
                current_sell_position += sell_volume
                assert sell_volume < 0
                trading_orders.append(Order(product_id, bid_price, sell_volume))
                self.market_taking.append((product_id, sell_volume, True))

        # Additional sell orders based on remaining quota
        if -position_limit - current_sell_position < 0:
            self.handle_additional_sell_orders(
                trading_orders,
                product_id,
                current_sell_position,
                max_sell_price,
                aggression_level,
                lowest_sell_price,
            )

        return trading_orders

    def handle_additional_buy_orders(
        self,
        trading_orders,
        product_id,
        current_position,
        buy_threshold,
        aggression,
        max_price,
    ):
        """
        Handles placing additional buy orders based on remaining capacity and thresholds.

        Args:
            trading_orders (List[Order]): List to append new orders to.
            product_id (str): The product identifier for which orders are being placed.
            current_position (int): Current holding position of the product.
            buy_threshold (int): Minimum price threshold for buying.
            aggression (int): Level of price aggression to adjust the buy price.
            max_price (int): Highest current buy price in the market order book.
        """
        # Constants for thresholds; these should be defined similarly as in your main trading logic
        over_threshold = self.THRESHOLDS[product_id]["over"]
        mid_threshold = self.THRESHOLDS[product_id]["mid"]
        position_limit = self.POSITION_LIMIT[product_id]

        # Additional buying up to different thresholds
        if current_position < over_threshold:
            target_price = min(buy_threshold - aggression, max_price + 1)
            volume_needed = over_threshold - current_position
            trading_orders.append(Order(product_id, target_price, volume_needed))
            current_position += volume_needed

        if over_threshold <= current_position <= mid_threshold:
            target_price = min(buy_threshold - 1 - aggression, max_price + 1)
            volume_needed = mid_threshold - current_position
            trading_orders.append(Order(product_id, target_price, volume_needed))
            current_position += volume_needed

        if current_position > mid_threshold:
            target_price = min(buy_threshold - 2 - aggression, max_price + 1)
            volume_needed = position_limit - current_position
            trading_orders.append(Order(product_id, target_price, volume_needed))
            current_position += volume_needed

    def handle_additional_sell_orders(
        self,
        trading_orders,
        product_id,
        current_position,
        sell_threshold,
        aggression,
        min_price,
    ):
        """
        Handles placing additional sell orders based on remaining quota and thresholds.

        Args:
            trading_orders (List[Order]): List to append new orders to.
            product_id (str): The product identifier for which orders are being placed.
            current_position (int): Current holding position of the product.
            sell_threshold (int): Maximum price threshold for selling.
            aggression (int): Level of price aggression to adjust the sell price.
            min_price (int): Lowest current sell price in the market order book.
        """
        # Constants for thresholds; these should be defined similarly as in your main trading logic
        over_threshold = self.THRESHOLDS[product_id]["over"]
        mid_threshold = self.THRESHOLDS[product_id]["mid"]
        position_limit = self.POSITION_LIMIT[product_id]

        # Additional selling up to different thresholds
        if current_position > -over_threshold:
            target_price = max(sell_threshold + aggression, min_price - 1)
            volume_needed = current_position + over_threshold
            trading_orders.append(Order(product_id, target_price, -volume_needed))
            current_position -= volume_needed

        if -over_threshold >= current_position >= -mid_threshold:
            target_price = max(sell_threshold + 1 + aggression, min_price - 1)
            volume_needed = current_position + mid_threshold
            trading_orders.append(Order(product_id, target_price, -volume_needed))
            current_position -= volume_needed

        if -mid_threshold > current_position:
            target_price = max(sell_threshold + 2 + aggression, min_price - 1)
            volume_needed = -position_limit - current_position
            trading_orders.append(Order(product_id, target_price, -volume_needed))
            current_position -= volume_needed
            

    # compute if we want to make a conversion or not
    def conversion_opp(self):
        conversions = [1]
        prods = ["ORCHIDS"]

        return sum(conversions)

    # compute orders for basket
    def compute_orders_basket(self, order_depth):

        orders = {"CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}
        prods = ["CHOCOLATE", "STRAWBERRIES", "ROSES", "GIFT_BASKET"]
        (osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell) = ({}, {}, {}, {}, {}, {}, {}, {}, {})

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol
                if vol_buy[p] >= self.POSITION_LIMIT[p] / 10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol
                if vol_sell[p] >= self.POSITION_LIMIT[p] / 10:
                    break

        res_buy = (mid_price["GIFT_BASKET"] - mid_price["CHOCOLATE"] * 4 - mid_price["STRAWBERRIES"] * 6 - mid_price["ROSES"] - 388)
        res_sell = (mid_price["GIFT_BASKET"] - mid_price["CHOCOLATE"] * 4 - mid_price["STRAWBERRIES"] * 6 - mid_price["ROSES"] - 388)

        trade_at = self.basket_std * 0.5
        close_at = self.basket_std * (-1000)

        pb_pos = self.position["GIFT_BASKET"]
        pb_neg = self.position["GIFT_BASKET"]

        rose_pos = self.position["ROSES"]
        rose_neg = self.position["ROSES"]

        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position["GIFT_BASKET"] == self.POSITION_LIMIT["GIFT_BASKET"]:
            self.cont_buy_basket_unfill = 0
        if self.position["GIFT_BASKET"] == -self.POSITION_LIMIT["GIFT_BASKET"]:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position["GIFT_BASKET"] + self.POSITION_LIMIT["GIFT_BASKET"]
            self.cont_buy_basket_unfill = 0  # no need to buy rn
            assert vol >= 0
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders["GIFT_BASKET"].append(
                    Order("GIFT_BASKET", worst_buy["GIFT_BASKET"], -vol)
                )
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                # uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT["GIFT_BASKET"] - self.position["GIFT_BASKET"]
            self.cont_sell_basket_unfill = 0  # no need to sell rn
            assert vol >= 0
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders["GIFT_BASKET"].append(
                    Order("GIFT_BASKET", worst_sell["GIFT_BASKET"], vol)
                )
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        if int(round(self.person_position['Rhianna']['ROSES'])) > 0:

            val_ord = self.POSITION_LIMIT['ROSES'] - rose_pos
            if val_ord > 0:
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], val_ord))
        if int(round(self.person_position['Rhianna']['ROSES'])) < 0:

            val_ord = -(self.POSITION_LIMIT['ROSES'] + rose_neg)
            if val_ord < 0:
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], val_ord))


        return orders

    def calc_next_price_coconut(self):
        # coconut cache stores price from 1 day ago, current day resp
        # by price, here we mean mid price

        coef = [0.0171, -0.0126, -0.0088, -0.0436]
        intercept = 6.96667
        nxt_price = intercept
        for i, val in enumerate(self.coconuts_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    # coupon orders

    def compute_orders_coupons(self, order_depth):
        orders = {"COCONUT": [], "COCONUT_COUPON": []}
        prods = ["COCONUT", "COCONUT_COUPON"]
        (osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell) = ({}, {}, {}, {}, {}, {}, {}, {}, {})

        for p in prods:
            osell[p] = collections.OrderedDict(
                sorted(order_depth[p].sell_orders.items())
            )
            obuy[p] = collections.OrderedDict(
                sorted(order_depth[p].buy_orders.items(), reverse=True)
            )

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol
                if vol_buy[p] >= self.POSITION_LIMIT[p] / 10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol
                if vol_sell[p] >= self.POSITION_LIMIT[p] / 10:
                    break

        # calculating coconut coupon, $637.63 is fair value
        u = math.exp(self.stdev_cc * math.sqrt(self.years / self.Tstep))
        d = 1 / u
        probup = ((math.exp((self.rate) * self.years / self.Tstep)) - d) / (u - d)
        discount_factor = self.rate / self.Tstep
        duration_of_time_step = self.years / self.Tstep
        payoffs = []
        for n in range(self.Tstep + 1):
            payoffs.append(
                max(0,(mid_price["COCONUT"] * (u ** ((self.Tstep) - n)) * (d**n) - self.strike)))

        for x in reversed(range(1, self.Tstep + 1)):
            discounting1 = []
            for i in range(0, x):
                discounting1.append((((probup) * payoffs[i]) + ((1 - probup) * payoffs[i + 1])) / (math.exp(discount_factor)))

            payoffs.clear()
            payoffs.extend(discounting1)

        calculated_ccoupon = discounting1[0]

        # ************************************* End of calculations ************************************

        if worst_sell["COCONUT_COUPON"] < calculated_ccoupon:  # fair value is $637.63
            self.buy_ccoupon = True
        if worst_buy['COCONUT_COUPON'] > calculated_ccoupon:
            self.sell_ccoupon = True

        if self.position["COCONUT_COUPON"] == self.POSITION_LIMIT["COCONUT_COUPON"]:
            self.buy_ccoupon = False
        if self.position["COCONUT_COUPON"] == -self.POSITION_LIMIT["COCONUT_COUPON"]:
            self.sell_ccoupon = False

        if self.buy_ccoupon:
            vol = self.position["COCONUT"]
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", worst_sell["COCONUT_COUPON"], vol))
        if self.sell_ccoupon:
            vol = self.position["COCONUT"]
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_buy["COCONUT_COUPON"], -vol))

        return orders

    # compute orders
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "AMETHYSTS":
            return self.compute_orders_amethysts(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == "COCONUT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
            
    
    def get_acceptable_price(self, state: TradingState, product: str):
        """
        Calculate the acceptable price for a given product based on the current market conditions and position limits.

        Args:
            state (TradingState): Current trading market state including order depths.
            product (str): Identifier for the trading product.

        Returns:
            Optional[int]: The acceptable price for the product based on the current market conditions.
        """
        order_depth = state.order_depths[product]
        if product == "AMETHYSTS":
            return self.calc_amethyst_price(state)
        if product == "STARFRUIT":
            return self.calc_starfruit_price(state)
        if product == "COCONUT":
            return self.calc_coconut_price(state)
        return None

    # RUN function, Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict
        result = {
            "AMETHYSTS": [],
            "STARFRUIT": [],
            "ORCHIDS": [],
            "CHOCOLATE": [],
            "STRAWBERRIES": [],
            "ROSES": [],
            "GIFT_BASKET": [],
            "COCONUT": [],
            "COCONUT_COUPON": [],
        }

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f"{key} position: {val}")

        # assert abs(self.position.get('ROSES', 0)) <= self.POSITION_LIMIT['ROSES']

        timestamp = state.timestamp

        if len(self.starfruit_cache) == self.starfruit_dim:
            self.starfruit_cache.pop(0)

        _, bs_starfruit = self.values_extract(
            collections.OrderedDict(
                sorted(state.order_depths["STARFRUIT"].sell_orders.items())
            )
        )
        _, bb_starfruit = self.values_extract(
            collections.OrderedDict(
                sorted(state.order_depths["STARFRUIT"].buy_orders.items(), reverse=True)
            ),
            1,
        )

        self.starfruit_cache.append((bs_starfruit + bb_starfruit) / 2)

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

        _, bs_coconut = self.values_extract(
            collections.OrderedDict(
                sorted(state.order_depths["COCONUT"].sell_orders.items())
            )
        )
        _, bb_coconut = self.values_extract(
            collections.OrderedDict(
                sorted(state.order_depths["COCONUT"].buy_orders.items(), reverse=True)
            ),
            1,
        )

        self.coconuts_cache.append((bs_coconut + bb_coconut) / 2)

        coconut_lb = -INF
        coconut_ub = INF

        if len(self.coconuts_cache) == self.coconut_dim:
            coconut_lb = self.calc_next_price_coconut() - 1
            coconut_ub = self.calc_next_price_coconut() + 1

        # CHANGE FROM HERE

        acc_bid = {
            "AMETHYSTS": amethysts_lb,
            "STARFRUIT": starfruit_lb,
            "COCONUT": coconut_lb,
        }  # we want to buy at slightly below
        acc_ask = {
            "AMETHYSTS": amethysts_ub,
            "STARFRUIT": starfruit_ub,
            "COCONUT": coconut_ub,
        }  # we want to sell at slightly above

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
        for product in ['ORCHIDS','CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET', 'COCONUT_COUPON']:
            if product == "ORCHIDS":
                orders = self.arbitrage(state, product)
                result[product] = orders
                continue
            
            product_acceptable_price = self.get_acceptable_price(state, product)
            if product_acceptable_price is None:
                continue
            
            else:
                
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
                if trade.timestamp != state.timestamp - 100:
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
            print(
                f"For product {product}, {settled_pnl + self.cpnl[product]}, {(settled_pnl+self.cpnl[product])/(self.volume_traded[product]+1e-20)}"
            )

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
