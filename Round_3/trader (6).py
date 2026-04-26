"""
IMC Prosperity Round 3 – "Gloves Off"
======================================
Strategy Summary
----------------
1. HYDROGEL_PACK  : Market-make around fair value ~10000 using EMA of mid-price.
                    Spread is consistently ~16, so we post inside to capture alpha.
2. VELVETFRUIT_EXTRACT (VE): Market-make around running EMA.
                    Also serves as delta hedge for our options book.
3. VEV_* options  : Price every voucher with Black-Scholes using a fitted vol-smile
                    (IV = a*m² + b*m + c, m = log(K/S)).
                    Buy when market ask < fair – threshold.
                    Sell when market bid > fair + threshold.
                    After each option trade, delta-hedge residual exposure with VE.

Vol smile calibration (from historical days 0-2):
    a ≈ 8.0  (curvature / wings)
    b ≈ 0.0  (skew, nearly symmetric)
    c ≈ 0.245  (ATM vol ~24.5%)

Round 3 TTE mapping:
    TTE (days) = 5 - timestamp / 10_000
    TTE (years) = TTE_days / 365
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import math
import json


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300",
    "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

# Vol-smile parameters (fitted from historical data)
# IV(m) = SMILE_A * m^2 + SMILE_B * m + SMILE_C,  m = log(K / S)
SMILE_A = 8.0
SMILE_B = 0.0
SMILE_C = 0.245   # ~24.5% ATM annualised vol

# HYDROGEL fair value anchor (mean ≈ 9991, rounds to 10000)
HP_FAIR = 10000.0
HP_EMA_ALPHA = 0.05   # slow EMA for stability
HP_SPREAD_HALF = 6    # post inside the ~16-tick market spread (so bid @-6, ask @+6)
HP_CLIP = 8           # max order size per side

# VE market-making
VE_EMA_ALPHA = 0.1
VE_SPREAD_HALF = 2    # post inside the 5-tick market spread
VE_CLIP = 15          # max order size per side

# Options arb thresholds
OPT_BUY_THRESH  = 0.5   # buy if market_ask < fair - threshold
OPT_SELL_THRESH = 0.5   # sell if market_bid > fair + threshold
OPT_MAX_POS = 200        # max net position per voucher (< 300 limit)

# Delta-hedge aggressiveness
DELTA_HEDGE_CLIP = 20   # max hedge units per tick


# ─────────────────────────────────────────────────────────────
# Black-Scholes helpers  (pure Python, no scipy dependency)
# ─────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price via Black-Scholes (r = 0)."""
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    d2 = d1 - sq
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta of a European call (r = 0)."""
    if T <= 1e-9 or sigma <= 1e-9:
        return 1.0 if S > K else 0.0
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / sq
    return _norm_cdf(d1)


def smile_iv(K: float, S: float) -> float:
    """Implied vol from fitted quadratic smile."""
    m = math.log(K / S)
    iv = SMILE_A * m * m + SMILE_B * m + SMILE_C
    return max(iv, 0.01)


def fair_option_price(K: float, S: float, T: float) -> float:
    """BS fair price using the vol-smile IV."""
    iv = smile_iv(K, S)
    return bs_call(S, K, T, iv)


def option_delta(K: float, S: float, T: float) -> float:
    """BS delta using vol-smile IV."""
    iv = smile_iv(K, S)
    return bs_delta(S, K, T, iv)


# ─────────────────────────────────────────────────────────────
# Order-book helpers
# ─────────────────────────────────────────────────────────────
def best_bid(od: OrderDepth):
    """(price, volume) of best bid, or (None, 0)."""
    if od.buy_orders:
        p = max(od.buy_orders)
        return p, od.buy_orders[p]
    return None, 0


def best_ask(od: OrderDepth):
    """(price, volume) of best ask, or (None, 0)."""
    if od.sell_orders:
        p = min(od.sell_orders)
        return p, od.sell_orders[p]
    return None, 0


def mid_price(od: OrderDepth):
    bp, _ = best_bid(od)
    ap, _ = best_ask(od)
    if bp is not None and ap is not None:
        return (bp + ap) / 2.0
    if bp is not None:
        return float(bp)
    if ap is not None:
        return float(ap)
    return None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ─────────────────────────────────────────────────────────────
# Trader class
# ─────────────────────────────────────────────────────────────
class Trader:
    """
    Stateful trader.  Per-tick state persisted via trader_data (JSON).
    """

    # ---------- state keys ----------
    def _load_state(self, raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _save_state(self, state: dict) -> str:
        return json.dumps(state)

    # ---------- EMA ----------
    def _ema(self, prev: float | None, new_val: float, alpha: float) -> float:
        if prev is None or prev == 0:
            return new_val
        return alpha * new_val + (1.0 - alpha) * prev

    # ========================================================
    # Main entry point
    # ========================================================
    def run(self, state: TradingState):
        # Load persistent state
        sd = self._load_state(state.traderData)

        orders: Dict[str, List[Order]] = {}
        conversions = 0

        timestamp = state.timestamp   # 0, 100, 200, …, 9900
        ods = state.order_depths
        pos = state.position          # dict product -> int

        # ── 1. Estimate fair value for VELVETFRUIT_EXTRACT (underlying) ──
        ve_fair = self._estimate_ve_fair(sd, ods, timestamp)

        # ── 2. TTE for options ──
        tte_days = 5.0 - timestamp / 10_000.0   # decreases 5→~4 over the round
        tte_years = max(tte_days / 365.0, 1e-9)

        # ── 3. Market-make HYDROGEL_PACK ──
        if "HYDROGEL_PACK" in ods:
            hp_orders = self._trade_hydrogel(sd, ods["HYDROGEL_PACK"],
                                             pos.get("HYDROGEL_PACK", 0))
            if hp_orders:
                orders["HYDROGEL_PACK"] = hp_orders

        # ── 4. Market-make VELVETFRUIT_EXTRACT (baseline, before hedge) ──
        ve_base_orders = []
        if "VELVETFRUIT_EXTRACT" in ods and ve_fair is not None:
            ve_base_orders = self._trade_ve_base(
                sd, ods["VELVETFRUIT_EXTRACT"],
                pos.get("VELVETFRUIT_EXTRACT", 0), ve_fair)

        # ── 5. Options arb + delta computation ──
        opt_orders, total_delta = self._trade_options(
            sd, ods, pos, ve_fair, tte_years)
        orders.update(opt_orders)

        # ── 6. Delta-hedge residual with VE ──
        ve_pos = pos.get("VELVETFRUIT_EXTRACT", 0)
        hedge_orders = self._delta_hedge(
            sd, ods.get("VELVETFRUIT_EXTRACT"), ve_pos,
            total_delta, ve_base_orders)
        if hedge_orders:
            orders["VELVETFRUIT_EXTRACT"] = hedge_orders
        elif ve_base_orders:
            orders["VELVETFRUIT_EXTRACT"] = ve_base_orders

        return orders, conversions, self._save_state(sd)

    # ========================================================
    # VE fair-value estimation (EMA of mid)
    # ========================================================
    def _estimate_ve_fair(self, sd, ods, timestamp) -> float | None:
        if "VELVETFRUIT_EXTRACT" not in ods:
            return sd.get("ve_ema")
        mp = mid_price(ods["VELVETFRUIT_EXTRACT"])
        if mp is None:
            return sd.get("ve_ema")
        prev = sd.get("ve_ema")
        new_ema = self._ema(prev, mp, VE_EMA_ALPHA)
        sd["ve_ema"] = new_ema
        return new_ema

    # ========================================================
    # HYDROGEL_PACK market making
    # ========================================================
    def _trade_hydrogel(self, sd, od: OrderDepth, pos: int) -> List[Order]:
        """Market-make HP around its fair value ~10000."""
        mp = mid_price(od)
        if mp is None:
            return []

        # EMA of HP mid-price
        prev = sd.get("hp_ema", HP_FAIR)
        hp_ema = self._ema(prev, mp, HP_EMA_ALPHA)
        sd["hp_ema"] = hp_ema

        # Fair value: weighted blend of EMA and 10000 anchor
        fair = 0.7 * hp_ema + 0.3 * HP_FAIR

        limit = POSITION_LIMITS["HYDROGEL_PACK"]
        result: List[Order] = []

        # Inventory skew: if long, lower prices slightly to encourage selling
        skew = -int(pos / limit * HP_SPREAD_HALF * 0.5)

        bid_price_target = int(round(fair - HP_SPREAD_HALF + skew))
        ask_price_target = int(round(fair + HP_SPREAD_HALF + skew))

        # Don't post a bid above best ask or ask below best bid
        bp, bvol = best_bid(od)
        ap, avol = best_ask(od)
        if ap is not None:
            bid_price_target = min(bid_price_target, ap - 1)
        if bp is not None:
            ask_price_target = max(ask_price_target, bp + 1)

        # Buy capacity
        buy_capacity = limit - pos
        if buy_capacity > 0 and bid_price_target > 0:
            qty = clamp(HP_CLIP, 1, buy_capacity)
            result.append(Order("HYDROGEL_PACK", bid_price_target, qty))

        # Sell capacity
        sell_capacity = limit + pos   # pos can be negative
        if sell_capacity > 0 and ask_price_target > 0:
            qty = clamp(HP_CLIP, 1, sell_capacity)
            result.append(Order("HYDROGEL_PACK", ask_price_target, -qty))

        # Also aggressively take any mis-priced orders in the book
        # Buy below fair - spread_half (i.e. any ask <= bid_target)
        for ask_p in sorted(od.sell_orders.keys()):
            if ask_p <= bid_price_target and buy_capacity > 0:
                take_qty = min(abs(od.sell_orders[ask_p]), buy_capacity, HP_CLIP * 2)
                result.append(Order("HYDROGEL_PACK", ask_p, take_qty))
                buy_capacity -= take_qty
        # Sell above fair + spread_half
        for bid_p in sorted(od.buy_orders.keys(), reverse=True):
            if bid_p >= ask_price_target and sell_capacity > 0:
                take_qty = min(od.buy_orders[bid_p], sell_capacity, HP_CLIP * 2)
                result.append(Order("HYDROGEL_PACK", bid_p, -take_qty))
                sell_capacity -= take_qty

        return result

    # ========================================================
    # VELVETFRUIT_EXTRACT baseline market making
    # ========================================================
    def _trade_ve_base(self, sd, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        """Market-make VE around fair.  Returns proposed orders (may be overridden by hedge)."""
        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        result: List[Order] = []

        skew = -int(pos / limit * VE_SPREAD_HALF * 0.5)
        bid_p = int(round(fair - VE_SPREAD_HALF + skew))
        ask_p = int(round(fair + VE_SPREAD_HALF + skew))

        bp, _ = best_bid(od)
        ap, _ = best_ask(od)
        if ap is not None:
            bid_p = min(bid_p, ap - 1)
        if bp is not None:
            ask_p = max(ask_p, bp + 1)

        buy_cap = limit - pos
        sell_cap = limit + pos
        if buy_cap > 0:
            result.append(Order("VELVETFRUIT_EXTRACT", bid_p, min(VE_CLIP, buy_cap)))
        if sell_cap > 0:
            result.append(Order("VELVETFRUIT_EXTRACT", ask_p, -min(VE_CLIP, sell_cap)))
        return result

    # ========================================================
    # Options arb (VEV_*)
    # ========================================================
    def _trade_options(self, sd, ods, pos, S: float | None,
                       T: float) -> tuple[dict, float]:
        """
        For each VEV, compute BS fair, compare to market, place arb orders.
        Returns (orders_dict, net_delta_of_option_book).
        """
        if S is None:
            return {}, 0.0

        orders: Dict[str, List[Order]] = {}
        net_delta = 0.0

        # First compute existing net delta from carried positions
        for prod, K in VEV_STRIKES.items():
            curr_pos = pos.get(prod, 0)
            if curr_pos != 0:
                net_delta += curr_pos * option_delta(K, S, T)

        for prod, K in VEV_STRIKES.items():
            if prod not in ods:
                continue
            od = ods[prod]
            curr_pos = pos.get(prod, 0)
            limit = POSITION_LIMITS[prod]

            fair = fair_option_price(K, S, T)
            delta = option_delta(K, S, T)
            intrinsic = max(S - K, 0.0)

            # Skip deeply ITM (price ~= intrinsic, no edge) or worthless OTM
            time_value = fair - intrinsic
            if time_value < 0.2:
                continue

            result: List[Order] = []

            # ── Buy cheap options ──
            ap, avol = best_ask(od)
            if ap is not None:
                edge = fair - ap
                if edge >= OPT_BUY_THRESH:
                    # Scale qty by edge strength, capped at position limit
                    qty_target = clamp(int(edge * 20), 1, OPT_MAX_POS)
                    qty = clamp(qty_target, 1, limit - curr_pos)
                    if qty > 0:
                        result.append(Order(prod, ap, qty))
                        net_delta += qty * delta

            # ── Sell expensive options ──
            bp, bvol = best_bid(od)
            if bp is not None:
                edge = bp - fair
                if edge >= OPT_SELL_THRESH:
                    qty_target = clamp(int(edge * 20), 1, OPT_MAX_POS)
                    qty = clamp(qty_target, 1, limit + curr_pos)
                    if qty > 0:
                        result.append(Order(prod, bp, -qty))
                        net_delta -= qty * delta

            # ── Passive market-making inside the spread ──
            # Only for liquid near-ATM options (time_value > 5)
            if time_value > 5.0 and bp is not None and ap is not None:
                spread = ap - bp
                if spread >= 2:
                    # Post bid and ask one tick inside
                    mm_bid = bp + 1
                    mm_ask = ap - 1
                    if mm_bid < mm_ask:
                        buy_cap = limit - curr_pos
                        sell_cap = limit + curr_pos
                        if buy_cap > 0:
                            result.append(Order(prod, mm_bid, min(5, buy_cap)))
                        if sell_cap > 0:
                            result.append(Order(prod, mm_ask, -min(5, sell_cap)))

            if result:
                orders[prod] = result

        return orders, net_delta

    # ========================================================
    # Delta hedge with VE
    # ========================================================
    def _delta_hedge(self, sd, od: OrderDepth | None, ve_pos: int,
                     net_delta: float, base_orders: List[Order]) -> List[Order]:
        """
        Target VE position = -round(net_delta) to flatten option-book delta.
        Try to achieve this while staying within position limit.
        """
        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        target_ve = clamp(-int(round(net_delta)), -limit, limit)
        delta_needed = target_ve - ve_pos

        if abs(delta_needed) < 2:
            # Close enough – keep base orders for revenue
            return base_orders

        # We need to trade delta_needed units of VE
        result: List[Order] = []
        if od is None:
            return result

        remaining = delta_needed
        if remaining > 0:
            # Need to buy VE
            for ask_p in sorted(od.sell_orders.keys()):
                if remaining <= 0:
                    break
                take = min(abs(od.sell_orders[ask_p]), remaining, DELTA_HEDGE_CLIP)
                result.append(Order("VELVETFRUIT_EXTRACT", ask_p, take))
                remaining -= take
            # Post a passive bid for remainder
            if remaining > 0:
                bp, _ = best_bid(od)
                if bp is not None:
                    post_qty = min(remaining, DELTA_HEDGE_CLIP)
                    result.append(Order("VELVETFRUIT_EXTRACT", bp + 1, post_qty))
        else:
            # Need to sell VE
            needed = -remaining
            for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                if needed <= 0:
                    break
                take = min(od.buy_orders[bid_p], needed, DELTA_HEDGE_CLIP)
                result.append(Order("VELVETFRUIT_EXTRACT", bid_p, -take))
                needed -= take
            # Post a passive ask for remainder
            if needed > 0:
                ap, _ = best_ask(od)
                if ap is not None:
                    post_qty = min(needed, DELTA_HEDGE_CLIP)
                    result.append(Order("VELVETFRUIT_EXTRACT", ap - 1, -post_qty))

        # Merge with base orders where capacity remains
        merged = list(result)
        final_ve = ve_pos + delta_needed
        for o in base_orders:
            if o.quantity > 0 and final_ve < limit:
                merged.append(o)
            elif o.quantity < 0 and final_ve > -limit:
                merged.append(o)

        return merged
