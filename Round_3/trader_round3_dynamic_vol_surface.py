"""
IMC Prosperity Round 3 - Dynamic Vol-Surface Trader
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_4000 ... VEV_6500

Core changes vs older versions:
1. Uses current underlying mid/microprice for option pricing, not a laggy EMA.
2. Correct Round 3 TTE mapping: 5 - timestamp / 1_000_000 days.
3. Fits the option IV smile live each tick using reliable strikes, then prices each
   voucher with leave-one-out style residual logic.
4. Treats VEV_5400 as the main long-vol/cheap option anomaly and VEV_5500/5300
   as the main rich option anomalies, but still requires live edge before trading.
5. Scales target position by edge size instead of blindly maxing every signal.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import math
import json

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

# Reliable smile strikes. Deep ITM options have almost no time value, and 6000/6500
# are pinned at the 0.5 floor, so they are deliberately excluded from calibration.
CALIBRATION_PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5500"]

TICKS_PER_ROUND = 1_000_000
ROUND3_START_TTE_DAYS = 5.0

# Historical all-day fit around near-money strikes: IV ~= 1.655*m^2 + 0.0147*m + 0.2395
# Used only as fallback if live IV calibration is not available.
FALLBACK_A = 1.655
FALLBACK_B = 0.0147
FALLBACK_C = 0.2395

# Product-specific historical bias from the three provided days.
# Positive means market tends to price IV rich vs smooth smile; negative means cheap.
HIST_IV_RESID = {
    "VEV_5000": -0.0008,
    "VEV_5100": -0.0009,
    "VEV_5200":  0.0028,
    "VEV_5300":  0.0052,
    "VEV_5400": -0.0116,
    "VEV_5500":  0.0053,
}

# Trading thresholds in price units.
EDGE_TAKE = {
    "VEV_5000": 1.2,
    "VEV_5100": 1.0,
    "VEV_5200": 0.8,
    "VEV_5300": 0.6,
    "VEV_5400": 0.35,   # very persistent anomaly and tight spread
    "VEV_5500": 0.35,   # very persistent anomaly and tight spread
}

# Max strategic target per option. Keep some room below hard limits for fills.
MAX_TARGET = {
    "VEV_5000": 80,
    "VEV_5100": 100,
    "VEV_5200": 120,
    "VEV_5300": 220,
    "VEV_5400": 300,
    "VEV_5500": 260,
}

HP_ANCHOR = 9991.0
HP_ALPHA = 0.06
VE_ALPHA = 0.18


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return max(S - K, 0.0)
    if T <= 1e-9 or sigma <= 1e-9:
        return max(S - K, 0.0)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 1e-9:
        return 1.0 if S > K else 0.0
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    return norm_cdf(d1)


def implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if T <= 1e-9 or price <= intrinsic + 0.05:
        return None
    lo, hi = 0.001, 2.0
    for _ in range(45):
        mid = (lo + hi) / 2.0
        if bs_call(S, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def best_bid(od: OrderDepth) -> Tuple[Optional[int], int]:
    if not od.buy_orders:
        return None, 0
    p = max(od.buy_orders)
    return p, od.buy_orders[p]


def best_ask(od: OrderDepth) -> Tuple[Optional[int], int]:
    if not od.sell_orders:
        return None, 0
    p = min(od.sell_orders)
    return p, abs(od.sell_orders[p])


def mid_price(od: OrderDepth) -> Optional[float]:
    bp, _ = best_bid(od)
    ap, _ = best_ask(od)
    if bp is not None and ap is not None:
        return (bp + ap) / 2.0
    if bp is not None:
        return float(bp)
    if ap is not None:
        return float(ap)
    return None


def micro_price(od: OrderDepth) -> Optional[float]:
    bp, bv = best_bid(od)
    ap, av = best_ask(od)
    if bp is None or ap is None or bv <= 0 or av <= 0:
        return mid_price(od)
    # If ask size is small relative to bid size, price leans upward, and vice versa.
    return (bp * av + ap * bv) / (av + bv)


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def ema(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def fit_quadratic(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Small no-numpy quadratic least-squares fit y = a*x^2 + b*x + c.
    If not enough points, return historical fallback.
    """
    if len(points) < 3:
        return FALLBACK_A, FALLBACK_B, FALLBACK_C

    # Normal equations for columns [x^2, x, 1]
    s40 = s30 = s20 = s10 = n = 0.0
    y20 = y10 = y00 = 0.0
    for x, y in points:
        x2 = x * x
        s40 += x2 * x2
        s30 += x2 * x
        s20 += x2
        s10 += x
        n += 1.0
        y20 += y * x2
        y10 += y * x
        y00 += y

    A = [[s40, s30, s20], [s30, s20, s10], [s20, s10, n]]
    b = [y20, y10, y00]

    # Gaussian elimination with fallback for singular matrices.
    try:
        for i in range(3):
            pivot = i
            for r in range(i + 1, 3):
                if abs(A[r][i]) > abs(A[pivot][i]):
                    pivot = r
            if abs(A[pivot][i]) < 1e-12:
                return FALLBACK_A, FALLBACK_B, FALLBACK_C
            if pivot != i:
                A[i], A[pivot] = A[pivot], A[i]
                b[i], b[pivot] = b[pivot], b[i]
            div = A[i][i]
            for j in range(i, 3):
                A[i][j] /= div
            b[i] /= div
            for r in range(3):
                if r == i:
                    continue
                factor = A[r][i]
                for j in range(i, 3):
                    A[r][j] -= factor * A[i][j]
                b[r] -= factor * b[i]
        a, bb, c = b
        # Keep fit sane; live data can be noisy at boundaries.
        if c < 0.15 or c > 0.35 or abs(a) > 20:
            return FALLBACK_A, FALLBACK_B, FALLBACK_C
        return a, bb, c
    except Exception:
        return FALLBACK_A, FALLBACK_B, FALLBACK_C


def smile_iv(K: float, S: float, coeffs: Tuple[float, float, float]) -> float:
    a, b, c = coeffs
    m = math.log(K / S)
    return max(0.05, min(0.80, a * m * m + b * m + c))


class Trader:
    def _load(self, data: str) -> Dict:
        if not data:
            return {}
        try:
            return json.loads(data)
        except Exception:
            return {}

    def _save(self, sd: Dict) -> str:
        return json.dumps(sd, separators=(",", ":"))

    def run(self, state: TradingState):
        sd = self._load(state.traderData)
        orders: Dict[str, List[Order]] = {}
        pos = state.position
        ods = state.order_depths

        # --- Underlying fair values ---
        ve_mid = None
        if "VELVETFRUIT_EXTRACT" in ods:
            ve_mid = micro_price(ods["VELVETFRUIT_EXTRACT"])
            if ve_mid is not None:
                sd["ve_ema"] = ema(sd.get("ve_ema"), ve_mid, VE_ALPHA)
        S = ve_mid if ve_mid is not None else sd.get("ve_ema", 5250.0)

        tte_days = max(0.01, ROUND3_START_TTE_DAYS - state.timestamp / TICKS_PER_ROUND)
        T = tte_days / 365.0

        # --- Base market making ---
        if "HYDROGEL_PACK" in ods:
            hp_orders = self.trade_hydrogel(ods["HYDROGEL_PACK"], pos.get("HYDROGEL_PACK", 0), sd)
            if hp_orders:
                orders["HYDROGEL_PACK"] = hp_orders

        ve_base_orders: List[Order] = []
        if "VELVETFRUIT_EXTRACT" in ods:
            ve_base_orders = self.trade_velvetfruit(ods["VELVETFRUIT_EXTRACT"], pos.get("VELVETFRUIT_EXTRACT", 0), S)

        # --- Options: live IV surface + historical residual bias ---
        opt_orders, target_option_delta = self.trade_options(ods, pos, S, T)
        orders.update(opt_orders)

        # --- Hedge option-book delta, but do not overpay unless exposure is meaningful ---
        ve_orders = self.trade_delta_hedge(
            ods.get("VELVETFRUIT_EXTRACT"),
            pos.get("VELVETFRUIT_EXTRACT", 0),
            target_option_delta,
            ve_base_orders
        )
        if ve_orders:
            orders["VELVETFRUIT_EXTRACT"] = ve_orders
        elif ve_base_orders:
            orders["VELVETFRUIT_EXTRACT"] = ve_base_orders

        return orders, 0, self._save(sd)

    def trade_hydrogel(self, od: OrderDepth, position: int, sd: Dict) -> List[Order]:
        mp = micro_price(od)
        if mp is None:
            return []
        sd["hp_ema"] = ema(sd.get("hp_ema", HP_ANCHOR), mp, HP_ALPHA)
        fair = 0.65 * sd["hp_ema"] + 0.35 * HP_ANCHOR

        result: List[Order] = []
        limit = POSITION_LIMITS["HYDROGEL_PACK"]
        bp, bv = best_bid(od)
        ap, av = best_ask(od)

        # Take only clearly stale top-of-book quotes.
        if ap is not None and ap < fair - 4 and position < limit:
            q = min(av, 25, limit - position)
            if q > 0:
                result.append(Order("HYDROGEL_PACK", ap, q))
                position += q
        if bp is not None and bp > fair + 4 and position > -limit:
            q = min(bv, 25, limit + position)
            if q > 0:
                result.append(Order("HYDROGEL_PACK", bp, -q))
                position -= q

        skew = position / limit
        bid = int(math.floor(fair - 6.5 - 3.0 * skew))
        ask = int(math.ceil(fair + 6.5 - 3.0 * skew))
        if ap is not None:
            bid = min(bid, ap - 1)
        if bp is not None:
            ask = max(ask, bp + 1)

        buy_cap = limit - position
        sell_cap = limit + position
        if buy_cap > 0:
            result.append(Order("HYDROGEL_PACK", bid, min(12, buy_cap)))
        if sell_cap > 0:
            result.append(Order("HYDROGEL_PACK", ask, -min(12, sell_cap)))
        return result

    def trade_velvetfruit(self, od: OrderDepth, position: int, fair: float) -> List[Order]:
        result: List[Order] = []
        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        bp, bv = best_bid(od)
        ap, av = best_ask(od)

        # The spread is tight, so be selective on taking and mostly earn spread.
        if ap is not None and ap < fair - 1.5 and position < limit:
            q = min(av, 18, limit - position)
            if q > 0:
                result.append(Order("VELVETFRUIT_EXTRACT", ap, q))
                position += q
        if bp is not None and bp > fair + 1.5 and position > -limit:
            q = min(bv, 18, limit + position)
            if q > 0:
                result.append(Order("VELVETFRUIT_EXTRACT", bp, -q))
                position -= q

        skew = position / limit
        bid = int(math.floor(fair - 2.0 - 1.5 * skew))
        ask = int(math.ceil(fair + 2.0 - 1.5 * skew))
        if ap is not None:
            bid = min(bid, ap - 1)
        if bp is not None:
            ask = max(ask, bp + 1)

        if limit - position > 0:
            result.append(Order("VELVETFRUIT_EXTRACT", bid, min(12, limit - position)))
        if limit + position > 0:
            result.append(Order("VELVETFRUIT_EXTRACT", ask, -min(12, limit + position)))
        return result

    def live_iv_points(self, ods: Dict[str, OrderDepth], S: float, T: float, exclude: Optional[str] = None) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for prod in CALIBRATION_PRODUCTS:
            if prod == exclude or prod not in ods:
                continue
            mp = mid_price(ods[prod])
            if mp is None:
                continue
            K = STRIKES[prod]
            iv = implied_vol(mp, S, K, T)
            if iv is None or iv < 0.05 or iv > 0.80:
                continue
            pts.append((math.log(K / S), iv))
        return pts

    def trade_options(self, ods: Dict[str, OrderDepth], pos: Dict[str, int], S: float, T: float) -> Tuple[Dict[str, List[Order]], float]:
        orders: Dict[str, List[Order]] = {}
        target_delta = 0.0

        # Delta of existing options.
        base_coeffs = fit_quadratic(self.live_iv_points(ods, S, T, exclude=None))
        for prod, K in STRIKES.items():
            current = pos.get(prod, 0)
            if current != 0:
                iv = smile_iv(K, S, base_coeffs)
                target_delta += current * bs_delta(S, K, T, iv)

        trade_products = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
        for prod in trade_products:
            if prod not in ods:
                continue

            od = ods[prod]
            K = STRIKES[prod]
            position = pos.get(prod, 0)
            limit = POSITION_LIMITS[prod]
            bp, bv = best_bid(od)
            ap, av = best_ask(od)
            if bp is None or ap is None:
                continue

            coeffs = fit_quadratic(self.live_iv_points(ods, S, T, exclude=prod))
            iv_model = smile_iv(K, S, coeffs)

            # Historical residual adjustment:
            # If a voucher historically sits rich vs the smooth smile, lower fair IV.
            # If it historically sits cheap vs the smooth smile, raise fair IV.
            # This keeps the model focused on true smooth-surface value instead of
            # following the same persistent anomaly we are trying to exploit.
            hist = HIST_IV_RESID.get(prod, 0.0)
            fair_iv = max(0.05, min(0.80, iv_model - hist))
            fair = bs_call(S, K, T, fair_iv)
            delta = bs_delta(S, K, T, fair_iv)
            mid = (bp + ap) / 2.0
            threshold = EDGE_TAKE.get(prod, 0.8)

            # Directional conviction from history. 5400 is cheap; 5300/5500 are rich.
            prefer_buy = hist < -0.004
            prefer_sell = hist > 0.004

            result: List[Order] = []

            # Live price edge in the natural direction.
            buy_edge = fair - ap
            sell_edge = bp - fair

            if buy_edge >= threshold and (prefer_buy or (not prefer_sell and buy_edge >= threshold + 0.8)):
                raw_target = int(min(MAX_TARGET.get(prod, 100), max(20, buy_edge * 75)))
                if prefer_buy:
                    raw_target = min(MAX_TARGET.get(prod, 100), max(raw_target, 180))
                needed = min(raw_target - position, limit - position)
                if needed > 0:
                    take = min(av, needed, 45)
                    if take > 0:
                        result.append(Order(prod, ap, take))
                        position += take
                        target_delta += take * delta
                        needed -= take
                    # Passive bid for remaining desired inventory.
                    if needed > 0:
                        bid_px = int(math.floor(fair - threshold * 0.45))
                        bid_px = min(bid_px, ap - 1)
                        bid_px = max(bid_px, bp + 1) if bp + 1 < ap else bp
                        q = min(35, needed)
                        if q > 0 and bid_px > 0:
                            result.append(Order(prod, bid_px, q))

            if sell_edge >= threshold and (prefer_sell or (not prefer_buy and sell_edge >= threshold + 0.8)):
                raw_target = -int(min(MAX_TARGET.get(prod, 100), max(20, sell_edge * 75)))
                if prefer_sell:
                    raw_target = -min(MAX_TARGET.get(prod, 100), max(abs(raw_target), 160))
                needed = max(raw_target - position, -limit - position)  # negative if need to sell
                if needed < 0:
                    take = min(bv, -needed, 45)
                    if take > 0:
                        result.append(Order(prod, bp, -take))
                        position -= take
                        target_delta -= take * delta
                        needed += take
                    # Passive ask for remaining desired short inventory.
                    if needed < 0:
                        ask_px = int(math.ceil(fair + threshold * 0.45))
                        ask_px = max(ask_px, bp + 1)
                        ask_px = min(ask_px, ap - 1) if bp + 1 < ap else ap
                        q = min(35, -needed)
                        if q > 0 and ask_px > 0:
                            result.append(Order(prod, ask_px, -q))

            # Low-risk spread capture only when not fighting the historical residual.
            if not result and ap - bp >= 2 and prod in ("VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"):
                if position < limit and fair > bp + 1.0:
                    bid_px = min(bp + 1, ap - 1)
                    result.append(Order(prod, bid_px, min(6, limit - position)))
                if position > -limit and fair < ap - 1.0:
                    ask_px = max(ap - 1, bp + 1)
                    result.append(Order(prod, ask_px, -min(6, limit + position)))

            if result:
                orders[prod] = result

        return orders, target_delta

    def trade_delta_hedge(self, od: Optional[OrderDepth], ve_pos: int, option_delta: float, base_orders: List[Order]) -> List[Order]:
        if od is None:
            return base_orders

        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        target_ve = clamp(int(round(-option_delta)), -limit, limit)
        need = target_ve - ve_pos

        # Small residual delta is not worth crossing the spread for.
        if abs(need) < 15:
            return base_orders

        result: List[Order] = []
        bp, bv = best_bid(od)
        ap, av = best_ask(od)

        if need > 0 and ap is not None:
            q = min(35, need, av, limit - ve_pos)
            if q > 0:
                result.append(Order("VELVETFRUIT_EXTRACT", ap, q))
                ve_pos += q
                need -= q
            if need > 0 and bp is not None and ve_pos < limit:
                result.append(Order("VELVETFRUIT_EXTRACT", bp + 1, min(25, need, limit - ve_pos)))

        elif need < 0 and bp is not None:
            q = min(35, -need, bv, limit + ve_pos)
            if q > 0:
                result.append(Order("VELVETFRUIT_EXTRACT", bp, -q))
                ve_pos -= q
                need += q
            if need < 0 and ap is not None and ve_pos > -limit:
                result.append(Order("VELVETFRUIT_EXTRACT", ap - 1, -min(25, -need, limit + ve_pos)))

        # Keep base market-making orders if they do not push us away from hedge too much.
        for o in base_orders:
            if o.quantity > 0 and ve_pos < limit and target_ve >= ve_pos - 40:
                result.append(o)
            elif o.quantity < 0 and ve_pos > -limit and target_ve <= ve_pos + 40:
                result.append(o)

        return result
