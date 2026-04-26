"""
IMC Prosperity Round 3 - "Gloves Off"
=====================================
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_4000..VEV_6500

KEY FINDINGS FROM DATA ANALYSIS:
---------------------------------
1. HYDROGEL_PACK: Mean-reverting around ~9990, spread ~15.7, half-life ~189 ticks
   → Market make with EMA fair value tracking

2. VELVETFRUIT_EXTRACT: Mean-reverting around ~5250, spread ~5
   → Market make + use as underlying for BS option pricing

3. VEV OPTIONS (Black-Scholes with σ=23.2%):
   - VEV_5400: Consistently UNDERPRICED by ~2.1 (IV ~22.2% vs consensus 23.2%) → BUY MAX
   - VEV_5500: Consistently OVERPRICED by ~0.9 (IV ~23.9% vs consensus 23.2%) → SELL MAX
   - VEV_5300: Often overpriced by ~1.3 → SELL moderately
   - VEV_5200: Mixed signal, market make around fair
   - VEV_5100/5000: Deep ITM, mostly intrinsic → gentle market make
   - VEV_4000/4500: Pure intrinsic (delta≈1) → track underlying, small spread
   - VEV_6000/6500: Floor at 0.5, no opportunity

TTE in Round 3:
   - Starts at 5 days (TTE=5d), ends at 4 days
   - At timestamp t (0..1,000,000): TTE = (5 - t/1_000_000) days
   - T_years = TTE_days / 365
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import json

# ───────────────────────────────────────────────────────────────────────────────
# Compatibility shims for the Prosperity trading framework
# ───────────────────────────────────────────────────────────────────────────────
try:
    from datamodel import (
        OrderDepth, TradingState, Order, ConversionObservation,
        Observation, ProsperityEncoder, Symbol, Product, Position,
        Trade, Listing
    )
except ImportError:
    # Minimal stubs so the file parses in isolation
    Symbol = str
    Product = str
    Position = int

    @dataclass
    class Order:
        symbol: str
        price: int
        quantity: int

    @dataclass
    class OrderDepth:
        buy_orders: Dict[int, int] = field(default_factory=dict)
        sell_orders: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class Trade:
        symbol: str
        price: int
        quantity: int
        buyer: str = ""
        seller: str = ""
        timestamp: int = 0

    @dataclass
    class TradingState:
        timestamp: int = 0
        listings: Dict = field(default_factory=dict)
        order_depths: Dict[str, OrderDepth] = field(default_factory=dict)
        own_trades: Dict[str, List[Trade]] = field(default_factory=dict)
        market_trades: Dict[str, List[Trade]] = field(default_factory=dict)
        position: Dict[str, int] = field(default_factory=dict)
        observations: object = None
        traderData: str = ""

    class ProsperityEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return super().default(obj)


# ───────────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────────
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

VOUCHER_STRIKES = {
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

# Consensus implied vol derived from ATM options across all historical days
# VEV_5200 and VEV_5300 both show ~23.2% consistently
SIGMA_CONSENSUS = 0.232

# Round 3 starts with TTE = 5 days; ends at 4 days
# Within round: timestamp goes 0..1,000,000
# TTE(t) = (5 - t/1_000_000) days   → T_years = TTE_days / 365
TTE_START_DAYS = 5.0
TICKS_PER_ROUND = 1_000_000


# ───────────────────────────────────────────────────────────────────────────────
# Black-Scholes helpers  (no external libraries needed)
# ───────────────────────────────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation of the normal CDF (max error < 7.5e-8)."""
    if x < -6:
        return 0.0
    if x > 6:
        return 1.0
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    approx = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return 0.5 + sign * (approx - 0.5)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """European call price under Black-Scholes (r=0)."""
    if T <= 1e-8 or sigma <= 1e-8:
        return max(0.0, S - K)
    if S <= 0 or K <= 0:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta for a European call."""
    if T <= 1e-8 or sigma <= 1e-8:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return _norm_cdf(d1)


def implied_vol(price: float, S: float, K: float, T: float,
                lo: float = 0.001, hi: float = 5.0, iterations: int = 50) -> float:
    """Bisection implied vol solver."""
    intrinsic = max(0.0, S - K)
    if price <= intrinsic + 0.01 or T <= 1e-8:
        return lo
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if bs_call_price(S, K, T, mid) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


# ───────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average helper
# ───────────────────────────────────────────────────────────────────────────────
class EMA:
    def __init__(self, alpha: float, initial: Optional[float] = None):
        self.alpha = alpha
        self.value = initial

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ───────────────────────────────────────────────────────────────────────────────
# Trader state persisted across ticks via traderData JSON
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class TraderState:
    hydrogel_ema: float = 9990.0
    vev_ema: float = 5250.0
    # Store per-strike IVs from last tick for smoothing
    iv_ema: Dict[str, float] = field(default_factory=dict)


def load_state(trader_data: str) -> TraderState:
    if not trader_data:
        return TraderState()
    try:
        d = json.loads(trader_data)
        s = TraderState()
        s.hydrogel_ema = d.get("hydrogel_ema", 9990.0)
        s.vev_ema = d.get("vev_ema", 5250.0)
        s.iv_ema = d.get("iv_ema", {})
        return s
    except Exception:
        return TraderState()


def save_state(s: TraderState) -> str:
    return json.dumps({
        "hydrogel_ema": s.hydrogel_ema,
        "vev_ema": s.vev_ema,
        "iv_ema": s.iv_ema,
    })


# ───────────────────────────────────────────────────────────────────────────────
# Order helpers
# ───────────────────────────────────────────────────────────────────────────────
def best_bid(depth: OrderDepth) -> Optional[Tuple[int, int]]:
    if not depth.buy_orders:
        return None
    price = max(depth.buy_orders)
    return price, depth.buy_orders[price]


def best_ask(depth: OrderDepth) -> Optional[Tuple[int, int]]:
    if not depth.sell_orders:
        return None
    price = min(depth.sell_orders)
    return price, depth.sell_orders[price]


def mid_price(depth: OrderDepth) -> Optional[float]:
    b = best_bid(depth)
    a = best_ask(depth)
    if b and a:
        return (b[0] + a[0]) / 2.0
    if b:
        return float(b[0])
    if a:
        return float(a[0])
    return None


def clamp_qty(qty: int, pos: int, limit: int) -> int:
    """Clamp order quantity to stay within position limits."""
    if qty > 0:
        return min(qty, limit - pos)
    else:
        return max(qty, -limit - pos)


# ───────────────────────────────────────────────────────────────────────────────
# Strategy functions
# ───────────────────────────────────────────────────────────────────────────────
def trade_hydrogel(
    depth: OrderDepth,
    position: int,
    ema: EMA,
    orders: List[Order],
) -> None:
    """
    Market make HYDROGEL_PACK around an EMA fair value.
    Spread ~15.7 in market; we quote at fair ± 7 to undercut by 1 tick.
    Soft position skew: narrow quotes when flat, tighten when skewed.
    """
    LIMIT = POSITION_LIMITS["HYDROGEL_PACK"]
    mp = mid_price(depth)
    if mp is None:
        return

    fair = ema.update(mp)
    skew = position / LIMIT  # -1 to +1

    # Quote spread: base ±7, skewed by position
    bid_offset = 7 + 3 * skew    # widen bid if long (harder to buy more)
    ask_offset = 7 - 3 * skew    # widen ask if short

    # Make sure offsets are at least 1
    bid_offset = max(1.0, bid_offset)
    ask_offset = max(1.0, ask_offset)

    my_bid = math.floor(fair - bid_offset)
    my_ask = math.ceil(fair + ask_offset)

    # Also aggressively take any obvious mispriced orders
    b = best_bid(depth)
    a = best_ask(depth)

    # Take cheap asks
    if a is not None and a[0] < fair - 3:
        qty = clamp_qty(min(a[1], 30), position, LIMIT)
        if qty > 0:
            orders.append(Order("HYDROGEL_PACK", a[0], qty))
            position += qty

    # Take rich bids
    if b is not None and b[0] > fair + 3:
        qty = clamp_qty(-min(b[1], 30), position, LIMIT)
        if qty < 0:
            orders.append(Order("HYDROGEL_PACK", b[0], qty))
            position += qty

    # Passive quotes
    bid_qty = clamp_qty(15, position, LIMIT)
    ask_qty = clamp_qty(-15, position, LIMIT)

    if bid_qty > 0:
        orders.append(Order("HYDROGEL_PACK", my_bid, bid_qty))
    if ask_qty < 0:
        orders.append(Order("HYDROGEL_PACK", my_ask, ask_qty))


def trade_velvetfruit(
    depth: OrderDepth,
    position: int,
    ema: EMA,
    orders: List[Order],
) -> None:
    """
    Market make VELVETFRUIT_EXTRACT around EMA fair value.
    Spread ~5; we quote at fair ± 2 (inside market by 1 on each side).
    """
    LIMIT = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
    mp = mid_price(depth)
    if mp is None:
        return

    fair = ema.update(mp)
    skew = position / LIMIT

    bid_offset = 2.5 + 1.5 * skew
    ask_offset = 2.5 - 1.5 * skew
    bid_offset = max(1.0, bid_offset)
    ask_offset = max(1.0, ask_offset)

    my_bid = math.floor(fair - bid_offset)
    my_ask = math.ceil(fair + ask_offset)

    # Take obvious mispricings
    a = best_ask(depth)
    b = best_bid(depth)
    if a and a[0] < fair - 2:
        qty = clamp_qty(min(a[1], 20), position, LIMIT)
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", a[0], qty))
            position += qty
    if b and b[0] > fair + 2:
        qty = clamp_qty(-min(b[1], 20), position, LIMIT)
        if qty < 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", b[0], qty))
            position += qty

    bid_qty = clamp_qty(20, position, LIMIT)
    ask_qty = clamp_qty(-20, position, LIMIT)
    if bid_qty > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", my_bid, bid_qty))
    if ask_qty < 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", my_ask, ask_qty))


def trade_vouchers(
    timestamp: int,
    depths: Dict[str, OrderDepth],
    positions: Dict[str, int],
    vev_fair: float,
    state: TraderState,
    orders: Dict[str, List[Order]],
) -> None:
    """
    Options trading strategy:
    
    Core alpha:
    - VEV_5400: Consistently underpriced (IV ~22.2% vs consensus 23.2%) → BUY to +300
    - VEV_5500: Consistently overpriced (IV ~23.9% vs consensus 23.2%) → SELL to -300
    - VEV_5300: Often overpriced → SELL to -200
    
    Other vouchers:
    - VEV_5000/5100/5200: Market make around BS fair value with small spread
    - VEV_4000/4500: Deep ITM, track intrinsic (delta ≈ 1)
    - VEV_6000/6500: Floor at 0.5, avoid
    
    TTE decreases through the round: TTE = (5 - ts/1e6) days
    """
    # Compute TTE
    tte_days = TTE_START_DAYS - (timestamp / TICKS_PER_ROUND)
    tte_days = max(0.01, tte_days)  # never negative
    T = tte_days / 365.0
    S = vev_fair

    # ── VEV_5400: CORE BUY ──────────────────────────────────────────────────
    v5400 = "VEV_5400"
    K5400 = 5400
    LIMIT = POSITION_LIMITS[v5400]
    pos = positions.get(v5400, 0)
    depth = depths.get(v5400)
    if depth:
        fair_5400 = bs_call_price(S, K5400, T, SIGMA_CONSENSUS)
        a = best_ask(depth)
        b = best_bid(depth)

        # Aggressively buy if ask < fair (we know it's underpriced)
        if a and a[0] < fair_5400 + 0.5:
            qty = clamp_qty(min(a[1], 50), pos, LIMIT)
            if qty > 0:
                orders[v5400].append(Order(v5400, a[0], qty))
                pos += qty

        # Post aggressive passive bid (just below fair, will get filled)
        # Since VEV_5400 is always underpriced, we can bid AT fair to buy
        target_pos = LIMIT  # want maximum long
        remaining = target_pos - pos
        if remaining > 0:
            # Bid at fair rounded down to be conservative
            bid_px = math.floor(fair_5400) - 1
            # But also check we're not bidding above best ask
            if a:
                bid_px = min(bid_px, a[0] - 1)
            if b:
                bid_px = max(bid_px, b[0])  # don't go below existing bids
            bid_qty = clamp_qty(min(remaining, 30), pos, LIMIT)
            if bid_qty > 0 and bid_px > 0:
                orders[v5400].append(Order(v5400, bid_px, bid_qty))

    # ── VEV_5500: CORE SELL ─────────────────────────────────────────────────
    v5500 = "VEV_5500"
    K5500 = 5500
    LIMIT = POSITION_LIMITS[v5500]
    pos = positions.get(v5500, 0)
    depth = depths.get(v5500)
    if depth:
        fair_5500 = bs_call_price(S, K5500, T, SIGMA_CONSENSUS)
        a = best_ask(depth)
        b = best_bid(depth)

        # Aggressively sell if bid > fair (overpriced)
        if b and b[0] > fair_5500 - 0.5:
            qty = clamp_qty(-min(b[1], 50), pos, LIMIT)
            if qty < 0:
                orders[v5500].append(Order(v5500, b[0], qty))
                pos += qty

        # Post aggressive passive ask
        target_pos = -LIMIT  # want maximum short
        remaining = target_pos - pos  # negative
        if remaining < 0:
            ask_px = math.ceil(fair_5500) + 1
            if b:
                ask_px = max(ask_px, b[0] + 1)
            if a:
                ask_px = min(ask_px, a[0])
            ask_qty = clamp_qty(max(remaining, -30), pos, LIMIT)
            if ask_qty < 0 and ask_px > 0:
                orders[v5500].append(Order(v5500, ask_px, ask_qty))

    # ── VEV_5300: SELL (overpriced day 1/2, neutral day 0) ─────────────────
    v5300 = "VEV_5300"
    K5300 = 5300
    LIMIT = POSITION_LIMITS[v5300]
    TARGET5300 = -200  # moderate short target
    pos = positions.get(v5300, 0)
    depth = depths.get(v5300)
    if depth:
        fair_5300 = bs_call_price(S, K5300, T, SIGMA_CONSENSUS)
        b = best_bid(depth)
        a = best_ask(depth)

        # Sell when bid > fair + 0.8 (clear overpricing signal)
        if b and b[0] > fair_5300 + 0.8:
            qty = clamp_qty(-min(b[1], 30), pos, LIMIT)
            if qty < 0:
                orders[v5300].append(Order(v5300, b[0], qty))
                pos += qty

        remaining = TARGET5300 - pos  # negative
        if remaining < 0:
            ask_px = math.ceil(fair_5300 + 1)
            if b:
                ask_px = max(ask_px, b[0] + 1)
            if a:
                ask_px = min(ask_px, a[0])
            ask_qty = clamp_qty(max(remaining, -20), pos, LIMIT)
            if ask_qty < 0 and ask_px > 0:
                orders[v5300].append(Order(v5300, ask_px, ask_qty))

    # ── VEV_5200: Market make around BS fair ────────────────────────────────
    v5200 = "VEV_5200"
    K5200 = 5200
    LIMIT = POSITION_LIMITS[v5200]
    pos = positions.get(v5200, 0)
    depth = depths.get(v5200)
    if depth:
        fair_5200 = bs_call_price(S, K5200, T, SIGMA_CONSENSUS)
        mp = mid_price(depth)
        if mp is not None:
            skew = pos / LIMIT
            bid_px = math.floor(fair_5200 - 2 - 2 * skew)
            ask_px = math.ceil(fair_5200 + 2 - 2 * skew)
            bid_qty = clamp_qty(15, pos, LIMIT)
            ask_qty = clamp_qty(-15, pos, LIMIT)
            if bid_qty > 0 and bid_px > 0:
                orders[v5200].append(Order(v5200, bid_px, bid_qty))
            if ask_qty < 0 and ask_px > 0:
                orders[v5200].append(Order(v5200, ask_px, ask_qty))

    # ── VEV_5100: Market make (slightly ITM) ────────────────────────────────
    v5100 = "VEV_5100"
    K5100 = 5100
    LIMIT = POSITION_LIMITS[v5100]
    pos = positions.get(v5100, 0)
    depth = depths.get(v5100)
    if depth:
        fair_5100 = bs_call_price(S, K5100, T, SIGMA_CONSENSUS)
        intrinsic = max(0.0, S - K5100)
        # Don't go below intrinsic
        fair_5100 = max(fair_5100, intrinsic)
        mp = mid_price(depth)
        if mp is not None:
            skew = pos / LIMIT
            bid_px = math.floor(fair_5100 - 3 - 2 * skew)
            ask_px = math.ceil(fair_5100 + 3 - 2 * skew)
            bid_qty = clamp_qty(10, pos, LIMIT)
            ask_qty = clamp_qty(-10, pos, LIMIT)
            if bid_qty > 0 and bid_px > 0:
                orders[v5100].append(Order(v5100, bid_px, bid_qty))
            if ask_qty < 0 and ask_px > 0:
                orders[v5100].append(Order(v5100, ask_px, ask_qty))

    # ── VEV_5000: Deep ITM market make ──────────────────────────────────────
    v5000 = "VEV_5000"
    K5000 = 5000
    LIMIT = POSITION_LIMITS[v5000]
    pos = positions.get(v5000, 0)
    depth = depths.get(v5000)
    if depth:
        fair_5000 = bs_call_price(S, K5000, T, SIGMA_CONSENSUS)
        intrinsic = max(0.0, S - K5000)
        fair_5000 = max(fair_5000, intrinsic)
        mp = mid_price(depth)
        if mp is not None:
            skew = pos / LIMIT
            bid_px = math.floor(fair_5000 - 3 - 3 * skew)
            ask_px = math.ceil(fair_5000 + 3 - 3 * skew)
            bid_qty = clamp_qty(10, pos, LIMIT)
            ask_qty = clamp_qty(-10, pos, LIMIT)
            if bid_qty > 0 and bid_px > 0:
                orders[v5000].append(Order(v5000, bid_px, bid_qty))
            if ask_qty < 0 and ask_px > 0:
                orders[v5000].append(Order(v5000, ask_px, ask_qty))

    # ── VEV_4500: Very deep ITM, tracks intrinsic + small time value ─────────
    v4500 = "VEV_4500"
    K4500 = 4500
    LIMIT = POSITION_LIMITS[v4500]
    pos = positions.get(v4500, 0)
    depth = depths.get(v4500)
    if depth:
        intrinsic = max(0.0, S - K4500)
        time_val = bs_call_price(S, K4500, T, SIGMA_CONSENSUS) - intrinsic
        fair_4500 = intrinsic + time_val
        mp = mid_price(depth)
        if mp is not None:
            # Take only if obvious edge
            a = best_ask(depth)
            b_ord = best_bid(depth)
            if a and a[0] < fair_4500 - 2:
                qty = clamp_qty(min(a[1], 10), pos, LIMIT)
                if qty > 0:
                    orders[v4500].append(Order(v4500, a[0], qty))
            if b_ord and b_ord[0] > fair_4500 + 2:
                qty = clamp_qty(-min(b_ord[1], 10), pos, LIMIT)
                if qty < 0:
                    orders[v4500].append(Order(v4500, b_ord[0], qty))

    # ── VEV_4000: Fully intrinsic, minimal trading ───────────────────────────
    v4000 = "VEV_4000"
    K4000 = 4000
    LIMIT = POSITION_LIMITS[v4000]
    pos = positions.get(v4000, 0)
    depth = depths.get(v4000)
    if depth:
        intrinsic = max(0.0, S - K4000)
        fair_4000 = intrinsic  # basically all intrinsic
        a = best_ask(depth)
        b_ord = best_bid(depth)
        if a and a[0] < fair_4000 - 2:
            qty = clamp_qty(min(a[1], 10), pos, LIMIT)
            if qty > 0:
                orders[v4000].append(Order(v4000, a[0], qty))
        if b_ord and b_ord[0] > fair_4000 + 2:
            qty = clamp_qty(-min(b_ord[1], 10), pos, LIMIT)
            if qty < 0:
                orders[v4000].append(Order(v4000, b_ord[0], qty))

    # VEV_6000 / VEV_6500: Floor at 0.5, no edge. Skip.


# ───────────────────────────────────────────────────────────────────────────────
# Main Trader class
# ───────────────────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self):
        # EMA objects initialized lazily from traderData
        self._hydrogel_ema: Optional[EMA] = None
        self._vev_ema: Optional[EMA] = None
        self._state: Optional[TraderState] = None

    def _init_emas(self, state: TraderState) -> None:
        # EMA alpha: α = 2/(N+1). Use N=100 for hydrogel, N=50 for VEV.
        if self._hydrogel_ema is None:
            self._hydrogel_ema = EMA(alpha=2 / 101, initial=state.hydrogel_ema)
        if self._vev_ema is None:
            self._vev_ema = EMA(alpha=2 / 51, initial=state.vev_ema)

    def run(self, trading_state: TradingState):
        # ── Load persisted state ────────────────────────────────────────────
        state = load_state(trading_state.traderData)
        self._init_emas(state)

        timestamp = trading_state.timestamp
        positions = trading_state.position
        depths = trading_state.order_depths

        result: Dict[str, List[Order]] = {sym: [] for sym in depths}

        # ── HYDROGEL_PACK ───────────────────────────────────────────────────
        if "HYDROGEL_PACK" in depths:
            pos = positions.get("HYDROGEL_PACK", 0)
            trade_hydrogel(
                depth=depths["HYDROGEL_PACK"],
                position=pos,
                ema=self._hydrogel_ema,
                orders=result["HYDROGEL_PACK"],
            )

        # ── VELVETFRUIT_EXTRACT ─────────────────────────────────────────────
        vev_fair = self._vev_ema.value or 5250.0
        if "VELVETFRUIT_EXTRACT" in depths:
            pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            mp = mid_price(depths["VELVETFRUIT_EXTRACT"])
            if mp is not None:
                vev_fair = self._vev_ema.update(mp)
            trade_velvetfruit(
                depth=depths["VELVETFRUIT_EXTRACT"],
                position=pos,
                ema=self._vev_ema,
                orders=result["VELVETFRUIT_EXTRACT"],
            )

        # ── OPTIONS (all vouchers) ──────────────────────────────────────────
        trade_vouchers(
            timestamp=timestamp,
            depths=depths,
            positions=positions,
            vev_fair=vev_fair,
            state=state,
            orders=result,
        )

        # ── Persist state ───────────────────────────────────────────────────
        state.hydrogel_ema = self._hydrogel_ema.value
        state.vev_ema = self._vev_ema.value
        trader_data = save_state(state)

        conversions = 0
        return result, conversions, trader_data
