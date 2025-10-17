"""
CTC Derivatives Game — Enhanced Market-Making Strategy (Bachelier + Risk Controls)

Key Upgrades vs. Template:
1) Proper option pricing under Normal (Bachelier) model using mean/variance of REMAINING dice.
2) Futures fair = expected settlement value (using realized sum + remaining EV).
3) Dynamic spreads scale with uncertainty (sigma_U) and tighten as subrounds progress.
4) Inventory-aware skew: futures skew by net position; options skew by delta exposure.
5) Strike selection: quote only strikes near current mean to maximize matches and stay within limits.

Assumptions:
- product.id formats:
    Futures: "S,F,N"                   (settles on sum of first N subrounds * 2000 dice)
    Call:    "S,C,STRIKE,EXPIRY"       (European, expires at EXPIRY subround)
    Put:     "S,P,STRIKE,EXPIRY"
- Quantity convention: each trade has quantity +1 (buy) / -1 (sell); position.position is net qty.

Author: you ✅
"""

from autograder.sdk.strategy_interface import AbstractTradingStrategy
from typing import Any, Dict, Tuple, List, Optional
from statistics import mean, pvariance
import math

class MyTradingStrategy(AbstractTradingStrategy):
    def __init__(self):
        # --- Quoting controls ---
        self.base_spread_tick = 0.1          # Minimum tick to keep markets well-formed
        self.k_fut = 0.002                   # Futures spread scale vs sigma_U (tune)
        self.k_opt = 0.004                   # Options spread scale vs (sigma_U * phi(d)) (tune)

        # Inventory skew (positive numbers push you to unwind exposure)
        self.inv_alpha_f = 0.1               # Price skew per futures unit of inventory
        self.inv_alpha_o = 0.1               # Price skew per delta-equivalent option inventory

        # Strike filter: quote only strikes within +/- N_sig of current mean
        self.option_strike_sigmas = 1.2      # quote strikes within +/- 1.2 * sigma_U around mean
        self.max_option_quotes = 6           # cap number of option strikes quoted per subround

        # Game params (will be set on start)
        self.dice_sides = 6
        self.team_name = "Unknown"

    # ---------------- Lifecycle ----------------
    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.dice_sides = config.get("dice_sides", self.dice_sides)
        self.team_name  = config.get("team_name", self.team_name)
        print(f"[{self.team_name}] Start — dice_sides={self.dice_sides}")

    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("pnl", 0.0)
        print(f"[{self.team_name}] Round end — PnL: {pnl:.2f}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        print(f"[{self.team_name}] Game end — Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}")

    # ---------------- Core loop ----------------
    def make_market(
        self, *, marketplace: Any, training_rolls: List[float], my_trades: Any,
        current_rolls: List[float], round_info: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:

        quotes: Dict[str, Tuple[float, float]] = {}

        # 1) Estimate per-roll mean/variance from training + current
        mu_roll, var_roll = self._mean_var_per_roll(training_rolls, current_rolls)

        # Precompute realized (known) sum
        known_sum = sum(current_rolls)

        # 2) Decide which products to quote (all futures + selected option strikes near mean)
        products = list(marketplace.get_products())

        # Categorize
        futures, calls, puts = [], [], []
        for p in products:
            kind = self._kind(p.id)
            if kind == "F": futures.append(p)
            elif kind == "C": calls.append(p)
