from dataclasses import dataclass


def mn(val):
    """format numeric input as money"""

    # We could use locale instead like:
    # >>> import locale
    # >>> locale.setlocale(locale.LC_ALL, '')
    # >>> locale.currency(300000.22, grouping=True)

    # but the format is faster and doesn't require extra "stuff"
    # (locale benchmarks at 11 us per output;
    #  python format + replace benchmarks at 600 ns per output)

    # If we eventually need additional currencies, just use the locale
    # approach instead.
    return f"${val:,.2f}".replace("$-", "-$")


@dataclass
class OptionFees:
    # Note: we may not be doing proper rounding here.
    #       Brokers will round up each sub-calculation to the next cent, but
    #       here we're carrying through the fractional cents and letting
    #       python display whatever final truncated cent value it formats.

    # Conditions (buys/sells) taken from: https://www.webull.com/pricing
    # Fee structure taken from "Fee Schedule" link at:
    # https://tradier.com/individuals/pricing
    # https://www.webull.com/pricing

    rate_sec: float = 0.00002780  # per dollar sold, min 0.01 per leg, SELLS ONLY
    rate_taf: float = 0.00279  # per CONTRACT, min 0.01 per leg, SELLS ONLY
    rate_orf: float = 0.02685  # per contract, all, no maximum
    rate_occ: float = 0.02  # per contract, all, max $55.00 per leg

    # Described as tuples (contractCount, contractPrice)
    # note: 'contractPrice' is the contract price, not the total price
    #       (i.e. 2.30 instead of 230.00)
    legs_buy: list[tuple[int, float]] | None = None
    legs_sell: list[tuple[int, float]] | None = None

    @staticmethod
    def legsTwo(count, buy, sell):
        return OptionFees(legs_buy=[(count, buy)], legs_sell=[(count, sell)])

    @staticmethod
    def legsFour(count, buy1, sell1, buy2, sell2):
        return OptionFees(
            legs_buy=[(count, buy1), (count, buy2)],
            legs_sell=[(count, sell1), (count, sell2)],
        )

    @property
    def buys(self):
        """Returns list for all buy legs: [(contractCounts), (contractPrices)]"""
        return zip(*self.legs_buy)

    @property
    def sells(self):
        """Returns list for all sell legs: [(contractCounts), (contractPrices)]"""
        return zip(*self.legs_sell)

    def priceOf(self, legs):
        """Total dollar amount price for a leg.

        `legs` is either `self.legs_buy` or `self.legs_sell`
        """
        return sum([x[1] * 100 for x in legs])

    @property
    def sellPrice(self):
        """Total monetary inflow for the spread"""
        return self.priceOf(self.legs_sell) * self.asells

    @property
    def buyPrice(self):
        """Total monetary outflow for the spread"""
        return self.priceOf(self.legs_buy) * self.abuys

    @property
    def netPrice(self):
        """Total amount of spread sold or bought"""
        return self.buyPrice - self.sellPrice

    @property
    def abuys(self):
        """Contract count for all buy legs"""
        return sum([x[0] for x in self.legs_buy])

    @property
    def asells(self):
        """Contract count for all sell legs"""
        return sum([x[0] for x in self.legs_sell])

    @property
    def contracts(self):
        """Contract count for all legs"""
        return self.abuys + self.asells

    def mathPerLeg(self, legs, evalUsing):
        return sum(evalUsing(legContracts) for legContracts in legs)

    @property
    def sec(self):
        # Sells based on total price only
        # 0.01 floor PER LEG

        # Calculated against the total dollar value of the sale,
        # so we take the contract price, multiply for 100 shares, then multiply
        # by total contracts in each sell leg.
        # We run this per leg because each leg must charge a minimum 0.01 fee.
        return self.mathPerLeg(
            self.legs_sell,
            lambda legContracts: max(
                0.01, self.rate_sec * legContracts[1] * 100 * legContracts[0]
            ),
        )

    @property
    def taf(self):
        # Sells based on contract count only
        # 0.01 floor PER LEG minimum
        # 5.95 ceiling PER LEG maximum
        return self.mathPerLeg(
            self.legs_sell,
            lambda legContracts: min(5.95, max(0.01, self.rate_taf * legContracts[0])),
        )

    @property
    def orf(self):
        # All contracts
        # aggregate value across all legs, no per-leg math needed
        return self.rate_orf * self.contracts

    @property
    def occ(self):
        # All contracts
        # $55.00 ceiling (1,222 contracts max fee) PER LEG
        # https://www.theocc.com/Company-Information/Schedule-of-Fees
        return self.mathPerLeg(
            self.legs_sell + self.legs_buy,
            lambda legContracts: min(55.00, self.rate_occ * legContracts[0]),
        )

    @property
    def total(self):
        return self.sec + self.taf + self.orf + self.occ

    def __repr__(self):
        fees = f"""BUY {self.abuys}; SELL {self.asells}; TOTAL {self.contracts}
Selling: {mn(self.sellPrice)}
Buying:  {mn(self.buyPrice)}
Net:     {mn(self.netPrice)}
SEC: {mn(self.sec)}
TAF: {mn(self.taf)}
ORF: {mn(self.orf)}
OCC: {mn(self.occ)}
Total estimate: {mn(self.total)}"""

        return fees
