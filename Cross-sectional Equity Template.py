"""
This algorithm demonstrates the concept of long-short equity. It uses a
combination of factors to construct a ranking of securities in a liquid
tradable universe. It then goes long on the highest-ranked securities and short
on the lowest-ranked securities.
For information on long-short equity strategies, please see the corresponding
lecture on our lectures page:
https://www.quantopian.com/lectures
This algorithm was developed as part of Quantopian's Lecture Series. Please
direct and questions, feedback, or corrections to feedback@quantopian.com
"""

import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals

MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600
MAX_SHORT_POSITION_SIZE = 5.5 / TOTAL_POSITIONS
MAX_LONG_POSITION_SIZE = 2.7 / TOTAL_POSITIONS

def initialize(context):
    """
    A core function called automatically once at the beginning of a backtest.
    Use this function for initializing state or other bookkeeping.
    Parameters
    ----------
    context : AlgorithmContext
        An object that can be used to store state that you want to maintain in 
        your algorithm. context is automatically passed to initialize, 
        before_trading_start, handle_data, and any functions run via schedule_function.
        context provides the portfolio attribute, which can be used to retrieve information 
        about current positions.
    """
    
    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    # Attach the pipeline for the risk model factors that we
    # want to neutralize in the optimization step. The 'risk_factors' string is 
    # used to retrieve the output of the pipeline in before_trading_start below.
    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    # Schedule our rebalance function
    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    # Record our portfolio variables at the end of day
    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


def make_pipeline(): 
    revenue_growth = Fundamentals.revenue_growth.latest
    roa= Fundamentals.roa.latest
    margin= Fundamentals.net_margin.latest
    growth_score=Fundamentals.growth_score.latest
    gross_margin=Fundamentals.gross_margin.latest
    common_stock=Fundamentals.common_stock_dividend_paid.latest
    universe = QTradableStocksUS()
    
    revenue_growth_winsorized= revenue_growth.winsorize(min_percentile=0.1, max_percentile=0.9)
    roa_winsorized= roa.winsorize(min_percentile=0.1, max_percentile=0.9)
    margin_winsorized= margin.winsorize(min_percentile=0.1, max_percentile=0.9)
    growth_score_winsorized= growth_score.winsorize(min_percentile=0.1, max_percentile=0.9)
    gross_margin_winsorized= gross_margin.winsorize(min_percentile=0.1, max_percentile=0.9)
    common_stock_winsorized=common_stock.winsorize(min_percentile=0.1, max_percentile=0.9)

    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
        2.8*gross_margin_winsorized.zscore()+
        0.9*common_stock_winsorized.zscore() +
        1.2*growth_score_winsorized.zscore()+
        1.3*roa_winsorized.zscore()+
        0.2*revenue_growth_winsorized.zscore()+
        3*margin_winsorized.zscore()
    )

     # Build Filters representing the top and bottom baskets of stocks by our
    # combined ranking system. We'll use these as our tradeable universe each
    # day.
    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    # The final output of our pipeline should only include
    # the top/bottom 300 stocks by our criteria
    long_short_screen = (longs | shorts)

    # Create pipeline
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):

    context.pipeline_data = algo.pipeline_output('long_short_equity_template')


    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):

    algo.record(num_positions=len(context.portfolio.positions))


def rebalance(context, data):

    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings


    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)


    constraints = []

    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))


    constraints.append(opt.DollarNeutral())


    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)


    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )
