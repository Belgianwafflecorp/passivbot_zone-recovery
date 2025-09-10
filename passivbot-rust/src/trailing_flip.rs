use numpy::{PyReadonlyArray3 ,PyReadonlyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use ndarray::s;

use crate::ta::adx::Adx;
use crate::utils::{calc_pnl_long, calc_pnl_short};

#[derive(Debug, Copy, Clone)]
pub struct Position {
    pub qty: f64,   // >0 long, <0 short, 0 flat
    pub price: f64, // avg entry
}

#[derive(Debug, Copy, Clone)]
pub struct Order {
    pub qty: f64, // >0 buy, <0 sell
    pub price: f64,
}

#[derive(Debug, Copy, Clone)]
pub enum FillType {
    EntryLong,
    EntryShort,
    CloseLong,
    CloseShort,
}

impl FillType {
    fn as_str(&self) -> &'static str {
        match self {
            FillType::EntryLong => "entry_long",
            FillType::EntryShort => "entry_short",
            FillType::CloseLong => "close_long",
            FillType::CloseShort => "close_short",
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Fill {
    pub qty: f64, // positive buys, negative sells
    pub price: f64,
    pub pnl: f64,                  // realised PnL BEFORE fees
    pub pos_size_after_fill: f64,  // position size after this fill
    pub pos_price_after_fill: f64, // position price after this fill
    pub fill_type: FillType,
    pub fee_paid: f64, // abs(qty) * price * fee_rate (>= 0), applied to balance
}

#[derive(Debug, Copy, Clone)]
pub struct TrailingExtrema {
    pub max_since_open: f64,
    pub min_since_max: f64,
    pub min_since_open: f64,
    pub max_since_min: f64,
}

/// Update trailing high/low values used for trailing stop logic.
fn update_trailing_prices(extrema: &mut TrailingExtrema, high: f64, low: f64, close: f64) {
    // Update the low watermark and record the close at which it occurred
    if low < extrema.min_since_open {
        extrema.min_since_open = low;
        extrema.max_since_min = close;
    } else {
        // Otherwise keep track of the highest high since the last low
        extrema.max_since_min = high.max(extrema.max_since_min);
    }
    // Update the high watermark and record the close at which it occurred
    if high > extrema.max_since_open {
        extrema.max_since_open = high;
        extrema.min_since_max = close;
    } else {
        // Otherwise keep track of the lowest low since the last high
        extrema.min_since_max = low.min(extrema.min_since_max);
    }
}

/// Pure position update given a signed fill. No PnL calculation.
fn update_pos(pos: Position, fill: Order) -> Position {
    let pos_qty = pos.qty;
    let pos_price = pos.price;
    let f_qty = fill.qty;
    let f_price = fill.price;

    if f_qty == 0.0 {
        return pos;
    }
    if pos_qty == 0.0 {
        return Position {
            qty: f_qty,
            price: f_price,
        };
    }
    if pos_qty * f_qty > 0.0 {
        let w_pos = pos_qty.abs();
        let w_fill = f_qty.abs();
        let new_qty = pos_qty + f_qty;
        let new_price = (w_pos * pos_price + w_fill * f_price) / (w_pos + w_fill);
        return Position {
            qty: new_qty,
            price: new_price,
        };
    }
    if f_qty.abs() < pos_qty.abs() {
        return Position {
            qty: pos_qty + f_qty,
            price: pos_price,
        };
    } else if (f_qty.abs() - pos_qty.abs()) == 0.0 {
        return Position {
            qty: 0.0,
            price: 0.0,
        };
    } else {
        return Position {
            qty: pos_qty + f_qty,
            price: f_price,
        };
    }
}

/// Processes a fill against a current position, returning a list of detailed fill records.
/// Each record contains realised PnL (pre-fee) and fee_paid (>=0).
fn process_fill(position: Position, open_order: Order, fee_rate: f64) -> Vec<Fill> {
    let pos_qty = position.qty;
    let pos_price = position.price;
    let ord_qty = open_order.qty;
    let ord_price = open_order.price;

    let mut fills: Vec<Fill> = Vec::new();

    if ord_qty == 0.0 {
        return fills;
    }

    // Flat or same direction: single entry (no realised PnL), but fee applies
    if pos_qty == 0.0 || pos_qty * ord_qty > 0.0 {
        let new_pos = update_pos(position, open_order);
        let typ = if ord_qty > 0.0 {
            FillType::EntryLong
        } else {
            FillType::EntryShort
        };
        fills.push(Fill {
            qty: ord_qty,
            price: ord_price,
            pnl: 0.0,
            pos_size_after_fill: new_pos.qty,
            pos_price_after_fill: new_pos.price,
            fill_type: typ,
            fee_paid: ord_qty.abs() * ord_price * fee_rate,
        });
        return fills;
    }

    // Opposite direction: some or all of the order closes the existing position
    let close_qty = pos_qty.abs().min(ord_qty.abs());
    // Signed close fill quantity: negative for a sell closing a long, positive for a buy closing a short
    let close_fill_qty = if pos_qty > 0.0 { -close_qty } else { close_qty };

    let (pnl_close, typ_close) = if pos_qty > 0.0 && ord_qty < 0.0 {
        // Closing a long with a sell
        (close_qty * (ord_price - pos_price), FillType::CloseLong)
    } else {
        // Closing a short with a buy
        (close_qty * (pos_price - ord_price), FillType::CloseShort)
    };

    // Apply the closing part of the order
    let new_pos_after_close = update_pos(
        position,
        Order {
            qty: close_fill_qty,
            price: ord_price,
        },
    );
    fills.push(Fill {
        qty: close_fill_qty,
        price: ord_price,
        pnl: pnl_close,
        pos_size_after_fill: new_pos_after_close.qty,
        pos_price_after_fill: new_pos_after_close.price,
        fill_type: typ_close,
        fee_paid: close_fill_qty.abs() * ord_price * fee_rate,
    });

    // Any remainder of the order opens a new (or adds to flipped) position
    let remainder = ord_qty - close_fill_qty;
    if remainder != 0.0 {
        let new_pos = update_pos(
            new_pos_after_close,
            Order {
                qty: remainder,
                price: ord_price,
            },
        );
        let typ_entry = if remainder > 0.0 {
            FillType::EntryLong
        } else {
            FillType::EntryShort
        };
        fills.push(Fill {
            qty: remainder,
            price: ord_price,
            pnl: 0.0,
            pos_size_after_fill: new_pos.qty,
            pos_price_after_fill: new_pos.price,
            fill_type: typ_entry,
            fee_paid: remainder.abs() * ord_price * fee_rate,
        });
    }
    fills
}

/// Clips an order's quantity so that wallet exposure does not exceed the specified limit.
fn clip_order_qty_to_WE_limit(
    wallet_exposure_limit: f64,
    balance: f64,
    pos: Position,
    order: Order,
) -> Order {
    let pos_if_filled = update_pos(pos, order);
    let we_if_filled = if balance != 0.0 {
        (pos_if_filled.qty.abs() * pos_if_filled.price) / balance
    } else {
        0.0
    };
    if we_if_filled > wallet_exposure_limit {
        let psize_mod = if we_if_filled != 0.0 {
            pos_if_filled.qty * (wallet_exposure_limit / we_if_filled)
        } else {
            0.0
        };
        let diff = pos_if_filled.qty - psize_mod;
        let order_qty_mod = order.qty - diff;
        Order {
            qty: order_qty_mod,
            price: order.price,
        }
    } else {
        order
    }
}

/// Determines the next open order (if any) and unrealised PnL given the current trailing state.
fn calc_open_order_and_upnl_with_flip_count(
    trailing_threshold_pct_profit: f64,
    trailing_retracement_pct_profit: f64,
    trailing_threshold_pct_loss: f64,
    trailing_retracement_pct_loss: f64,
    wallet_exposure_limit: f64,
    double_down_factor: f64,
    initial_qty_pct: f64,
    balance: f64,
    pos: Position,
    high: f64,
    low: f64,
    close: f64,
    extrema: TrailingExtrema,
) -> (Order, f64, bool) {
    let pos_qty = pos.qty;
    let pos_price = pos.price;
    let mut upnl = 0.0;

    if pos_qty > 0.0 {
        // Long position: use the low to compute unrealised PnL
        upnl = calc_pnl_long(pos_price, low, pos_qty, 1.0);
        // Trailing take profit condition
        if close >= pos_price * (1.0 + trailing_threshold_pct_profit) {
            if extrema.min_since_max
                <= extrema.max_since_open * (1.0 - trailing_retracement_pct_profit)
            {
                return (
                    Order {
                        qty: -pos_qty,
                        price: close,
                    },
                    upnl, false
                );
            }
        }
        // Trailing stop loss / add to losing long condition
        if close < pos_price * (1.0 - trailing_threshold_pct_loss) {
            if extrema.max_since_min
                >= extrema.min_since_open * (1.0 + trailing_retracement_pct_loss)
            {
                let close_qty = -pos_qty - pos_qty * double_down_factor;
                let order = Order {
                    qty: close_qty,
                    price: close,
                };
                let clipped =
                    clip_order_qty_to_WE_limit(wallet_exposure_limit, balance, pos, order);
                return (clipped, upnl, true); // flip triggered
            }
        }
    } else if pos_qty < 0.0 {
        // Short position: use the high to compute unrealised PnL
        upnl = calc_pnl_short(pos_price, high, pos_qty, 1.0);
        // Trailing take profit condition for shorts
        if close <= pos_price * (1.0 - trailing_threshold_pct_profit) {
            if extrema.max_since_min
                >= extrema.min_since_open * (1.0 + trailing_retracement_pct_profit)
            {
                return (
                    Order {
                        qty: -pos_qty,
                        price: close,
                    },
                    upnl, false
                );
            }
        }
        // Trailing stop loss / add to losing short condition
        if close >= pos_price * (1.0 + trailing_threshold_pct_loss) {
            if extrema.min_since_max
                <= extrema.max_since_open * (1.0 - trailing_retracement_pct_loss)
            {
                let close_qty = -pos_qty - pos_qty * double_down_factor;
                let order = Order {
                    qty: close_qty,
                    price: close,
                };
                let clipped =
                    clip_order_qty_to_WE_limit(wallet_exposure_limit, balance, pos, order);
                return (clipped, upnl, true); // flip triggered
            }
        }
    } else {
        // Flat: open a new position
        if close != 0.0 {
            let qty = (balance / close) * initial_qty_pct;
            return (Order { qty, price: close }, 0.0, false);
        } else {
            return (
                Order {
                    qty: 0.0,
                    price: 0.0,
                },
                0.0, false
            );
        }
    }
    // No order triggered; return zeros
    (
        Order {
            qty: 0.0,
            price: 0.0,
        },
        upnl, false
    )
}

/// Run a simple backtest using trailing-stop and flip logic.
///
/// Returns `(fills, equities)` where:
///  - `fills` is a vector of `(index, pnl, fee_paid, balance, qty, price, pos_size, pos_price, upnl, typ)`
///    for each fill. `pnl` is realised PnL **before** fees; `balance` reflects fees deducted.
///  - `equities` is the equity curve after each bar.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn backtest_trailing_flip<'py>(
    py: Python<'py>,
    hlcv: PyReadonlyArray2<'py, f64>, // accept ndarray directly
    initial_qty_pct: f64,
    double_down_factor: f64,
    wallet_exposure_limit: f64,
    trailing_threshold_pct_profit: f64,
    trailing_retracement_pct_profit: f64,
    trailing_threshold_pct_loss: f64,
    trailing_retracement_pct_loss: f64,
    fee_rate: f64, // e.g. 0.00055 for 0.055%
    adx_scale_higher_width: f64, // e.g. 1.5
    adx_scale_lower_width: f64, // e.g. 0.7
    max_flips_per_cycle: usize, // max flips per cycle (0 = no limit)
) -> PyResult<(
    Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, String)>,
    Vec<f64>,
)> {
    let _ = py; // silence unused if not needed
    let hlcv = hlcv.as_array(); // 2-D read-only ndarray view
    let mut adx_1 = Adx::new(15);
    let mut adx_2 = Adx::new(30);

    let mut extrema = TrailingExtrema {
        max_since_open: 0.0,
        min_since_max: f64::INFINITY,
        min_since_open: f64::INFINITY,
        max_since_min: 0.0,
    };
    let mut balance = 100.0f64;
    let mut pos = Position {
        qty: 0.0,
        price: 0.0,
    };
    let mut open_order = Order {
        qty: 0.0,
        price: 0.0,
    };
    let mut fills_out: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, String)> = Vec::new();
    let mut equities: Vec<f64> = Vec::new();

    let mut flip_count = 0;

    for (i, row) in hlcv.outer_iter().enumerate() {
        let high = row[0];
        let low = row[1];
        let close = row[2];
        let _vol = row[3];
        let adx_1_val = adx_1.next(high, low, close).unwrap_or(0.0);
        let adx_2_val = adx_2.next(high, low, close).unwrap_or(0.0);

        // Check if the open order would have been filled this bar.
        if (open_order.qty > 0.0 && low < open_order.price)
            || (open_order.qty < 0.0 && high > open_order.price)
        {
            // Fill the order(s) and reset trailing extrema.
            let fills_batch: Vec<Fill> = process_fill(pos, open_order, fee_rate);
            for fill in fills_batch {
                // Apply realised PnL and deduct fee from balance
                balance += fill.pnl - fill.fee_paid;

                pos = Position {
                    qty: fill.pos_size_after_fill,
                    price: fill.pos_price_after_fill,
                };
                fills_out.push((
                    i,
                    fill.pnl,      // realised PnL (pre-fee)
                    fill.fee_paid, // fee paid for this fill
                    balance,       // balance AFTER fee deduction
                    fill.qty,
                    fill.price,
                    fill.pos_size_after_fill,
                    fill.pos_price_after_fill,
                    0.0, // upnl set later
                    fill.fill_type.as_str().to_string(),
                ));
            }
            extrema = TrailingExtrema {
                max_since_open: 0.0,
                min_since_max: f64::INFINITY,
                min_since_open: f64::INFINITY,
                max_since_min: 0.0,
            };
        } else {
            // Update trailing highs/lows when no fill occurs.
            update_trailing_prices(&mut extrema, high, low, close);
        }

        // Determine next open order and unrealised PnL.

        // Scale thresholds dynamically based on ADX value.
        let adx_scale = if adx_1_val > 30.0 && adx_2_val > 25.0 {
            adx_scale_higher_width
        } else if adx_1_val < 20.0 && adx_2_val < 20.0 {
            adx_scale_lower_width
        } else {
            1.0
        };
    
        let trailing_threshold_pct_profit_dyn = trailing_threshold_pct_profit * adx_scale;
        let trailing_threshold_pct_loss_dyn = trailing_threshold_pct_loss * adx_scale;
        let trailing_retracement_pct_profit_dyn = trailing_retracement_pct_profit * adx_scale;
        let trailing_retracement_pct_loss_dyn = trailing_retracement_pct_loss * adx_scale;
    
        let (new_order, upnl, flip_triggered) = calc_open_order_and_upnl_with_flip_count(
            trailing_threshold_pct_profit_dyn,
            trailing_retracement_pct_profit_dyn,
            trailing_threshold_pct_loss_dyn,
            trailing_retracement_pct_loss_dyn,
            wallet_exposure_limit,
            double_down_factor,
            initial_qty_pct,
            balance,
            pos,
            high,
            low,
            close,
            extrema,
        );

        if flip_triggered && max_flips_per_cycle > 0 {
            flip_count += 1;
        }
        
        if max_flips_per_cycle > 0 && flip_count >= max_flips_per_cycle {
            // Close position and reset flip count
            open_order = Order { qty: -pos.qty, price: close };
            flip_count = 0;
        } else {
            open_order = new_order;
        }

        equities.push(balance + upnl);

        // If we recorded any fills at this bar, update their upnl to the current value.
        if let Some(last) = fills_out.last_mut() {
            if last.0 == i {
                last.8 = upnl; // index 8 is upnl in the output tuple
            }
        }
    }

    Ok((fills_out, equities))
}

/// Core logic for a single cycle on one coin, starting at any bar.
/// Returns (fills, equities, bars_processed)
fn backtest_trailing_flip_single_cycle(
    hlcv: &ndarray::ArrayView2<f64>,
    start_balance: f64,
    initial_qty_pct: f64,
    double_down_factor: f64,
    wallet_exposure_limit: f64,
    trailing_threshold_pct_profit: f64,
    trailing_retracement_pct_profit: f64,
    trailing_threshold_pct_loss: f64,
    trailing_retracement_pct_loss: f64,
    fee_rate: f64,
    adx_scale_higher_width: f64,
    adx_scale_lower_width: f64,
    max_flips_per_cycle: usize,
) -> (Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, String)>, Vec<f64>, usize) {
    
    let mut adx_1 = Adx::new(15);
    let mut adx_2 = Adx::new(30);

    let mut extrema = TrailingExtrema {
        max_since_open: 0.0,
        min_since_max: f64::INFINITY,
        min_since_open: f64::INFINITY,
        max_since_min: 0.0,
    };
    let mut balance = start_balance;
    let mut pos = Position { qty: 0.0, price: 0.0 };
    let mut open_order = Order { qty: 0.0, price: 0.0 };
    let mut fills_out: Vec<(usize, f64, f64, f64, f64, f64, f64, f64, f64, String)> = Vec::new();
    let mut equities: Vec<f64> = Vec::new();

    let mut flip_count = 0;

    let n_bars = hlcv.shape()[0];

    for i in 0..n_bars {
        let row = hlcv.row(i);
        let high = row[0];
        let low = row[1];
        let close = row[2];
        let _vol = row[3];
        let adx_1_val = adx_1.next(high, low, close).unwrap_or(0.0);
        let adx_2_val = adx_2.next(high, low, close).unwrap_or(0.0);

        // Check if the open order would have been filled this bar.
        if (open_order.qty > 0.0 && low < open_order.price)
            || (open_order.qty < 0.0 && high > open_order.price)
        {
            // Fill the order(s) and reset trailing extrema.
            let fills_batch: Vec<Fill> = process_fill(pos, open_order, fee_rate);
            for fill in fills_batch {
                // Apply realised PnL and deduct fee from balance
                balance += fill.pnl - fill.fee_paid;

                pos = Position {
                    qty: fill.pos_size_after_fill,
                    price: fill.pos_price_after_fill,
                };
                fills_out.push((
                    i,
                    fill.pnl,      // realised PnL (pre-fee)
                    fill.fee_paid, // fee paid for this fill
                    balance,       // balance AFTER fee deduction
                    fill.qty,
                    fill.price,
                    fill.pos_size_after_fill,
                    fill.pos_price_after_fill,
                    0.0, // upnl set later
                    fill.fill_type.as_str().to_string(),
                ));
            }
            extrema = TrailingExtrema {
                max_since_open: 0.0,
                min_since_max: f64::INFINITY,
                min_since_open: f64::INFINITY,
                max_since_min: 0.0,
            };
        } else {
            // Update trailing highs/lows when no fill occurs.
            update_trailing_prices(&mut extrema, high, low, close);
        }

        // Scale thresholds dynamically based on ADX value.
        let adx_scale = if adx_1_val > 30.0 && adx_2_val > 25.0 {
            adx_scale_higher_width
        } else if adx_1_val < 20.0 && adx_2_val < 20.0 {
            adx_scale_lower_width
        } else {
            1.0
        };

        let trailing_threshold_pct_profit_dyn = trailing_threshold_pct_profit * adx_scale;
        let trailing_threshold_pct_loss_dyn = trailing_threshold_pct_loss * adx_scale;
        let trailing_retracement_pct_profit_dyn = trailing_retracement_pct_profit * adx_scale;
        let trailing_retracement_pct_loss_dyn = trailing_retracement_pct_loss * adx_scale;

        let (new_order, upnl, flip_triggered) = calc_open_order_and_upnl_with_flip_count(
            trailing_threshold_pct_profit_dyn,
            trailing_retracement_pct_profit_dyn,
            trailing_threshold_pct_loss_dyn,
            trailing_retracement_pct_loss_dyn,
            wallet_exposure_limit,
            double_down_factor,
            initial_qty_pct,
            balance,
            pos,
            high,
            low,
            close,
            extrema,
        );

        if flip_triggered && max_flips_per_cycle > 0 {
            flip_count += 1;
        }

        // End cycle if max flips reached (force close)
        if max_flips_per_cycle > 0 && flip_count >= max_flips_per_cycle {
            // Close position and reset flip count
            open_order = Order { qty: -pos.qty, price: close };
            flip_count = 0;
        } else {
            open_order = new_order;
        }

        equities.push(balance + upnl);

        // If we recorded any fills at this bar, update their upnl to the current value.
        if let Some(last) = fills_out.last_mut() {
            if last.0 == i {
                last.8 = upnl; // index 8 is upnl in the output tuple
            }
        }

        // --- CYCLE END LOGIC ---
        // End cycle if position is flat (closed in profit or after max flips)
        if pos.qty == 0.0 && !fills_out.is_empty() {
            // End of cycle: return everything up to this bar
            return (fills_out, equities, i);
        }
    }

    // If we reach the end of the slice without closing, return what we have
    (fills_out, equities, n_bars.saturating_sub(1))
}

/// Multi-coin cycling backtest: only one coin is traded at a time, cycles through coins.
/// Each cycle starts at the next bar after the previous cycle ends.
/// Returns (fills, equities, coin_indices) for all cycles.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn backtest_trailing_flip_multi<'py>(
    py: Python<'py>,
    hlcvs: PyReadonlyArray3<'py, f64>,    // coins × bars × 4
    initial_qty_pct: f64,
    double_down_factor: f64,
    wallet_exposure_limit: f64,
    trailing_threshold_pct_profit: f64,
    trailing_retracement_pct_profit: f64,
    trailing_threshold_pct_loss: f64,
    trailing_retracement_pct_loss: f64,
    fee_rate: f64,
    adx_scale_higher_width: f64,
    adx_scale_lower_width: f64,
    max_flips_per_cycle: usize,
) -> PyResult<(
    Vec<(usize, usize, f64, f64, f64, f64, f64, f64, f64, f64, String)>, // (bar, coin, pnl, fee, bal, qty, price, psize, pprice, upnl, typ)
    Vec<f64>, // equity curve
    Vec<usize>, // coin index per cycle
)> {
    let _ = py;
    let hlcvs = hlcvs.as_array();

    let n_coins = hlcvs.shape()[0];
    let n_bars = hlcvs.shape()[1];

    let mut balance = 100.0f64;
    let mut bar = 0usize;
    let mut coin_idx = 0usize;
    let mut fills_all = Vec::new();
    let mut equities = Vec::new();
    let mut coin_indices = Vec::new();

    while bar < n_bars {
        // Select coin
        let hlcv = hlcvs.index_axis(ndarray::Axis(0), coin_idx);

        // Run a single-cycle backtest starting at `bar`
        // You will need to refactor `backtest_trailing_flip` to allow:
        // - starting from a given bar
        // - passing in the current balance
        // - returning the ending bar of the cycle

        let (fills, eqs, cycle_end_bar) = backtest_trailing_flip_single_cycle(
            &hlcv.slice(s![bar.., ..]),
            balance,
            initial_qty_pct,
            double_down_factor,
            wallet_exposure_limit,
            trailing_threshold_pct_profit,
            trailing_retracement_pct_profit,
            trailing_threshold_pct_loss,
            trailing_retracement_pct_loss,
            fee_rate,
            adx_scale_higher_width,
            adx_scale_lower_width,
            max_flips_per_cycle,
        );

        // Adjust indices to global bar index
        for (i, fill) in fills.into_iter().enumerate() {
            let global_bar = bar + i;
            fills_all.push((
                global_bar,      // bar
                coin_idx,        // coin_idx
                fill.1,          // pnl
                fill.2,          // fee_paid
                fill.3,          // balance
                fill.4,          // qty
                fill.5,          // price
                fill.6,          // psize
                fill.7,          // pprice
                fill.8,          // upnl
                fill.9.clone(),  // type (String)
            ));
        }
        for eq in eqs {
            equities.push(eq);
        }
        coin_indices.push(coin_idx);

        // Update balance for next cycle
        if let Some(last_eq) = equities.last() {
            balance = *last_eq;
        }

        // Advance to next bar and coin
        bar = bar + cycle_end_bar + 1;
        coin_idx = (coin_idx + 1) % n_coins;
    }

    Ok((fills_all, equities, coin_indices))
}