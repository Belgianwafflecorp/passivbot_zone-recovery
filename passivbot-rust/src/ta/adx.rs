pub struct Adx {
    window: usize,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    prev_close: Option<f64>,
    tr_ema: Option<f64>,
    plus_dm_ema: Option<f64>,
    minus_dm_ema: Option<f64>,
    adx_ema: Option<f64>,
    count: usize,
}

impl Adx {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            prev_high: None,
            prev_low: None,
            prev_close: None,
            tr_ema: None,
            plus_dm_ema: None,
            minus_dm_ema: None,
            adx_ema: None,
            count: 0,
        }
    }

    pub fn next(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let (tr, plus_dm, minus_dm) = if let (Some(ph), Some(pl), Some(pc)) = (self.prev_high, self.prev_low, self.prev_close) {
            let tr = (high - low)
                .max((high - pc).abs())
                .max((low - pc).abs());
            let up_move = high - ph;
            let down_move = pl - low;
            let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
            (tr, plus_dm, minus_dm)
        } else {
            (0.0, 0.0, 0.0)
        };

        self.prev_high = Some(high);
        self.prev_low = Some(low);
        self.prev_close = Some(close);

        // Wilder's smoothing
        let alpha = 1.0 / self.window as f64;
        self.tr_ema = Some(match self.tr_ema {
            Some(ema) => ema * (1.0 - alpha) + tr * alpha,
            None => tr,
        });
        self.plus_dm_ema = Some(match self.plus_dm_ema {
            Some(ema) => ema * (1.0 - alpha) + plus_dm * alpha,
            None => plus_dm,
        });
        self.minus_dm_ema = Some(match self.minus_dm_ema {
            Some(ema) => ema * (1.0 - alpha) + minus_dm * alpha,
            None => minus_dm,
        });

        self.count += 1;
        if self.count < self.window {
            return None;
        }

        let tr_ema = self.tr_ema.unwrap();
        let plus_di = if tr_ema != 0.0 { 100.0 * self.plus_dm_ema.unwrap() / tr_ema } else { 0.0 };
        let minus_di = if tr_ema != 0.0 { 100.0 * self.minus_dm_ema.unwrap() / tr_ema } else { 0.0 };
        let dx = if plus_di + minus_di != 0.0 {
            100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        } else {
            0.0
        };

        self.adx_ema = Some(match self.adx_ema {
            Some(ema) => ema * (1.0 - alpha) + dx * alpha,
            None => dx,
        });

        Some(self.adx_ema.unwrap())
    }
}