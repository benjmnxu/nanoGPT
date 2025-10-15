#include <optional>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>

struct DipStartHighRate {
    // ---- Tunables (pick sane defaults, then tune on your traces) ----
    double ema_alpha_power   = 0.15;   // 0..1  (EMA for power smoothing; small lag)
    double ema_alpha_dabs    = 0.05;   // EMA for |d/dt| to estimate derivative scale
    double min_separation_s  = 0.20;   // seconds: debounce between dip starts
    double period_alpha      = 0.15;   // EMA for period (start->start)
    double thresh_k          = 3.0;    // dip start needs slope < -k * E[|d/dt|]
    double slope_eps         = 0.0;    // extra fixed deadband near 0 slope (W/s)

    // ---- State ----
    bool   has_p        = false;
    double p_smooth     = 0.0;       // smoothed power
    double t_prev_s     = 0.0;       // previous timestamp [s] when we updated

    bool   has_d        = false;
    double d_prev       = 0.0;       // previous slope (W/s)

    bool   has_dscale   = false;
    double d_abs_ema    = 0.0;       // E[|d/dt|] for adaptive threshold

    double last_start_s = -1.0;      // last detected dip START time [s]
    double period_ema   = -1.0;      // learned period [s]

    static inline double ns_to_s(int64_t t_ns) {
        return (double)t_ns * 1e-9;
    }

    // Feed one sample (monotonic timestamp in nanoseconds, power in watts).
    // Returns predicted next dip START time [s] if available.
    std::optional<double> update(int64_t t_ns, double power_w) {
        double t = ns_to_s(t_ns);

        // 1) minimal-lag smoothing (EMA)
        if (!has_p) {
            p_smooth = power_w;
            has_p = true;
            t_prev_s = t;
            return std::nullopt;
        }
        double p_prev = p_smooth;
        p_smooth = ema_alpha_power * power_w + (1.0 - ema_alpha_power) * p_smooth;

        // 2) slope estimate (current sample, no look-ahead)
        double dt = std::max(1e-12, t - t_prev_s);   // protect against jitter
        double d  = (p_smooth - p_prev) / dt;        // W/s

        // 3) update noise/scale model of derivative
        if (!has_dscale) {
            d_abs_ema = std::fabs(d);
            has_dscale = true;
        } else {
            d_abs_ema = ema_alpha_dabs * std::fabs(d) + (1.0 - ema_alpha_dabs) * d_abs_ema;
        }
        double neg_required = std::max(slope_eps, thresh_k * d_abs_ema); // adaptive

        // 4) detect dip START when slope crosses from >=0 to "sufficiently negative"
        if (has_d && d_prev >= 0.0 && d < -neg_required) {
            // Interpolate the zero-slope crossing between (t_prev_s, t)
            // d_prev at t_prev_s, d at t → linear interpolation for t_cross
            double t_cross = t_prev_s + (0.0 - d_prev) / (d - d_prev) * dt;

            // Debounce (avoid multiple triggers on the same descent)
            if (last_start_s < 0.0 || (t_cross - last_start_s) >= min_separation_s) {
                // Learn period from consecutive starts
                if (last_start_s >= 0.0) {
                    double p = t_cross - last_start_s;
                    period_ema = (period_ema < 0.0) ? p
                               : period_alpha * p + (1.0 - period_alpha) * period_ema;
                }
                last_start_s = t_cross;
                // Optional: hook for logging/markers
                // std::cout << "[DipStart] t=" << t_cross << " s\n";
            }
        }

        // 5) advance state
        d_prev   = d;
        has_d    = true;
        t_prev_s = t;

        // 6) predict next dip START if we have a learned period
        if (last_start_s >= 0.0 && period_ema > 0.0)
            return last_start_s + period_ema;
        return std::nullopt;
    }
};

int main() {
    DipStartHighRate det;

    // Example: 100 kHz sampler emits (t_ns, power_w)
    const double fs = 100000.0;
    const double dt = 1.0 / fs;
    int64_t t_ns = 0;

    // Synthetic: 2 s period, dip starts at 1.6 s
    auto synth = [](double t)->double {
        double T = 2.0, start = 1.6, len = 0.15;
        double phase = fmod(t, T);
        double base = 150.0;
        if (phase >= start && phase < start + len) return base - 60.0; // dip segment
        return base;
    };

    for (int n = 0; n < int(5 * fs); ++n) {
        double t = n * dt;
        double p = synth(t);
        auto next = det.update(t_ns, p);
        if (next) {
            std::cout << "Predicted NEXT dip START at t ≈ " << *next << " s\n";
        }
        t_ns += (int64_t)std::llround(dt * 1e9);
    }
}
