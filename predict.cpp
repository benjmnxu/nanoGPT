// predict.cpp
#include <cstdint>
#include <cmath>
#include <vector>
#include <optional>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <random>

struct PeriodRemainderForecast {
    // ---------- Tunables ----------
    int    TEMPLATE_BINS     = 1024;   // phase bins per period template
    double template_alpha    = 0.3;    // EMA when updating the template
    double ema_alpha_power   = 0.15;   // smoothing for power
    double ema_alpha_dabs    = 0.05;   // smoothing for |d/dt|
    double thresh_k          = 3.0;    // start requires slope < -k * E|d/dt|
    double slope_eps         = 0.0;    // extra fixed deadband on slope [W/s]
    double min_sep_s         = 0.20;   // debounce time between dip starts [s]
    double period_alpha      = 0.15;   // EMA for period (start→start)
    bool   amplitude_scale   = true;   // per-cycle amplitude/offset normalization

    // ---------- State ----------
    bool   has_s = false;
    double s_prev = 0.0;
    double t_prev = 0.0;

    bool   has_d = false;
    double d_prev = 0.0;
    bool   has_dscale = false;
    double d_abs_ema = 0.0;

    double last_start = -1.0;     // last dip START time [s]
    double period_ema = -1.0;     // learned period [s]

    // Current-cycle raw samples (smoothed) to carve a full period
    std::vector<double> cyc_t;    // absolute times [s]
    std::vector<double> cyc_p;    // smoothed power [W]

    // Learned template over normalized phase [0..1)
    bool   has_template = false;
    std::vector<double> tpl;      // size = TEMPLATE_BINS
    // Optional amplitude/offset adaptation (helps when plateaus drift)
    double cycle_mean_prev = 0.0;
    double cycle_amp_prev  = 1.0;

    static inline double ns_to_s(int64_t t_ns) { return (double)t_ns * 1e-9; }

    // Linear interpolation helper on (x[i], y[i]) monotonic in x
    static double lerp_series(const std::vector<double>& xs,
                              const std::vector<double>& ys,
                              double x) {
        if (xs.empty()) return 0.0;
        if (x <= xs.front()) return ys.front();
        if (x >= xs.back())  return ys.back();
        auto it = std::upper_bound(xs.begin(), xs.end(), x);
        size_t j = std::max<size_t>(1, (size_t)(it - xs.begin())) - 1;
        double x0 = xs[j], x1 = xs[j+1];
        double y0 = ys[j], y1 = ys[j+1];
        double u  = (x - x0) / std::max(1e-12, x1 - x0);
        return y0 + u * (y1 - y0);
    }

    // Resample one cycle (absolute times t0..t1, powers) to TEMPLATE_BINS over phase 0..1
    void resample_cycle_to_template(double t0, double t1,
                                    const std::vector<double>& ts,
                                    const std::vector<double>& ps,
                                    std::vector<double>& out_tpl) {
        out_tpl.assign(TEMPLATE_BINS, 0.0);
        double P = std::max(1e-12, t1 - t0);
        for (int k = 0; k < TEMPLATE_BINS; ++k) {
            double phi = (k + 0.5) / TEMPLATE_BINS;       // bin center
            double t   = t0 + phi * P;
            out_tpl[k] = lerp_series(ts, ps, t);
        }
    }

    // EMA update of template bins (optionally normalize by mean & amplitude)
    void merge_template(const std::vector<double>& new_tpl) {
        if (!has_template) {
            tpl = new_tpl;
            has_template = true;
            // initialize ref stats
            cycle_mean_prev = std::accumulate(tpl.begin(), tpl.end(), 0.0) / tpl.size();
            double mn = *std::min_element(tpl.begin(), tpl.end());
            double mx = *std::max_element(tpl.begin(), tpl.end());
            cycle_amp_prev = std::max(1e-9, mx - mn);
            return;
        }
        // Optional per-cycle normalization to handle amplitude/offset drift
        double new_mean = std::accumulate(new_tpl.begin(), new_tpl.end(), 0.0) / new_tpl.size();
        double new_min = *std::min_element(new_tpl.begin(), new_tpl.end());
        double new_max = *std::max_element(new_tpl.begin(), new_tpl.end());
        double new_amp = std::max(1e-9, new_max - new_min);

        for (int k = 0; k < TEMPLATE_BINS; ++k) {
            double x = new_tpl[k];
            if (amplitude_scale) {
                // Normalize new_tpl to (mean,amp) of existing template before merging
                double x_n = (x - new_mean) / new_amp;                    // zero-mean, unit-amp
                double x_s = x_n * cycle_amp_prev + cycle_mean_prev;      // scale to old stats
                tpl[k] = template_alpha * x_s + (1.0 - template_alpha) * tpl[k];
            } else {
                tpl[k] = template_alpha * x + (1.0 - template_alpha) * tpl[k];
            }
        }
        // Update stored stats slowly (optional)
        cycle_mean_prev = 0.9 * cycle_mean_prev + 0.1 * new_mean;
        cycle_amp_prev  = 0.9 * cycle_amp_prev  + 0.1 * new_amp;
    }

    // Sample template at a given phase in [0,1)
    double sample_template(double phase) const {
        if (!has_template) return 0.0;
        double u = std::fmod(std::max(0.0, phase), 1.0) * TEMPLATE_BINS;
        int    i = (int)std::floor(u) % TEMPLATE_BINS;
        int    j = (i + 1) % TEMPLATE_BINS;
        double w = u - std::floor(u);
        return tpl[i] * (1.0 - w) + tpl[j] * w;
    }

    // Main streaming update.
    // Input: timestamp (ns), power (W).
    // Output: if possible, returns predicted (time_s, power_W) for the remainder of the current period.
    std::optional<std::vector<std::pair<double,double>>> update(int64_t t_ns, double power_w) {
        const double t = ns_to_s(t_ns);

        // 1) EMA smoothing of power
        double s = has_s ? (ema_alpha_power * power_w + (1.0 - ema_alpha_power) * s_prev) : power_w;

        // 2) derivative and its scale
        if (!has_s) {
            has_s = true; s_prev = s; t_prev = t;
            cyc_t.clear(); cyc_p.clear();
            cyc_t.push_back(t); cyc_p.push_back(s);
            return std::nullopt;
        }
        double dt = std::max(1e-12, t - t_prev);
        double d  = (s - s_prev) / dt;

        if (!has_dscale) { d_abs_ema = std::fabs(d); has_dscale = true; }
        else             { d_abs_ema = ema_alpha_dabs * std::fabs(d) + (1.0 - ema_alpha_dabs) * d_abs_ema; }
        double neg_req = std::max(slope_eps, thresh_k * d_abs_ema);

        // 3) Detect dip START (slope ≥ 0 to < -neg_req), interpolate crossing time
        bool dip_start_now = false;
        double t_start = 0.0;
        if (has_d && d_prev >= 0.0 && d < -neg_req) {
            t_start = t_prev + (0.0 - d_prev) / (d - d_prev) * dt; // sub-sample time
            if (last_start < 0.0 || (t_start - last_start) >= min_sep_s) {
                dip_start_now = true;
            }
        }

        // 4) Append current sample to current-cycle buffer
        cyc_t.push_back(t);
        cyc_p.push_back(s);

        // 5) If a new start fired and we already had a previous start => close previous cycle
        if (dip_start_now) {
            if (last_start >= 0.0) {
                // Close previous cycle [last_start, t_start] using samples in cyc_{t,p}
                auto it0 = std::lower_bound(cyc_t.begin(), cyc_t.end(), last_start);
                auto it1 = std::lower_bound(cyc_t.begin(), cyc_t.end(), t_start);
                if (it0 == cyc_t.end()) it0 = cyc_t.end()-1;
                if (it1 == cyc_t.end()) it1 = cyc_t.end()-1;
                size_t idx0 = std::max<size_t>(0, (size_t)(it0 - cyc_t.begin()));
                size_t idx1 = std::max<size_t>(idx0+1, (size_t)(it1 - cyc_t.begin()));

                std::vector<double> seg_t(cyc_t.begin()+idx0, cyc_t.begin()+idx1+1);
                std::vector<double> seg_p(cyc_p.begin()+idx0, cyc_p.begin()+idx1+1);

                // Make sure the exact boundaries are present (insert if needed)
                if (seg_t.front() > last_start) {
                    seg_t.insert(seg_t.begin(), last_start);
                    seg_p.insert(seg_p.begin(), lerp_series(cyc_t, cyc_p, last_start));
                }
                if (seg_t.back() < t_start) {
                    seg_t.push_back(t_start);
                    seg_p.push_back(lerp_series(cyc_t, cyc_p, t_start));
                }

                // Resample this cycle to phase template and merge
                std::vector<double> new_tpl;
                resample_cycle_to_template(last_start, t_start, seg_t, seg_p, new_tpl);
                merge_template(new_tpl);

                // Learn period
                double P = t_start - last_start;
                period_ema = (period_ema < 0.0) ? P : period_alpha * P + (1.0 - period_alpha) * period_ema;
            }

            // IMPORTANT FIX: compute boundary value BEFORE clearing buffers
            double boundary_power = lerp_series(cyc_t, cyc_p, t_start);

            // Start a new cycle with the exact boundary point
            last_start = t_start;
            cyc_t.clear(); cyc_p.clear();
            cyc_t.push_back(t_start);
            cyc_p.push_back(boundary_power);
        }

        // Keep last sample in the new buffer as well
        if (cyc_t.empty() || cyc_t.back() < t) { cyc_t.push_back(t); cyc_p.push_back(s); }

        // 6) Advance state
        d_prev = d; has_d = true; s_prev = s; t_prev = t;

        // 7) If we have a template & period, return forecast for remainder of current period
        if (has_template && last_start >= 0.0 && period_ema > 0.0) {
            double P   = period_ema;
            double t0  = last_start;
            double t1  = t0 + P;
            if (t < t1 - 1e-12) {
                // Use the last observed dt as output step
                double dt_out = dt; // you can clamp to a max if needed
                int    N      = (int)std::ceil((t1 - t) / dt_out);
                std::vector<std::pair<double,double>> out;
                out.reserve(N);

                // Optional amplitude/offset adjust based on current partial cycle
                double adj_mean = 0.0, adj_amp = 1.0;
                if (amplitude_scale && cyc_p.size() >= 4) {
                    double mn = *std::min_element(cyc_p.begin(), cyc_p.end());
                    double mx = *std::max_element(cyc_p.begin(), cyc_p.end());
                    adj_mean = std::accumulate(cyc_p.begin(), cyc_p.end(), 0.0) / (double)cyc_p.size();
                    adj_amp  = std::max(1e-9, mx - mn);
                } else {
                    adj_mean = cycle_mean_prev;
                    adj_amp  = cycle_amp_prev;
                }

                for (int k = 0; k < N; ++k) {
                    double tk   = t + k * dt_out;
                    double phi  = (tk - t0) / P; // phase in [0,1)
                    double y    = sample_template(phi);
                    if (amplitude_scale) {
                        // Normalize template to zero-mean, unit-amp based on stored stats,
                        // then scale to current cycle's partial stats (adj_*).
                        double y_n = (y - cycle_mean_prev) / cycle_amp_prev;
                        y = y_n * adj_amp + adj_mean;
                    }
                    out.emplace_back(tk, y);
                }
                return out; // prediction for the remainder of this period
            }
        }

        return std::nullopt; // not enough info yet or period already completed
    }
};

// ------------------------------
// Test harness
// ------------------------------

// Synthetic GPU-like power: plateau with a sharp periodic dip + small ripple + noise.
// Period T, dip starts at dip_start_frac*T, lasts dip_len_frac*T, depth = dip_depth.
double synthetic_power(double t,
                       double T = 2.0,
                       double dip_start_frac = 0.8,
                       double dip_len_frac   = 0.10,
                       double base = 150.0,
                       double dip_depth = 60.0,
                       double ripple_amp = 3.0)
{
    double phase = std::fmod(t, T);
    double dip_start = dip_start_frac * T;
    double dip_end   = dip_start + dip_len_frac * T;

    // base + mild workload ripple
    double p = base + ripple_amp * std::sin(2 * M_PI * phase / T * 6.0);

    // dip as a flat valley (you can change to triangular/exponential if you prefer)
    if (phase >= dip_start && phase < dip_end) {
        p -= dip_depth;
    }
    return p;
}

int main() {
    PeriodRemainderForecast det;
    det.TEMPLATE_BINS   = 512;   // try 512..4096
    det.ema_alpha_power = 0.12;  // smoothing (increase if your data is noisy)
    det.ema_alpha_dabs  = 0.05;  // derivative scale EMA
    det.thresh_k        = 3.0;   // edge sharpness requirement
    det.min_sep_s       = 0.20;  // debounce (must be < true period)
    det.amplitude_scale = true;  // adjust template to slow drift in amplitude/offset

    // Sampling: 100 kHz (adjust to your setup)
    const double fs = 100000.0;
    const double dt = 1.0 / fs;
    const double sim_seconds = 8.0;

    // Optional white noise to make it realistic
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.5); // ~0.5W sigma

    std::cout << std::fixed << std::setprecision(6);

    int64_t t_ns = 0;
    double last_reported_start = -1.0;

    for (int n = 0; n < (int)(sim_seconds * fs); ++n) {
        double t = n * dt;

        double p = synthetic_power(t);
        p += noise(rng); // add small noise

        auto forecast = det.update(t_ns, p);

        // Print when a new dip START was detected
        if (det.last_start >= 0.0 && det.last_start != last_reported_start) {
            std::cout << "[Dip START] t = " << det.last_start << " s";
            if (det.period_ema > 0.0)
                std::cout << "   (period ≈ " << det.period_ema << " s)";
            std::cout << "\n";
            last_reported_start = det.last_start;
        }

        // Print a short preview when we first have a forecast in a cycle
        static double last_forecast_cycle_t0 = -1.0;
        if (forecast && det.last_start != last_forecast_cycle_t0) {
            std::cout << "[Forecast issued at t=" << t
                      << " s] " << forecast->size()
                      << " points until next dip. Showing first 8:\n";
            for (size_t i = 0; i < std::min<size_t>(8, forecast->size()); ++i) {
                std::cout << "  " << (*forecast)[i].first
                          << " s -> " << (*forecast)[i].second << " W\n";
            }
            std::cout << "  ...\n";
            last_forecast_cycle_t0 = det.last_start;
        }

        t_ns += (int64_t)std::llround(dt * 1e9);
    }

    std::cout << "Done.\n";
    return 0;
}
