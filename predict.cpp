#include <iostream>
#include <vector>
#include <cmath>
#include <optional>
#include <algorithm>
#include <deque>

struct NextDipPredictor {
    double smooth_alpha = 0.2;
    double period_alpha = 0.2;
    double min_separation = 0.25;
    double prominence = 0.05;

    bool has_smooth = false;
    double s_prev = 0.0;
    std::deque<double> last3;
    std::deque<double> recentVals;
    size_t rangeWindow = 200;

    double last_time = 0.0;
    double last_dip_time = -1.0;
    double period_ema = -1.0;

    std::optional<double> update(double t, double power) {
        double s = has_smooth ? (smooth_alpha * power + (1.0 - smooth_alpha) * s_prev) : power;
        has_smooth = true;
        s_prev = s;

        last3.push_back(s);
        if (last3.size() > 3) last3.pop_front();
        if (recentVals.size() >= rangeWindow) recentVals.pop_front();
        recentVals.push_back(s);

        if (last3.size() == 3) {
            double a = last3[0], b = last3[1], c = last3[2];
            bool localMin = (a > b && c > b);
            double rmin = *std::min_element(recentVals.begin(), recentVals.end());
            double rmax = *std::max_element(recentVals.begin(), recentVals.end());
            double rng = std::max(1e-9, rmax - rmin);
            bool prominent = ((std::max(a, c) - b) >= prominence * rng);
            double dt = t - last_time;
            double t_center = t - 0.5 * dt;

            if (localMin && prominent) {
                if (last_dip_time < 0.0 || (t_center - last_dip_time) >= min_separation) {
                    if (last_dip_time >= 0.0) {
                        double p = t_center - last_dip_time;
                        period_ema = (period_ema < 0.0) ? p :
                                     period_alpha * p + (1.0 - period_alpha) * period_ema;
                    }
                    last_dip_time = t_center;
                    std::cout << "Dip detected at t=" << t_center << " s\n";
                }
            }
        }
        last_time = t;

        if (last_dip_time >= 0.0 && period_ema > 0.0)
            return last_dip_time + period_ema;
        return std::nullopt;
    }
};

int main() {
    NextDipPredictor pred;

    // Simulated periodic power: sinusoid with noise
    double dt = 0.05;  // 50 ms sampling
    double T = 2.0;    // period of 2 seconds
    for (double t = 0; t < 20; t += dt) {
        double power = 50 + 20 * std::sin(2 * M_PI * t / T);
        if (auto next = pred.update(t, power)) {
            std::cout << "Predicted next dip around t=" << *next << " s\n";
        }
    }
    return 0;
}
