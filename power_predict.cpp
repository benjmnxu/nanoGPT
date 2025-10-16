// Implementation for periodic GPU power modeling & prediction,
// plus a testable main().
//
// Build:  g++ -std=c++17 -O2 power_predict.cpp -o power_predict
// Demo:   ./power_predict
// Live:   tail -f power_values.txt | ./power_predict --stdin
//
// How the period is identified:
// We compute normalized autocorrelation over a rolling window for lags in a plausible range,
// pick the best peak (with a tiny parabolic refinement), then learn a template for one period,
// and predict by repeating it (with per-period scale/offset adaptation).

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>
#include <cstddef>
#include <optional>
#include <iostream>
#include <string>
#include <sstream>

namespace power {

// ---------- RingBuffer ----------
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t cap=0): buf_(cap), cap_(cap) {}
    void resize(size_t cap) { buf_.assign(cap, T{}); cap_=cap; head_=0; size_=0; }
    void push(T v){
        if(cap_==0) return;
        buf_[head_] = v;
        head_ = (head_+1)%cap_;
        if(size_<cap_) ++size_;
    }
    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }
    std::vector<T> last(size_t n) const {
        n = std::min(n,size_);
        std::vector<T> out; out.reserve(n);
        size_t start = (head_ + cap_ - n)%cap_;
        for(size_t i=0;i<n;++i){
            out.push_back(buf_[(start+i)%cap_]);
        }
        return out;
    }
    std::vector<T> to_vector() const { return last(size_); }
private:
    std::vector<T> buf_;
    size_t cap_{0}, head_{0}, size_{0};
};

// ---------- Math utils ----------
inline double mean(const std::vector<double>& x){
    if(x.empty()) return 0.0;
    double s = std::accumulate(x.begin(), x.end(), 0.0);
    return s / x.size();
}
inline void detrend(std::vector<double>& x){
    double m = mean(x);
    for(auto& v: x) v -= m;
}
inline void apply_hann(std::vector<double>& x){
    size_t n=x.size(); if(n<2) return;
    for(size_t i=0;i<n;++i){
        double w = 0.5 - 0.5*std::cos(2.0*M_PI*static_cast<double>(i)/(n-1));
        x[i] *= w;
    }
}
inline std::vector<double> circshift(const std::vector<double>& x, int k){
    size_t n = x.size(); if(n==0) return {};
    std::vector<double> y(n);
    int kk = ((k % static_cast<int>(n)) + static_cast<int>(n)) % static_cast<int>(n);
    for(size_t i=0;i<n;++i) y[(i+kk)%n] = x[i];
    return y;
}

// ---------- PeriodEstimator ----------
class PeriodEstimator {
public:
    PeriodEstimator(double window_sec, double sample_interval_sec,
                    int min_period_samples, int max_period_samples,
                    int reestimate_stride_samples = 10)
    : minP_(min_period_samples),
      maxP_(max_period_samples),
      stride_(std::max(1, reestimate_stride_samples))
    {
        size_t cap = static_cast<size_t>(std::round(window_sec / sample_interval_sec));
        history_.resize(std::max<size_t>(cap, static_cast<size_t>(maxP_*3)));
    }

    std::optional<int> update(double sample){
        history_.push(sample);
        ++since_last_;
        if(history_.size() < static_cast<size_t>(std::max(64, maxP_*2))) return std::nullopt;
        if(since_last_ < stride_) return std::nullopt;
        since_last_ = 0;
        int p = estimate_period_autocorr();
        if(p>0){ last_period_ = p; return p; }
        return std::nullopt;
    }

    int period() const { return last_period_; }

private:
    int estimate_period_autocorr(){
        std::vector<double> x = history_.to_vector();
        detrend(x);
        apply_hann(x);

        int N = static_cast<int>(x.size());
        int minLag = std::max(1, std::min(minP_, N/4));
        int maxLag = std::min(std::max(minLag+1, maxP_), N/2);
        if(maxLag <= minLag) return -1;

        auto norm_corr = [&](int lag){
            double s=0.0, e1=0.0, e2=0.0;
            for(int i=0; i<N-lag; ++i){
                double a=x[i], b=x[i+lag];
                s += a*b; e1 += a*a; e2 += b*b;
            }
            return s / (std::sqrt(e1*e2) + 1e-12);
        };

        double bestScore = -1.0;
        int bestLag = -1;
        for(int lag=minLag; lag<=maxLag; ++lag){
            double r = norm_corr(lag);
            if(r > bestScore){ bestScore = r; bestLag = lag; }
        }

        // Sub-sample refinement via parabolic interpolation around the peak
        if(bestLag>minLag && bestLag<maxLag){
            double y1 = norm_corr(bestLag-1);
            double y2 = bestScore;
            double y3 = norm_corr(bestLag+1);
            double denom = (y1 - 2*y2 + y3);
            if(std::fabs(denom) > 1e-9){
                double delta = 0.5*(y1 - y3)/denom; // shift in [-1,1]
                refined_period_frac_ = bestLag + delta; // available if you need it
            } else {
                refined_period_frac_ = bestLag;
            }
        } else {
            refined_period_frac_ = bestLag;
        }
        return bestLag;
    }

    RingBuffer<double> history_;
    int minP_, maxP_, stride_;
    int since_last_{0};
    int last_period_{-1};
    double refined_period_frac_{-1.0};
};

// ---------- PatternModel ----------
class PatternModel {
public:
    explicit PatternModel(int period=0, int max_recent_periods=6, double ema_alpha=0.3)
    : P_(period), max_recent_(max_recent_periods), alpha_(ema_alpha) {}

    void reset(int newP){
        P_ = newP;
        template_.assign(P_, 0.0);
        filled_ = 0;
        recent_.clear();
    }

    int period() const { return P_; }
    bool ready() const { return P_>0 && filled_>=2; }

    void add_period(const std::vector<double>& raw){
        if(P_<=0 || raw.empty()) return;

        // resample to exactly P_ samples if needed (linear)
        std::vector<double> x = (raw.size()==static_cast<size_t>(P_)) ? raw : resample_linear(raw, P_);

        // phase alignment to current template (skip on first fill)
        if(filled_>0){
            int shift = best_circular_shift(x, template_);
            x = circshift(x, shift);
        }

        // EMA update of template
        if(filled_==0) template_ = x;
        else {
            for(int i=0;i<P_;++i){
                template_[i] = alpha_*x[i] + (1.0-alpha_)*template_[i];
            }
        }
        ++filled_;

        // store recent
        recent_.push_back(x);
        if(static_cast<int>(recent_.size()) > max_recent_) recent_.pop_front();
    }

    std::pair<double,double> fit_scale_offset() const {
        if(recent_.empty() || template_.empty()) return {1.0, 0.0};
        const auto& y = recent_.back();
        const auto& x = template_;
        double mx = mean(x), my = mean(y);
        double num=0.0, den=0.0;
        for(size_t i=0;i<x.size();++i){ num += (x[i]-mx)*(y[i]-my); den += (x[i]-mx)*(x[i]-mx); }
        double scale = (den>1e-12) ? (num/den) : 1.0;
        double offset = my - scale*mx;
        return {scale, offset};
    }

    std::vector<double> predict(size_t H, int phase_idx=0, bool adjust_amplitude=true) const {
        std::vector<double> out; out.reserve(H);
        double s=1.0, b=0.0;
        if(adjust_amplitude){
            auto so = fit_scale_offset(); s=so.first; b=so.second;
        }
        for(size_t k=0;k<H;++k){
            int idx = (phase_idx + static_cast<int>(k)) % P_;
            out.push_back(s*template_[idx] + b);
        }
        return out;
    }

    const std::vector<double>& pattern() const { return template_; }

private:
    static int best_circular_shift(const std::vector<double>& a, const std::vector<double>& b){
        int n = static_cast<int>(std::min(a.size(), b.size()));
        if(n==0) return 0;
        std::vector<double> aa = a, bb = b;
        detrend(aa); detrend(bb);
        double best=-1e300; int bestk=0;
        for(int k=0;k<n;++k){
            double s=0.0;
            for(int i=0;i<n;++i) s += aa[i]*bb[(i+k)%n];
            if(s>best){ best=s; bestk=k; }
        }
        return bestk;
    }

    static std::vector<double> resample_linear(const std::vector<double>& x, int M){
        std::vector<double> y(M,0.0);
        if(x.empty() || M<=0) return y;
        double N = static_cast<double>(x.size());
        for(int i=0;i<M;++i){
            double t = (i*(N-1))/(M-1.0);
            int t0 = static_cast<int>(std::floor(t));
            int t1 = std::min(static_cast<int>(N)-1, t0+1);
            double w = t - t0;
            y[i] = (1.0-w)*x[t0] + w*x[t1];
        }
        return y;
    }

    int P_{0}, max_recent_{6};
    double alpha_{0.3};
    int filled_{0};
    std::vector<double> template_;
    std::deque<std::vector<double>> recent_;
};

// ---------- PowerAnalyzer (public API) ----------
class PowerAnalyzer {
public:
    struct Config {
        double sample_interval_sec = 0.1; // 100ms
        double estimator_window_sec = 30.0;
        int    min_period_samples = 5;    // 0.5s
        int    max_period_samples = 600;  // 60s
        int    reestimate_stride_samples = 5;
        int    max_recent_periods = 6;
        double ema_alpha = 0.3;
    };

    explicit PowerAnalyzer(const Config& cfg)
    : cfg_(cfg),
      estimator_(cfg.estimator_window_sec, cfg.sample_interval_sec,
                 cfg.min_period_samples, cfg.max_period_samples,
                 cfg.reestimate_stride_samples),
      model_(/*period=*/0, cfg.max_recent_periods, cfg.ema_alpha)
    {}

    PowerAnalyzer() : PowerAnalyzer(Config{}) {}

    void push(double value){
        raw_history_.push_back(value);

        if(auto maybeP = estimator_.update(value)){ on_period_change(*maybeP); }

        if(P_>0){
            current_period_.push_back(value);
            if(static_cast<int>(current_period_.size()) == P_){
                model_.add_period(current_period_);
                current_period_.clear();
            }
        }
    }

    std::vector<double> predict(size_t H) const {
        if(P_<=0 || !model_.ready()){
            // fallback: hold-last
            std::vector<double> out(H, raw_history_.empty()?0.0:raw_history_.back());
            return out;
        }
        int phase = static_cast<int>(current_period_.size()) % P_;
        return model_.predict(H, phase, /*adjust_amplitude=*/true);
    }

    int period_samples() const { return P_; }
    double period_seconds() const { return (P_>0) ? P_*cfg_.sample_interval_sec : -1.0; }
    const std::vector<double>& pattern() const { return model_.pattern(); }

private:
    void on_period_change(int newP){
        if(newP<=0) return;
        if(newP != P_){
            P_ = newP;
            model_.reset(P_);
            current_period_.clear();
            // Backfill a few recent full periods (if available)
            const int max_backfill = 3;
            int n = static_cast<int>(raw_history_.size());
            int k = std::min(max_backfill, n / P_);
            for(int i=k; i>=1; --i){
                std::vector<double> seg(raw_history_.end()-i*P_, raw_history_.end()-(i-1)*P_);
                model_.add_period(seg);
            }
        }
    }

    Config cfg_;
    PeriodEstimator estimator_;
    int P_{-1};
    std::vector<double> current_period_;
    std::vector<double> raw_history_;
    PatternModel model_;
};

} // namespace power

// ------------------- main(): two test modes -------------------
#include <random>
#include <chrono>
#include <thread>

int main(int argc, char** argv){
    using namespace power;

    // Config tuned for ~100ms cadence
    PowerAnalyzer::Config cfg;
    cfg.sample_interval_sec = 0.1;
    cfg.estimator_window_sec = 30.0;
    cfg.min_period_samples = 8;   // 0.8s min
    cfg.max_period_samples = 120; // 12s max
    cfg.reestimate_stride_samples = 5;
    cfg.max_recent_periods = 6;
    cfg.ema_alpha = 0.3;

    PowerAnalyzer analyzer(cfg);

    auto print_status = [&](int t){
        if(analyzer.period_samples()>0){
            auto pred = analyzer.predict(10);
            std::cout<<"t="<<t
                     << "  P="<<analyzer.period_samples()
                     << " ("<<analyzer.period_seconds()<<" s)"
                     << "  next_pred="<<(pred.empty()?0.0:pred.front())
                     << "\n";
        }
    };

    // Mode A: read real-time power values (one per line) from stdin
    if(argc>1 && std::string(argv[1])=="--stdin"){
        std::cerr<<"Reading power values from stdin (one value per line). "
                 <<"Printing status every ~1s of samples.\n";
        std::string line;
        int t=0;
        while(std::getline(std::cin, line)){
            if(line.empty()) continue;
            std::stringstream ss(line);
            double v;
            if(!(ss>>v)) continue;
            analyzer.push(v);
            if(++t % 10 == 0) print_status(t);
        }
        if(analyzer.period_samples()>0){
            auto pred = analyzer.predict(20);
            std::cout<<"Final estimated period: "<<analyzer.period_samples()
                     <<" samples ("<<analyzer.period_seconds()<<" s)\n";
            std::cout<<"First few predictions:";
            for(size_t i=0;i<std::min<size_t>(pred.size(),10);++i) std::cout<<" "<<pred[i];
            std::cout<<"\n";
        }
        return 0;
    }

    // Mode B (default): synthetic test demonstrating periodic detection & prediction
    std::cout<<"Synthetic demo (default). Use --stdin to feed real values.\n";
    // Create a synthetic periodic power trace:
    // baseline 120W, sinusoid +/-20W, small spike every 10 samples, Gaussian noise.
    int trueP = 32;                    // true period in samples -> 3.2s at 100ms
    std::vector<double> pattern(trueP);
    for(int i=0;i<trueP;++i){
        pattern[i] = 120.0 + 20.0*std::sin(2*M_PI*i/trueP) + 6.0*(i%10==0);
    }
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 2.5);

    // Stream samples; periodically print estimator state and a short prediction
    for(int t=0; t<1800; ++t){
        // Optional: vary amplitude/offset slowly to test adaptation
        double amp = 1.0 + 0.10*std::sin(2*M_PI*t/800.0);
        double offset = 0.5*std::sin(2*M_PI*t/1000.0);
        double val = amp*pattern[t%trueP] + offset + noise(rng);

        analyzer.push(val);

        if(t%50==0) print_status(t);

        // If you want to simulate wall-clock ingestion at 100ms, uncomment:
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout<<"Estimated period: "<<analyzer.period_samples()
             <<" samples ("<<analyzer.period_seconds()<<" s)\n";
    auto pred = analyzer.predict(16);
    std::cout<<"Next 16 predicted values:";
    for(double v: pred) std::cout<<" "<<v;
    std::cout<<"\n";

    // Show a tiny excerpt of the learned template
    const auto& tmpl = analyzer.pattern();
    if(!tmpl.empty()){
        std::cout<<"Template (first 12 samples):";
        for(size_t i=0;i<std::min<size_t>(tmpl.size(),12);++i) std::cout<<" "<<tmpl[i];
        std::cout<<"\n";
    }

    // Basic sanity check vs. true period (not required, but informative)
    if(analyzer.period_samples()>0){
        int err = std::abs(analyzer.period_samples() - trueP);
        std::cout<<"|Estimated P - True P| = "<<err<<" samples\n";
    }

    return 0;
}
