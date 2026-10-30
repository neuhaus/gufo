#ifndef GUFO_SERVER_TP_SNAPSHOT_GATE_HPP_
#define GUFO_SERVER_TP_SNAPSHOT_GATE_HPP_

#include <atomic>
#include <cstdint>
#include <utility>

namespace gufo::server {

// Decode waits outside the existing recursive call mutex, allowing an already
// waiting snapshot to acquire it before another multi-token decode cycle.
// This does not preempt an active call or impose ordering on prefill.
class TpSnapshotGate {
public:
  class Pending {
  public:
    explicit Pending(TpSnapshotGate& gate) noexcept : gate_(&gate) {
      gate_->pending_.fetch_add(1, std::memory_order_acq_rel);
    }
    Pending(const Pending&) = delete;
    Pending& operator=(const Pending&) = delete;
    ~Pending() { Acquired(); }

    void Acquired() noexcept {
      if (auto* gate = std::exchange(gate_, nullptr)) {
        gate->pending_.fetch_sub(1, std::memory_order_release);
        gate->pending_.notify_all();
      }
    }

  private:
    TpSnapshotGate* gate_;
  };

  void Wait() const noexcept {
    auto pending = pending_.load(std::memory_order_acquire);
    while (pending != 0) {
      pending_.wait(pending, std::memory_order_acquire);
      pending = pending_.load(std::memory_order_acquire);
    }
  }

private:
  std::atomic<std::uint32_t> pending_{0};
};

}  // namespace gufo::server
#endif
