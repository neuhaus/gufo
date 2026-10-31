#ifndef GUFO_SERVER_TP_SNAPSHOT_GATE_HPP_
#define GUFO_SERVER_TP_SNAPSHOT_GATE_HPP_

#include <atomic>
#include <cstdint>
#include <utility>

namespace gufo::server {

// Decode waits outside the existing recursive call mutex, allowing an already
// waiting snapshot to acquire it before another decode or advance call.
// This does not preempt an active call or impose ordering on prefill.
class TpSnapshotGate {
public:
  // A fallback DecodeStep may hold the call mutex while calling Advance.
  // Only the outer turn waits: waiting again from that nested Advance would
  // prevent the pending capture from ever acquiring the same mutex.
  class Turn {
  public:
    explicit Turn(const TpSnapshotGate& gate) noexcept
        : gate_(&gate), previous_(current_) {
      bool nested = false;
      for (auto* turn = current_; turn; turn = turn->previous_) {
        if (turn->gate_ == gate_) {
          nested = true;
          break;
        }
      }
      if (!nested)
        gate.Wait();
      current_ = this;
    }
    Turn(const Turn&) = delete;
    Turn& operator=(const Turn&) = delete;
    ~Turn() { current_ = previous_; }

  private:
    const TpSnapshotGate* gate_;
    Turn* previous_;
    inline static thread_local Turn* current_{nullptr};
  };

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
