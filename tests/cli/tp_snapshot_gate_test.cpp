#include "src/cli/serve/tp_snapshot_gate.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>

using gufo::server::TpSnapshotGate;
using namespace std::chrono_literals;

int main() {
  // Pending captures hold decode at the boundary until every reservation
  // clears.
  for (bool abandon : {false, true}) {
    TpSnapshotGate gate;
    std::future<void> decoder;
    std::latch entered(1);
    {
      TpSnapshotGate::Pending first(gate);
      TpSnapshotGate::Pending second(gate);
      decoder = std::async(std::launch::async, [&] {
        entered.count_down();
        gate.Wait();
      });
      entered.wait();
      assert(decoder.wait_for(20ms) == std::future_status::timeout);
      first.Acquired();
      first.Acquired();  // Idempotence must not release another reservation.
      assert(decoder.wait_for(20ms) == std::future_status::timeout);
      if (!abandon)
        second.Acquired();
    }
    assert(decoder.wait_for(2s) == std::future_status::ready);
    decoder.get();
  }
  // Mirror the production recursive lock: nested Advance remains legal while a
  // capture waits, then a subsequent decode yields outside that same mutex.
  for (int round = 0; round != 100; ++round) {
    TpSnapshotGate gate;
    std::recursive_mutex calls;
    std::latch registered(1);
    std::vector<int> order;
    std::unique_lock active(calls);
    auto capture = std::async(std::launch::async, [&] {
      TpSnapshotGate::Pending pending(gate);
      registered.count_down();
      std::lock_guard lock(calls);
      pending.Acquired();
      order.push_back(2);
    });
    registered.wait();
    {
      std::lock_guard nested(calls);
      order.push_back(1);
    }
    auto next = std::async(std::launch::async, [&] {
      gate.Wait();
      std::lock_guard lock(calls);
      order.push_back(3);
    });
    active.unlock();
    assert(capture.wait_for(2s) == std::future_status::ready);
    assert(next.wait_for(2s) == std::future_status::ready);
    capture.get();
    next.get();
    assert((order == std::vector<int>{1, 2, 3}));
  }
  // Concurrent reservations and decodes finish without losing registrations.
  TpSnapshotGate gate;
  std::recursive_mutex calls;
  std::atomic<int> snapshots{0}, decodes{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker != 6; ++worker) {
    workers.emplace_back([&, worker] {
      for (int i = 0; i != 2000; ++i) {
        if (worker < 3) {
          TpSnapshotGate::Pending pending(gate);
          std::lock_guard lock(calls);
          pending.Acquired();
          ++snapshots;
        } else {
          gate.Wait();
          std::lock_guard lock(calls);
          ++decodes;
        }
      }
    });
  }
  for (auto& worker : workers)
    worker.join();
  assert(snapshots == 6000 && decodes == 6000);
  gate.Wait();
  std::cout << "PASS: multiple reservations, abandonment, idempotence, "
               "recursive lock handoff, concurrent completion\n";
}
