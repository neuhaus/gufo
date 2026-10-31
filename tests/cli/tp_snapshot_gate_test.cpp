#include "src/cli/serve/tp_snapshot_gate.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <latch>
#include <mutex>
#include <stdexcept>
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
    TpSnapshotGate::Turn active_turn(gate);
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
      TpSnapshotGate::Turn nested_turn(gate);
      std::lock_guard nested(calls);
      order.push_back(1);
    }
    auto next = std::async(std::launch::async, [&] {
      TpSnapshotGate::Turn next_turn(gate);
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
  // An outer turn on another runner must not hide an earlier recursive owner.
  // Pending captures cannot finish while these existing owners hold their
  // call mutexes, so neither nested turn may wait for them.
  {
    TpSnapshotGate first, second;
    TpSnapshotGate::Turn outer(first);
    TpSnapshotGate::Pending pending_first(first);
    TpSnapshotGate::Turn other(second);
    TpSnapshotGate::Pending pending_second(second);
    TpSnapshotGate::Turn nested_first(first);
    TpSnapshotGate::Turn nested_second(second);
  }
  // Exceptions must restore the per-thread chain. A later independent turn
  // on that same thread still waits for a pending capture.
  {
    TpSnapshotGate gate;
    try {
      TpSnapshotGate::Turn turn(gate);
      throw std::runtime_error("unwind turn");
    } catch (const std::runtime_error&) {
    }
    TpSnapshotGate::Pending pending(gate);
    std::atomic<bool> released{false};
    std::latch entered(1);
    auto waiter = std::async(std::launch::async, [&] {
      entered.count_down();
      std::this_thread::sleep_for(20ms);
      released.store(true);
      pending.Acquired();
    });
    entered.wait();
    {
      TpSnapshotGate::Turn next(gate);
    }
    assert(released.load());
    waiter.get();
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
          TpSnapshotGate::Turn turn(gate);
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
               "recursive AR/decode turns, cross-runner nesting, exception "
               "unwind, lock handoff, concurrent completion\n";
}
