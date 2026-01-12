// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/dataflow_buffer_interface.h"
#include "debug/assert.h"

// TODO: make this the top level api header but then separate out 1xx and 2xx implementations

#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#endif

#include "experimental/lock.h"

namespace experimental {

template <AccessPattern PAP, AccessPattern CAP>
class DataflowBuffer {
public:
    DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {
        static_assert(PAP != AccessPattern::BLOCKED, "Only SxS and SxB are supported");
#ifdef ARCH_QUASAR
        uint64_t hartid;
        asm volatile("csrr %0, mhartid" : "=r"(hartid));
        hart_id_ = hartid;
#else
        hart_id_ = 0;
#endif
    }

    uint32_t get_entry_size() const { return g_dfb_interface[logical_dfb_id_].entry_size; }

    // Explicit sync APIs
    void reserve_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = g_dfb_interface[logical_dfb_id_].packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        static_assert(false, "Not implemented");
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        while (fast_llk_intf_get_free_space(tensix_id, tc_id) < num_entries);
#endif
    }

    void push_back(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id _];
        PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        static_assert(false, "Not implemented");
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        fast_llk_intf_inc_posted(tensix_id, tc_id, num_entries);
#endif

        local_dfb_interface.wr_ptr[counter_idx_] += (num_pages * local_dfb_interface.stride_size);
        if (local_dfb_interface.wr_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
            local_dfb_interface.wr_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
        }

        counter_index_ = (counter_index_ + 1) % local_dfb_interface.num_tcs_to_rr;
    }

    void wait_front(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        PackedTileCounter packed_tc = g_dfb_interface[logical_dfb_id_].packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        static_assert(false, "Not implemented");
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        while (fast_llk_intf_get_occupancy(tensix_id, tc_id) < num_entries);
#endif
    }

    void pop_front(uint16_t num_entries) {
        ASSERT(num_entries == 1);
        LocalDFBInterface& local_dfb_interface = g_dfb_interface[logical_dfb_id_];
        PackedTileCounter packed_tc = local_dfb_interface.packed_tile_counter[counter_idx_];
        uint8_t tc_id = get_counter_id(packed_tc);
#ifdef COMPILE_FOR_TRISC
        static_assert(false, "Not implemented");
#else
        uint8_t tensix_id = get_tensix_id(packed_tc);
        fast_llk_intf_inc_acked(tensix_id, tc_id, num_entries);
#endif

        local_dfb_interface.rd_ptr[counter_idx_] += (num_pages * local_dfb_interface.stride_size);
        if (local_dfb_interface.rd_ptr[counter_idx_] == local_dfb_interface.limit[counter_idx_]) {
            local_dfb_interface.rd_ptr[counter_idx_] = local_dfb_interface.base_addr[counter_idx_];
        }
        counter_index_ = (counter_index_ + 1) % local_dfb_interface.num_tcs_to_rr;
    }
    // Explicit sync APIs end

    // Implicit sync APIs
    void read_in() {}

    void write_out() {}
    // Implicit sync APIs end

    // from pov of producer need to make sure all the entries get posted (check the raw posted per TC == raw acked per
    // TC)
    // also that there are no interrupts remaining...
    void finish() {}

    uint32_t get_write_ptr() const {
        // return byte address (wr_ptr is 16B address on Gen1XX)
        uint32_t wr_ptr_bytes = g_dfb_interface[logical_dfb_id_].wr_ptr[counter_idx_];
        return wr_ptr_bytes;
    }

    uint32_t get_read_ptr() const {
        // return byte address (rd_ptr is 16B address on Gen1XX)
        uint32_t rd_ptr_bytes = g_dfb_interface[logical_dfb_id_].rd_ptr[counter_idx_];
        return rd_ptr_bytes;
    }

    [[nodiscard]] auto scoped_lock() {
        // TODO: Register with the debugger to track the lock
        return Lock([this]() { release_scoped_lock(); });
    }

private:
    void release_scoped_lock() {
        // TODO: Unregister with the debugger
    }

    uint16_t logical_dfb_id_;
    uint8_t hart_id_ = 0;
    uint8_t counter_idx_ = 0;

    // TODO: update txn id isr handling
    uint8_t txn_id_index_ = 0;
    uint32_t txn_id_loop_cnt_ = 0;  // try to remove this
};

}  // namespace experimental
