// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/dataflow_buffer_interface.h"
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#endif

namespace experimental {

FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint8_t hart_bit = 1 << hartid;

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    volatile dfb_initializer_t* config_ptr = reinterpret_cast<volatile dfb_initializer_t*>(dfb_config_base);

    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        uint8_t risc_mask = config_ptr->dm_risc_mask;

        // Parse config, but only populate entries for this RISC
        // Remapper has to be configured before this because we reset and set tc capacities here
        volatile LocalDFBInterface* local_dfb_ptr = reinterpret_cast<volatile LocalDFBInterface*>(config_ptr + 1);
        if (risc_mask & hart_bit) {
            LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];
            dfb_interface = *local_dfb_ptr;
#ifndef COMPILE_FOR_TRISC
            if (local_dfb_ptr->set_capacity) {
                // Initialize all TCs that this risc round-robins over
                for (uint8_t tc = 0; tc < local_dfb_ptr->num_tcs_to_rr; tc++) {
                    uint8_t tensix_id = get_tensix_id(local_dfb_ptr->packed_tile_counter[tc]);
                    uint8_t tc_id = get_counter_id(local_dfb_ptr->packed_tile_counter[tc]);
                    llk_intf_reset(tensix_id, tc_id);
                    llk_intf_set_capacity(tensix_id, tc_id, local_dfb_ptr->capacity);
                }
            }
#endif
            // Host writes capacity on producer side to intialize the tile counters so the entry & stride sizes need to
            // be explicitly set
            dfb_interface.entry_size = local_dfb_ptr->entry_size;
            dfb_interface.stride_size = local_dfb_ptr->stride_size;
        }

        // jump to the next dfb_initializer_t
        config_ptr = reinterpret_cast<volatile dfb_initializer_t*>(local_dfb_ptr + 1);
    }
}

}  // namespace experimental
