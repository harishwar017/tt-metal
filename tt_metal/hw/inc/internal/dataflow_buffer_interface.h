// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace experimental {

enum AccessPattern : uint8_t {  // this should be put into experimental/hostdev or should it be a host file???
    STRIDED,
    BLOCKED,
    UNKNOWN,
};

using PackedTileCounter = uint8_t;  // top 2 bits identify tensix id, bottom 5 bits for counter id

inline __attribute__((always_inline)) constexpr uint8_t get_tensix_id(const PackedTileCounter& p) {
    return (p >> 5) & 0x03;
}

inline __attribute__((always_inline)) constexpr uint8_t get_counter_id(const PackedTileCounter& p) { return p & 0x1F; }

// move configs and LocalDFBInterface structs to hw/inc/hostdev
/*
    in memory:
        | dfb_initializer_t | logical dfb 0
        | LocalDFBInterface | (total of num_riscs max of 8 + 4)
        | dfb_initializer_t | logical dfb 1
        | LocalDFBInterface | (total of num_riscs max of 8 + 4)
        ...
*/
struct dfb_initializer_t {
    uint32_t logical_id;
    uint8_t dm_risc_mask;
    uint8_t tensix_risc_mask;
    uint8_t padding[2];
} __attribute__((packed));

// on WH/BH arrays will be sized to 1
struct LocalDFBInterface {
    uint32_t rd_ptr[4];
    uint32_t wr_ptr[4];
    uint32_t base_addr[4];
    uint32_t limit[4];

    union {
        struct {
            uint32_t entry_size;
            uint32_t stride_size;
        };
        struct {
            uint32_t set_capacity;  // host writes capacity to initialize
            uint32_t capacity;
        };
    };

    PackedTileCounter packed_tile_counter[4];
    uint8_t txn_ids[4];
    uint8_t num_tiles_per_txn_id;
    uint8_t num_tiles_per_txn_id_per_tc;
    uint8_t remapper_pair_index;
    uint8_t num_tcs_to_rr;
    uint8_t num_txn_ids;

    uint8_t padding[3];

    // #ifndef ARCH_QUASAR
    //     // used by packer for in-order packing ... is this still needed on Quasar?
    //     uint32_t wr_tile_ptr;

    //     // Save a cycle during init by writing 0 to the uint32 below
    //     union {
    //         uint32_t tiles_acked_received_init;
    //         struct {
    //             uint16_t tiles_acked;
    //             uint16_t tiles_received;
    //         };
    //     };
    // #endif
} __attribute__((packed));

}  // namespace experimental
