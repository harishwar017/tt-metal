// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    experimental::DataflowBuffer<experimental::AccessPattern::STRIDED, experimental::AccessPattern::STRIDED> dfb(0);

    for (uint32_t tile_id = 0; tile_id < 16; tile_id++) {
        dfb.wait_front(1);
        dfb.pop_front(1);
    }
}
