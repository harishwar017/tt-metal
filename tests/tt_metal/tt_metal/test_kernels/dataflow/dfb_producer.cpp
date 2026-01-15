// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/debug/dprint.h"

void kernel_main() {
    DPRINT << "here" << ENDL();

    experimental::DataflowBuffer<experimental::AccessPattern::STRIDED, experimental::AccessPattern::STRIDED> dfb(0);

    for (uint32_t tile_id = 0; tile_id < 16; tile_id++) {
        dfb.reserve_back(1);
        // noc read -- add 2.0 dev api support
        dfb.push_back(1);
    }
    dfb.finish();
}
