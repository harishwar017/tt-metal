// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/experimental/dataflow_buffer/dataflow_buffer.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::dfb {

uint32_t CreateDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const DataflowBufferConfig& config) {
    auto core_range_set = std::visit(
        ttsl::overloaded{
            [](const CoreCoord& core_spec) { return CoreRangeSet(CoreRange(core_spec, core_spec)); },
            [](const CoreRange& core_spec) { return CoreRangeSet(core_spec); },
            [](const CoreRangeSet& core_spec) { return core_spec; },
        },
        core_spec);

    return program.impl().add_dataflow_buffer(core_range_set, config);
}

namespace detail {

::experimental::PackedTileCounter TileCounterAllocator::allocate(uint8_t tensix_id) {
    // 16 exposed to overlay.
    TT_FATAL(next_tc_id_ < 16, "Out of tile counters for tensix {}", (uint32_t)tensix_id);
    uint8_t tc_id = next_tc_id_++;
    return static_cast<::experimental::PackedTileCounter>((tensix_id << 5) | tc_id);
}

uint8_t calculate_num_tile_counters(const DataflowBufferConfig& config, bool is_producer) {
    if (config.cap == ::experimental::AccessPattern::BLOCKED) {
        return is_producer ? config.num_consumers : 1;
    }
    return (config.num_consumers + config.num_producers - 1) / config.num_producers;
}

::experimental::PackedTileCounter get_shared_tc_for_consumer(
    const DataflowBufferImpl* dfb, uint8_t consumer_idx, uint8_t tc_idx) {
    // In strided mode, consumers share TCs with producers (unless remapper is used and we have diff 1:1 remappings)
    // TODO: this needs to be updated when remapper is added
    uint8_t producer_idx = (consumer_idx * dfb->config.num_producers) / dfb->config.num_consumers;
    return dfb->risc_configs[producer_idx].config.packed_tile_counter[tc_idx];
}

uint32_t DataflowBufferImpl::serialized_size() const {
    // One dfb_initializer_t + one LocalDFBInterface per risc
    return sizeof(::experimental::dfb_initializer_t) + risc_configs.size() * sizeof(::experimental::LocalDFBInterface);
}

std::vector<uint8_t> DataflowBufferImpl::serialize() const {
    std::vector<uint8_t> data;
    data.reserve(serialized_size());

    // Build dfb_initializer_t
    ::experimental::dfb_initializer_t init = {};
    init.logical_id = this->id;
    init.dm_risc_mask = this->dm_risc_mask;
    init.tensix_risc_mask = this->tensix_risc_mask;

    auto* init_bytes = reinterpret_cast<const uint8_t*>(&init);
    data.insert(data.end(), init_bytes, init_bytes + sizeof(init));

    // Write one LocalDFBInterface per risc
    for (const auto& rc : risc_configs) {
        ::experimental::LocalDFBInterface device_config = {};

        // Copy arrays - rd_ptr/wr_ptr start at base_addr
        for (int i = 0; i < 4; i++) {
            device_config.base_addr[i] = rc.config.base_addr[i] >> 4;
            device_config.limit[i] = rc.config.limit[i] >> 4;
            device_config.rd_ptr[i] = rc.config.base_addr[i] >> 4;
            device_config.wr_ptr[i] = rc.config.base_addr[i] >> 4;
            device_config.packed_tile_counter[i] = rc.config.packed_tile_counter[i];
            device_config.txn_ids[i] = rc.config.txn_ids[i];
        }

        // Union handling: producers write capacity, consumers write entry/stride
        if (rc.is_producer) {
            device_config.set_capacity = rc.config.set_capacity;
            device_config.capacity = rc.config.capacity;
        } else {
            device_config.entry_size = rc.config.entry_size >> 4;
            device_config.stride_size = rc.config.stride_size >> 4;
        }

        device_config.num_tiles_per_txn_id = rc.config.num_tiles_per_txn_id;
        device_config.num_tiles_per_txn_id_per_tc = rc.config.num_tiles_per_txn_id_per_tc;
        device_config.remapper_pair_index = rc.config.remapper_pair_index;
        device_config.num_tcs_to_rr = rc.config.num_tcs_to_rr;
        device_config.num_txn_ids = rc.config.num_txn_ids;

        auto* cfg_bytes = reinterpret_cast<const uint8_t*>(&device_config);
        data.insert(data.end(), cfg_bytes, cfg_bytes + sizeof(device_config));
    }

    return data;
}

uint32_t finalize_dfbs(
    uint32_t /*programmable_core_type_index*/,
    std::vector<std::shared_ptr<tt::tt_metal::KernelGroup>>& kernel_groups,
    const std::vector<std::shared_ptr<DataflowBufferImpl>>& dataflow_buffers,
    uint32_t base_offset,
    uint32_t& dfb_offset,
    uint32_t& dfb_size) {
    const auto& hal = MetalContext::instance().hal();

    dfb_offset = base_offset;
    dfb_size = 0;

    for (auto& kg : kernel_groups) {
        auto kernel_config = kg->launch_msg.view().kernel_config();
        kernel_config.local_cb_offset() = base_offset;

        // Calculate total DFB size for this kernel group
        uint32_t kg_dfb_size = 0;
        for (const auto& dfb : dataflow_buffers) {
            // Check if this DFB overlaps with any core in the kernel group
            bool dfb_on_kg = false;
            for (const CoreRange& kg_range : kg->core_ranges.ranges()) {
                if (dfb->core_ranges.intersects(kg_range)) {
                    dfb_on_kg = true;
                    break;
                }
            }
            if (dfb_on_kg) {
                kg_dfb_size += dfb->serialized_size();
            }
        }

        // Track max across all kernel groups
        dfb_size = std::max(dfb_size, kg_dfb_size);
    }

    return tt::align(base_offset + dfb_size, hal.get_alignment(HalMemType::L1));
}

}  // namespace detail

}  // namespace tt::tt_metal::experimental::dfb

// ProgramImpl methods must be defined in the correct namespace
namespace tt::tt_metal::detail {

using namespace tt::tt_metal::experimental::dfb;
using namespace tt::tt_metal::experimental::dfb::detail;

uint32_t ProgramImpl::add_dataflow_buffer(const CoreRangeSet& core_range_set, const DataflowBufferConfig& config) {
    TT_FATAL(this->compiled_.empty(), "Cannot add dataflow buffer to an already compiled program {}", this->id);

    TT_FATAL(this->circular_buffers_.empty(), "Cannot add dataflow buffer to a program with circular buffers");

    TT_FATAL(config.entry_size > 0, "Entry size must be > 0");
    TT_FATAL(config.num_entries > 0, "Num entries must be > 0");
    TT_FATAL(config.num_producers == 1, "DFB only supports one producer for now");
    TT_FATAL(config.num_consumers == 1, "DFB only supports one consumer for now");
    TT_FATAL(config.pap != ::experimental::AccessPattern::BLOCKED, "Blocked producer pattern not supported");

    TT_FATAL(config.cap != ::experimental::AccessPattern::BLOCKED, "Blocked consumer pattern not supported yet");

    auto dfb = std::make_shared<DataflowBufferImpl>();

    // Assign logical ID (0, 1, 2, ...)
    dfb->id = static_cast<uint32_t>(this->dataflow_buffers_.size());
    dfb->core_ranges = core_range_set.merge_ranges();
    dfb->config = config;

    // Initialize masks
    dfb->dm_risc_mask = 0;
    dfb->tensix_risc_mask = 0;  // Keep at 0 for now

    uint32_t capacity;
    switch (config.cap) {
        case ::experimental::AccessPattern::STRIDED:
            TT_FATAL(
                config.num_entries % std::max(config.num_producers, config.num_consumers) == 0,
                "Num entries in DFB {} must be divisible by max of num producers and consumers {}",
                config.num_entries,
                std::max(config.num_producers, config.num_consumers));
            capacity = config.num_entries / std::max(config.num_producers, config.num_consumers);
            break;
        case ::experimental::AccessPattern::BLOCKED:
            TT_FATAL(
                config.num_entries % config.num_producers == 0,
                "Num entries in DFB {} must be divisible by num producers {}",
                config.num_entries,
                config.num_producers);
            capacity = config.num_entries / config.num_producers;
            break;
        default: TT_FATAL(false, "Invalid access pattern", (uint32_t)config.cap);
    }

    uint8_t num_producer_tcs = calculate_num_tile_counters(config, true);
    uint8_t num_consumer_tcs = calculate_num_tile_counters(config, false);

    // Producer 0 uses DM risc 0 (BRISC)
    for (uint8_t p = 0; p < config.num_producers; p++) {
        uint8_t producer_risc_id = p;  // Producer 0 = DM risc 0
        dfb->dm_risc_mask |= (1 << producer_risc_id);

        DFBRiscConfig producer_config;
        producer_config.risc_id = producer_risc_id;
        producer_config.is_producer = true;
        producer_config.is_dm_risc = true;

        // Fill arrays for round-robin TCs
        for (uint8_t tc = 0; tc < num_producer_tcs; tc++) {
            producer_config.config.packed_tile_counter[tc] = tile_counter_allocator_.allocate(producer_risc_id);
        }
        producer_config.config.num_tcs_to_rr = num_producer_tcs;
        producer_config.config.entry_size = config.entry_size;
        producer_config.config.stride_size = config.entry_size * config.num_producers;
        producer_config.config.set_capacity = true;
        producer_config.config.capacity = capacity;

        dfb->risc_configs.push_back(std::move(producer_config));
    }

    // Consumer 0 uses DM risc 1 (NCRISC)
    for (uint8_t c = 0; c < config.num_consumers; c++) {
        uint8_t consumer_risc_id = config.num_producers + c;  // Consumer 0 = DM risc 1
        dfb->dm_risc_mask |= (1 << consumer_risc_id);

        DFBRiscConfig consumer_config;
        consumer_config.risc_id = consumer_risc_id;
        consumer_config.is_producer = false;
        consumer_config.is_dm_risc = true;

        // Fill arrays for round-robin TCs
        for (uint8_t tc = 0; tc < num_consumer_tcs; tc++) {
            if (config.cap == ::experimental::AccessPattern::STRIDED) {
                consumer_config.config.packed_tile_counter[tc] = get_shared_tc_for_consumer(dfb.get(), c, tc);
            } else {
                TT_FATAL(false, "Need to implement blocked consumer access pattern");
            }
        }
        consumer_config.config.num_tcs_to_rr = num_consumer_tcs;
        consumer_config.config.entry_size = config.entry_size;
        consumer_config.config.stride_size = config.entry_size * config.num_consumers;

        dfb->risc_configs.push_back(std::move(consumer_config));
    }

    this->dataflow_buffers_.push_back(dfb);
    this->dataflow_buffer_by_id_.insert({dfb->id, dfb});

    for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord logical_core(x, y);
                per_core_num_dfbs_[logical_core]++;
            }
        }
    }

    return dfb->id;
}

void ProgramImpl::invalidate_dataflow_buffer_allocation() {
    if (this->local_dataflow_buffer_allocation_needed_) {
        return;
    }
    for (CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
        dfb_allocator.reset_available_addresses();
    }
    this->local_dataflow_buffer_allocation_needed_ = true;
}

void ProgramImpl::allocate_dataflow_buffers(const IDevice* device) {
    if (not this->local_dataflow_buffer_allocation_needed_) {
        return;
    }

    uint64_t base_dfb_address = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (auto& dfb : this->dataflow_buffers_) {
        uint64_t computed_addr = base_dfb_address;
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            // Need the max available address across all cores dataflow buffer is placed on
            for (const CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
                if (dfb_allocator.core_range == core_range) {
                    computed_addr = std::max(computed_addr, dfb_allocator.get_cb_region_end());
                    break;
                }
            }
        }
        computed_addr = align(computed_addr, device->allocator()->get_alignment(BufferType::DRAM));
        for (const CoreRange& core_range : dfb->core_ranges.ranges()) {
            for (CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
                if (dfb_allocator.core_range.intersects(core_range)) {
                    if (dfb_allocator.core_range != core_range and computed_addr < dfb_allocator.get_cb_region_end()) {
                        // Intersecting core range has already been marked to have allocation at this address. This
                        // could have been marked by a dataflow buffer on a core range disjoint from current
                        // `core_range` but also intersecting `dfb_allocator.core_range`
                        continue;
                    }
                    dfb_allocator.mark_address(computed_addr, dfb->total_size(), base_dfb_address);
                }
            }
        }
        dfb->allocated_address = computed_addr;

        // Populate base_addr[] and limit[] arrays for each risc config
        uint32_t entry_size = dfb->config.entry_size;
        uint32_t max_prod_cons = std::max(dfb->config.num_producers, dfb->config.num_consumers);

        for (auto& rc : dfb->risc_configs) {
            for (uint8_t tc = 0; tc < rc.config.num_tcs_to_rr; tc++) {
                rc.config.base_addr[tc] = static_cast<uint32_t>(computed_addr) + (tc * entry_size);
                rc.config.limit[tc] =
                    rc.config.base_addr[tc] + ((entry_size * max_prod_cons) * (rc.config.capacity - 1)) + entry_size;
            }
        }
    }
    this->local_dataflow_buffer_allocation_needed_ = false;
}

void ProgramImpl::validate_dataflow_buffer_region(const IDevice* device) {
    std::optional<DeviceAddr> lowest_address =
        device->lowest_occupied_compute_l1_address(this->determine_sub_device_ids(device));
    uint32_t max_l1_size = device->l1_size_per_core();

    for (const CircularBufferAllocator& dfb_allocator : this->dfb_allocators_) {
        if (dfb_allocator.l1_regions.empty()) {
            continue;
        }
        uint64_t dfb_region_end = dfb_allocator.l1_regions.back().second;
        if (dfb_region_end > max_l1_size) {
            TT_THROW(
                "Statically allocated dataflow buffers on core range {} grow to {} B which is beyond max L1 size of {} "
                "B",
                dfb_allocator.core_range.str(),
                dfb_region_end,
                max_l1_size);
        }
        if (lowest_address.has_value() and lowest_address.value() < dfb_region_end) {
            TT_THROW(
                "Statically allocated dataflow buffers in program {} clash with L1 buffers on core range {}. L1 buffer "
                "allocated at {} and static dataflow buffer region ends at {}",
                this->id,
                dfb_allocator.core_range.str(),
                lowest_address.value(),
                dfb_region_end);
        }
    }
}

std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>>
ProgramImpl::dataflow_buffers_on_core(const CoreCoord& core) const {
    std::vector<std::shared_ptr<tt::tt_metal::experimental::dfb::detail::DataflowBufferImpl>> dfbs_on_core;
    for (const auto& dfb : dataflow_buffers_) {
        if (dfb->core_ranges.intersects(core)) {
            dfbs_on_core.push_back(dfb);
        }
    }
    return dfbs_on_core;
}

}  // namespace tt::tt_metal::detail
