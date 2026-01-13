// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "device_fixture.hpp"
#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"
#include "tt_metal/experimental/dataflow_buffer/dataflow_buffer.hpp"

namespace tt::tt_metal {

enum class DataDirection {
    READ_IN = 0,   // producer will read data into the DFB via noc
    WRITE_OUT = 1  // producer sends resident data to consumer and consumer writes data out via noc
};

bool run_dfb_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const experimental::dfb::DataflowBufferConfig& dfb_config,
    const DataDirection& data_direction) {
    Program program = CreateProgram();

    uint32_t buffer_size = dfb_config.entry_size * dfb_config.num_entries;
    distributed::DeviceLocalBufferConfig local_buffer_config{.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};
    auto buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    log_info(tt::LogTest, "Buffer: [address: {} B, size: {} B]", buffer->address(), buffer->size());

    CoreCoord logical_core = CoreCoord(0, 0);
    /*auto logical_dfb_id = */ experimental::dfb::CreateDataflowBuffer(program, logical_core, dfb_config);

    /*auto producer_kernel = */ CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer.cpp",
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

    /*auto consumer_kernel = */ CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer.cpp",
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1});

    // Execute using slow dispatch (DFBs not yet supported in MeshWorkload path)
    IDevice* device = mesh_device->get_devices()[0];
    detail::LaunchProgram(device, program, true /*wait_until_cores_done*/);

    return true;  // compare input to output to make sure dfb works
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx1SReadIn) {
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    EXPECT_TRUE(run_dfb_program(this->devices_.at(0), config, DataDirection::READ_IN));
}

TEST_F(MeshDeviceFixture, TensixTest1xDFB1Sx1SWriteOut) {
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    EXPECT_TRUE(run_dfb_program(this->devices_.at(0), config, DataDirection::WRITE_OUT));
}

}  // end namespace tt::tt_metal
