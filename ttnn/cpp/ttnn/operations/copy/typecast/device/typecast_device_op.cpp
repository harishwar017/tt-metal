// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_device_op.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::copy {

TypecastDeviceOperation::program_factory_t TypecastDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        log_debug(tt::LogOp, "Using TypecastShardedProgramFactory");
        return program::TypecastShardedProgramFactory{};
    }
    if (args.sub_core_grids.has_value()) {
        log_debug(tt::LogOp, "Using TypecastSubgridProgramFactory");
        return program::TypecastSubgridProgramFactory{};
    }

    const auto& input = tensor_args.input;
    if (input.layout() == Layout::ROW_MAJOR) {
        // FIXME(vtsilytskyi):
        // TypecastRowMajorChunkedProgramFactory uses streaming approach to handle large tensors.
        // Downside - it is slower than naive implementation, which uses unary eltwise kernels.
        // Despite RM support was recently added to unary eltwise kernels and added to TypecastProgramFactory
        // we cannot use it here yet, because it fails on input chunks > 1024 elements.
        // Once fixed, please uncomment heuristic check code below. Rows, that fits L1 memory
        // should be handled in naive way via TypecastProgramFactory.

        // constexpr uint32_t max_l1_budget_bytes = 512 * 1024;  // 512KB budget for typecast CBs
        // constexpr uint32_t num_input_pages = 2;               // Double buffering
        // constexpr uint32_t num_output_pages = 2;              // Double buffering

        // const tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
        // const tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(args.output_dtype);
        // const uint32_t input_element_size = tt::datum_size(input_data_format);
        // const uint32_t output_element_size = tt::datum_size(output_data_format);

        // const auto& padded_shape = input.padded_shape();
        // const uint32_t row_width_elements = padded_shape[padded_shape.rank() - 1];
        // const uint32_t input_row_size = row_width_elements * input_element_size;
        // const uint32_t output_row_size = row_width_elements * output_element_size;
        // const uint32_t total_cb_size = num_input_pages * input_row_size + num_output_pages * output_row_size;

        // // Use chunked factory if double buffering would exceed L1 budget
        // if (total_cb_size > max_l1_budget_bytes) {
        log_debug(tt::LogOp, "Using TypecastRowMajorChunkedProgramFactory");
        return program::TypecastRowMajorChunkedProgramFactory{};
        // }
    }

    log_debug(tt::LogOp, "Using TypecastProgramFactory");
    return program::TypecastProgramFactory{};
}

void TypecastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TypecastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Typecast operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to Typecast need to be allocated in buffers on the device. Buffer is null.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "Typecast operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        static_cast<int>(input_tensor.memory_config().memory_layout()),
        static_cast<int>(out_memory_config.memory_layout()));

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Typecast operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    } else {
        TT_FATAL(
            !args.sub_core_grids.has_value(),
            "Typecast operation has sub_core_grids support for non-sharded inputs only");
    }

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocted output tensor is used, Typecast operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        if (!input_tensor.is_sharded()) {
            TT_FATAL(
                preallocated_output_tensor.value().layout() == input_tensor.layout(),
                "Typecast operation requires input and output layouts to match. Input layout: {}, Output layout: {}",
                static_cast<int>(input_tensor.layout()),
                static_cast<int>(preallocated_output_tensor.value().layout()));
        }
    }
}

spec_return_value_t TypecastDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const Layout output_layout = tensor_args.input.layout();

    const Shape output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, args.output_memory_config));
}

tensor_return_value_t TypecastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t TypecastDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);

    operation::Hash hash;

    // For tile layout, only volume matters. For row-major, actual shape dimensions matter.
    if (input_tensor.layout() == Layout::TILE) {
        hash = operation::hash_operation<TypecastDeviceOperation>(
            args,
            program_factory.index(),
            input_tensor.dtype(),
            input_tensor.memory_config(),
            input_shape.volume(),
            input_tensor.layout());
    } else {
        hash = operation::hash_operation<TypecastDeviceOperation>(
            args,
            program_factory.index(),
            input_tensor.dtype(),
            input_tensor.memory_config(),
            input_shape,
            input_tensor.layout());
    }
    return hash;
}

bool TypecastDeviceOperation::skip_launch(
    const operation_attributes_t& /*attributes*/,
    const tensor_args_t& /*tensor_args*/,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::copy

namespace ttnn::prim {
ttnn::operations::copy::TypecastDeviceOperation::tensor_return_value_t typecast(
    const Tensor& input,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = ttnn::operations::copy::TypecastDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .input_dtype = input.dtype(),
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .bfp8_pack_precise = bfp8_pack_precise,
            .sub_core_grids = sub_core_grids,
        },
        OperationType::tensor_args_t{.input = input, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim
