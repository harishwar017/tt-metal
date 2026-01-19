// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/logical_topology_translator.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt_stl/assert.hpp>
#include <cstdio>

namespace tt::tt_fabric {

LogicalTopologyTranslator::LogicalTopologyTranslator(const MeshShape& mesh_shape, FabricConfig fabric_config) :
    mesh_shape_(mesh_shape),
    fabric_config_(fabric_config),
    is_1d_fabric_(tt::tt_fabric::is_1d_fabric_config(fabric_config)),
    is_ring_(fabric_config == FabricConfig::FABRIC_1D_RING),
    num_rows_(mesh_shape[0]),
    num_cols_(mesh_shape[1]),
    num_chips_(static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_)) {
    // Build the logical line order (zigzag for 1D, row-major for 2D)
    logical_line_order_.reserve(num_chips_);

    if (is_1d_fabric_) {
        // Zigzag pattern: row 0 L->R, row 1 R->L, row 2 L->R, ...
        for (uint32_t row = 0; row < num_rows_; ++row) {
            if (row % 2 == 0) {
                // Even rows: left to right
                for (uint32_t col = 0; col < num_cols_; ++col) {
                    ChipId chip_id = row * num_cols_ + col;
                    logical_line_order_.push_back(chip_id);
                }
            } else {
                // Odd rows: right to left
                for (int col = static_cast<int>(num_cols_) - 1; col >= 0; --col) {
                    ChipId chip_id = row * num_cols_ + static_cast<uint32_t>(col);
                    logical_line_order_.push_back(chip_id);
                }
            }
        }
    } else {
        // Row-major order for 2D
        for (size_t i = 0; i < num_chips_; ++i) {
            logical_line_order_.push_back(static_cast<ChipId>(i));
        }
    }

    // Build reverse mapping from chip_id to line index
    for (size_t i = 0; i < logical_line_order_.size(); ++i) {
        chip_to_line_idx_[logical_line_order_[i]] = i;
    }
}

size_t LogicalTopologyTranslator::compute_zigzag_index(const MeshCoordinate& coord) const {
    uint32_t row = coord[0];
    uint32_t col = coord[1];
    size_t base = static_cast<size_t>(row) * num_cols_;
    if (row % 2 == 0) {
        return base + col;  // Even rows: left to right
    } else {
        return base + (num_cols_ - 1 - col);  // Odd rows: right to left
    }
}

size_t LogicalTopologyTranslator::get_line_index(const MeshCoordinate& coord) const {
    ChipId chip_id = coordinate_to_chip_id(coord);
    return get_line_index(chip_id);
}

size_t LogicalTopologyTranslator::get_line_index(ChipId chip_id) const {
    auto it = chip_to_line_idx_.find(chip_id);
    TT_FATAL(it != chip_to_line_idx_.end(), "Chip {} not found in topology translator", chip_id);
    return it->second;
}

MeshCoordinate LogicalTopologyTranslator::get_coordinate_from_line_index(size_t line_idx) const {
    TT_FATAL(line_idx < num_chips_, "Line index {} out of range (num_chips={})", line_idx, num_chips_);
    ChipId chip_id = logical_line_order_[line_idx];
    return chip_id_to_coordinate(chip_id);
}

ChipId LogicalTopologyTranslator::get_chip_id_from_line_index(size_t line_idx) const {
    TT_FATAL(line_idx < num_chips_, "Line index {} out of range (num_chips={})", line_idx, num_chips_);
    return logical_line_order_[line_idx];
}

MeshCoordinate LogicalTopologyTranslator::chip_id_to_coordinate(ChipId chip_id) const {
    uint32_t row = chip_id / num_cols_;
    uint32_t col = chip_id % num_cols_;
    return MeshCoordinate(row, col);
}

ChipId LogicalTopologyTranslator::coordinate_to_chip_id(const MeshCoordinate& coord) const {
    return coord[0] * num_cols_ + coord[1];
}

RoutingDirection LogicalTopologyTranslator::physical_to_logical_direction(
    ChipId src_chip_id, ChipId dst_chip_id, RoutingDirection physical_direction) const {
    if (!is_1d_fabric_) {
        // For 2D fabric, physical = logical
        return physical_direction;
    }

    // For 1D fabric, translate to logical E/W based on zigzag position
    size_t src_line_idx = get_line_index(src_chip_id);
    size_t dst_line_idx = get_line_index(dst_chip_id);

    // Check if dst is the next chip in the logical line (forward = E)
    size_t next_line_idx = (src_line_idx + 1) % num_chips_;
    if (dst_line_idx == next_line_idx) {
        // Only valid if ring, or if not at end of line
        if (is_ring_ || src_line_idx < num_chips_ - 1) {
            return RoutingDirection::E;  // Forward in line = logical East
        }
    }

    // Check if dst is the previous chip in the logical line (backward = W)
    size_t prev_line_idx = (src_line_idx + num_chips_ - 1) % num_chips_;
    if (dst_line_idx == prev_line_idx) {
        // Only valid if ring, or if not at start of line
        if (is_ring_ || src_line_idx > 0) {
            return RoutingDirection::W;  // Backward in line = logical West
        }
    }

    // This physical connection is not part of the logical 1D topology
    return RoutingDirection::NONE;
}

RoutingDirection LogicalTopologyTranslator::get_logical_routing_direction(
    ChipId src_chip_id, ChipId dst_chip_id) const {
    if (src_chip_id == dst_chip_id) {
        return RoutingDirection::C;  // Same chip
    }

    if (!is_1d_fabric_) {
        // 2D fabric: use X-first routing (E/W first, then N/S)
        MeshCoordinate src_coord = chip_id_to_coordinate(src_chip_id);
        MeshCoordinate dst_coord = chip_id_to_coordinate(dst_chip_id);

        // X-first: check column first
        if (dst_coord[1] > src_coord[1]) {
            return RoutingDirection::E;
        } else if (dst_coord[1] < src_coord[1]) {
            return RoutingDirection::W;
        }

        // Same column, check row
        if (dst_coord[0] < src_coord[0]) {
            return RoutingDirection::N;
        } else if (dst_coord[0] > src_coord[0]) {
            return RoutingDirection::S;
        }

        // Should not reach here if src != dst
        return RoutingDirection::C;
    }

    // 1D fabric: determine shorter path around the line/ring
    size_t src_line_idx = get_line_index(src_chip_id);
    size_t dst_line_idx = get_line_index(dst_chip_id);

    // Calculate forward and backward distances
    int forward_dist, backward_dist;

    if (dst_line_idx > src_line_idx) {
        forward_dist = static_cast<int>(dst_line_idx - src_line_idx);
        backward_dist = is_ring_ ? static_cast<int>(src_line_idx + num_chips_ - dst_line_idx)
                                 : static_cast<int>(num_chips_);  // Infinite for linear
    } else {
        forward_dist = is_ring_ ? static_cast<int>(dst_line_idx + num_chips_ - src_line_idx)
                                : static_cast<int>(num_chips_);  // Infinite for linear
        backward_dist = static_cast<int>(src_line_idx - dst_line_idx);
    }

    // Choose shorter path
    if (forward_dist <= backward_dist) {
        return RoutingDirection::E;  // Forward in line
    } else {
        return RoutingDirection::W;  // Backward in line
    }
}

std::optional<ChipId> LogicalTopologyTranslator::get_logical_neighbor(
    ChipId src_chip_id, RoutingDirection logical_direction) const {
    if (!is_1d_fabric_) {
        // 2D fabric: return physical neighbor
        MeshCoordinate src_coord = chip_id_to_coordinate(src_chip_id);
        int row = static_cast<int>(src_coord[0]);
        int col = static_cast<int>(src_coord[1]);

        switch (logical_direction) {
            case RoutingDirection::N:
                if (row > 0) {
                    return coordinate_to_chip_id(MeshCoordinate(row - 1, col));
                }
                break;
            case RoutingDirection::S:
                if (row < static_cast<int>(num_rows_) - 1) {
                    return coordinate_to_chip_id(MeshCoordinate(row + 1, col));
                }
                break;
            case RoutingDirection::E:
                if (col < static_cast<int>(num_cols_) - 1) {
                    return coordinate_to_chip_id(MeshCoordinate(row, col + 1));
                }
                break;
            case RoutingDirection::W:
                if (col > 0) {
                    return coordinate_to_chip_id(MeshCoordinate(row, col - 1));
                }
                break;
            default: break;
        }
        return std::nullopt;
    }

    // 1D fabric: E = next in line, W = previous in line
    size_t src_line_idx = get_line_index(src_chip_id);

    if (logical_direction == RoutingDirection::E) {
        // Forward in line
        if (is_ring_ || src_line_idx < num_chips_ - 1) {
            size_t next_line_idx = (src_line_idx + 1) % num_chips_;
            return get_chip_id_from_line_index(next_line_idx);
        }
    } else if (logical_direction == RoutingDirection::W) {
        // Backward in line
        if (is_ring_ || src_line_idx > 0) {
            size_t prev_line_idx = (src_line_idx + num_chips_ - 1) % num_chips_;
            return get_chip_id_from_line_index(prev_line_idx);
        }
    }

    return std::nullopt;
}

}  // namespace tt::tt_fabric
