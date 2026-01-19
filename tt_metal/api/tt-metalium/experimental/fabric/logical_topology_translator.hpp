// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tt::tt_fabric {

/**
 * @brief Translates between physical (N/S/E/W) and logical (E/W for 1D) topology directions.
 *
 * This class provides the translation layer between:
 * - Physical topology: The actual hardware connections (N/S/E/W based on 2D grid layout)
 * - Logical topology: The user's view of the fabric (E/W for 1D ring/line, N/S/E/W for 2D)
 *
 * For 1D fabric configurations (FABRIC_1D, FABRIC_1D_RING), a zigzag path through the
 * physical 2D grid creates a logical 1D line. Physical turns (e.g., going from E to S)
 * are abstracted away - the user sees only logical E (forward) and W (backward).
 *
 * For 2D fabric configurations, physical and logical directions are the same.
 */
class LogicalTopologyTranslator {
public:
    /**
     * @brief Construct a translator for the given mesh shape and fabric config.
     *
     * @param mesh_shape The shape of the mesh (rows, cols)
     * @param fabric_config The fabric configuration (determines 1D vs 2D)
     */
    LogicalTopologyTranslator(const MeshShape& mesh_shape, FabricConfig fabric_config);

    /**
     * @brief Check if this translator is for a 1D fabric.
     */
    bool is_1d_fabric() const { return is_1d_fabric_; }

    /**
     * @brief Check if this translator is for a ring topology (wraps around).
     */
    bool is_ring() const { return is_ring_; }

    /**
     * @brief Get the logical line index for a chip at the given coordinate.
     *
     * For 1D fabric, this is the position in the zigzag path (0 to N-1).
     * For 2D fabric, this returns the linearized coordinate (row * cols + col).
     */
    size_t get_line_index(const MeshCoordinate& coord) const;

    /**
     * @brief Get the logical line index for a chip ID.
     */
    size_t get_line_index(ChipId chip_id) const;

    /**
     * @brief Get the mesh coordinate for a given logical line index.
     */
    MeshCoordinate get_coordinate_from_line_index(size_t line_idx) const;

    /**
     * @brief Get the ChipId for a given logical line index.
     */
    ChipId get_chip_id_from_line_index(size_t line_idx) const;

    /**
     * @brief Translate a physical direction to a logical direction.
     *
     * For 1D fabric: physical N/S/E/W -> logical E (forward) or W (backward) or NONE
     * For 2D fabric: pass-through (physical = logical)
     *
     * @param src_chip_id Source chip ID
     * @param dst_chip_id Destination chip ID (must be physically adjacent to src)
     * @param physical_direction The physical direction from src to dst
     * @return The logical direction, or NONE if not part of logical topology
     */
    RoutingDirection physical_to_logical_direction(
        ChipId src_chip_id, ChipId dst_chip_id, RoutingDirection physical_direction) const;

    /**
     * @brief Get the logical routing direction from src to dst chip.
     *
     * For 1D fabric: returns E (forward) or W (backward) based on shorter path in ring/line.
     * For 2D fabric: uses X-first routing (E/W first, then N/S).
     *
     * @param src_chip_id Source chip ID
     * @param dst_chip_id Destination chip ID
     * @return The logical routing direction to take from src toward dst
     */
    RoutingDirection get_logical_routing_direction(ChipId src_chip_id, ChipId dst_chip_id) const;

    /**
     * @brief Get the next logical neighbor chip in the given logical direction.
     *
     * For 1D fabric: E = next in line, W = previous in line
     * For 2D fabric: N/S/E/W based on grid position
     *
     * @param src_chip_id Source chip ID
     * @param logical_direction The logical direction to move
     * @return The neighbor chip ID, or std::nullopt if no neighbor in that direction
     */
    std::optional<ChipId> get_logical_neighbor(ChipId src_chip_id, RoutingDirection logical_direction) const;

    /**
     * @brief Get all chips in the logical line order.
     *
     * @return Vector of chip IDs in logical line order (for 1D) or row-major order (for 2D)
     */
    const std::vector<ChipId>& get_logical_line_order() const { return logical_line_order_; }

    /**
     * @brief Get the total number of chips.
     */
    size_t get_num_chips() const { return num_chips_; }

    /**
     * @brief Convert ChipId to MeshCoordinate.
     */
    MeshCoordinate chip_id_to_coordinate(ChipId chip_id) const;

    /**
     * @brief Convert MeshCoordinate to ChipId.
     */
    ChipId coordinate_to_chip_id(const MeshCoordinate& coord) const;

private:
    MeshShape mesh_shape_;
    [[maybe_unused]] FabricConfig fabric_config_;  // Kept for potential future use
    bool is_1d_fabric_;
    bool is_ring_;
    uint32_t num_rows_;
    uint32_t num_cols_;
    size_t num_chips_;

    // For 1D fabric: maps chip_id to its position in the logical zigzag line
    std::unordered_map<ChipId, size_t> chip_to_line_idx_;

    // For 1D fabric: the ordered list of chip IDs in the zigzag path
    std::vector<ChipId> logical_line_order_;

    // Helper to compute zigzag line index from coordinate
    size_t compute_zigzag_index(const MeshCoordinate& coord) const;
};

}  // namespace tt::tt_fabric
