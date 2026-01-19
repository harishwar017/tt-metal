// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <enchantum/enchantum.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <chrono>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>

namespace tt::tt_fabric {

RoutingTableGenerator::RoutingTableGenerator(const TopologyMapper& topology_mapper) :
    topology_mapper_(topology_mapper) {
    // Use IntraMeshConnectivity to size all variables
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();
    const auto& inter_mesh_connectivity = mesh_graph.get_inter_mesh_connectivity();
    this->intra_mesh_table_.resize(intra_mesh_connectivity.size());
    this->inter_mesh_table_.resize(intra_mesh_connectivity.size());
    this->exit_node_lut_.resize(intra_mesh_connectivity.size());
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < intra_mesh_connectivity.size(); mesh_id_val++) {
        this->intra_mesh_table_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        this->inter_mesh_table_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        this->exit_node_lut_[mesh_id_val].resize(intra_mesh_connectivity[mesh_id_val].size());
        for (auto& devices_in_mesh : this->intra_mesh_table_[mesh_id_val]) {
            // intra_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other chips in the mesh
            devices_in_mesh.resize(intra_mesh_connectivity[mesh_id_val].size());
        }
        for (auto& devices_in_mesh : this->inter_mesh_table_[mesh_id_val]) {
            // inter_mesh_table[mesh_id][chip_id] holds a vector of ports to route to other meshes
            devices_in_mesh.resize(intra_mesh_connectivity.size());
        }
        for (auto& devices_in_mesh : this->exit_node_lut_[mesh_id_val]) {
            // exit_node_lut_[mesh_id][chip_id] holds exit chip per destination mesh
            devices_in_mesh.resize(intra_mesh_connectivity.size());
        }
    }
    // Generate the intra mesh routing table
    this->generate_intramesh_routing_table(intra_mesh_connectivity);

    // Generate the inter mesh routing table
    this->generate_intermesh_routing_table(inter_mesh_connectivity, intra_mesh_connectivity);
}

void RoutingTableGenerator::generate_intramesh_routing_table(const IntraMeshConnectivity& intra_mesh_connectivity) {
    const auto get_shorter_direction_on_row_or_col = [&](std::uint32_t mesh_id_val,
                                                         std::uint32_t src_chip_id,
                                                         std::uint32_t dst_chip_id,
                                                         RoutingDirection a,
                                                         RoutingDirection b) -> RoutingDirection {
        // Loop through intra_mesh_connectivity starting with a or b direction and return direction that matches
        // dst_chip_id_first In case of tie, this function is returning a
        std::uint32_t curr_a = src_chip_id, curr_b = src_chip_id;
        bool a_valid = true, b_valid = true;
        while (a_valid or b_valid) {
            if (intra_mesh_connectivity[mesh_id_val][curr_a].contains(dst_chip_id) and
                intra_mesh_connectivity[mesh_id_val][curr_a].at(dst_chip_id).port_direction == a) {
                return a;
            }
            if (intra_mesh_connectivity[mesh_id_val][curr_b].contains(dst_chip_id) and
                intra_mesh_connectivity[mesh_id_val][curr_b].at(dst_chip_id).port_direction == b) {
                return b;
            }
            a_valid = false;
            b_valid = false;
            for (const auto& [next_chip_id, edge] : intra_mesh_connectivity[mesh_id_val][curr_a]) {
                if (edge.port_direction == a) {
                    curr_a = next_chip_id;
                    a_valid = true;
                    break;
                }
            }
            for (const auto& [next_chip_id, edge] : intra_mesh_connectivity[mesh_id_val][curr_b]) {
                if (edge.port_direction == b) {
                    curr_b = next_chip_id;
                    b_valid = true;
                    break;
                }
            }
        }
        TT_ASSERT(
            false,
            "No valid direction found for src_chip_id {} and dst_chip_id {} in mesh_id {}. "
            "This should not happen, check the intra_mesh_connectivity.",
            src_chip_id,
            dst_chip_id,
            mesh_id_val);
        return RoutingDirection::NONE;  // This line should never be reached
    };
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    // #region agent log - Log routing table generation with fabric config
    {
        auto fabric_config = tt::tt_fabric::GetFabricConfig();
        bool is_1d = tt::tt_fabric::is_1d_fabric_config(fabric_config);
        std::ofstream log("/localdev/snijjar/tt-metal/.cursor/debug.log", std::ios::app);
        log << "{\"location\":\"routing_table_generator:generate_intramesh\",\"hypothesisId\":\"FABRIC_CONFIG\","
            << "\"data\":{\"num_meshes\":" << this->intra_mesh_table_.size()
            << ",\"fabric_config\":" << static_cast<int>(fabric_config)
            << ",\"is_1d_fabric\":" << (is_1d ? "true" : "false")
            << "},\"timestamp\":" << std::chrono::system_clock::now().time_since_epoch().count() << "}\n";
    }
    // #endregion
    // #region agent log - Dump full intra_mesh_connectivity
    {
        std::ofstream log("/localdev/snijjar/tt-metal/.cursor/debug.log", std::ios::app);
        const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();
        for (std::uint32_t m = 0; m < intra_mesh_connectivity.size(); m++) {
            for (std::uint32_t src = 0; src < intra_mesh_connectivity[m].size(); src++) {
                for (const auto& [dst, edge] : intra_mesh_connectivity[m][src]) {
                    const char* dir_str = "?";
                    switch (edge.port_direction) {
                        case RoutingDirection::N: dir_str = "N"; break;
                        case RoutingDirection::E: dir_str = "E"; break;
                        case RoutingDirection::S: dir_str = "S"; break;
                        case RoutingDirection::W: dir_str = "W"; break;
                        case RoutingDirection::Z: dir_str = "Z"; break;
                        case RoutingDirection::C: dir_str = "C"; break;
                        default: dir_str = "NONE"; break;
                    }
                    log << "{\"location\":\"connectivity\",\"mesh\":" << m << ",\"src\":" << src << ",\"dst\":" << dst
                        << ",\"dir\":\"" << dir_str << "\"}\n";
                }
            }
        }
        log.flush();
    }
    // #endregion

    // Check fabric config for logging
    auto fabric_config = tt::tt_fabric::GetFabricConfig();
    bool is_1d_fabric = tt::tt_fabric::is_1d_fabric_config(fabric_config);

    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_table_.size(); mesh_id_val++) {
        MeshId mesh_id{mesh_id_val};

        // #region agent log - Log routing mode
        {
            std::ofstream log("/localdev/snijjar/tt-metal/.cursor/debug.log", std::ios::app);
            log << "{\"location\":\"routing_table:mode\",\"hypothesisId\":\"ROUTE_MODE\","
                << "\"data\":{\"mesh_id\":" << mesh_id_val << ",\"is_1d_fabric\":" << (is_1d_fabric ? "true" : "false")
                << ",\"fabric_config\":" << static_cast<int>(fabric_config)
                << "},\"timestamp\":" << std::chrono::system_clock::now().time_since_epoch().count() << "}\n";
        }
        // #endregion

        uint32_t num_chips = this->intra_mesh_table_[mesh_id_val].size();

        if (is_1d_fabric) {
            // For 1D fabric (ring or linear): use LOGICAL E/W directions only
            // E = forward in line/ring, W = backward in line/ring
            // The line/ring path is a snake/zigzag through ALL chips (not just perimeter)
            bool is_ring = (fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING);

            // Get mesh shape
            MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
            TT_FATAL(mesh_shape.dims() == 2, "1D fabric requires 2D mesh shape");
            uint32_t num_rows = mesh_shape[0];
            uint32_t num_cols = mesh_shape[1];

            // Generate line coordinates using zigzag/snake pattern through ALL chips
            // This matches the logic in MeshDeviceViewImpl::get_line_coordinates
            std::vector<MeshCoordinate> line_coords;
            line_coords.reserve(num_chips);

            // Zigzag pattern: alternate direction on each row
            for (uint32_t row = 0; row < num_rows; ++row) {
                if (row % 2 == 0) {
                    // Even rows: left to right
                    for (uint32_t col = 0; col < num_cols; ++col) {
                        line_coords.emplace_back(MeshCoordinate{row, col});
                    }
                } else {
                    // Odd rows: right to left
                    for (int col = static_cast<int>(num_cols - 1); col >= 0; --col) {
                        line_coords.emplace_back(MeshCoordinate{row, static_cast<uint32_t>(col)});
                    }
                }
            }

            // Build chip_id -> line_index map
            std::unordered_map<ChipId, size_t> chip_to_line_idx;
            for (size_t i = 0; i < line_coords.size(); ++i) {
                ChipId chip_id = mesh_graph.coordinate_to_chip(mesh_id, line_coords[i]);
                chip_to_line_idx[chip_id] = i;
            }
            size_t line_size = line_coords.size();

            for (ChipId src_chip_id = 0; src_chip_id < num_chips; src_chip_id++) {
                auto src_it = chip_to_line_idx.find(src_chip_id);
                if (src_it == chip_to_line_idx.end()) {
                    continue;  // Shouldn't happen
                }
                size_t src_idx = src_it->second;

                for (ChipId dst_chip_id = 0; dst_chip_id < num_chips; dst_chip_id++) {
                    if (src_chip_id == dst_chip_id) {
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::C;
                        continue;
                    }

                    auto dst_it = chip_to_line_idx.find(dst_chip_id);
                    if (dst_it == chip_to_line_idx.end()) {
                        continue;  // Shouldn't happen
                    }
                    size_t dst_idx = dst_it->second;

                    // Calculate forward and backward distances along the LINE/RING ORDER
                    size_t forward_dist, backward_dist;
                    if (dst_idx >= src_idx) {
                        forward_dist = dst_idx - src_idx;
                        backward_dist = is_ring ? (line_size - dst_idx + src_idx) : line_size;
                    } else {
                        forward_dist = is_ring ? (line_size - src_idx + dst_idx) : line_size;
                        backward_dist = src_idx - dst_idx;
                    }

                    // Choose shorter path - use LOGICAL direction E (forward) or W (backward)
                    if (forward_dist <= backward_dist) {
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::E;
                    } else {
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::W;
                    }
                }
            }
        } else {
            // 2D fabric: use 2D X-first routing with physical N/S/E/W directions
            for (ChipId src_chip_id = 0; src_chip_id < num_chips; src_chip_id++) {
                for (ChipId dst_chip_id = 0; dst_chip_id < num_chips; dst_chip_id++) {
                    auto src_mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, src_chip_id);
                    auto dst_mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, dst_chip_id);
                    // X first routing, traverse rows first
                    if (src_mesh_coord[0] != dst_mesh_coord[0]) {
                        // Move North or South
                        MeshCoordinate target_coord_on_column(dst_mesh_coord[0], src_mesh_coord[1]);
                        auto target_chip_id = mesh_graph.coordinate_to_chip(mesh_id, target_coord_on_column);
                        auto direction = get_shorter_direction_on_row_or_col(
                            mesh_id_val, src_chip_id, target_chip_id, RoutingDirection::N, RoutingDirection::S);
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = direction;
                    } else if (src_mesh_coord[1] != dst_mesh_coord[1]) {
                        // Move East or West
                        auto direction = get_shorter_direction_on_row_or_col(
                            mesh_id_val, src_chip_id, dst_chip_id, RoutingDirection::E, RoutingDirection::W);
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = direction;
                    } else {
                        // No movement
                        this->intra_mesh_table_[*mesh_id][src_chip_id][dst_chip_id] = RoutingDirection::C;
                    }
                }
            }
        }
    }
    // #region agent log - Dump full routing table
    {
        std::ofstream log("/localdev/snijjar/tt-metal/.cursor/debug.log", std::ios::app);
        log << "{\"location\":\"routing_table_dump\",\"type\":\"start\"}\n";
        for (std::uint32_t m = 0; m < this->intra_mesh_table_.size(); m++) {
            for (ChipId src = 0; src < this->intra_mesh_table_[m].size(); src++) {
                for (ChipId dst = 0; dst < this->intra_mesh_table_[m][src].size(); dst++) {
                    auto dir = this->intra_mesh_table_[m][src][dst];
                    const char* dir_str = "?";
                    switch (dir) {
                        case RoutingDirection::N: dir_str = "N"; break;
                        case RoutingDirection::E: dir_str = "E"; break;
                        case RoutingDirection::S: dir_str = "S"; break;
                        case RoutingDirection::W: dir_str = "W"; break;
                        case RoutingDirection::Z: dir_str = "Z"; break;
                        case RoutingDirection::C: dir_str = "C"; break;
                        default: dir_str = "NONE"; break;
                    }
                    if (src != dst) {  // Skip self-routes
                        log << "{\"rt\":{\"src\":" << src << ",\"dst\":" << dst << ",\"dir\":\"" << dir_str << "\"}}\n";
                    }
                }
            }
        }
        log << "{\"location\":\"routing_table_dump\",\"type\":\"end\"}\n";
        log.flush();
    }
    // #endregion
}

// Shortest Path
// TODO: Put into mesh algorithms?
std::vector<std::vector<std::vector<std::pair<ChipId, MeshId>>>> RoutingTableGenerator::get_paths_to_all_meshes(
    MeshId src, const InterMeshConnectivity& inter_mesh_connectivity) const {
    // TODO: add more tests for this
    std::uint32_t num_meshes = inter_mesh_connectivity.size();
    // avoid vector<bool> specialization
    std::vector<std::uint8_t> visited(num_meshes, false);

    // paths[target_mesh_id][path_count][next_chip and next_mesh];
    std::vector<std::vector<std::vector<std::pair<ChipId, MeshId>>>> paths;
    paths.resize(num_meshes);
    paths[*src] = {{{}}};

    std::vector<std::uint32_t> dist(num_meshes, std::numeric_limits<std::uint32_t>::max());
    dist[*src] = 0;

    std::queue<MeshId> q;
    q.push(src);
    visited[*src] = true;
    // BFS
    while (!q.empty()) {
        MeshId current_mesh_id = q.front();
        q.pop();

        // Captures paths at the chip level
        for (ChipId chip_in_mesh = 0; chip_in_mesh < inter_mesh_connectivity[*current_mesh_id].size(); chip_in_mesh++) {
            for (const auto& [connected_mesh_id, edge] : inter_mesh_connectivity[*current_mesh_id][chip_in_mesh]) {
                if (!visited[*connected_mesh_id]) {
                    q.push(connected_mesh_id);
                    visited[*connected_mesh_id] = true;
                }
                if (dist[*connected_mesh_id] > dist[*current_mesh_id] + 1) {
                    dist[*connected_mesh_id] = dist[*current_mesh_id] + 1;
                    paths[*connected_mesh_id] = paths[*current_mesh_id];
                    for (auto& path : paths[*connected_mesh_id]) {
                        path.push_back({chip_in_mesh, connected_mesh_id});
                    }
                } else if (dist[*connected_mesh_id] == dist[*current_mesh_id] + 1) {
                    // another possible path discovered
                    for (auto path : paths[*current_mesh_id]) {
                        path.push_back({chip_in_mesh, connected_mesh_id});
                        paths[*connected_mesh_id].push_back(path);
                    }
                }
            }
        }
    }
    /*
    for (MeshId mesh_id = 0; mesh_id < num_meshes; mesh_id++) {
        std::cout << "Path: from src " << src << " to " << mesh_id << " : ";
        for (const auto& path: paths[mesh_id]) {
          std::cout << std::endl;
          for (const auto& id: path) {
              std::cout << " chip " << id.first << " mesh " << id.second << " ";
          }
        }
        std::cout << std::endl;
    }*/
    return paths;
}

void RoutingTableGenerator::generate_intermesh_routing_table(
    const InterMeshConnectivity& inter_mesh_connectivity, const IntraMeshConnectivity& /*intra_mesh_connectivity*/) {
    for (std::uint32_t src_mesh_id_val = 0; src_mesh_id_val < this->inter_mesh_table_.size(); src_mesh_id_val++) {
        MeshId src_mesh_id{src_mesh_id_val};
        auto paths = get_paths_to_all_meshes(src_mesh_id, inter_mesh_connectivity);
        const auto& mesh_graph = topology_mapper_.get_mesh_graph();
        MeshShape mesh_shape = mesh_graph.get_mesh_shape(src_mesh_id);
        std::uint32_t ew_size = mesh_shape[1];
        for (ChipId src_chip_id = 0; src_chip_id < this->inter_mesh_table_[src_mesh_id_val].size(); src_chip_id++) {
            for (std::uint32_t dst_mesh_id_val = 0; dst_mesh_id_val < this->inter_mesh_table_.size();
                 dst_mesh_id_val++) {
                MeshId dst_mesh_id{dst_mesh_id_val};
                if (dst_mesh_id == src_mesh_id) {
                    // inter mesh table entry from mesh to itself
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = RoutingDirection::C;
                    this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = src_chip_id;
                    continue;
                }
                auto& candidate_paths = paths[dst_mesh_id_val];
                std::uint32_t min_load = std::numeric_limits<std::uint32_t>::max();
                std::uint32_t min_distance = std::numeric_limits<std::uint32_t>::max();
                if (candidate_paths.empty()) {
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = RoutingDirection::NONE;
                    this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        std::numeric_limits<ChipId>::max();
                    continue;
                }
                // TODO: This exit_chip_id doesn't make sense since it is always chip 0
                ChipId exit_chip_id = candidate_paths[0][1].first;
                MeshId next_mesh_id = candidate_paths[0][1].second;
                for (auto& path : candidate_paths) {
                    // First element is itself, second is next mesh
                    TT_ASSERT(!path.empty(), "Expecting at least two entries in path");
                    ChipId candidate_exit_chip_id = path[1].first;  // first element is the first hop to target mesh
                    MeshId candidate_next_mesh_id = path[1].second;
                    if (candidate_exit_chip_id == src_chip_id) {
                        // optimization for latency, always use src chip if it is an exit chip to next mesh, regardless
                        // of load on the edge
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                        break;
                    }
                    // TODO: Ideally this should take into account the shortest path through all of the meshes to get to
                    // the target mesh This is a simple implementation that only considers the shortest path to the next
                    // mesh
                    std::uint32_t ew_distance = std::abs(
                        static_cast<std::int32_t>(src_chip_id % ew_size) -
                        static_cast<std::int32_t>(candidate_exit_chip_id % ew_size));
                    std::uint32_t ns_distance = std::abs(
                        static_cast<std::int32_t>(src_chip_id / ew_size) -
                        static_cast<std::int32_t>(candidate_exit_chip_id / ew_size));
                    std::uint32_t distance = ew_distance + ns_distance;
                    if (distance < min_distance) {
                        // optimization for latency, always use the shortest path to next mesh, regardless of load on
                        // the edge
                        exit_chip_id = candidate_exit_chip_id;
                        next_mesh_id = candidate_next_mesh_id;
                        min_distance = distance;
                    } else if (distance == min_distance) {
                        const auto& edge =
                            inter_mesh_connectivity[*src_mesh_id][candidate_exit_chip_id].at(candidate_next_mesh_id);
                        if (edge.weight < min_load) {
                            min_load = edge.weight;
                            exit_chip_id = candidate_exit_chip_id;
                            next_mesh_id = candidate_next_mesh_id;
                        }
                    }
                }

                if (exit_chip_id == src_chip_id) {
                    // If src is already exit chip, use port directions to next mesh
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        inter_mesh_connectivity[*src_mesh_id][src_chip_id].at(next_mesh_id).port_direction;
                    // TODO: today we are not updating the weight of the edge, should we use weight to balance
                    //  routing traffic?
                    //  inter_mesh_connectivity[src_mesh_id][src_chip_id][next_mesh_id].weight += 1;
                } else {
                    // Use direction to exit chip from the intermesh routing table
                    this->inter_mesh_table_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] =
                        this->intra_mesh_table_[src_mesh_id_val][src_chip_id].at(exit_chip_id);
                    // Update weight for exit chip to next mesh and src chip to exit chip
                    // inter_mesh_connectivity[src_mesh_id][exit_chip_id][next_mesh_id].weight += 1;
                    // for (auto& edge: intra_mesh_connectivity[src_mesh_id][src_chip_id]) {
                    //   if (edge.second.port_direction ==
                    //   this->inter_mesh_table_[src_mesh_id][src_chip_id][dst_mesh_id]) {
                    //       edge.second.weight += 1;
                    //       break;
                    //    }
                    //  }
                }
                this->exit_node_lut_[src_mesh_id_val][src_chip_id][dst_mesh_id_val] = exit_chip_id;
                mesh_to_exit_nodes_[dst_mesh_id].push_back(FabricNodeId(MeshId{src_mesh_id}, exit_chip_id));
            }
        }
    }
}

void RoutingTableGenerator::load_intermesh_connections(const AnnotatedIntermeshConnections& intermesh_connections) {
    const auto& mesh_graph = topology_mapper_.get_mesh_graph();
    const_cast<MeshGraph&>(mesh_graph).load_intermesh_connections(intermesh_connections);
    this->generate_intermesh_routing_table(
        mesh_graph.get_inter_mesh_connectivity(), mesh_graph.get_intra_mesh_connectivity());
}

void RoutingTableGenerator::print_routing_tables() const {
    std::stringstream ss;
    ss << "Routing Table Generator: IntraMesh Routing Tables" << std::endl;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->intra_mesh_table_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (ChipId src_chip_id = 0; src_chip_id < this->intra_mesh_table_[mesh_id_val].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (ChipId dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->intra_mesh_table_[mesh_id_val][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->intra_mesh_table_[mesh_id_val][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << enchantum::to_string(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
    ss.str(std::string());
    ss << "Routing Table Generator: InterMesh Routing Tables" << std::endl;
    for (std::uint32_t mesh_id_val = 0; mesh_id_val < this->inter_mesh_table_.size(); mesh_id_val++) {
        ss << "M" << mesh_id_val << ":" << std::endl;
        for (ChipId src_chip_id = 0; src_chip_id < this->inter_mesh_table_[mesh_id_val].size(); src_chip_id++) {
            ss << "   D" << src_chip_id << ": ";
            for (ChipId dst_chip_or_mesh_id = 0;
                 dst_chip_or_mesh_id < this->inter_mesh_table_[mesh_id_val][src_chip_id].size();
                 dst_chip_or_mesh_id++) {
                auto direction = this->inter_mesh_table_[mesh_id_val][src_chip_id][dst_chip_or_mesh_id];
                ss << dst_chip_or_mesh_id << "(" << enchantum::to_string(direction) << ") ";
            }
            ss << std::endl;
        }
    }
    log_debug(tt::LogFabric, "{}", ss.str());
}

const std::vector<FabricNodeId>& RoutingTableGenerator::get_exit_nodes_routing_to_mesh(MeshId mesh_id) const {
    auto it = this->mesh_to_exit_nodes_.find(mesh_id);
    if (it != this->mesh_to_exit_nodes_.end()) {
        return it->second;
    }
    TT_THROW("No exit nodes found for mesh_id {}", *mesh_id);
}

FabricNodeId RoutingTableGenerator::get_exit_node_from_mesh_to_mesh(
    MeshId src_mesh_id, ChipId src_chip_id, MeshId dst_mesh_id) const {
    TT_FATAL(*src_mesh_id < this->exit_node_lut_.size(), "src_mesh_id out of range");
    TT_FATAL(src_chip_id < this->exit_node_lut_[*src_mesh_id].size(), "src_chip_id out of range");
    TT_FATAL(*dst_mesh_id < this->exit_node_lut_[*src_mesh_id][src_chip_id].size(), "dst_mesh_id out of range");

    ChipId exit_chip = this->exit_node_lut_[*src_mesh_id][src_chip_id][*dst_mesh_id];
    if (src_mesh_id == dst_mesh_id) {
        return FabricNodeId(src_mesh_id, src_chip_id);
    }
    TT_FATAL(
        exit_chip != std::numeric_limits<ChipId>::max(),
        "No exit chip mapped from M{}D{} to M{}",
        *src_mesh_id,
        src_chip_id,
        *dst_mesh_id);
    return FabricNodeId(src_mesh_id, exit_chip);
}
}  // namespace tt::tt_fabric
