# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set(DFB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dataflow_buffer.cpp)

add_library(dfb OBJECT ${DFB_SRC})

target_link_libraries(dfb PUBLIC common)

target_include_directories(
    dfb
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/hostdevcommon/api
)
