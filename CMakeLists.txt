cmake_minimum_required(VERSION 3.0)

project(training_data_loader)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS "-DYANEURAOU_ENGINE_NNUE -DEVAL_LEARN -DUNICODE -DUSE_AVX2 -msse4.1 -mbmi -mbmi2 -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -Ofast")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_library(training_data_loader SHARED
  training_data_loader.cpp
  YaneuraOu/source/bitboard.cpp
  YaneuraOu/source/book/apery_book.cpp
  YaneuraOu/source/book/book.cpp
  YaneuraOu/source/book/makebook.cpp
  YaneuraOu/source/book/makebook2015.cpp
  YaneuraOu/source/book/makebook2019.cpp
  YaneuraOu/source/book/makebook2021.cpp
  YaneuraOu/source/engine/dlshogi-engine/dlshogi_searcher.cpp
  YaneuraOu/source/engine/dlshogi-engine/Node.cpp
  YaneuraOu/source/engine/dlshogi-engine/PrintInfo.cpp
  YaneuraOu/source/engine/dlshogi-engine/UctSearch.cpp
  YaneuraOu/source/engine/dlshogi-engine/YaneuraOu_dlshogi_bridge.cpp
  YaneuraOu/source/engine/tanuki-mate-engine/tanuki-mate-search.cpp
  YaneuraOu/source/engine/user-engine/user-search.cpp
  YaneuraOu/source/engine/yaneuraou-engine/yaneuraou-search.cpp
  YaneuraOu/source/engine/yaneuraou-mate-engine/yaneuraou-mate-search.cpp
  YaneuraOu/source/engine/yo-cluster/ProcessNegotiator.cpp
  YaneuraOu/source/engine/yo-cluster/yo_cluster.cpp
  YaneuraOu/source/eval/deep/nn.cpp
  YaneuraOu/source/eval/deep/nn_onnx_runtime.cpp
  YaneuraOu/source/eval/deep/nn_tensorrt.cpp
  YaneuraOu/source/eval/deep/nn_types.cpp
  YaneuraOu/source/eval/evaluate.cpp
  YaneuraOu/source/eval/evaluate_bona_piece.cpp
  YaneuraOu/source/eval/evaluate_io.cpp
  YaneuraOu/source/eval/evaluate_mir_inv_tools.cpp
  YaneuraOu/source/eval/kppt/evaluate_kppt.cpp
  YaneuraOu/source/eval/kppt/evaluate_kppt_learner.cpp
  YaneuraOu/source/eval/kpp_kkpt/evaluate_kpp_kkpt.cpp
  YaneuraOu/source/eval/kpp_kkpt/evaluate_kpp_kkpt_learner.cpp
  YaneuraOu/source/eval/material/evaluate_material.cpp
  YaneuraOu/source/eval/nnue/evaluate_nnue.cpp
  YaneuraOu/source/eval/nnue/evaluate_nnue_learner.cpp
  YaneuraOu/source/eval/nnue/features/half_kp.cpp
  YaneuraOu/source/eval/nnue/features/half_kpe9.cpp
  YaneuraOu/source/eval/nnue/features/half_kp_vm.cpp
  YaneuraOu/source/eval/nnue/features/half_relative_kp.cpp
  YaneuraOu/source/eval/nnue/features/k.cpp
  YaneuraOu/source/eval/nnue/features/p.cpp
  YaneuraOu/source/eval/nnue/features/pe9.cpp
  YaneuraOu/source/eval/nnue/nnue_test_command.cpp
  YaneuraOu/source/eval/nnue/wasm_simd.cpp
  YaneuraOu/source/extra/bitop.cpp
  YaneuraOu/source/extra/long_effect.cpp
  YaneuraOu/source/extra/sfen_packer.cpp
  YaneuraOu/source/extra/super_sort.cpp
  YaneuraOu/source/learn/learner.cpp
  YaneuraOu/source/learn/learning_tools.cpp
  YaneuraOu/source/learn/multi_think.cpp
  YaneuraOu/source/mate/mate.cpp
  YaneuraOu/source/mate/mate1ply_without_effect.cpp
  YaneuraOu/source/mate/mate1ply_with_effect.cpp
  YaneuraOu/source/mate/mate_solver.cpp
  YaneuraOu/source/misc.cpp
  YaneuraOu/source/movegen.cpp
  YaneuraOu/source/movepick.cpp
  YaneuraOu/source/position.cpp
  YaneuraOu/source/testcmd/benchmark.cpp
  YaneuraOu/source/testcmd/mate_test_cmd.cpp
  YaneuraOu/source/testcmd/normal_test_cmd.cpp
  YaneuraOu/source/testcmd/unit_test.cpp
  YaneuraOu/source/thread.cpp
  YaneuraOu/source/timeman.cpp
  YaneuraOu/source/tt.cpp
  YaneuraOu/source/types.cpp
  YaneuraOu/source/usi.cpp
  YaneuraOu/source/usi_option.cpp
  )

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
