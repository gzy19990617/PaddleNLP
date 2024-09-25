# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import re


def get_candidate_tiles():
    base_configs = [("<64, 64, 64>", "<32, 32, 64>", "<16, 8, 32>")]

    base_configs.extend(
        [
            ("<32, 128, 64>", "<32, 32, 64>", "<16, 8, 32>"),
            ("<64, 128, 64>", "<32, 64, 64>", "<16, 8, 32>"),
            ("<64, 64, 128>", "<32, 64, 64>", "<16, 8, 32>"),
            ("<64, 128, 64>", "<64, 32, 64>", "<16, 8, 32>"),
            ("<128, 64, 64>", "<64, 32, 64>", "<16, 8, 32>"),
            ("<128, 128, 64>", "<64, 32, 64>", "<16, 8, 32>"),
            ("<128, 128, 64>", "<64, 64, 64>", "<16, 8, 32>"),
            ("<128, 128, 64>", "<128, 32, 64>", "<16, 8, 32>"),
            ("<128, 256, 64>", "<64, 64, 64>", "<16, 8, 32>"),
            ("<256, 128, 64>", "<64, 64, 64>", "<16, 8, 32>"),
            ("<16, 256, 128>", "<16, 64, 128>", "<16, 8, 32>"),
        ]
    )

    return base_configs


def get_candidate_configs(sm, min_split_k, max_split_k, min_stages, max_stages):
    tiles = get_candidate_tiles()
    candidate_configs = list()

    stages = tuple(i for i in range(min_stages, max_stages + 1, 1))
    splitks = tuple(i for i in range(min_split_k, max_split_k + 1, 1))
    hasbias = ("false", "true")

    for act_tag in [
        ("noact", "LinearCombination"),
        ("relu", "LinearCombinationRelu"),
        ("gelu", "LinearCombinationGELU"),
    ]:
        candidate_configs.extend([(stages, splitks, tiles, act_tag, hasbias)])

    return candidate_configs


# this is a file's header part
CommonHead = """// Generated by generate_code_gemm_fused_kernels.py - Do not edit.

#pragma once

#include "fp8_gemm_fused/fuse_gemm_{act_tag}_template.h"

"""


CommonTail = """

"""

GemmDeclare = """
template<>
bool dispatch_fuse_gemm_{act_tag}<phi::dtype::{input_type}, phi::dtype::{output_type},
                        cutlass::gemm::GemmShape{thread_block_shape}, cutlass::gemm::GemmShape{warp_shape},
                        cutlass::gemm::GemmShape{mma_shape}, {num_stages}, {hasbias}, {SM}>(GemmEpilogueAllParams);


"""


GemmSplitKDeclare = """
template<>
bool dispatch_fuse_gemm_split_k_{act_tag}<phi::dtype::{input_type}, phi::dtype::{output_type},
                        cutlass::gemm::GemmShape{thread_block_shape}, cutlass::gemm::GemmShape{warp_shape},
                        cutlass::gemm::GemmShape{mma_shape}, {num_stages}, {hasbias}, {SM}>(GemmEpilogueAllParams);


"""

LaunchGemmHead = """
#pragma once

#include "fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.h"

"""

LaunchGemmDeclare = """
bool launch_gemm_kernel_{gemm_config}(const int type_id, const int split_k, GemmEpilogueAllParams params);
"""

LaunchGemmPart0 = """
#pragma once

#include "launch_gemm_kernel.h"

bool launch_gemm_kernel_{gemm_config}(const int type_id, const int split_k, GemmEpilogueAllParams params){
    if(split_k < 2){
        params.split_k = 1;
        switch (type_id) {
"""

LaunchGemmPart1 = """
            case {type_id}:
                return dispatch_fuse_gemm_{act_tag}<phi::dtype::{input_type}, phi::dtype::{output_type},
                                        cutlass::gemm::GemmShape{thread_block_shape}, cutlass::gemm::GemmShape{warp_shape},
                                        cutlass::gemm::GemmShape{mma_shape}, {num_stages}, {hasbias}, {SM}>(params);
                break;
"""

LaunchGemmPart2 = """
            default:
                throw std::runtime_error("cutlass gemm config is invalid.");
                break;
        }
    }else{
        switch (type_id) {
"""

LaunchGemmPart3 = """
            case {type_id}:
                return dispatch_fuse_gemm_split_k_{act_tag}<phi::dtype::{input_type}, phi::dtype::{output_type},
                                        cutlass::gemm::GemmShape{thread_block_shape}, cutlass::gemm::GemmShape{warp_shape},
                                        cutlass::gemm::GemmShape{mma_shape}, {num_stages}, {hasbias}, {SM}>(params);
                break;
"""

LaunchGemmPart4 = """
            default:
                throw std::runtime_error("cutlass gemm config is invalid.");
                break;
        }
    }

    return false;
}
"""


code_part0 = """// Generated by generate_code_gemm_fused_kernels.py - Do not edit.

#include <map>
#include "fp8_fp8_gemm_scale_bias_act.h"
#include "launch_gemm_kernel.h"

COMMON_DECLARE_string(use_cutlass_device_best_config_path);

std::map<std::string, int> gemm_type_map{"""

code_part1 = """
    {"{input_type}_{output_type}_{hasbias}_{act_tag}",   {type_id}}, """

code_part2 = """
};

std::map<std::string, int> gemm_config_map{
"""

code_part3 = """    {"{thread_block_shape}, {warp_shape}, {mma_shape}, {num_stages}", {tile_id}},
"""

code_part4 = """};

bool launch_gemm_kernel(const int type_id, const int split_k, const int kernel_id, GemmEpilogueAllParams params){
    switch (kernel_id) {"""

code_part5 = """
        case {tile_id}:
            return launch_gemm_kernel_{gemm_config}(type_id, split_k, params);
            break;"""

code_part6 = """
        default:
            throw std::runtime_error("fp8_fp8_bf16_gemm_fused Config is invalid.");
            break;
    }
    return false;
}


bool fp8_fp8_gemm_scale_bias_act(GemmEpilogueAllParams params) {
  if (gemm_type_map.find(params.fuse_gemm_config) == gemm_type_map.end()) {
    throw std::runtime_error("fp8 gemm_fused config is invalid.");
  }

  int type_id = gemm_type_map[params.fuse_gemm_config];
  int M = (params.M+31)/32 *32;
  int N = params.N;
  int K = params.K;

  std::string mnk_string = "gemm<"+ std::to_string(M)+ ", " +std::to_string(N) + ", "+ std::to_string(K)+ ">";
  std::string mnk_split_k_string =  "gemm<"+ std::to_string(M)+ ", " +std::to_string(N) + ", "+ std::to_string(K)+ ">" + ", split_k";
  int split_k;
  int kernel_id;
  std::string best_config;
  CutlassGemmConfigMannager& best_config_mannager = CutlassGemmConfigMannager::getInstance();
  if(getenv("FLAGS_use_cutlass_device_best_config_path")){ // run kernel
    std::string config_file_path = getenv("FLAGS_use_cutlass_device_best_config_path");
    nlohmann::json* config_json = best_config_mannager.get_gemm_best_configs(config_file_path);
    if (config_json->contains(mnk_string)) {
        best_config = config_json->at(mnk_string);
    } else {
        std::cerr << "Can not find the config for this gemm shape, please tune this shape: " << mnk_string <<std::endl;
    }

    if (config_json->contains(mnk_split_k_string)) {
        split_k = config_json->at(mnk_split_k_string);
    } else {
        std::cerr << "Can not find the config(split_k) for this gemm shape, please tune this shape: " << mnk_string <<std::endl;
    }

    if (gemm_config_map.find(best_config) == gemm_config_map.end()) {
        throw std::runtime_error("This config'kernel not be generate, please check generate_code_gemm_fused_kernels.py and re-generate.");
    } else {
        kernel_id = gemm_config_map[best_config];
    }
    return launch_gemm_kernel(type_id, split_k, kernel_id, params);
  } else { // tune kernel
    int warm_up_times = 5;
    int tune_times = 10;
    std::string best_kernel_id = "";
    int best_split_k = -1;
    float duratation = 1000000.f;
    // tune all split_k, kernel_id kernels
    for(int i = 1; i < {max_split_k}+1; ++i){ // all split_k
        for(const auto& config_pair : gemm_config_map){
            bool is_valid = true;
            // warm up
            for(int num_time = 0; num_time < warm_up_times; ++num_time){
                if(!launch_gemm_kernel(type_id, i, config_pair.second, params)){
                    is_valid = false;
                    break;
                }
            }
            if(!is_valid){
                continue;
            }
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaStreamSynchronize(params.stream);
            cudaEventRecord(start, params.stream);
            for(int num_time = 0; num_time < tune_times; ++num_time){
                if(!launch_gemm_kernel(type_id, i, config_pair.second, params)){
                    is_valid = false;
                    break;
                };
            }
            cudaEventRecord(stop, params.stream);
            cudaEventSynchronize(stop);
            float elapsedTime;
            if(is_valid){
                cudaEventElapsedTime(&elapsedTime, start, stop);
            } else {
                continue;
            }
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            if(elapsedTime < duratation){
              best_kernel_id = config_pair.first;
              best_split_k = i;
              duratation = elapsedTime;
            }
        }
    }

    nlohmann::json new_json;
    new_json[mnk_string] = best_kernel_id;
    new_json[mnk_split_k_string] = best_split_k;
    best_config_mannager.up_date_configs(new_json);
    std::cout <<"Gemm tune result for " << mnk_string<< ": best config is: "<< best_kernel_id << ", split k: " << best_split_k << std::endl;
    return true;
  }
}
"""


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = f"\\{{{key}\\}}"
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="The argument for generating the generic_mixed_gemm_kernelLauncher instance."
    )
    parser.add_argument(
        "--cuda_arch",
        type=str,
        nargs="+",
        default=["89"],
        help="The CUDA architecture to be generated.",
    )

    parser.add_argument(
        "--min_split_k",
        type=int,
        default=2,
        help="The min split k for the gemm kernel.",
    )

    parser.add_argument(
        "--max_split_k",
        type=int,
        default=6,
        help="The max split k for the gemm kernel.",
    )

    parser.add_argument(
        "--min_stages",
        type=int,
        default=2,
        help="The min stages for the gemm kernel.",
    )

    parser.add_argument(
        "--max_stages",
        type=int,
        default=8,
        help="The max stages for the gemm kernel.",
    )

    args = parser.parse_args()
    return args


# generate source .cu
def generate_source_cu(
    inputs_type: (str),
    outputs_type: (str),
    stages: (int),
    tiles: (str),
    act_tag: str,
    hasbiases: (str),
    sm: str,
):
    value_dict = {
        "act_tag": act_tag,
    }
    all_code = SubstituteTemplate(CommonHead, value_dict)

    for input_type in inputs_type:
        for output_type in outputs_type:
            for stage in stages:
                for hasbias in hasbiases:
                    for tile_config in tiles:
                        value_dict = {
                            "input_type": input_type,
                            "output_type": output_type,
                            "thread_block_shape": tile_config[0],
                            "warp_shape": tile_config[1],
                            "mma_shape": tile_config[2],
                            "num_stages": str(stage),
                            "act_tag": act_tag,
                            "hasbias": hasbias,
                            "SM": sm,
                        }
                        all_code += SubstituteTemplate(GemmDeclare, value_dict)

    for input_type in inputs_type:
        for output_type in outputs_type:
            for stage in stages:
                for hasbias in hasbiases:
                    for tile_config in tiles:
                        value_dict = {
                            "input_type": input_type,
                            "output_type": output_type,
                            "thread_block_shape": tile_config[0],
                            "warp_shape": tile_config[1],
                            "mma_shape": tile_config[2],
                            "num_stages": str(stage),
                            "act_tag": act_tag,
                            "hasbias": hasbias,
                            "SM": sm,
                        }
                        all_code += SubstituteTemplate(GemmSplitKDeclare, value_dict)

    all_code += CommonTail
    return all_code


# generate gemm launch .cu
def generate_launch_gemm_cus(
    generate_dir: (str),
    inputs_type: (str),
    outputs_type: (str),
    stages: (int),
    split_ks: (int),
    tiles: (str),
    act_tags: (str),
    hasbiases: (str),
    sm: str,
    min_split_k: int,
    max_split_k: int,
):
    code_map = {}
    head_path = os.path.join(generate_dir, "launch_gemm_kernel.h")
    head_all_code = LaunchGemmHead
    for tile in tiles:
        blocks, warps, mmas = [s.replace(" ", "").strip("<>").split(",") for s in tile]
        gemm_config = f"block{blocks[0]}x{blocks[1]}x{blocks[2]}_warp{warps[0]}x{warps[1]}x{warps[2]}_mma{mmas[0]}x{mmas[1]}x{mmas[2]}"
        for stage in stages:
            gemm_config_str = gemm_config + f"_stage{stage}"
            value_dict = {
                "gemm_config": gemm_config_str,
            }
            head_all_code += SubstituteTemplate(LaunchGemmDeclare, value_dict)
    os.makedirs(generate_dir, exist_ok=True)
    with open(head_path, "w") as f:
        f.write(head_all_code)
        f.close()

    for tile in tiles:
        blocks, warps, mmas = [s.replace(" ", "").strip("<>").split(",") for s in tile]
        gemm_config = f"block{blocks[0]}x{blocks[1]}x{blocks[2]}_warp{warps[0]}x{warps[1]}x{warps[2]}_mma{mmas[0]}x{mmas[1]}x{mmas[2]}"
        for stage in stages:
            gemm_config_str = gemm_config + f"_stage{stage}"
            value_dict = {
                "gemm_config": gemm_config_str,
            }
            source_all_code = SubstituteTemplate(LaunchGemmPart0, value_dict)
            split_k_code = ""
            type_id = 0
            for input_type in inputs_type:
                for output_type in outputs_type:
                    for act_tag in act_tags:
                        for hasbias in hasbiases:
                            value_dict = {
                                "act_tag": act_tag,
                                "input_type": input_type,
                                "output_type": output_type,
                                "hasbias": hasbias,
                                "type_id": str(type_id),
                                "thread_block_shape": tile[0],
                                "warp_shape": tile[1],
                                "mma_shape": tile[2],
                                "num_stages": str(stage),
                                "SM": sm,
                            }
                            source_all_code += SubstituteTemplate(LaunchGemmPart1, value_dict)
                            split_k_code += SubstituteTemplate(LaunchGemmPart3, value_dict)
                            type_id += 1
            source_all_code += LaunchGemmPart2
            source_all_code += split_k_code
            source_all_code += LaunchGemmPart4
            code_map[gemm_config_str] = source_all_code
            source_path = os.path.join(generate_dir, f"launch_gemm_kernel_{gemm_config_str}.cu")
            with open(source_path, "w") as f:
                f.write(source_all_code)
                f.close()

    return head_all_code, code_map


# generate fp8_fp8_gemm_scale_bias_act.cu
def generate_dispatch_gemm_cu(
    inputs_type: (str),
    outputs_type: (str),
    stages: (int),
    split_ks: (int),
    tiles: (str),
    act_tags: (str),
    hasbiases: (str),
    sm: str,
    min_split_k: int,
    max_split_k: int,
):

    all_code = code_part0
    type_id = 0
    for input_type in inputs_type:
        for output_type in outputs_type:
            for act_tag in act_tags:
                for hasbias in hasbiases:
                    value_dict = {
                        "act_tag": act_tag,
                        "input_type": input_type,
                        "output_type": output_type,
                        "hasbias": hasbias,
                        "type_id": str(type_id),
                    }
                    all_code += SubstituteTemplate(code_part1, value_dict)
                    type_id += 1

    all_code += code_part2
    tile_id = 0
    for tile in tiles:
        for stage in stages:
            value_dict = {
                "thread_block_shape": tile[0],
                "warp_shape": tile[1],
                "mma_shape": tile[2],
                "num_stages": str(stage),
                "tile_id": str(tile_id),
            }
            all_code += SubstituteTemplate(code_part3, value_dict)
            tile_id += 1
    all_code += code_part4
    tile_id = 0
    for tile in tiles:
        blocks, warps, mmas = [s.replace(" ", "").strip("<>").split(",") for s in tile]
        gemm_config = f"block{blocks[0]}x{blocks[1]}x{blocks[2]}_warp{warps[0]}x{warps[1]}x{warps[2]}_mma{mmas[0]}x{mmas[1]}x{mmas[2]}"
        for stage in stages:
            gemm_config_str = gemm_config + f"_stage{stage}"
            value_dict = {
                "tile_id": str(tile_id),
                "gemm_config": gemm_config_str,
            }
            all_code += SubstituteTemplate(code_part5, value_dict)
            tile_id += 1
    value_dict.update(
        {
            "min_split_k": str(min_split_k),
            "max_split_k": str(max_split_k),
        }
    )
    all_code += SubstituteTemplate(code_part6, value_dict)
    return all_code


if __name__ == "__main__":
    args = parse_args()
    archs = args.cuda_arch
    min_split_k = args.min_split_k
    max_split_k = args.max_split_k
    min_stages = args.min_stages
    max_stages = args.max_stages
    inputs_type = ("float8_e4m3fn", "float8_e5m2")
    outputs_type = ("float16", "bfloat16")
    sm_dict = {"89": "cutlass::arch::Sm89", "90": "cutlass::arch::Sm90"}

    for sm in archs:
        if sm == "89":
            fuse_gemm_configs = get_candidate_configs(sm, min_split_k, max_split_k, min_stages, max_stages)
            for fuse_gemm_config in fuse_gemm_configs:
                file_name = f"gpu/cutlass_kernels/fp8_gemm_fused/autogen/generic_gemm_kernel_sm{sm}_{fuse_gemm_config[3][0]}.cu"
                all_code = generate_source_cu(
                    inputs_type,
                    outputs_type,
                    fuse_gemm_config[0],
                    fuse_gemm_config[2],
                    fuse_gemm_config[3][0],
                    fuse_gemm_config[4],
                    sm_dict[sm],
                )
                file_dir = os.path.dirname(file_name)
                os.makedirs(file_dir, exist_ok=True)
                with open(file_name, "w") as f:
                    f.write(all_code)
                    f.close()

            fuse_gemm_config = list(fuse_gemm_configs)[0]

            act_tags = ["noact", "relu", "gelu"]
            # Compile parallelization
            generate_launch_gemm_cus(
                "gpu/cutlass_kernels/fp8_gemm_fused/autogen",
                inputs_type,
                outputs_type,
                fuse_gemm_config[0],
                fuse_gemm_config[1],
                fuse_gemm_config[2],
                act_tags,
                fuse_gemm_config[4],
                sm_dict[sm],
                min_split_k,
                max_split_k,
            )

            # hard code for act_tag

            file_name = "gpu/cutlass_kernels/fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.cu"
            all_code = generate_dispatch_gemm_cu(
                inputs_type,
                outputs_type,
                fuse_gemm_config[0],
                fuse_gemm_config[1],
                fuse_gemm_config[2],
                act_tags,
                fuse_gemm_config[4],
                sm_dict[sm],
                min_split_k,
                max_split_k,
            )
            file_dir = os.path.dirname(file_name)
            os.makedirs(file_dir, exist_ok=True)
            with open(file_name, "w") as f:
                f.write(all_code)
                f.close()

        elif sm == 90:
            print("Not supported yet.")
            exit(0)
        else:
            raise ValueError(f"Unsupported SM: {sm}")
