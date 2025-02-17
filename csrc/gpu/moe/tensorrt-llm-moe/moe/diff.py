


import paddle

from paddle.incubate.nn.functional import (
    fused_bias_act,
    fused_layer_norm,
    fused_moe)


path = "/root/paddlejob/workspace/env_run/gaoziyuan/PaddleNLP/moe_input"
input_down = paddle.load(path)





from paddlenlp_ops import trt_llm_fused_moe



gate_weight = input_down["gate_weights[i]"]
tmp_out = input_down["tmp_out"]
ffn1_weights = input_down["ffn1_weights[i]"]
ffn2_weights = input_down["ffn2_weights[i]"]

print(gate_weight.shape)
# print(ffn1_weights.shape)
# print(ffn2_weights.shape)
# [64, 2048, 2816]
# [64, 1408, 2048]

# ffn1_weights = paddle.ones([64, 2048, 2816]).cast("bfloat16")
# ffn2_weights = paddle.ones([64, 1408, 2048]).cast("bfloat16")
# gate_weight = paddle.ones([2048, 64]).cast("float32")

# ffn1_weights = paddle.zeros_like(ffn1_weights)
# ffn1_weights[0] = 1

gate_out = paddle.matmul(tmp_out.cast("float32"), gate_weight)

# shape1 = ffn1_weights.shape
# shape2 = ffn2_weights.shape
# w1= paddle.transpose(ffn1_weights, perm=[0, 2, 1]).reshape(shape1)
# w2 = paddle.transpose(ffn2_weights, perm=[0, 2, 1]).reshape(shape2)

#ffn1_weights_1 = paddle.concat([ffn1_weights[:,:,0:-1:2], ffn1_weights[:,:,1:-1:2]],axis=-1)

fused_moe_out_1 = trt_llm_fused_moe(
            tmp_out,
            gate_out,
            ffn1_weights,
            ffn2_weights,
            # w1,
            # w2,
            None,
            None,
            None,
            6,
            0,
            "none",
        )

print(fused_moe_out_1)
fused_moe_out = fused_moe(
            tmp_out,
            gate_weight,
            ffn1_weights,
            ffn2_weights,
            None,
            None,
            None,
            None,
            "None",
            6,
            False,
        )
print(fused_moe_out)
diff = fused_moe_out - fused_moe_out_1

diff = diff
print(diff)

print(paddle.max(paddle.abs(diff)))


out1 = paddle.count_nonzero(diff)
print(out1)


