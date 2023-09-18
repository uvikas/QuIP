# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 2 --quant ldlq --incoh_processing
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 3 --quant ldlq --incoh_processing
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-350m c4 --wbits 4 --quant ldlq --incoh_processing

CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 2 --quant ldlq --incoh_processing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 3 --quant ldlq --incoh_processing
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --wbits 4 --quant ldlq --incoh_processing