from bmt.tokenization.motion_tokenizers import DeltaDeltaTokenizer, START_ACTION, END_ACTION, DeltaTokenizer
from bmt.tokenization.biycle_tokenizer import BicycleModelTokenizer, BicycleModelInterpolatedTokenizer, BicycleModelTokenizerNoAVG, BicycleModelTokenizerFixed0124


def get_tokenizer(config):

    if config.TOKENIZATION.TOKENIZATION_METHOD == "delta":
        return DeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "delta_delta":
        return DeltaDeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "precomputed_delta_delta":
        return PrecomputedDeltaDeltaTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle":
        return BicycleModelTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle_noavg":
        return BicycleModelTokenizerNoAVG(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "bicycle_interpolated":
        return BicycleModelInterpolatedTokenizer(config)
    elif config.TOKENIZATION.TOKENIZATION_METHOD == "BicycleModelTokenizerFixed0124":
        return BicycleModelTokenizerFixed0124(config)
    else:
        raise ValueError("Unknown tokenizer: {}".format(config.TOKENIZATION.TOKENIZATION_METHOD))


def get_action_dim(config):
    t = get_tokenizer(config)
    return t.num_actions
