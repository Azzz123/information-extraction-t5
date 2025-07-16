# information_extraction_t5/predict.py (THE FINAL, ULTIMATE, CORRECTED VERSION)
import configargparse
import torch
from pytorch_lightning import Trainer

from information_extraction_t5.models.qa_model import LitQA
from information_extraction_t5.data.qa_data import QADataModule


def main():
    """Predict using a specified checkpoint with a guaranteed hparams override."""

    parser = configargparse.ArgParser(
        'Prediction script for the T5 QA model',
        config_file_parser_class=configargparse.YAMLConfigFileParser)

    # --- 手术刀修改 1: 添加一个必须的、用于指定检查点路径的新参数 ---
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the model checkpoint (.ckpt) to use for prediction.')

    # 严格保留原作者的所有参数定义，确保兼容性
    parser.add_argument('-c', '--my-config', required=True, is_config_file=True, help='config file path')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--use_cached_predictions", action="store_true",
                        help="If true, reload the cache to post-process and re-evaluate.")

    # 严格保持原作者的参数加载顺序
    parser = LitQA.add_model_specific_args(parser)
    parser = QADataModule.add_model_specific_args(parser)
    args, _ = parser.parse_known_args()

    print(f'Loading model from specified checkpoint: {args.ckpt_path}')

    # 严格遵循原作者的加载逻辑，只替换检查点路径
    model = LitQA.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        hparams_file=None,
        map_location=None,
        hparams=args,  # 关键：始终传递hparams，满足__init__的契约
        strict=False
    )

    # --- 手术刀修改 2: 在模型加载后，进行决定性的“强制覆盖” ---
    # 无论模型从ckpt里加载了什么旧设置，我们在这里用命令行的值进行强制覆盖！
    model.hparams.use_cached_predictions = args.use_cached_predictions

    # 严格保留原作者的后续逻辑
    dm = QADataModule(args)
    dm.setup('test')

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(gpus=gpus)

    print("\n--- Starting Final Evaluation ---")
    if model.hparams.use_cached_predictions:
        print("Mode: Using cached predictions. Inference will be skipped.")
    else:
        print("Mode: Running full inference.")

    trainer.test(model, datamodule=dm)

    print("\n--- Prediction and Evaluation Finished ---")


if __name__ == "__main__":
    main()