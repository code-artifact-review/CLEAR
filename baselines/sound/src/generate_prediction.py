import os, argparse, pickle

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from gensim.models import Word2Vec

from tqdm import tqdm

from baselines.sound.src.models.DeepLineDP_model import HierarchicalAttentionNetwork
from baselines.sound.src.utils.config_for_deeplinedp import *
from clear.src.my_utils.helper import linedp_projects

torch.manual_seed(0)

arg = argparse.ArgumentParser()

arg.add_argument('-dataset', type=str, default='ambari', help='software project name (lowercase)')
arg.add_argument('-embed_dim', type=int, default=50, help='word embedding size')
arg.add_argument('-word_gru_hidden_dim', type=int, default=64, help='word attention hidden size')
arg.add_argument('-sent_gru_hidden_dim', type=int, default=64, help='sentence attention hidden size')
arg.add_argument('-word_gru_num_layers', type=int, default=1, help='number of GRU layer at word level')
arg.add_argument('-sent_gru_num_layers', type=int, default=1, help='number of GRU layer at sentence level')
arg.add_argument('-exp_name', type=str, default='')
arg.add_argument('-target_epochs', type=str, default='7', help='the epoch to load model')
arg.add_argument('-dropout', type=float, default=0.2, help='dropout rate')

args = arg.parse_args()

weight_dict = {}

max_grad_norm = 5
embed_dim = args.embed_dim
word_gru_hidden_dim = args.word_gru_hidden_dim
sent_gru_hidden_dim = args.sent_gru_hidden_dim
word_gru_num_layers = args.word_gru_num_layers
sent_gru_num_layers = args.sent_gru_num_layers
word_att_dim = 64
sent_att_dim = 64
use_layer_norm = True
dropout = args.dropout

save_every_epochs = 5
exp_name = args.exp_name

save_model_dir = './output/model/DeepLineDP/'
intermediate_output_dir = '../output/intermediate_output/DeepLineDP/within-release/'
prediction_dir = '../output/prediction/DeepLineDP/within-release/'

os.makedirs(prediction_dir, exist_ok=True)

# file_lvl_gt = '../datasets/preprocessed_data/'

DATASETS_PROJECTS = {
    #'glance_dataset': all_releases,
     'linedp_dataset': linedp_projects,
}

PREPROCESSED_DATA_ROOT = '../../../dataset'

if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)


def get_dataset_df(which_dataset, release):
    data_path = os.path.join(
        PREPROCESSED_DATA_ROOT,
        which_dataset,
        'preprocessed_data',
        release + '.csv',
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Preprocessed data file not found: {data_path}'
        )

    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.fillna('')
    dataframe = dataframe[dataframe['is_blank'] == False]
    dataframe = dataframe[dataframe['is_test_file'] == False]

    return dataframe


def predict_defective_files_in_releases(
        which_dataset,
        dataset_name,
        target_epochs,
):
    releases = DATASETS_PROJECTS[which_dataset][dataset_name]

    for i in range(len(releases) - 1):
        train_rel = releases[i]
        test_rel = releases[i + 1]

        for run in range(5):

            actual_save_model_dir = (
                    save_model_dir
                    + which_dataset
                    + '/'
                    + dataset_name
                    + '/'
                    + train_rel
                    + '/'
                    + str(run)
                    + '/'
            )

            # w2v_dir = get_w2v_path()
            w2v_dir = os.path.join(
                '../src/output/Word2Vec_model/',
                which_dataset,
            )

            word2vec_file_dir = os.path.join(w2v_dir, train_rel + '-' + str(embed_dim) + 'dim.bin')

            word2vec = Word2Vec.load(word2vec_file_dir)
            print(
                'load Word2Vec for',
                which_dataset,
                dataset_name,
                'finished',
            )

            total_vocab = len(word2vec.wv.key_to_index)

            vocab_size = total_vocab + 1

            model = HierarchicalAttentionNetwork(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                word_gru_hidden_dim=word_gru_hidden_dim,
                sent_gru_hidden_dim=sent_gru_hidden_dim,
                word_gru_num_layers=word_gru_num_layers,
                sent_gru_num_layers=sent_gru_num_layers,
                word_att_dim=word_att_dim,
                sent_att_dim=sent_att_dim,
                use_layer_norm=use_layer_norm,
                dropout=dropout)

            if exp_name == '':
                checkpoint = torch.load(actual_save_model_dir + 'checkpoint_' + target_epochs + 'epochs.pth')

            else:
                checkpoint = torch.load(
                    actual_save_model_dir + exp_name + '/checkpoint_' + target_epochs + 'epochs.pth')

            model.load_state_dict(checkpoint['model_state_dict'])

            model.sent_attention.word_attention.freeze_embeddings(True)

            model = model.cuda()
            model.eval()

            rel = test_rel
            print('generating prediction of release:', rel)

            actual_intermediate_output_dir = (
                    intermediate_output_dir
                    + which_dataset
                    + '/'
                    + dataset_name
                    + '/'
                    + rel
                    + '/'
                    + str(run)
                    + '/'
            )

            if not os.path.exists(actual_intermediate_output_dir):
                os.makedirs(actual_intermediate_output_dir)

            test_df = get_dataset_df(which_dataset, rel)

            row_list = []

            for filename, df in tqdm(test_df.groupby('filename')):

                file_label = bool(df['file-label'].unique())
                line_label = df['line-label'].tolist()
                line_number = df['line_number'].tolist()
                is_comments = df['is_comment'].tolist()

                code = df['code_line'].tolist()

                code2d = prepare_code2d(code, True)

                code3d = [code2d]

                codevec = get_x_vec(code3d, word2vec)

                save_file_path = actual_intermediate_output_dir + filename.replace('/', '_').replace('.java',
                                                                                                     '') + '_' + target_epochs + '_epochs.pkl'

                if not os.path.exists(save_file_path):
                    with torch.no_grad():
                        codevec_padded_tensor = torch.tensor(codevec)
                        # output, word_att_weights, line_att_weight, _ = model(codevec_padded_tensor)
                        codevec_padded_tensor = codevec_padded_tensor.contiguous()

                        try:
                            output, word_att_weights, line_att_weight, _ = model(
                                codevec_padded_tensor
                            )

                        except RuntimeError as exc:
                            if 'CUDNN_STATUS_NOT_SUPPORTED' not in str(exc):
                                raise

                            print(
                                f'\ncuDNN does not support the current GRU input. '
                                f'Retrying without cuDNN: '
                                f'release={rel}, '
                                f'sample_index={i}, '
                                f'input_shape={tuple(codevec_padded_tensor.shape)}'
                            )

                            torch.cuda.empty_cache()

                            with torch.backends.cudnn.flags(enabled=False):
                                output, word_att_weights, line_att_weight, _ = model(
                                    codevec_padded_tensor
                                )
                        file_prob = output.item()
                        prediction = bool(round(output.item()))

                        torch.cuda.empty_cache()

                        output_dict = {
                            'filename': filename,
                            'file-label': file_label,
                            'prob': file_prob,
                            'pred': prediction,
                            'word_attention_mat': word_att_weights,
                            'line_attention_mat': line_att_weight,
                            'line-label': line_label,
                            'line-number': line_number
                        }

                        pickle.dump(output_dict, open(save_file_path, 'wb'))

                else:
                    output_dict = pickle.load(open(save_file_path, 'rb'))
                    file_prob = output_dict['prob']
                    prediction = output_dict['pred']
                    word_att_weights = output_dict['word_attention_mat']
                    line_att_weight = output_dict['line_attention_mat']

                numpy_word_attn = word_att_weights[0].cpu().detach().numpy()
                numpy_line_attn = line_att_weight[0].cpu().detach().numpy()

                for i in range(0, len(code)):
                    cur_line = code[i]
                    cur_line_label = line_label[i]
                    cur_line_number = line_number[i]
                    cur_is_comment = is_comments[i]
                    cur_line_attn = numpy_line_attn[i]

                    token_list = cur_line.strip().split()

                    max_len = min(len(token_list), 50)

                    for j in range(0, max_len):
                        tok = token_list[j]
                        word_attn = numpy_word_attn[i][j]

                        row_dict = {
                            'project': dataset_name,
                            'train': train_rel,
                            'test': rel,
                            'filename': filename,
                            'file-level-ground-truth': file_label,
                            'prediction-prob': file_prob,
                            'prediction-label': prediction,
                            'line-number': cur_line_number,
                            'line-level-ground-truth': cur_line_label,
                            'is-comment-line': cur_is_comment,
                            'token': tok,
                            'token-attention-score': word_attn,
                            'line-attention-score': cur_line_attn
                        }

                        row_list.append(row_dict)

            df = pd.DataFrame(row_list)

            save_df_name = (
                    prediction_dir
                    + '/'
                    + which_dataset
                    + '/'
                    + str(run)
                    + '/'
                    + rel
                    + '.csv'
            )
            os.makedirs(os.path.dirname(save_df_name), exist_ok=True)
            df.to_csv(save_df_name, index=False)

            print('finished release', rel, 'for run', run)


target_epochs = args.target_epochs

if __name__ == "__main__":
    for which_dataset, projects in DATASETS_PROJECTS.items():
        for dataset_name in projects:
            predict_defective_files_in_releases(
                which_dataset,
                dataset_name,
                target_epochs,
            )
