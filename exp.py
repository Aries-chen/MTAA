import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

import transformers
from transformers import logging as hf_logging
import argparse
from evaluation import *
from model import *
from datasets import *
import warnings
warnings.filterwarnings("ignore")

hf_logging.set_verbosity_error()


def train_and_test(task_type, train_ds, test_ds, MODEL_PATH, cfg, model_type, is_train=True, ckpt_path=None, epoch=1, lr=1e-5, lr_decay=0.3, every_epoch=1, alpha=10):
    if is_train:
        print('Train the model: ')
        # Train from the scratch
        model = MTAA(MODEL_PATH, cfg, model_type, task_type)
        loss_metric = tf.keras.metrics.Mean(name='train_loss')
        train_metric = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')
        val_metric = tf.keras.metrics.RootMeanSquaredError(name='test_rmse')
        backbone_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        w_all = []
        head_num = get_head_num(task_type)
        for epoch in range(epoch):
            loss_metric.reset_states()
            train_metric.reset_states()
            val_metric.reset_states()
            avg_w_per = []

            # Initialize weights
            if epoch == 0 or epoch == 1:
                loss_relative = [1 for _ in range(head_num)]
            else:
                # Calculate w
                loss_t_1, loss_t_2 = w_all[-2], w_all[-1]
                loss_relative = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]
            task_weight_list = adjust_weights(loss_relative, alpha)
            print(f'epoch:{epoch}', 'task_weight_list:', task_weight_list)

            if 1 < epoch and epoch % every_epoch == 0:
                updated_learning_rate = backbone_optimizer.learning_rate * lr_decay
                backbone_optimizer.learning_rate.assign(updated_learning_rate)
                print(f'epoch:{epoch} optimizer_backbone lr:{backbone_optimizer.learning_rate.numpy()}')

            for i, (X, y) in enumerate(train_ds):
                with tf.GradientTape(persistent=True) as tape:
                    predictions = model(X, y, training=True)
                    losses = model.losses
                    avg_w_per.append(losses)

                if epoch == 0 or epoch == 1:
                    initial_grads = tape.gradient(losses, model.trainable_variables)
                    backbone_optimizer.apply_gradients(zip(initial_grads, model.trainable_variables))
                else:
                    backbone_trainable_variables = model.backbone.trainable_variables
                    grads_backbone = tape.gradient(losses, backbone_trainable_variables)
                    backbone_optimizer.apply_gradients(zip(grads_backbone, backbone_trainable_variables))

                    task_trainable_variables = [extract_task_weights(model, i) for i in range(1, head_num+1)]
                    task_grads = [tape.gradient(losses, task_trainable_variables[i]) for i in range(head_num)]

                    for i in range(head_num):
                        task_learning_rate = updated_learning_rate * task_weight_list[i]
                        task_optimizer = tf.keras.optimizers.Adam(learning_rate=task_learning_rate)
                        task_optimizer.apply_gradients(zip(task_grads[i], task_trainable_variables[i]))

                loss_metric.update_state(sum(losses))
                train_metric.update_state(y, predictions)

            w_epoch = np.average(avg_w_per, axis=0)
            w_all.append(w_epoch)
            print(f'Epoch {epoch}, Loss {loss_metric.result()}, Rmse {train_metric.result()}, \n')
        # Save the model
        model.save_weights(ckpt_path)

    # Load the pre-trained model
    print('Load the pre-trained model: ')
    model = MTAA(MODEL_PATH, cfg, model_type, task_type)
    model.load_weights(ckpt_path)

    # testing loop
    test_metric = tf.keras.metrics.RootMeanSquaredError(name='test_rmse')
    pred = []
    for (X, y) in test_ds:
        # predictions = model(X, y=None, training=False)
        predictions = model(X, y, training=False)
        batch_pred = np.concatenate([i for i in predictions], axis=1)
        pred.append(batch_pred)
        test_metric.update_state(y, predictions)

    print(f'test_rmse:{test_metric.result().numpy()}')
    pred_all = np.concatenate([i for i in pred], axis=0)

    return pred_all


def run(task_type, train_data_path, test_data_path, model_type, text_type, dr, is_train, ckpt_path, batch_size,
         epoch=3, lr=1e-5, lr_decay=0.3, every_epoch=1, alpha=10):
    print('start model_type:', model_type)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    MODEL_PATH, tokenizer, cfg = model_selection(model_type, dr)
    train_ds, test_ds = get_df(task_type, train_data, test_data, tokenizer, text_type, batch_size)
    y_preds = train_and_test(task_type, train_ds, test_ds, MODEL_PATH, cfg, model_type, is_train, ckpt_path,
                                epoch, lr, lr_decay, every_epoch, alpha)

    y_trues = get_y_true(task_type, test_data)
    result = get_metrics(task_type, y_trues, y_preds)
    print('results: \n', result)
    result.to_csv(f'results/{task_type}.csv')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define all arguments
    parser.add_argument('--task_type', type=str, default='ell', help="Task type (asap_12, asap_36, ell)")
    parser.add_argument('--train_path', type=str, default='dataset/ELLIPSE/ell_train.csv',
                        help="Training data path")
    parser.add_argument('--test_path', type=str, default='dataset/ELLIPSE/ell_test.csv', help="Test data path")
    parser.add_argument('--ckpt_path', type=str, default='ckpt/ell_ckpt/checkpoint/ell_checkpoints',
                        help="Checkpoint path")
    parser.add_argument('--text_type', type=str, default='full_text',
                        help="Text type, 'full_text' in ell, 'essay' in asap_12, asap_36")
    parser.add_argument('--is_train', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Train or evaluate (True/False)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--epoch', type=int, default=6, help="Number of epochs")
    parser.add_argument('--alpha', type=int, default=10, help="Alpha parameter")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.3, help="Learning rate decay")
    parser.add_argument('--every_epoch', type=int, default=1, help="Decay interval (every N epochs)")
    parser.add_argument('--dropout_rate', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--model_type', type=str, default='deberta', help="Model type (e.g., deberta)")

    args = parser.parse_args()

    result = run(args.task_type, args.train_path, args.test_path, args.model_type, args.text_type,
                 args.dropout_rate,
                 args.is_train, args.ckpt_path, args.batch_size, args.epoch, args.lr, args.lr_decay,
                 args.every_epoch, args.alpha)
