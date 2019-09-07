import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import nets.Network as Segception
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics, evaluate
import argparse

# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path",
                        default=r'datas/lanes')
    parser.add_argument("--model_path", help="Model path", default='weights/model')
    parser.add_argument("--n_classes", help="number of classes to classify", default=5)
    parser.add_argument("--batch_size", help="batch size", default=8)
    parser.add_argument("--epochs", help="number of epochs to train", default=0)  # 500
    parser.add_argument("--width", help="number of epochs to train", default=352)
    parser.add_argument("--height", help="number of epochs to train", default=224)
    parser.add_argument("--lr", help="init learning rate", default=1e-3)
    parser.add_argument("--n_gpu", help="number of the gpu", default=4)
    args = parser.parse_args()

    n_gpu = int(args.n_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    n_classes = int(args.n_classes)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    width = int(args.width)
    height = int(args.height)
    lr = float(args.lr)

    channels = 1  # input of 6 channels
    channels_image = 1
    channels_events = channels - channels_image
    folder_best_model = args.model_path
    name_best_model = os.path.join(folder_best_model, 'best')  # vita
    dataset_path = args.dataset
    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, channels=channels_image, channels_events=channels_events)

    #if not os.path.exists(folder_best_model):
    #    os.makedirs(folder_best_model)

    # build model and optimizer
    model = Segception.Segception_small(num_classes=n_classes, weights=None, input_shape=(None, None, channels))

    # optimizer
    learning_rate = tfe.Variable(lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Init models (optional, just for get_params function)
    init_model(model, input_shape=(batch_size, width, height, channels))

    variables_to_restore = model.variables  # [x for x in model.variables if 'block1_conv1' not in x.name]
    variables_to_save = model.variables
    variables_to_optimize = model.variables

    # Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
    saver_model = tfe.Saver(var_list=variables_to_save)
    restore_model = tfe.Saver(var_list=variables_to_restore)

    # restore if model saved and show number of params
    restore_state(restore_model, name_best_model)
    get_params(model)

    # Test best model
    print('Testing model')
    evaluate(loader, model, loader.n_classes, train=True, flip_inference=True, scales=[1, 0.75, 1.5], write_images=True, preprocess_mode=None)

    #test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True,
    #                                  scales=[1, 0.75, 1.5],
    #                                  write_images=True, preprocess_mode=None)
    #print('Test accuracy: ' + str(test_acc.numpy()))
    #print('Test miou: ' + str(test_miou))
