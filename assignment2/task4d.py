import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    neurons_per_layer = [64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid = True,
        use_improved_weight_init = True,
        use_relu = False
    )
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    neurons_per_layer = [60, 60, 10]
    model_2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid = True,
        use_improved_weight_init = True,
        use_relu = False
    )
    trainer_2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_2, val_history_2 = trainer_2.train(num_epochs)

    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    #neurons_per_layer = [64, 64, 10]
    model_3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu = True
    )
    trainer_3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_3, val_history_3 = trainer_3.train(num_epochs)


    plt.subplot(1, 2, 1)
    plt.ylim([0, 0.55])
    utils.plot_loss(train_history["loss"],"1 hidden train", npoints_to_average=10)
    utils.plot_loss(val_history["loss"],"1 hidden val")
    utils.plot_loss(train_history_2["loss"],"2 hidden train", npoints_to_average=10)
    utils.plot_loss(val_history_2["loss"],"2 hidden val")
    utils.plot_loss(train_history_3["loss"],"relu 10 hidden train", npoints_to_average=10)
    utils.plot_loss(val_history_3["loss"],"relu 10 hidden val")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.97, 1.0])
    utils.plot_loss(val_history["accuracy"], "1 hidden")
    utils.plot_loss(val_history_2["accuracy"], "2 hidden")
    utils.plot_loss(val_history_3["accuracy"], "relu 10 hidden")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4f.png")
    plt.show()

if __name__ == "__main__":
    main()
