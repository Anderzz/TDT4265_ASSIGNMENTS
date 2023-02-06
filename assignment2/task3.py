import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    # this is for task 4 a) and b).
    # the commented out code below was used for task 3.

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    model_32 = SoftmaxModel(
        [32, 10],
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_32 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32 = trainer_32.train(num_epochs)

    model_128 = SoftmaxModel(
        [128, 10],
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_128 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128 = trainer_128.train(num_epochs)

    plt.subplot(1, 2, 1)
    plt.ylim([0, 0.55])
    utils.plot_loss(train_history["loss"],"64 units", npoints_to_average=10)
    utils.plot_loss(train_history_32["loss"],"32 units", npoints_to_average=10)
    utils.plot_loss(train_history_128["loss"],"128 units", npoints_to_average=10)
    plt.ylabel("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.97, 1.0])
    utils.plot_loss(val_history["accuracy"], "64 units")
    utils.plot_loss(val_history_32["accuracy"], "32 units")
    utils.plot_loss(val_history_128["accuracy"], "128 units")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4a.png")
    plt.show()

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # use_improved_weight_init = True

    # # Train a new model with new parameters
    # model_glorot = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init,
    #     use_relu)
    # trainer_glorot = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_glorot, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_glorot, val_history_glorot = trainer_glorot.train(
    #     num_epochs)

    # use_improved_weight_init = True
    # use_improved_sigmoid = True

    # # Train a new model with new parameters
    # model_glorot_sigmoid = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init,
    #     use_relu)
    # trainer_glorot_sigmoid = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_glorot_sigmoid, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_glorot_sigmoid, val_history_glorot_sigmoid = trainer_glorot_sigmoid.train(
    #     num_epochs)

    # use_improved_weight_init = True
    # use_improved_sigmoid = True
    # use_momentum = True
    # learning_rate = .02

    # # Train a new model with new parameters
    # model_momentum = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init,
    #     use_relu)
    # trainer_momentum = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_momentum, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_momentum, val_history_momentum = trainer_momentum.train(
    #     num_epochs)

    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history["loss"],
    #                 "Task 2 Model", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_glorot["loss"], "Task 2 Model - Xavier initialization", npoints_to_average=10)
    # # add glorot_sigmoid
    # utils.plot_loss(
    #     train_history_glorot_sigmoid["loss"], "Task 2 Model - Xavier + improved Sigmoid", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_momentum["loss"], "Task 2 Model - Momentum + rest", npoints_to_average=10)
    # plt.ylabel("Training Loss")
    # plt.legend()
    
    # #plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # #plt.ylim([0.85, .95])
    # utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    # utils.plot_loss(
    #     val_history_glorot["accuracy"], "Task 2 Model - Xavier initialization")
    # utils.plot_loss(
    #     val_history_glorot_sigmoid["accuracy"], "Task 2 Model - Xavier + improved Sigmoid")
    # utils.plot_loss(
    #     val_history_momentum["accuracy"], "Task 2 Model - Momentum + rest")
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # plt.show()
    # plt.savefig("task3.png")


if __name__ == "__main__":
    main()
